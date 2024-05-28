from alignment import H4ArgumentParser, ModelArguments, DataArguments, DPOConfig
from transformers import AutoTokenizer
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams
import llm_blender
from tqdm import tqdm
import numpy as np

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")
@torch.no_grad()
def generate_response_vllm(model, tokenizer, dataset):
    with torch.inference_mode():
        sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1024, stop=tokenizer.eos_token, skip_special_tokens=True)
        chosen_messages = dataset['chosen']
        chat_prompts = []
        for chosen_message in chosen_messages:
            prompt_message = chosen_message[:-1]
            chat_prompts.append(tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True))
        existing_chosen_responses = []
        for idx, r in enumerate(dataset['chosen']):
            res_content = r[1]["content"]
            existing_chosen_responses.append(res_content)
        existing_rejected_responses = []
        for idx, r in enumerate(dataset['rejected']):
            res_content = r[1]["content"]
            existing_rejected_responses.append(res_content)

        responses = model.generate(chat_prompts, sampling_params)
        responses_list = [response.outputs[0].text.strip() for response in responses]
        dataset = dataset.add_column("reference_response", responses_list)
        candidates_texts = [[responses_list[idx]] + [existing_chosen_responses[idx]] +
                            [existing_rejected_responses[idx]] for idx in range(len(responses_list))]
        prompts = dataset['prompt']
        rank = blender.rank(prompts, candidates_texts, return_scores=False)
        chosen_indices = np.argmin(rank, axis=1)
        rejected_indices = np.argmax(rank, axis=1)
        chosen_texts = np.array(candidates_texts)[np.arange(len(candidates_texts)), chosen_indices]
        rejected_texts = np.array(candidates_texts)[np.arange(len(candidates_texts)), rejected_indices]
        chosen_responses_dict = np.array([{"content": res, "role": "assistant"} for res in chosen_texts])
        rejected_responses_dict = np.array([{"content": res, "role": "assistant"} for res in rejected_texts])
        chosen_np = np.array(dataset['chosen'])
        reject_np = np.array(dataset['rejected'])
        update_chosen_column = np.column_stack((chosen_np[:, 0], chosen_responses_dict))  # -1 for Gemma
        update_reject_column = np.column_stack((reject_np[:, 0], rejected_responses_dict))
    dataset = dataset.remove_columns(["chosen", "rejected", "score_chosen", "score_rejected"])
    dataset = dataset.add_column("chosen", update_chosen_column.tolist())
    dataset = dataset.add_column("rejected", update_reject_column.tolist())
    return dataset

if __name__ == "__main__":
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()
    if type(data_args.dataset_mixer) == str:
        data_args.dataset_mixer = eval(data_args.dataset_mixer)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    try:
        ref_model = LLM(model=model_args.model_name_or_path, tokenizer=model_args.model_name_or_path,
                        gpu_memory_utilization=0.6, swap_space=4, tensor_parallel_size=torch.cuda.device_count(),
                        trust_remote_code=True, dtype="auto")
        updated_dataset_name, iter_str = data_args.dataset_mixer["updated"].split("_iter")
        original_test_dataset = load_dataset(data_args.dataset_mixer["original"], split=data_args.dataset_splits[1])
        new_test_dataset = generate_response_vllm(ref_model, tokenizer, original_test_dataset)
        new_test_dataset.push_to_hub(updated_dataset_name, private=False, split="test_prefs"+iter_str)
        original_train_dataset = load_dataset(data_args.dataset_mixer["original"], split=data_args.dataset_splits[0])
        new_train_dataset = generate_response_vllm(ref_model, tokenizer, original_train_dataset)
        new_train_dataset.push_to_hub(updated_dataset_name, private=False, split="train_prefs"+iter_str)
    except Exception as e:
        print(e)