import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel

hf_cache_dir = "/home/hpc/empk/empk004h/.cache/huggingface"
# token = "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"
token = "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"

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def load_model_and_tokenizer(model_name, bnb_config, n_gpus, max_memory):
    model = AutoModel.from_pretrained(
        '/home/hpc/empk/empk004h/.cache/huggingface/hub/models--openchat--openchat-3.5-0106/snapshots/dfcf6be1e44eb54db7af0d05d2760fb1d4969845',
        #model_name,
        quantization_config=bnb_config,
        #device_map="auto",
        max_memory={i: max_memory for i in range(n_gpus)},
        #token=token,
        #cache_dir=hf_cache_dir
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained('/home/hpc/empk/empk004h/.cache/huggingface/hub/models--openchat--openchat-3.5-0106/snapshots/dfcf6be1e44eb54db7af0d05d2760fb1d4969845', local_files_only=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


if __name__ == "__main__":
    
    # for model in models:
    n_gpus = 1
    max_memory = '15960MB'
    bnb_config = create_bnb_config()
    # Load model and tokenizer
    guidance = load_model_and_tokenizer("openchat/openchat-3.5-0106", bnb_config, n_gpus, max_memory)

    