import bitsandbytes as bnb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig 
from peft import PeftModel
import os

os.environ['CURL_CA_BUNDLE'] = ''

max_memory = f'{39960}MB'
token = "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"

id2label = {0: "severe", 1: "moderate", 2: "no-depression"}
label2id = {"severe": 0, "moderate": 1, "no-depression": 2}

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
        token=token,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
        # cache_dir="/home/hpc/empk/empk004h/.cache/huggingface"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=token) # 

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def llama_inference(texts, llama_model_name, llama_adapter_dir):
    # texts is simply an array of texts

    bnb_config = create_bnb_config()
    model, tokenizer = load_model(llama_model_name, bnb_config)

    model = PeftModel.from_pretrained(model, llama_adapter_dir)

    outputs = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to('cuda')

        # predict with trainer.model
        output = model(**inputs)
        logits = output.logits
        # pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]

        outputs.append(probs)

    return outputs

def run_inference(texts):
    model_path = "/home/hpc/empk/empk004h/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852"

    output = llama_inference(texts, model_path, './EDI_finetune_llama_output/checkpoint-15000')
    print('output: ', output)
    return output

if __name__ == '__main__':
    texts = ["I am feeling happy today", "I feel pretty sad", "There is a dog in the house"]
    run_inference(texts)

