import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
import torch
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPTQConfig
from datasets import load_dataset
import json
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding
import csv
from datasets import load_dataset
import os

# setting a different cache dir for hugging_face
# os.environ['HF_HOME'] = '/home/woody/empk/empk004h/huggingface_cache/.cache'

os.environ['CURL_CA_BUNDLE'] = ''

token = "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"

training_data_path = "/home/hpc/empk/empk004h/depression-detection/data/EDI_deproberta/preprocessed_dataset/train.csv"
validation_data_path = "/home/hpc/empk/empk004h/depression-detection/data/EDI_deproberta/preprocessed_dataset/dev.csv"
max_memory = f'{79960}MB'

training_data = []
validation_data = []

# load dataset using csv
with open(training_data_path, 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Check if the row is not the header
        if row[2].lower() != 'labels':
            # Each row is a list of values representing the columns
            training_data.append({"text": row[1], "label": int(row[2])})
            # if row[2].lower() != 1 and row[2].lower() != 0:
            #     print('Wrong label: if row[2].lower() = ', row[2].lower())

    # remove the header
    training_data.pop(0)

#print('training_data: ', training_data)

with open(validation_data_path, 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        if row[2].lower() != 'labels':
            # Each row is a list of values representing the columns
            validation_data.append({"text": row[1], "label": int(row[2])})

    # remove the header
    validation_data.pop(0)

# export them as json files
with open('training_data_tmp.json', 'w') as f:
    json.dump(training_data, f)

with open('validation_data_tmp.json', 'w') as f:
    json.dump(validation_data, f)

id2label = {0: "severe", 1: "moderate", 2: "no-depression"}
label2id = {"severe": 0, "moderate": 1, "no-depression": 2}

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()

    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5,resume_download=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
        token=token,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)
    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)

dataset = load_dataset('json', data_files='./training_data_tmp.json')
dataset = dataset['train']

dataset_val = load_dataset('json', data_files='./validation_data_tmp.json')
dataset_val = dataset_val['train']      # is it corrcet? train?

tokenized_train = dataset.map(preprocess_function, batched=True)
tokenized_val = dataset_val.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# accuracy = evaluate.load("accuracy")
accuracy = evaluate.load("evaluate-main/metrics/accuracy/accuracy.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=8,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="SEQ_CLS",
    )

    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "HuggingFaceH4/zephyr-7b-beta"

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)

model.gradient_checkpointing_enable()

# 2 - Using the prepare_model_for_kbit_training method from PEFT
model = prepare_model_for_kbit_training(model)

# Get lora module names
modules = find_all_linear_names(model)

# Create PEFT config for these modules and wrap the model to PEFT
peft_config = create_peft_config(modules)
model = get_peft_model(model, peft_config)

# Print information about the percentage of trainable parameters
print_trainable_parameters(model)

training_args = TrainingArguments(
    output_dir="./EDI_finetune_llama_output",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    load_best_model_at_end=True,
    fp16=True,
    # push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    # tokenizer=tokenizer,
    # data_collator=data_collator,
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=1),
    compute_metrics=compute_metrics,
)

model.config.use_cache = False

trainer.train()