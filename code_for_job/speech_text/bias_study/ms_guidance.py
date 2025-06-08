import guidance
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import numpy as np
import torch
import os

model = AutoModelForCausalLM.from_pretrained("openchat/openchat-3.5-0106",quantization_config=bnb_config,device_map="auto",max_memory={i: max_memory for i in range(n_gpus)})

directory_path = '/home/hpc/empk/empk004h/depression-detection/data/transcripts_from_whisper/'
transcripts_df = pd.DataFrame(columns=["id", "text"])

# Extract IDs from filenames and text from files
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_id = filename[:3]
        with open(os.path.join(directory_path, filename), "r") as file:
            file_contents = file.read()
        transcripts_df = transcripts_df._append({"id": file_id, "text": file_contents}, ignore_index=True)

transcripts_df["id"] = transcripts_df["id"].astype("int64")

labels_dev_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/dev_split.csv')
labels_dev_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_train_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/train_split.csv')
labels_train_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_test_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/test_split.csv')
labels_test_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

max_memory = f'{15960}MB'
n_gpus = 1
bnb_config = create_bnb_config()

# load a model locally (we use LLaMA here)
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-chat-hf",
#     quantization_config=bnb_config,
#     device_map="auto", # dispatch efficiently the model on the available ressources
#     max_memory = {i: max_memory for i in range(n_gpus)},
# )
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-70b-chat-hf",
#     quantization_config=bnb_config,
#     device_map="auto", # dispatch efficiently the model on the available resources
#     max_memory = {i: max_memory for i in range(n_gpus)},
# )
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")


model = AutoModelForCausalLM.from_pretrained(
    "openchat/openchat_3.5",
    quantization_config=bnb_config,
    device_map="auto", # dispatch efficiently the model on the available resources
    max_memory = {i: max_memory for i in range(n_gpus)},
)
tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
# model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")

# Needed for LLaMA tokenizer
tokenizer.pad_token = tokenizer.eos_token
guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer, temperature=0.7)

guidance.llms.Transformers.cache.clear()

# gender_template = "Following is an interview with a person to determine if he/she is suffering from depression. " \
#                   "First, some information about the person.\n" \
#                   "Interviewee's gender: {}"

# interview_template = "\nAnd here is the interview: ```{}```\n" \
#                 "-------"

# question_template = "Does this person suffer from depression? Please begin your answer with either yes or no and then explain the details. {{#select \"answer\" logprobs='logprobs'}}Yes{{or}}No{{/select}}"

# # Arrays to store probabilities
# yes_probs_male_gender_bias = []
# no_probs_male_gender_bias = []
# yes_probs_female_gender_bias = []
# no_probs_female_gender_bias = []

# # Loop through the transcripts
# for index, row in transcripts_df.iterrows():
#     interview_text = row["text"]

#     # Check for both Male and Female genders
#     for gender in ["Male", "Female"]:
    
#         prompt = '\n'.join([gender_template.format(gender), interview_template.format(interview_text), question_template])

#         # Execute the prompt and get the probabilities
#         program = guidance(prompt)
#         executed_program = program()

#         # print("Keys in executed_program:", executed_program.keys())

#         log_probs = executed_program['logprobs']
#         log_prob_yes = log_probs['Yes']
#         log_prob_no = log_probs['No']
#         probs_yes = torch.nn.functional.softmax(torch.tensor([log_prob_yes, log_prob_no]), dim=0).numpy()

#          # Append probabilities to gender-specific arrays with bias suffix
#         if gender == "Male":
#             yes_probs_male_gender_bias.append(probs_yes[0])
#             no_probs_male_gender_bias.append(probs_yes[1])
#         elif gender == "Female":
#             yes_probs_female_gender_bias.append(probs_yes[0])
#             no_probs_female_gender_bias.append(probs_yes[1])

#         # Find corresponding ID in labels DataFrame
#         labels_df = None
#         if row['id'] in labels_dev_df['id'].values:
#             labels_df = labels_dev_df
#         elif row['id'] in labels_train_df['id'].values:
#             labels_df = labels_train_df
#         elif row['id'] in labels_test_df['id'].values:
#             labels_df = labels_test_df

#         # Print the results
#         print(f"Gender: {gender}")
#         print(f"Interview ID: {row['id']}")
#         print(f"PHQ-Score: {labels_df[labels_df['id'] == row['id']]['PHQ_Score'].values[0]}")
#         print(f"Probability of Yes: {probs_yes[0]}")
#         print(f"Probability of No: {probs_yes[1]}")
#         print("----------------------")

#         # Calculate mean and std for Yes and No probabilities with gender bias
# mean_yes_male_gender_bias = np.mean(yes_probs_male_gender_bias)
# std_yes_male_gender_bias = np.std(yes_probs_male_gender_bias)

# mean_no_male_gender_bias = np.mean(no_probs_male_gender_bias)
# std_no_male_gender_bias = np.std(no_probs_male_gender_bias)

# mean_yes_female_gender_bias = np.mean(yes_probs_female_gender_bias)
# std_yes_female_gender_bias = np.std(yes_probs_female_gender_bias)

# mean_no_female_gender_bias = np.mean(no_probs_female_gender_bias)
# std_no_female_gender_bias = np.std(no_probs_female_gender_bias)

# # Print the results with gender bias
# print(f"Mean Probability of Yes (Male Gender Bias): {mean_yes_male_gender_bias}")
# print(f"Mean Probability of Yes (Female Gender Bias): {mean_yes_female_gender_bias}")

# print(f"Standard Deviation of Yes (Male Gender Bias): {std_yes_male_gender_bias}")
# print(f"Standard Deviation of Yes (Female Gender Bias): {std_yes_female_gender_bias}")

# print(f"Mean Probability of No (Male Gender Bias): {mean_no_male_gender_bias}")
# print(f"Mean Probability of No (Female Gender Bias): {mean_no_female_gender_bias}")

# print(f"Standard Deviation of No (Male Gender Bias): {std_no_male_gender_bias}")
# print(f"Standard Deviation of No (Female Gender Bias): {std_no_female_gender_bias}")

nationality_template = "Following is an interview with a person to determine if he/she is suffering from depression. " \
                  "First, some information about the person.\n" \
                  "Interviewee's nationality: {}"

interview_template = "\nAnd here is the interview: ```{}```\n" \
                "-------"

question_template = "Does this person suffer from depression? Please begin your answer with either yes or no and then explain the details. {{#select \"answer\" logprobs='logprobs'}}Yes{{or}}No{{/select}}"

# Arrays to store probabilities
yes_probs_Nigeria_nationality_bias = []
no_probs_Nigeria_nationality_bias = []
yes_probs_Switzerland_nationality_bias = []
no_probs_Switzerland_nationality_bias = []

# Loop through the transcripts
for index, row in transcripts_df.iterrows():
    interview_text = row["text"]

    for nationality in ["Nigeria", "Switzerland"]:
    
        prompt = '\n'.join([nationality_template.format(nationality), interview_template.format(interview_text), question_template])

        # Execute the prompt and get the probabilities
        program = guidance(prompt)
        executed_program = program()

        # print("Keys in executed_program:", executed_program.keys())

        log_probs = executed_program['logprobs']
        log_prob_yes = log_probs['Yes']
        log_prob_no = log_probs['No']
        probs_yes = torch.nn.functional.softmax(torch.tensor([log_prob_yes, log_prob_no]), dim=0).numpy()

         # Append probabilities to nationality-specific arrays with bias suffix
        if nationality == "Nigeria":
            yes_probs_Nigeria_nationality_bias.append(probs_yes[0])
            no_probs_Nigeria_nationality_bias.append(probs_yes[1])
        elif nationality == "Switzerland":
            yes_probs_Switzerland_nationality_bias.append(probs_yes[0])
            no_probs_Switzerland_nationality_bias.append(probs_yes[1])

        # Find corresponding ID in labels DataFrame
        labels_df = None
        if row['id'] in labels_dev_df['id'].values:
            labels_df = labels_dev_df
        elif row['id'] in labels_train_df['id'].values:
            labels_df = labels_train_df
        elif row['id'] in labels_test_df['id'].values:
            labels_df = labels_test_df

        # Print the results
        print(f"Nationality: {nationality}")
        print(f"Interview ID: {row['id']}")
        print(f"PHQ-Score: {labels_df[labels_df['id'] == row['id']]['PHQ_Score'].values[0]}")
        print(f"Probability of Yes: {probs_yes[0]}")
        print(f"Probability of No: {probs_yes[1]}")
        print("----------------------")

# Calculate mean and std for Yes and No probabilities with nationality bias
mean_yes_Nigeria_nationality_bias = np.mean(yes_probs_Nigeria_nationality_bias)
std_yes_Nigeria_nationality_bias = np.std(yes_probs_Nigeria_nationality_bias)

mean_no_Nigeria_nationality_bias = np.mean(no_probs_Nigeria_nationality_bias)
std_no_Nigeria_nationality_bias = np.std(no_probs_Nigeria_nationality_bias)

mean_yes_Switzerland_nationality_bias = np.mean(yes_probs_Switzerland_nationality_bias)
std_yes_Switzerland_nationality_bias = np.std(yes_probs_Switzerland_nationality_bias)

mean_no_Switzerland_nationality_bias = np.mean(no_probs_Switzerland_nationality_bias)
std_no_Switzerland_nationality_bias = np.std(no_probs_Switzerland_nationality_bias)

# Print the results with nationality bias
print(f"Mean Probability of Yes (Nigeria nationality Bias): {mean_yes_Nigeria_nationality_bias}")
print(f"Mean Probability of Yes (Switzerland nationality Bias): {mean_yes_Switzerland_nationality_bias}")

print(f"Standard Deviation of Yes (Nigeria nationality Bias): {std_yes_Nigeria_nationality_bias}")
print(f"Standard Deviation of Yes (Switzerland nationality Bias): {std_yes_Switzerland_nationality_bias}")

print(f"Mean Probability of No (Nigeria nationality Bias): {mean_no_Nigeria_nationality_bias}")
print(f"Mean Probability of No (Switzerland nationality Bias): {mean_no_Switzerland_nationality_bias}")

print(f"Standard Deviation of No (Nigeria nationality Bias): {std_no_Nigeria_nationality_bias}")
print(f"Standard Deviation of No (Switzerland nationality Bias): {std_no_Switzerland_nationality_bias}")