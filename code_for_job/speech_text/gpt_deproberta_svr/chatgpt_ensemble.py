import os
import openai
import pandas as pd
import requests
from transformers import GPT2TokenizerFast

openai.api_key = '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' # for my personal account

label_dir = '/home/vault/empk/empk004h/DAIC/labels/'
openface_features_dir = '/home/vault/empk/empk004h/DAIC/DAIC_openface_features/'
splitted_dataset_dir = '/home/vault/empk/empk004h/DAIC/splitted_dataset/'

directory_path = '/home/vault/empk/empk004h/DAIC/transcripts_from_whisper/'
transcripts_df = pd.DataFrame(columns=["id", "text"])
# ------------------------------------------------------------------------------------------------
# extract IDs from filenames and text from files
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_id = filename[:3]
        with open(os.path.join(directory_path, filename), "r") as file:
            file_contents = file.read()
        transcripts_df = transcripts_df.append({"id": file_id, "text": file_contents}, ignore_index=True)

transcripts_df["id"] = transcripts_df["id"].astype("int64")

labels_train_df = pd.read_csv('/home/vault/empk/empk004h/DAIC/labels/train_split.csv')
labels_train_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_dev_df = pd.read_csv('/home/vault/empk/empk004h/DAIC/labels/dev_split.csv')
labels_dev_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_test_df = pd.read_csv('/home/vault/empk/empk004h/DAIC/labels/test_split.csv')
labels_test_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']
# ------------------------------------------------------------------------------------------------
# merge dataframes on ID
df_train = pd.merge(labels_train_df, transcripts_df, on='id')
df_dev = pd.merge(labels_dev_df, transcripts_df, on='id')
df_test = pd.merge(labels_test_df, transcripts_df, on='id')

df_dev = df_dev.sort_values(by="id")
df_train = df_train.sort_values(by="id")
df_test = df_test.sort_values(by="id")

print('def_dev: ', df_dev.head())
print('max PHQ score in df_train: ', max(df_train['PHQ_Score']))
# ------------------------------------------------------------------------------------------------
# !!! These are the prompts that we will use for gpt-3.5-turbo API !!!
prompt1 = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point.""" 
# we already have the results of prompt2
prompt2 = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point. It is very essential that you write your answer in the first-person perspective, as if the interviewee is narrating about himself or herself. """
prompt3 = """ After reading the interview, briefly summarize the main aspects that pertain to the person's depression. """
prompt4 = """ Based on the interview, highlight the key factors that might be indicative of the interviewee's depression. """
prompt5 = """ Analyzing the interview, condense the significant points that relate to the person's depression. """
prompt6 = """ Your task is to summarize the interviewee's main points that could be linked to their depression. Keep it concise. """
prompt7 = """ After reading the interview, identify and summarize the main challenges or difficulties the interviewee faces that are indicative of depression. """
prompt8 = """ Based on the interview, provide a concise analysis of the interviewee's emotional state and behaviors that may indicate the presence of depression. """
prompt9 = """ Read the interview carefully and extract the most significant indicators of depression exhibited by the interviewee. Summarize them concisely. """
prompt10 = """ Your task is to analyze the interviewee's responses and highlight the key signs or symptoms of depression that are evident in the interview. """
prompt11 = """ Summarize the key points related to the interviewee's depression in a concise manner. """
prompt12 = """ Provide a concise summary of the interviewee's points that may be relevant to their depression. """
prompt13 = """ Read the interview and provide a brief summary focusing on the aspects that may indicate the interviewee's depression. """
prompt14 = """ After reading the interview, extract and summarize the key elements that could be connected to the interviewee's depression. """
# ------------------------------------------------------------------------------------------------
# GPT-3 cost estimator for all completions
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")   # should it be gpt2? ask Reza

token_counts = []
# for all tokens in all transcripts in dev, train, and test sets
for text in df_dev['text'].tolist() + df_train['text'].tolist() + df_test['text'].tolist():
    # tokenize text
    tokens = tokenizer.encode(text)
    # tokenize prompt
    prompt_tokens = tokenizer.encode(prompt3)

    # append token count to list
    token_counts.append(len(tokens) + len(prompt_tokens) + 512)

# Final cost estimataion
print(f'Estimated cost: ${(sum(token_counts) / 1000 * 0.002)}')
# ------------------------------------------------------------------------------------------------
# making the prompt and sending requests to the API
def make_prompt(prompt, text):
    return """{}\nHere is the interview between triple backticks:\n```{}```""".format(prompt, text)

# extract ChatGPT completions for each transcript
def get_completions(prompt, text):
    prompt = make_prompt(prompt, text)
    # send request to API
    response = requests.post(
        "http://localhost:4100/parallel-requests",
        json={"prompts": prompt}
    )
    # get response
    response = response.json()
    return response["response"]

def get_completions_batch(prompt, texts):
    prompts = [make_prompt(prompt, text) for text in texts]
    prompts = "---".join(prompts)

    # send request to API
    response = requests.post(
        "http://localhost:4100/parallel-requests",
        json={"prompts": prompts}
    )

    # get response
    response = response.json()
    return response["response"]
# ------------------------------------------------------------------------------------------------
# Explore the result for one example ID
# get completion for the first transcript in the dev set
# id = 300
# print('ID: ', id)
# # find dv_dev row with id
# row = df_dev[df_dev['id'] == id]
# # get the value of the text column
# text = row['text'].values[0]
# print('------------------------------------')
# print('This is the completion:')
# completions = get_completions(prompt3, text)
# print(completions)
# ------------------------------------------------------------------------------------------------
