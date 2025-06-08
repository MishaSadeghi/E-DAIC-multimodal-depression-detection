import os
import re
import glob
import joblib
import openai
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers import BertForSequenceClassification, BertTokenizer

directory_path = '/home/hpc/empk/empk004h/depression-detection/data/transcripts_from_whisper/'
transcripts_df = pd.DataFrame(columns=["id", "text"])
# ------------------------------------------------------------------------------------------------
# Extract IDs from filenames and text from files
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_id = filename[:3]
        with open(os.path.join(directory_path, filename), "r") as file:
            file_contents = file.read()
        transcripts_df = transcripts_df.append({"id": file_id, "text": file_contents}, ignore_index=True)

transcripts_df["id"] = transcripts_df["id"].astype("int64")

labels_dev_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/dev_split.csv')
labels_dev_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_train_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/train_split.csv')
labels_train_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']

labels_test_df = pd.read_csv('/home/hpc/empk/empk004h/depression-detection/data/labels/test_split.csv')
labels_test_df.columns = ['id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']
# ------------------------------------------------------------------------------------------------
# Merge dataframes on ID
df_train = pd.merge(labels_train_df, transcripts_df, on='id')
df_dev = pd.merge(labels_dev_df, transcripts_df, on='id')
df_test = pd.merge(labels_test_df, transcripts_df, on='id')

df_dev = df_dev.sort_values(by="id")
df_train = df_train.sort_values(by="id")
df_test = df_test.sort_values(by="id")

print('def_dev: ', df_dev.head())
print('max PHQ score in df_train: ', max(df_train['PHQ_Score']))
# ------------------------------------------------------------------------------------------------
# Reading from the CSV files so we don't have to send requests over and over to gpt

# Define the input directory path
input_directory = "/home/hpc/empk/empk004h/depression-detection/notebooks/prompts_outputs/"
prompt_number = 2

# Define the folder path for the prompt number
prompt_directory = os.path.join(input_directory, str(prompt_number))

# Read the CSV files for each dataframe
dev_filepath = os.path.join(prompt_directory, f"dev_prompt{prompt_number}.csv")
train_filepath = os.path.join(prompt_directory, f"train_prompt{prompt_number}.csv")
test_filepath = os.path.join(prompt_directory, f"test_prompt{prompt_number}.csv")

# Read the dataframes from CSV files each have these columns: 'id', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity', 'text', 'completions'
df_dev_prompt2 = pd.read_csv(dev_filepath)
df_train_prompt2 = pd.read_csv(train_filepath)
df_test_prompt2 = pd.read_csv(test_filepath)

print(df_dev_prompt2.head())

# ------------------------------------------------------------------------------------

# # Load BERT model and tokenizer
# bert_model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
# bert_model = AutoModel.from_pretrained(bert_model_name)

# Preprocess completions text and generate BERT embeddings
def preprocess_text(text):
    # Preprocess your text here if needed
    return text

def generate_bert_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings

# # Apply preprocessing and generate BERT embeddings for train, dev, and test data
# train_embeddings = df_train_prompt2["completions"].apply(preprocess_text).apply(generate_bert_embeddings).tolist()
# dev_embeddings = df_dev_prompt2["completions"].apply(preprocess_text).apply(generate_bert_embeddings).tolist()
# test_embeddings = df_test_prompt2["completions"].apply(preprocess_text).apply(generate_bert_embeddings).tolist()

# # Convert embeddings to NumPy arrays
# train_features = torch.cat(train_embeddings).numpy()
# dev_features = torch.cat(dev_embeddings).numpy()
# test_features = torch.cat(test_embeddings).numpy()

# # Define the target variables
# train_labels = df_train_prompt2["PHQ_Score"]
# dev_labels = df_dev_prompt2["PHQ_Score"]
# test_labels = df_test_prompt2["PHQ_Score"]

# # Train an SVR model
# svr_model = SVR()
# svr_model.fit(train_features, train_labels)

# # Predict on dev and test sets
# dev_predictions = svr_model.predict(dev_features)
# test_predictions = svr_model.predict(test_features)

# # Evaluate the model
# dev_mae = mean_absolute_error(dev_labels, dev_predictions)
# test_mae = mean_absolute_error(test_labels, test_predictions)

# print("Dev MAE:", dev_mae)
# print("Test MAE:", test_mae)

# Results: 
# Dev MAE: 4.488440202595507
# Test MAE: 5.3719295201739765

# ------------------------------------------------------------------------------------
# Load a pre-trained BERT model and tokenizer
# model_name = "bert-base-uncased"
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
# tokenizer = BertTokenizer.from_pretrained(model_name)

# # Fine-tuning data
# train_texts = df_train_prompt2["completions"].apply(preprocess_text).tolist()
# train_labels = df_train_prompt2["PHQ_Score"].tolist()

# dev_texts = df_dev_prompt2["completions"].apply(preprocess_text).tolist()
# dev_labels = df_dev_prompt2["PHQ_Score"].tolist()

# test_texts = df_test_prompt2["completions"].apply(preprocess_text).tolist()
# test_labels = df_test_prompt2["PHQ_Score"].tolist()

# # Tokenize the text and create DataLoader
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
# train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels))
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# # Define DataLoader for dev and test sets
# dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, return_tensors="pt")
# dev_dataset = TensorDataset(dev_encodings["input_ids"], dev_encodings["attention_mask"], torch.tensor(dev_labels))
# dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False)  # No need to shuffle for dev

# test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")
# test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_labels))
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # No need to shuffle for test

# # Fine-tune the BERT model
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# for epoch in range(20):
#     total_loss = 0.0
#     for step, batch in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         labels = batch[2].unsqueeze(1).float()
        
#         outputs = model(**inputs)
        
#         loss = torch.nn.MSELoss()(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         if step % 10 == 0:
#             print(f"Epoch [{epoch + 1}/20] | Step [{step + 1}/{len(train_dataloader)}] | Loss: {loss.item():.4f}")
    
#     avg_epoch_loss = total_loss / len(train_dataloader)
#     print(f"Epoch [{epoch + 1}/20] | Average Loss: {avg_epoch_loss:.4f}")
    
#     # Evaluate on train set
#     train_predictions = []
#     with torch.no_grad():
#         for train_batch in train_dataloader:
#             train_inputs = {"input_ids": train_batch[0], "attention_mask": train_batch[1]}
#             train_outputs = model(**train_inputs)
#             train_predictions.append(train_outputs.logits.numpy().flatten())
            
#         train_predictions = np.concatenate(train_predictions)
#         train_mae = mean_absolute_error(train_labels, train_predictions)
#         print(f"Epoch [{epoch + 1}/20] | Train MAE: {train_mae:.4f}")
        
#     # Evaluate on dev set
#     dev_predictions = []
#     with torch.no_grad():
#         for dev_batch in dev_dataloader:
#             dev_inputs = {"input_ids": dev_batch[0], "attention_mask": dev_batch[1]}
#             dev_outputs = model(**dev_inputs)
#             dev_predictions.append(dev_outputs.logits.numpy().flatten())
            
#         dev_predictions = np.concatenate(dev_predictions)
#         dev_mae = mean_absolute_error(dev_labels, dev_predictions)
#         print(f"Epoch [{epoch + 1}/20] | Dev MAE: {dev_mae:.4f}")

# # Predict on train set
# train_logits = []

# # Predict on dev and test sets
# dev_logits = []
# test_logits = []

# with torch.no_grad():
#     for batch in train_dataloader:
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         outputs = model(**inputs)
#         train_logits.append(outputs.logits)

# train_logits = torch.cat(train_logits)

# # Convert logits to predictions
# train_predictions = train_logits.numpy().flatten()

# # Compute MAE on train set
# train_mae = mean_absolute_error(train_labels, train_predictions)
# print("Train MAE:", train_mae)

# with torch.no_grad():
#     for batch in dev_dataloader:
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         outputs = model(**inputs)
#         dev_logits.append(outputs.logits)

#     for batch in test_dataloader:
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         outputs = model(**inputs)
#         test_logits.append(outputs.logits)

# dev_logits = torch.cat(dev_logits)
# test_logits = torch.cat(test_logits)

# # Convert logits to predictions
# dev_predictions = dev_logits.numpy().flatten()
# test_predictions = test_logits.numpy().flatten()

# # Evaluate the model
# dev_mae = mean_absolute_error(dev_labels, dev_predictions)
# test_mae = mean_absolute_error(test_labels, test_predictions)

# print("Dev MAE:", dev_mae)
# print("Test MAE:", test_mae)


# ------------------------------------------------------------------------------------
# Mental Bert
# Fine-tuning data

# tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased", use_auth_token=True)
# model = AutoModelForMaskedLM.from_pretrained("mental/mental-bert-base-uncased", use_auth_token=True)

# # model_name = "mental/mental-bert-base-uncased"
# # model = AutoModel.from_pretrained(model_name)
# # tokenizer = AutoTokenizer.from_pretrained(model_name)

# train_texts = df_train_prompt2["completions"].apply(preprocess_text).tolist()
# train_labels = df_train_prompt2["PHQ_Score"].tolist()

# dev_texts = df_dev_prompt2["completions"].apply(preprocess_text).tolist()
# dev_labels = df_dev_prompt2["PHQ_Score"].tolist()

# test_texts = df_test_prompt2["completions"].apply(preprocess_text).tolist()
# test_labels = df_test_prompt2["PHQ_Score"].tolist()

# # Tokenize the text and create DataLoader
# train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
# train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels))
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# # Define DataLoader for dev and test sets
# dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, return_tensors="pt")
# dev_dataset = TensorDataset(dev_encodings["input_ids"], dev_encodings["attention_mask"], torch.tensor(dev_labels))
# dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False)  # No need to shuffle for dev

# test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")
# test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_labels))
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # No need to shuffle for test

# # Fine-tune the BERT model
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# for epoch in range(20):
#     total_loss = 0.0
#     for step, batch in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         labels = batch[2].unsqueeze(1).float()
        
#         outputs = model(**inputs)
        
#         loss = torch.nn.MSELoss()(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         if step % 10 == 0:
#             print(f"Epoch [{epoch + 1}/20] | Step [{step + 1}/{len(train_dataloader)}] | Loss: {loss.item():.4f}")
    
#     avg_epoch_loss = total_loss / len(train_dataloader)
#     print(f"Epoch [{epoch + 1}/20] | Average Loss: {avg_epoch_loss:.4f}")
    
#     # Evaluate on train set
#     train_predictions = []
#     with torch.no_grad():
#         for train_batch in train_dataloader:
#             train_inputs = {"input_ids": train_batch[0], "attention_mask": train_batch[1]}
#             train_outputs = model(**train_inputs)
#             train_predictions.append(train_outputs.logits.numpy().flatten())
            
#         train_predictions = np.concatenate(train_predictions)
#         train_mae = mean_absolute_error(train_labels, train_predictions)
#         print(f"Epoch [{epoch + 1}/20] | Train MAE: {train_mae:.4f}")
        
#     # Evaluate on dev set
#     dev_predictions = []
#     with torch.no_grad():
#         for dev_batch in dev_dataloader:
#             dev_inputs = {"input_ids": dev_batch[0], "attention_mask": dev_batch[1]}
#             dev_outputs = model(**dev_inputs)
#             dev_predictions.append(dev_outputs.logits.numpy().flatten())
            
#         dev_predictions = np.concatenate(dev_predictions)
#         dev_mae = mean_absolute_error(dev_labels, dev_predictions)
#         print(f"Epoch [{epoch + 1}/20] | Dev MAE: {dev_mae:.4f}")

# # Predict on train set
# train_logits = []

# # Predict on dev and test sets
# dev_logits = []
# test_logits = []

# with torch.no_grad():
#     for batch in train_dataloader:
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         outputs = model(**inputs)
#         train_logits.append(outputs.logits)

# train_logits = torch.cat(train_logits)

# # Convert logits to predictions
# train_predictions = train_logits.numpy().flatten()

# # Compute MAE on train set
# train_mae = mean_absolute_error(train_labels, train_predictions)
# print("Train MAE:", train_mae)

# with torch.no_grad():
#     for batch in dev_dataloader:
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         outputs = model(**inputs)
#         dev_logits.append(outputs.logits)

#     for batch in test_dataloader:
#         inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#         outputs = model(**inputs)
#         test_logits.append(outputs.logits)

# dev_logits = torch.cat(dev_logits)
# test_logits = torch.cat(test_logits)

# # Convert logits to predictions
# dev_predictions = dev_logits.numpy().flatten()
# test_predictions = test_logits.numpy().flatten()

# # Evaluate the model
# dev_mae = mean_absolute_error(dev_labels, dev_predictions)
# test_mae = mean_absolute_error(test_labels, test_predictions)

# print("Dev MAE:", dev_mae)
# print("Test MAE:", test_mae)

# -------------------------------------------------------------------------
# Mental Bert another implementation

# Fine-tune the model 
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased", use_auth_token=True)
model = AutoModelForMaskedLM.from_pretrained("mental/mental-bert-base-uncased", num_labels=1, use_auth_token=True)

# Set up device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define a function for fine-tuning the model 
def fine_tune_model(train_texts, train_labels, dev_texts, dev_labels, max_epochs=20, device=device):
    
    # Prepare the data for the current training set
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, return_tensors='pt')
    dev_dataset = TensorDataset(dev_encodings['input_ids'], dev_encodings['attention_mask'], torch.tensor(dev_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    best_dev_loss = float('inf')
    best_model_state_dict = None

    # Unfreeze the last 4 layers of the base model's encoder
    # for name, param in model.named_parameters():
    #     print('model layer name: ', name)
    #     print('Parameter size:', param.size())
    #     print('--------------------------------------')

    for epoch in range(max_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Batch {step}, Train Loss: {loss.item():.3f}")
        
        model.eval()
        with torch.no_grad():
            dev_loss = 0
            dev_total = 0
            for dev_batch in dev_loader:
                dev_input_ids, dev_attn_mask, dev_labels = tuple(t.to(device) for t in dev_batch)
                dev_outputs = model(dev_input_ids, attention_mask=dev_attn_mask, labels=dev_labels)
                dev_loss += dev_outputs.loss.item() * dev_labels.size(0)
                dev_total += dev_labels.size(0)
            
            dev_avg_loss = dev_loss / dev_total
            print(f"Epoch {epoch}, Dev Loss: {dev_avg_loss:.3f}")
            
            if dev_avg_loss < best_dev_loss:
                best_dev_loss = dev_avg_loss
                best_model_state_dict = model.state_dict()
                print("The best model has been updated!")
    
    return best_model_state_dict

train_texts = df_train_prompt2["completions"].tolist()
train_labels = np.array(df_train_prompt2["PHQ_Binary"].tolist())

dev_texts = df_dev_prompt2["completions"].tolist()
dev_labels = np.array(df_dev_prompt2["PHQ_Binary"].tolist())

test_texts = df_test_prompt2["completions"].tolist()
test_labels = np.array(df_test_prompt2["PHQ_Binary"].tolist())    

best_model_state_dict = fine_tune_model(train_texts, train_labels, dev_texts, dev_labels, device=device)
print('best_model_state_dict: ', best_model_state_dict)