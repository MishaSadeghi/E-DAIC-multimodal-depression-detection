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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification

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
# !!! These are the prompts that we will use for gpt-3.5-turbo API !!!
prompt1 = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point.""" 
# keep that in mind that you are now using the prompt2 outputs from previous model which was gpt3.5-turbo 
# because they were better than the new model regarding being in the first person narrating 
prompt2 = """ Your task is to read the following text which is an interview with a person and to summarize the key points that might be related to the depression of the person. Be concise and to the point. It is very essential that you write your answer in the first-person perspective, as if the interviewee is narrating about himself or herself. """
prompt3 = """ After reading the interview, briefly summarize the main aspects that pertain to the person's depression. """
prompt4 = """ Based on the interview, highlight the key factors that might be indicative of the interviewee's depression. """
prompt5 = """ Your task is to summarize the interviewee's main points that could be linked to their depression. Keep it concise. """
prompt6 = """ After reading the interview, identify and summarize the main challenges or difficulties the interviewee faces that are indicative of depression. """
prompt7 = """ Based on the interview, provide a concise analysis of the interviewee's emotional state and behaviors that may indicate the presence of depression. """
prompt8 = """ Read the interview carefully and extract the most significant indicators of depression exhibited by the interviewee. Summarize them concisely. """
prompt9 = """ Your task is to analyze the interviewee's responses and highlight the key signs or symptoms of depression that are evident in the interview. """
prompt10 = """ Provide a brief summary of the interview, focusing on aspects that strongly suggest the presence of depression in the interviewee. """

prompts = {str(i): eval(f"prompt{i}") for i in range(1, 11)}
# ------------------------------------------------------------------------------------------------
# Reading from the CSV files so we don't have to send requests over and over to gpt

# Define the input directory path
input_directory = "/home/hpc/empk/empk004h/depression-detection/notebooks/prompts_outputs/"
prompt_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Iterate through the results
for prompt_number in prompt_numbers:

    # Define the folder path for the prompt number
    prompt_directory = os.path.join(input_directory, str(prompt_number))

    # Read the CSV files for each dataframe
    dev_filepath = os.path.join(prompt_directory, f"dev_prompt{prompt_number}.csv")
    train_filepath = os.path.join(prompt_directory, f"train_prompt{prompt_number}.csv")
    test_filepath = os.path.join(prompt_directory, f"test_prompt{prompt_number}.csv")

    # Read the dataframes from CSV files
    df_dev = pd.read_csv(dev_filepath)
    df_train = pd.read_csv(train_filepath)
    df_test = pd.read_csv(test_filepath)

    # Rename the dataframes using exec()
    exec(f"df_dev_{prompt_number} = df_dev")
    exec(f"df_train_{prompt_number} = df_train")
    exec(f"df_test_{prompt_number} = df_test")

# ------------------------------------------------------------------------------------------------

# Initialize dictionaries to store completions for each ID
id_completions_train = {}
id_completions_dev = {}
id_completions_test = {}

for prompt_number, prompt in prompts.items():
    
    print('prompt_number: ', prompt_number)
    print('prompt: ', prompt)
    
    # Get the dataframe for the current prompt number
    df_dev = eval(f"df_dev_{prompt_number}")
    df_train = eval(f"df_train_{prompt_number}")
    df_test = eval(f"df_test_{prompt_number}")

    # Iterate through each row and accumulate completions for each ID
    for idx, row in df_train.iterrows():
        id_ = row['id']
        # print('id: ', id_)
        completions = row['completions']
        # print('completions: ', completions)
        if id_ not in id_completions_train:
            id_completions_train[id_] = []
        id_completions_train[id_].append(completions)
    
    for idx, row in df_dev.iterrows():
        id_ = row['id']
        completions = row['completions']
        if id_ not in id_completions_dev:
            id_completions_dev[id_] = []
        id_completions_dev[id_].append(completions)
    
    for idx, row in df_test.iterrows():
        id_ = row['id']
        completions = row['completions']
        if id_ not in id_completions_test:
            id_completions_test[id_] = []
        id_completions_test[id_].append(completions)

# Now construct the combined dataframes using the accumulated completions
combined_train_data = []
combined_dev_data = []
combined_test_data = []

for id_, completions in id_completions_train.items():
    combined_train_data.append({
        'id': id_,
        'completions': ' '.join(completions)
    })

for id_, completions in id_completions_dev.items():
    combined_dev_data.append({
        'id': id_,
        'completions': ' '.join(completions)
    })

for id_, completions in id_completions_test.items():
    combined_test_data.append({
        'id': id_,
        'completions': ' '.join(completions)
    })

combined_train = pd.DataFrame(combined_train_data)
combined_dev = pd.DataFrame(combined_dev_data)
combined_test = pd.DataFrame(combined_test_data)

combined_train.to_csv('combined_train.csv', index=False)

# List of statements to remove
statements_to_remove = [
    r"\bKey points related to depression from the interview\b",
    r"\bThe main aspects that pertain to the person's depression are\b",
    r"\bKey factors that might be indicative of the interviewee's depression include\b",
    r"\bThe main points that could be linked to the interviewee's depression are\b",
    r"\bBased on the interview, the interviewee's emotional state appears to be fluctuating\b",
    r"\bSignificant indicators of depression exhibited by the interviewee include\b",
    r"\bKey signs or symptoms of depression evident in the interview are\b",
    r"\bIn this interview, several aspects strongly suggest the presence of depression in the interviewee\b",
    r"\bThe main challenges or difficulties that the interviewee faces that are indicative of depression include\b"
    r"\bKey points related to the depression of the person\b",
    r"\bBased on the interview, the interviewee displays several behaviors and statements that may indicate the presence of depression\b",
    r"\bBased on the interview, the main challenges or difficulties indicative of depression that the interviewee faces are\b",
    r"\bKey points related to depression\b",
    r"\bIn the interview, the interviewee mentioned the following points that could be linked to their depression\b",
    r"\bBased on the interview, the interviewee displays several behaviors and statements that may indicate the presence of depression. These include\b",
    r"\bBased on the interview, the interviewee displays some behaviors and emotions that may indicate the presence of depression\b",
    r"\bKey points related to the person's depression\b",
    r"\bKey signs or symptoms of depression evident in the interviewee's responses include:\b",
    r"\bThe main challenges or difficulties the interviewee faces that are indicative of depression include\b"         
]

# Function to remove specific statements from text using regular expressions
def remove_statements(text):
    for statement in statements_to_remove:
        text = re.sub(statement, '', text, flags=re.IGNORECASE)
    return text.strip()

# Apply the function to 'completions' column of each DataFrame
combined_train['completions'] = combined_train['completions'].apply(remove_statements)
combined_dev['completions'] = combined_dev['completions'].apply(remove_statements)
combined_test['completions'] = combined_test['completions'].apply(remove_statements)

df_train = df_train.drop('completions', axis=1)
df_dev = df_dev.drop('completions', axis=1)
df_test = df_test.drop('completions', axis=1)

# Merge each set with its combined set
df_train = pd.merge(df_train, combined_train, on='id')
df_dev = pd.merge(df_dev, combined_dev, on='id')
df_test = pd.merge(df_test, combined_test, on='id')

# Save the modified DataFrames to CSV files
df_train.to_csv('df_train.csv', index=False)
df_dev.to_csv('df_dev.csv', index=False)
df_test.to_csv('df_test.csv', index=False)

# ------------------------------------------------------------------------------------------------
# # First step for fine tuning the Deproberta model is to create labels similar to the Deproberta model labels

# # Define a function to map PHQ scores to categories
# def map_phq_score_to_category(score):
#     if score >= 14:
#         # "severe" group
#         return 0
#     elif score >= 7 and score <= 13:
#         # "moderate" group
#         return 1
#     elif score < 7:
#         # "not depression" group
#         return 2
    
# # Add a new column based on the PHQ scores
# df_train["PHQ_Group"] = df_train["PHQ_Score"].apply(map_phq_score_to_category)
# df_dev["PHQ_Group"] = df_dev["PHQ_Score"].apply(map_phq_score_to_category)
# df_test["PHQ_Group"] = df_test["PHQ_Score"].apply(map_phq_score_to_category)

# ------------------------------------------------------------------------------------------------
# Using BERT instead of Deproberta

# Load your dataset

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)  # Regression head

# Tokenize and encode text data
def tokenize_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Adjust this as needed
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )

# Create DataLoader for training and development sets
def create_dataloader(df):
    encoded_texts = [tokenize_text(text) for text in df['completions']]
    input_ids = torch.cat([enc['input_ids'] for enc in encoded_texts])
    attention_masks = torch.cat([enc['attention_mask'] for enc in encoded_texts])
    labels = torch.tensor(df['PHQ_Score'], dtype=torch.float32)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

train_dataloader = create_dataloader(df_train)
dev_dataloader = create_dataloader(df_dev)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # Adjust learning rate
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 20  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        total_mae = 0
        total_rmse = 0
        num_samples = 0
        
        for batch in dev_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.squeeze()
            
            mae = torch.abs(predictions - labels).sum().item()
            rmse = torch.sqrt(torch.mean((predictions - labels)**2)).item()
            
            print('predictions dev: ', predictions)
            print('labels dev: ', labels)

            total_mae += mae
            total_rmse += rmse
            num_samples += len(labels)

            print('len(labels): ', len(labels))
        
        avg_mae = total_mae / num_samples
        avg_rmse = total_rmse / num_samples
        print(f"Validation MAE: {avg_mae:.4f}, Validation RMSE: {avg_rmse:.4f}")



# # ------------------------------------------------------------------------------------------------
# # Fine tuning the DepRoberta model

# model_dir = "/home/hpc/empk/empk004h/depression-detection/model/prompts_combined/"

# # Define a function for fine-tuning the model on a specific training set
# def fine_tune_model(train_texts, train_labels, dev_texts, dev_labels, max_epochs=20):
#     # Prepare the data for the current training set
#     train_encodings = tokenizer(train_texts, truncation=True, padding=True)
#     train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), 
#                                   torch.tensor(train_encodings['attention_mask']),
#                                   torch.tensor(train_labels))

#     dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
#     dev_dataset = TensorDataset(torch.tensor(dev_encodings['input_ids']), 
#                                 torch.tensor(dev_encodings['attention_mask']),
#                                 torch.tensor(dev_labels))
#     dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

#     # Define optimizer and other fine-tuning parameters here
#     # optim = torch.optim.AdamW(model.parameters(), lr=5e-6) #=> 68% accuracy on dev & 61% accuracy on test
#     # optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
#     # optim = torch.optim.AdamW(model.classifier.parameters(), lr=5e-6) # for fine-tuning only the final layer 
#     # optim = torch.optim.AdamW(model.classifier.out_proj.parameters(), lr=5e-6) # for fine-tuning last 2 fully connected layers

#     # for param in model.base_model.parameters():
#     #     param.requires_grad = False

#     # for param in model.classifier.dense.parameters():   # for fine-tuning last 2 fully connected layers
#     #     param.requires_grad = False

#     # for name, param in model.named_parameters():
#     #     if 'classifier' not in name:  # Unfreeze classifier layer
#     #         param.requires_grad = False


#     # Unfreeze the last 4 layers of the base model's encoder
#     for name, param in model.named_parameters():
#         if 'classifier' not in name and 'encoder.layer' in name:
#             layer_num = int(name.split('encoder.layer.')[-1].split('.')[0])
#             if layer_num >= (model.config.num_hidden_layers - 4):  # Unfreeze last 4 layers
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
    
#     optim = torch.optim.AdamW(
#     [
#         {'params': model.classifier.parameters()},
#         {'params': model.base_model.encoder.layer[-10:].parameters(), 'lr': 1e-5}  # Fine-tuning last 4 layers
#     ],
#     lr=5e-6
#     )
    
#     # Define the scheduler
#         # scheduler = CosineAnnealingLR(optim, T_max=5, eta_min=1e-7)

#     # Parameters for early stopping
#     best_dev_loss = float('inf')  # Track the best development loss
#     best_epoch = -1  # Track the epoch with the best development loss
#     epochs_since_best_loss = 0  # Count the number of epochs since the best loss was updated
#     max_epochs_without_improvement = 3  # Number of epochs without improvement to trigger early stopping

#     # Train the model
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

#     # Start fine-tuning
#     for epoch in range(max_epochs):
#         model.train() 
#         print('len train_loader: ', len(train_loader))
#         for step, batch in enumerate(train_loader):
#             optim.zero_grad()
#             input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
#             outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optim.step()
#             # scheduler.step()
            
#             current_lr = optim.param_groups[0]['lr']
#             print(f"Epoch {epoch}, Batch {step}, Learning Rate: {current_lr:.8f}, Train Loss: {loss.item():.3f}")
            
#             # if step % 100 == 0:
#             #     print(f"Epoch {epoch}, Batch {step}, Train Loss: {loss.item():.3f}")
            
#         # Calculate dev loss after each epoch
#         model.eval()
#         with torch.no_grad():
#             dev_loss = 0
#             dev_total = 0
#             for dev_batch in dev_loader:
#                 dev_input_ids, dev_attn_mask, dev_labels = tuple(t.to(device) for t in dev_batch)
#                 dev_outputs = model(dev_input_ids, attention_mask=dev_attn_mask, labels=dev_labels)
#                 dev_loss += dev_outputs.loss.item() * dev_labels.size(0)
#                 dev_total += dev_labels.size(0)
#             dev_avg_loss = dev_loss / dev_total
#             print(f"Epoch {epoch}, Batch {step}, Dev Loss: {dev_avg_loss:.3f}")
            
#             # Check for early stopping
#             if dev_avg_loss < best_dev_loss:
#                 best_dev_loss = dev_avg_loss
#                 best_epoch = epoch  # Update the best epoch
#                 print('best_epoch: ', best_epoch)
#                 epochs_since_best_loss = 0

#                 # Update the best model's state dict
#                 best_model_state_dict = model.state_dict()
                
#                 # Save the model at the best epoch
#                 model_name = f"fine_tuned_model_combined_prompts_best_epoch_{best_epoch}"
#                 model.save_pretrained(model_dir + model_name)
#                 print("The best model has been saved!")
#             else:
#                 epochs_since_best_loss += 1
                
#         print('epochs_since_best_loss: ', epochs_since_best_loss)
# #         if epochs_since_best_loss >= max_epochs_without_improvement:
# #             print("Early stopping triggered. No improvement in dev loss.")
# #             break

#     return model


# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")
# model = AutoModelForSequenceClassification.from_pretrained("rafalposwiata/deproberta-large-depression", num_labels=3)

# # Set up device (GPU or CPU)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

# best_model_state_dict = None

# print('df train: ', df_train.head())
# print('df dev: ', df_dev.head())
# print('df test: ', df_test.head())

# train_texts = df_train['completions'].tolist()
# train_labels = np.array(df_train['PHQ_Group'].tolist())
# dev_texts = df_dev['completions'].tolist()
# dev_labels = np.array(df_dev['PHQ_Group'].tolist())

# # Fine-tune the model on the current training set
# model = fine_tune_model(train_texts, train_labels, dev_texts, dev_labels)
# print('best_model_state_dict: ', best_model_state_dict)
# if best_model_state_dict is not None:
#     # Save the fine-tuned model
#     model_name = "fine_tuned_model_combined_prompts"
#     model.load_state_dict(best_model_state_dict)
#     model.save_pretrained(model_dir + model_name)
#     print("The model has been saved!")

# # ------------------------------------------------------------------------------------------------
# # Feature extraction based on fine-tuned deproberta
# # Extracting deproberta features (probabilities) from completions based on the fine-tuned model

# tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")

# def extract_features(df_train, df_dev, df_test, model):
#     for df in [df_train, df_dev, df_test]:
#         # Remove newlines and non-meaningful characters
#         df['completions'] = df['completions'].replace(r'\n', ' ', regex=True)  # Replace newlines with spaces
#         df['completions'] = df['completions'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Remove non-alphanumeric and non-space characters
#         df['completions'] = df['completions'].replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space

#     X_train = df_train['completions']
#     X_dev = df_dev['completions']
#     X_test = df_test['completions']
    
#     print(len(X_train))
#     print(len(X_dev))
#     print(len(X_test))

#     # Extract features from train data
#     X_train_features = []
#     for i in range(len(X_train)):
#         input_ids = torch.tensor(tokenizer.encode(X_train[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids)
#         logits = outputs[0]
#         probs = torch.softmax(logits, dim=1)
#         predicted_label_index = torch.argmax(probs, dim=1).item()
#         X_train_features.append(probs.detach().numpy())
#         print('train i: ', i)

#     # Extract features from dev data
#     X_dev_features = []
#     for i in range(len(X_dev)):
#         input_ids = torch.tensor(tokenizer.encode(X_dev[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids)
#         logits = outputs[0]
#         probs = torch.softmax(logits, dim=1)
#         predicted_label_index = torch.argmax(probs, dim=1).item()
#         X_dev_features.append(probs.detach().numpy())
#         print('dev i: ', i)

#     # Extract features from test data
#     X_test_features = []
#     for i in range(len(X_test)):
#         input_ids = torch.tensor(tokenizer.encode(X_test[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids)
#         logits = outputs[0]
#         probs = torch.softmax(logits, dim=1)
#         predicted_label_index = torch.argmax(probs, dim=1).item()
#         X_test_features.append(probs.detach().numpy())
#         print('test i: ', i)

#     return (X_train_features, X_dev_features, X_test_features)

# # Function to extract the epoch number from the model file name
# def get_epoch_number(file_path):
#     return int(file_path.split("_")[-1])

# y_train = np.array(df_train['PHQ_Score'])
# y_dev = np.array(df_dev['PHQ_Score'])
# y_test = np.array(df_test['PHQ_Score'])

# # Find the corresponding model file based on the prompt number
# model_files = glob.glob(f"{model_dir}/fine_tuned_model_combined_prompts_best_epoch_*")

# # Select the model with the greatest epoch
# latest_model_file = max(model_files, key=get_epoch_number)
# print('latest_model_file: ', latest_model_file)

# # Load the model 
# model = AutoModelForSequenceClassification.from_pretrained(latest_model_file)

# X_train_features, X_dev_features, X_test_features = extract_features(df_train, df_dev, df_test, model)

# print('Succefullly extracted features from combined prompts')
# print('---------------------------------------------------')

# # ------------------------------------------------------------------------------------------------
# # Training and Evaluating and Testing

# # Convert to numpy arrays
# deproberta_features_train = np.array(X_train_features)
# deproberta_features_dev = np.array(X_dev_features)
# deproberta_features_test = np.array(X_test_features)

# y_train = np.array(df_train['PHQ_Score'])
# y_dev = np.array(df_dev['PHQ_Score'])
# y_test = np.array(df_test['PHQ_Score'])

# print('train shape: ', np.shape(deproberta_features_train))
# print('dev shape: ', np.shape(deproberta_features_dev))
# print('test shape: ', np.shape(deproberta_features_test))

# # Reshape the features
# deproberta_features_train = deproberta_features_train.reshape(deproberta_features_train.shape[0], deproberta_features_train.shape[2])
# deproberta_features_dev = deproberta_features_dev.reshape(deproberta_features_dev.shape[0], deproberta_features_dev.shape[2])
# deproberta_features_test = deproberta_features_test.reshape(deproberta_features_test.shape[0], deproberta_features_test.shape[2])

# # Normalize X_train, X_dev, and X_test
# #     scaler = StandardScaler()
# #     X_train = scaler.fit_transform(deproberta_features_train)
# #     X_dev = scaler.transform(deproberta_features_dev)
# #     X_test = scaler.transform(deproberta_features_test)
    
# X_train = deproberta_features_train
# X_dev = deproberta_features_dev
# X_test = deproberta_features_test
    
# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'kernel': ['linear', 'rbf', 'poly'],
#     'C': [1, 5, 10],
#     'gamma': ['scale', 'auto']
#     }

# # Perform grid search to find the best parameters
# grid_search = GridSearchCV(SVR(), param_grid, cv=5)
# grid_search.fit(X_dev, y_dev)

# # Get the best SVR model from the grid search
# svr = grid_search.best_estimator_

# # Train the SVR model on X_train and y_train
# svr.fit(X_train, y_train)

# # Predict on X_train and calculate the mean squared error and mean absolute error
# y_pred_train = svr.predict(X_train)
# mse_train = mean_squared_error(y_train, y_pred_train)
# mae_train = mean_absolute_error(y_train, y_pred_train)
# print('RMSE for train: ', np.sqrt(mse_train))
# print('MAE for train: ', mae_train)

# # Predict on X_dev and calculate the mean squared error and mean absolute error
# y_pred_dev = svr.predict(X_dev)
# mse_dev = mean_squared_error(y_dev, y_pred_dev)
# mae_dev = mean_absolute_error(y_dev, y_pred_dev)
# print('RMSE for dev: ', np.sqrt(mse_dev))
# print('MAE for dev: ', mae_dev)

# # Predict on X_test and calculate the mean squared error and mean absolute error
# y_pred_test = svr.predict(X_test)
# mse_test = mean_squared_error(y_test, y_pred_test)
# mae_test = mean_absolute_error(y_test, y_pred_test)
# print('RMSE for test: ', np.sqrt(mse_test))
# print('MAE for test: ', mae_test)
