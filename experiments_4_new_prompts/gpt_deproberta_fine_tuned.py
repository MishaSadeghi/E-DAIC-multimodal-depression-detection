import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW, GPT2TokenizerFast, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer

# Reading generated CSV files for 4 new prompts and creating corresponding dataframes
data_path = '/home/hpc/empk/empk004h/depression-detection/data/'

# Function to read CSV files and store them in dataframes
def read_csv_files(directory):
    dfs = {}  # Dictionary to store dataframes
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                df_name = os.path.splitext(file)[0]  # Extract dataframe name from file name
                dfs[df_name] = pd.read_csv(filepath)  # Store dataframe in dictionary

    return dfs

# Read CSV files from original_transcripts_completions directory
original_dfs = read_csv_files(os.path.join(data_path, 'original_transcripts_completions'))

# Read CSV files from revised_transcripts_completions directory
revised_dfs = read_csv_files(os.path.join(data_path, 'revised_transcripts_completions'))

# print('df_dev_prompt1\n', original_dfs['df_dev_prompt1'].head())
print('Original transcripts keys:', original_dfs.keys())
print('Revised transcripts keys:', revised_dfs.keys())

print('Original transcripts df_train_prompt1: ', original_dfs['df_train_prompt1'].head())
print('max PHQ score in df_train_prompt1: ', max(original_dfs['df_train_prompt1']['PHQ_Score']))

# 'df_train_prompt1', 'df_dev_prompt1', 'df_test_prompt1'
# 'df_train_prompt2', 'df_dev_prompt2', 'df_test_prompt2'
# 'df_train_prompt3', 'df_dev_prompt3', 'df_test_prompt3'
# 'df_train_Q10', 'df_dev_Q10', 'df_test_Q10'
# ------------------------------------------------------------------------------------------------
# # Fine tuning Deproberta
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

# # Loop through each dictionary in the results list
# for result in results:
#     # Get the dataframe for the current prompt number
#     df_train = result['df_train']
#     df_dev = result['df_dev']
#     df_test = result['df_test']
    
#     # Add a new column based on the PHQ scores
#     df_train["PHQ_Group"] = df_train["PHQ_Score"].apply(map_phq_score_to_category)
#     df_dev["PHQ_Group"] = df_dev["PHQ_Score"].apply(map_phq_score_to_category)
#     df_test["PHQ_Group"] = df_test["PHQ_Score"].apply(map_phq_score_to_category)

# Fine tuning the DepRoberta model
model_dir = "/home/hpc/empk/empk004h/depression-detection/model/ensemble/"

# Define a function for fine-tuning the model on a specific training set
def fine_tune_model(train_texts, train_labels, dev_texts, dev_labels, max_epochs=20):
    # Prepare the data for the current training set
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), 
                                  torch.tensor(train_encodings['attention_mask']),
                                  torch.tensor(train_labels))

    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)
    dev_dataset = TensorDataset(torch.tensor(dev_encodings['input_ids']), 
                                torch.tensor(dev_encodings['attention_mask']),
                                torch.tensor(dev_labels))
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    # Define optimizer and other fine-tuning parameters here
    # optim = torch.optim.AdamW(model.parameters(), lr=5e-6) #=> 68% accuracy on dev & 61% accuracy on test
    # optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optim = torch.optim.AdamW(model.classifier.parameters(), lr=5e-6) # for fine-tuning only the final layer 
    # optim = torch.optim.AdamW(model.classifier.out_proj.parameters(), lr=5e-6) # for fine-tuning last 2 fully connected layers

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # for param in model.classifier.dense.parameters():   # for fine-tuning last 2 fully connected layers
    #     param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if 'classifier' not in name:  # Unfreeze classifier layer
    #         param.requires_grad = False


    # Unfreeze the last 4 layers of the base model's encoder
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'encoder.layer' in name:
            layer_num = int(name.split('encoder.layer.')[-1].split('.')[0])
            if layer_num >= (model.config.num_hidden_layers - 4):  # Unfreeze last 4 layers
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    optim = torch.optim.AdamW(
    [
        {'params': model.classifier.parameters()},
        {'params': model.base_model.encoder.layer[-10:].parameters(), 'lr': 1e-5}  # Fine-tuning last 4 layers
    ],
    lr=5e-6
    )
    
    # Define the scheduler
        # scheduler = CosineAnnealingLR(optim, T_max=5, eta_min=1e-7)

    # Parameters for early stopping
    best_dev_loss = float('inf')  # Track the best development loss
    best_epoch = -1  # Track the epoch with the best development loss
    epochs_since_best_loss = 0  # Count the number of epochs since the best loss was updated
    max_epochs_without_improvement = 3  # Number of epochs without improvement to trigger early stopping

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Start fine-tuning
    for epoch in range(max_epochs):
        model.train() 
        print('len train_loader: ', len(train_loader))
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            # scheduler.step()
            
            current_lr = optim.param_groups[0]['lr']
            print(f"Epoch {epoch}, Batch {step}, Learning Rate: {current_lr:.8f}, Train Loss: {loss.item():.3f}")
            
            # if step % 100 == 0:
            #     print(f"Epoch {epoch}, Batch {step}, Train Loss: {loss.item():.3f}")
            
        # Calculate dev loss after each epoch
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
            print(f"Epoch {epoch}, Batch {step}, Dev Loss: {dev_avg_loss:.3f}")
            
            # Check for early stopping
            if dev_avg_loss < best_dev_loss:
                best_dev_loss = dev_avg_loss
                best_epoch = epoch  # Update the best epoch
                print('best_epoch: ', best_epoch)
                epochs_since_best_loss = 0

                # Update the best model's state dict
                best_model_state_dict = model.state_dict()
                
                # Save the model at the best epoch
                model_name = f"fine_tuned_model_prompt_{result['prompt_number']}_best_epoch_{best_epoch}"
                model.save_pretrained(model_dir + model_name)
                print("The best model has been saved!")
            else:
                epochs_since_best_loss += 1
                
        print('epochs_since_best_loss: ', epochs_since_best_loss)
#         if epochs_since_best_loss >= max_epochs_without_improvement:
#             print("Early stopping triggered. No improvement in dev loss.")
#             break

    return model

# # Fine-tune the model for each training set in the results list
# for result in results:
#     # Load the tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")
#     model = AutoModelForSequenceClassification.from_pretrained("rafalposwiata/deproberta-large-depression", num_labels=3)

#     # Set up device (GPU or CPU)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     model.to(device)

#     best_model_state_dict = None

#     print('prompt_number: ', result['prompt_number'])
#     df_train = result['df_train']
#     df_dev = result['df_dev']
#     train_texts = df_train['completions'].tolist()
#     train_labels = np.array(df_train['PHQ_Group'].tolist())
#     dev_texts = df_dev['completions'].tolist()
#     dev_labels = np.array(df_dev['PHQ_Group'].tolist())
    
#     # Fine-tune the model on the current training set
#     model = fine_tune_model(train_texts, train_labels, dev_texts, dev_labels)
#     print('best_model_state_dict: ', best_model_state_dict)
#     if best_model_state_dict is not None:
#         # Save the fine-tuned model
#         model_name = f"fine_tuned_model_prompt_{result['prompt_number']}"
#         model.load_state_dict(best_model_state_dict)
#         model.save_pretrained(model_dir + model_name)
#         print("The model has been saved!")
# ------------------------------------------------------------------------------------------------
# Feature extraction based on fine-tuned deproberta
# Extracting deproberta features (probabilities) from completions for all of the prompts results

tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")

# Load the latest fine-tuned Deproberta model 
latest_model_file = "/home/hpc/empk/empk004h/depression-detection/model/best_deproberta_finetuned_train_3rd_phq_range_12_ep_lr_5e-6"
model = AutoModelForSequenceClassification.from_pretrained(latest_model_file)

def extract_features(df, model):
    # for df in [df_train, df_dev, df_test]:
        # # Remove newlines and non-meaningful characters
        # df['completions'] = df['completions'].replace(r'\n', ' ', regex=True)  # Replace newlines with spaces
        # df['completions'] = df['completions'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Remove non-alphanumeric and non-space characters
        # df['completions'] = df['completions'].replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space

    X = df['completions']
    print('len X: ', len(X))

    # Extract features
    X_features = []
    for i in range(len(X)):
        input_ids = torch.tensor(tokenizer.encode(X[i], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        predicted_label_index = torch.argmax(probs, dim=1).item()
        X_features.append(probs.detach().numpy())

    return X_features

def extract_features_for_all_dfs(dataframes_dict, model):
    for key, df in dataframes_dict.items():
        if 'Q10' not in key:
            print('name of the df: ', key)
            print(df.head())
            features = extract_features(df, model)  
            df['features_deproberta'] = features
            print('df after feature extraction: ', df.head())
    
    return dataframes_dict

# Update original transcripts dataframes with extracted features
original_dfs_with_features = extract_features_for_all_dfs(original_dfs, model)

# Update revised transcripts dataframes with extracted features
revised_dfs_with_features = extract_features_for_all_dfs(revised_dfs, model)

def save_dataframes_to_csv(dataframes_dict, directory):
    for key, df in dataframes_dict.items():
        filename = os.path.join(directory, f"{key}.csv")
        df.to_csv(filename, index=False)
        print(f"DataFrame '{key}' saved to '{filename}'")

output_directory_org = '/home/hpc/empk/empk004h/depression-detection/experiments_4_new_prompts/deproberta_features/original/'
output_directory_revised = '/home/hpc/empk/empk004h/depression-detection/experiments_4_new_prompts/deproberta_features/revised/'

# Save original transcripts dataframes with extracted features
save_dataframes_to_csv(original_dfs_with_features, output_directory_org)

# Save revised transcripts dataframes with extracted features
save_dataframes_to_csv(revised_dfs_with_features, output_directory_revised)

# ------------------------------------------------------------------------------------------------
# # Training and Evaluating and Testing

# # original transcripts
# # prompt 1
# features_directory = '/home/hpc/empk/empk004h/depression-detection/experiments_4_new_prompts/csv_files/revised/prompt1/'

# csv_files = [file for file in os.listdir(features_directory) if file.endswith('.csv')]

# df_dev = pd.DataFrame()
# df_train = pd.DataFrame()
# df_test = pd.DataFrame()

# for file in csv_files:
#     file_path = os.path.join(features_directory, file)
#     if 'dev' in file:
#         df_dev = pd.concat([df_dev, pd.read_csv(file_path)], ignore_index=True)
#     elif 'train' in file:
#         df_train = pd.concat([df_train, pd.read_csv(file_path)], ignore_index=True)
#     elif 'test' in file:
#         df_test = pd.concat([df_test, pd.read_csv(file_path)], ignore_index=True)

# X_train = np.array(df_train['features_deproberta'])
# X_dev = np.array(df_dev['features_deproberta'])
# X_test = np.array(df_test['features_deproberta'])

# y_train = np.array(df_train['PHQ_Score'])
# y_dev = np.array(df_dev['PHQ_Score'])
# y_test = np.array(df_test['PHQ_Score'])
    
# # deproberta_features_train = np.array(X_train_features)
# # deproberta_features_dev = np.array(X_dev_features)
# # deproberta_features_test = np.array(X_test_features)

# print('train shape: ', np.shape(X_train))
# print('dev shape: ', np.shape(X_dev))
# print('test shape: ', np.shape(X_test))

# # Reshape the features
# deproberta_features_train = deproberta_features_train.reshape(deproberta_features_train.shape[0], deproberta_features_train.shape[2])
# deproberta_features_dev = deproberta_features_dev.reshape(deproberta_features_dev.shape[0], deproberta_features_dev.shape[2])
# deproberta_features_test = deproberta_features_test.reshape(deproberta_features_test.shape[0], deproberta_features_test.shape[2])

# X_train = deproberta_features_train
# X_dev = deproberta_features_dev
# X_test = deproberta_features_test

# # Define the parameter grid for GridSearchCV
# param_grid = {
#     'kernel': ['rbf', 'poly'],
#     'C': [0.01, 0.1, 10],
#     'gamma': ['scale', 'auto'],
#     'coef0': [0.0, 1.0, 2.0],
#     'degree': [2, 3]
# }

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

# print('SRV results based on fine-tuned DepRoberta model: ')

# # Print predicted and true values for train set
# print("Train set:")
# for true, pred in zip(y_train, y_pred_train):
#     print("True:", true, "\tPredicted:", pred)

# # Predict on X_dev and calculate the mean squared error and mean absolute error
# y_pred_dev = svr.predict(X_dev)
# mse_dev = mean_squared_error(y_dev, y_pred_dev)
# mae_dev = mean_absolute_error(y_dev, y_pred_dev)
# print('RMSE for dev: ', np.sqrt(mse_dev))
# print('MAE for dev: ', mae_dev)

# # Print predicted and true values for dev set
# print("Dev set:")
# for true, pred in zip(y_dev, y_pred_dev):
#     print("True:", true, "\tPredicted:", pred)

# # Predict on X_test and calculate the mean squared error and mean absolute error
# y_pred_test = svr.predict(X_test)
# mse_test = mean_squared_error(y_test, y_pred_test)
# mae_test = mean_absolute_error(y_test, y_pred_test)
# print('RMSE for test: ', np.sqrt(mse_test))
# print('MAE for test: ', mae_test)

# # Print predicted and true values for test set
# print("Test set:")
# for true, pred in zip(y_test, y_pred_test):
#     print("True:", true, "\tPredicted:", pred)




