import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AdamW, GPT2TokenizerFast, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer

# Set up device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir_org = '/home/hpc/empk/empk004h/depression-detection/data/original_transcripts_completions/'
data_dir_revised = '/home/hpc/empk/empk004h/depression-detection/data/revised_transcripts_completions/'

# # Define a function to map PHQ scores to categories
def map_phq_score_to_category(score):
    if score >= 14:
        # "severe" group
        return 0
    elif score >= 7 and score <= 13:
        # "moderate" group
        return 1
    elif score < 7:
        # "not depression" group
        return 2

# Define a function to map PHQ scores to categories
# def map_phq_score_to_category(score):
#     if score >= 15:
#         # "severe" group
#         return 0
#     elif score >= 10 and score <= 14:
#         # "moderate" group
#         return 1
#     elif score < 10:
#         # "not depression" group
#         return 2

# mapping base PHQ 8: (24 - 4) —> because of 20 in the dev set OR avg(23,20,22) —> 21
# * None to Mild: PHQ scores from 0 to 6.
# * Moderate: PHQ scores from 7 to 11.
# * Moderately Severe to Severe: PHQ scores from 12 to 21.
# def map_phq_score_to_category(score):
#     if score >= 12:
#         # "severe" group
#         return 0
#     elif score >= 7 and score <= 11:
#         # "moderate" group
#         return 1
#     elif score < 7:
#         # "not depression" group
#         return 2

def map_phq_score_for_all_prompts(prompt, org_or_revised_dir):
    df_dev = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    folder_path = os.path.join(org_or_revised_dir, prompt)
    print('folder_path: ', folder_path)
    
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    print('csv_files: ', csv_files)

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        if 'train' in file:
            df_train = pd.read_csv(file_path)
        elif 'dev' in file:
            df_dev = pd.read_csv(file_path)
            # df_dev.drop(df_dev[df_dev['id'] == 347].index, inplace=True)
        elif 'test' in file:
            df_test = pd.read_csv(file_path)
                
    # Add a new column based on the PHQ scores
    df_train["PHQ_Group"] = df_train["PHQ_Score"].apply(map_phq_score_to_category)
    df_dev["PHQ_Group"] = df_dev["PHQ_Score"].apply(map_phq_score_to_category)
    df_test["PHQ_Group"] = df_test["PHQ_Score"].apply(map_phq_score_to_category)

    return (df_train, df_dev, df_test)


# Fine tuning Deproberta
model_dir = '/home/woody/empk/empk004h/models/fine_tuned_deproberta/only_text/'

# Define a function for fine-tuning the deproberta model on a specific training set
def fine_tune_deproberta_model(prompt, org_revised, train_texts, train_labels, dev_texts, dev_labels, max_epochs=20):
    print('fine tuning for prompt: ', prompt)
    tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")
    model = AutoModelForSequenceClassification.from_pretrained("rafalposwiata/deproberta-large-depression", num_labels=3)
    model.to(device)

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

    optim = torch.optim.AdamW(model.parameters(), lr=5e-7)

    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # for param in model.classifier.dense.parameters():   # for fine-tuning last 2 fully connected layers
    #     param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if 'classifier' not in name:  # Unfreeze classifier layer
    #         param.requires_grad = False


    # # Unfreeze the last 4 layers of the base model's encoder
    # for name, param in model.named_parameters():
    #     if 'classifier' not in name and 'encoder.layer' in name:
    #         layer_num = int(name.split('encoder.layer.')[-1].split('.')[0])
    #         if layer_num >= (model.config.num_hidden_layers - 4):  # Unfreeze last 4 layers
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    
    # optim = torch.optim.AdamW(
    # [
    #     {'params': model.classifier.parameters()},
    #     {'params': model.base_model.encoder.layer[-10:].parameters(), 'lr': 1e-5}  # Fine-tuning last 4 layers
    # ],
    # lr=5e-6
    # )

    # optim = torch.optim.AdamW(
    # [
    #     {'params': model.classifier.parameters()},
    #     {'params': model.base_model.encoder.layer[-10:].parameters(), 'lr': 1e-6}  # Fine-tuning last 4 layers
    # ],
    # lr=5e-7
    # )
    
    # Define the scheduler
    # scheduler = CosineAnnealingLR(optim, T_max=5, eta_min=1e-7)

    # Parameters for early stopping
    best_dev_loss = float('inf')  # Track the best development loss
    best_epoch = -1  # Track the epoch with the best development loss
    epochs_since_best_loss = 0  # Count the number of epochs since the best loss was updated
    max_epochs_without_improvement = 3  # Number of epochs without improvement to trigger early stopping

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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

            # # Implement early stopping based on validation loss
            # if dev_avg_loss < best_dev_loss:
            #     best_dev_loss = dev_avg_loss
            #     best_epoch = epoch
            #     epochs_since_best_loss = 0
            #     # Save the best model if needed
            #     # torch.save(model.state_dict(), 'best_model.pth')
            # else:
            #     epochs_since_best_loss += 1
            #     if epochs_since_best_loss >= max_epochs_without_improvement:
            #         print("Early stopping triggered. No improvement in dev loss.")
            #         break

        if org_revised == 'org':
            # Create the folder named according to the prompt
            prompt_folder = os.path.join(model_dir, 'org', prompt)
            os.makedirs(prompt_folder, exist_ok=True)

            # Save the model inside the prompt folder with a specific name
            model_file_path = os.path.join(prompt_folder, f"fine_tuned_deproberta_model_epoch_{epoch}")
            model.save_pretrained(model_file_path)

        else:
            # Create the folder named according to the prompt
            prompt_folder = os.path.join(model_dir, 'revised', prompt)
            os.makedirs(prompt_folder, exist_ok=True)

            # Save the model inside the prompt folder with a specific name
            model_file_path = os.path.join(prompt_folder, f"fine_tuned_deproberta_model_epoch_{epoch}")
            model.save_pretrained(model_file_path)
        
    return model

def fine_tune_deproberta_model_all_prompts(prompt, org_revised, org_or_revised_dir):
    df_train, df_dev, df_test = map_phq_score_for_all_prompts(prompt, org_or_revised_dir)
    train_texts = df_train['text'].tolist()
    train_labels = np.array(df_train['PHQ_Group'].tolist())
    dev_texts = df_dev['text'].tolist()
    dev_labels = np.array(df_dev['PHQ_Group'].tolist())

    max_epochs = 15
    model = fine_tune_deproberta_model(prompt, org_revised, train_texts, train_labels, dev_texts, dev_labels, max_epochs)

# fine_tune_deproberta_model_all_prompts('prompt_1', 'org', data_dir_org)
# fine_tune_deproberta_model_all_prompts('prompt_2', 'org', data_dir_org)
fine_tune_deproberta_model_all_prompts('prompt_3', 'org', data_dir_org)

# fine_tune_deproberta_model_all_prompts('prompt_1', 'revised', data_dir_revised)
# fine_tune_deproberta_model_all_prompts('prompt_2', 'revised', data_dir_revised)
# fine_tune_deproberta_model_all_prompts('prompt_3', 'revised', data_dir_revised)

# best_model_state_dict = None
# print('best_model_state_dict: ', best_model_state_dict)
# if best_model_state_dict is not None:
#     # Save the fine-tuned model
#     model_name = f"fine_tuned_model_prompt_{result['prompt_number']}"
#     model.load_state_dict(best_model_state_dict)
#     model.save_pretrained(model_dir + model_name)
#     print("The model has been saved!")
