import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class TextDataset(Dataset):
    """
    Custom PyTorch Dataset for loading transcript text and PHQ scores.
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data.text
        self.targets = self.data.PHQ_Group
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

def map_phq_score_to_category(score):
    """Maps a PHQ8 score to a severity category."""
    if score >= 14:
        return 0  # "severe"
    elif score >= 7 and score <= 13:
        return 1  # "moderate"
    else: # score < 7
        return 2  # "not depression"

def load_and_prepare_text_data(data_dir, prompt_name):
    """
    Loads transcript data for a specific prompt, applies PHQ mapping, 
    and returns train, dev, and test DataFrames.
    """
    folder_path = os.path.join(data_dir, prompt_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Prompt folder not found: {folder_path}")

    df_train, df_dev, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for file in os.listdir(folder_path):
        if not file.endswith('.csv'):
            continue
        file_path = os.path.join(folder_path, file)
        if 'train' in file:
            df_train = pd.read_csv(file_path)
        elif 'dev' in file:
            df_dev = pd.read_csv(file_path)
        elif 'test' in file:
            df_test = pd.read_csv(file_path)
    
    # Apply the PHQ score mapping
    for df in [df_train, df_dev, df_test]:
        if not df.empty and 'PHQ_Score' in df.columns:
            df["PHQ_Group"] = df["PHQ_Score"].apply(map_phq_score_to_category)

    return df_train, df_dev, df_test

def create_text_data_loaders(data_dir, prompt_name, model_name, max_len, batch_size):
    """
    Creates and returns train, validation, and test DataLoaders for the text model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df_train, df_dev, df_test = load_and_prepare_text_data(data_dir, prompt_name)

    train_dataset = TextDataset(df_train, tokenizer, max_len)
    val_dataset = TextDataset(df_dev, tokenizer, max_len)
    test_dataset = TextDataset(df_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_and_extract_text_features(label_dir, text_data_path, model_name, device):
    """
    Loads labels and ChatGPT-processed text, then extracts features using a pre-trained
    transformer model like DepRoBERTa.

    Args:
        label_dir (str): Path to the directory with train/dev/test label CSVs.
        text_data_path (str): Path to the Excel file with ChatGPT processed text.
        model_name (str): Name of the Hugging Face model to use for feature extraction.
        device (torch.device): The device to run the model on.

    Returns:
        A tuple of (X_train, y_train, X_dev, y_dev, X_test, y_test)
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Load and merge dataframes
    chatgpt_text_df = pd.read_excel(text_data_path)
    chatgpt_text_df.columns = ['id', 'text']
    
    df_train = pd.merge(pd.read_csv(os.path.join(label_dir, 'train_split.csv')), chatgpt_text_df, on='id')
    df_dev = pd.merge(pd.read_csv(os.path.join(label_dir, 'dev_split.csv')), chatgpt_text_df, on='id')
    df_test = pd.merge(pd.read_csv(os.path.join(label_dir, 'test_split.csv')), chatgpt_text_df, on='id')

    # Helper function for feature extraction
    def _extract_features(texts):
        all_features = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                # We get the logits from the classification head as features
                features = model(**inputs).logits.cpu().numpy()
                all_features.append(features)
        return np.vstack(all_features)

    # Extract features for all splits
    X_train = _extract_features(df_train['text'].tolist())
    X_dev = _extract_features(df_dev['text'].tolist())
    X_test = _extract_features(df_test['text'].tolist())
    
    # Get labels
    y_train = df_train['PHQ_Score'].values
    y_dev = df_dev['PHQ_Score'].values
    y_test = df_test['PHQ_Score'].values
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test 