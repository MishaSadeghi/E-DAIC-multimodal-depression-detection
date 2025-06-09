import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

from src.models.video_model import VideoLSTMModel 

def str_to_np_array(s):
    """Converts a string representation of a numpy array back to a numpy array."""
    s = s.strip('[]')
    return np.fromstring(s, sep=' ')

class OpenFaceDataset(Dataset):
    """Dataset for loading OpenFace features for video feature extraction."""
    def __init__(self, labels_df, data_folder):
        self.data_folder = data_folder
        self.data_info = labels_df

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        participant_id = row['Participant_ID']
        if 'PHQ_Score' in row:
            phq_score = row['PHQ_Score']
        else:
            phq_score = -1 

        filepath = os.path.join(self.data_folder, f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        
        try:
            features_df = pd.read_csv(filepath)
            
            # Basic cleaning from the notebook
            features_df = features_df.drop(features_df[features_df[' confidence'] < 0.9].index)
            features_df = features_df.iloc[:, 5:]  # Remove frame, timestamp, confidence, success
            features_df = features_df.dropna(axis=1, how='all') # drop columns with all NaNs
            features_df = features_df.fillna(0) # Fill any remaining NaNs
            
            # Select feature set corresponding to 'pose_gaze_au_r' from notebook
            pose_columns = [col for col in features_df.columns if 'pose_' in col]
            gaze_columns = [col for col in features_df.columns if 'gaze_' in col]
            au_r_columns = [col for col in features_df.columns if '_r' in col]
            
            selected_columns = pose_columns + gaze_columns + au_r_columns
            features = features_df[selected_columns].values

        except FileNotFoundError:
            print(f"Warning: File not found for participant {participant_id}. Returning empty tensor.")
            features = np.zeros((1, 31))

        return torch.tensor(features, dtype=torch.float32), torch.tensor(phq_score, dtype=torch.float32)


def extract_video_features(model, labels_df, data_folder, device):
    """Extracts features from a dataset using the pretrained LSTM model."""
    dataset = OpenFaceDataset(labels_df=labels_df, data_folder=data_folder)
    extracted_features = []
    model.eval()
    with torch.no_grad():
        for features, _ in dataset:
            features = features.unsqueeze(0).to(device)
            output = model.get_feature_representation(features).squeeze().cpu().numpy()
            extracted_features.append(output)
    return np.array(extracted_features)

def load_multimodal_data(video_feature_dir, text_feature_dir, video_model_path, device):
    """
    Loads all data required for the multimodal model.
    1. Loads the pretrained video model.
    2. Extracts video features for train/dev/test sets.
    3. Loads text (DeBERTa) features.
    4. Loads questionnaire features.
    5. Merges them all.
    """
    print("Loading multimodal data...")

    # 1. Load video model
    video_model = VideoLSTMModel(input_size=31, hidden_size=64, num_layers=3, dropout_rate=0.3)
    video_model.load_state_dict(torch.load(video_model_path, map_location=device))
    video_model.to(device)
    print("Video model loaded.")

    # 2. Define paths and load label files to get participant IDs
    labels_path = os.path.join(video_feature_dir, 'labels')
    openface_path = os.path.join(video_feature_dir, 'DAIC_openface_features')

    train_labels_df = pd.read_csv(os.path.join(labels_path, 'train_split.csv'))
    dev_labels_df = pd.read_csv(os.path.join(labels_path, 'dev_split.csv'))
    test_labels_df = pd.read_csv(os.path.join(labels_path, 'test_split.csv'))
    
    # Correct column name from 'Participant_ID' to 'id' for later merging
    train_labels_df.rename(columns={'Participant_ID': 'id'}, inplace=True)
    dev_labels_df.rename(columns={'Participant_ID': 'id'}, inplace=True)
    test_labels_df.rename(columns={'Participant_ID': 'id'}, inplace=True)

    # 3. Extract video features
    print("Extracting video features for train set...")
    vid_feat_train = extract_video_features(video_model, train_labels_df, os.path.join(openface_path, "train"), device)
    print("Extracting video features for dev set...")
    vid_feat_dev = extract_video_features(video_model, dev_labels_df, os.path.join(openface_path, "dev"), device)
    print("Extracting video features for test set...")
    vid_feat_test = extract_video_features(video_model, test_labels_df, os.path.join(openface_path, "test"), device)

    vid_feat_train_df = pd.DataFrame(vid_feat_train, index=train_labels_df['id'])
    vid_feat_dev_df = pd.DataFrame(vid_feat_dev, index=dev_labels_df['id'])
    vid_feat_test_df = pd.DataFrame(vid_feat_test, index=test_labels_df['id'])

    # 4. Load text and questionnaire features
    print("Loading text and questionnaire features...")
    deberta_prompt = 'prompt3' 
    
    txt_feat_train_df = pd.read_csv(os.path.join(text_feature_dir, f"df_train_{deberta_prompt}.csv"))
    txt_feat_dev_df = pd.read_csv(os.path.join(text_feature_dir, f"df_dev_{deberta_prompt}.csv"))
    txt_feat_test_df = pd.read_csv(os.path.join(text_feature_dir, f"df_test_{deberta_prompt}.csv"))
    
    quest_train_df = pd.read_csv(os.path.join(text_feature_dir, 'df_train_Q10_org.csv'))
    quest_dev_df = pd.read_csv(os.path.join(text_feature_dir, 'df_dev_Q10_org.csv'))
    quest_test_df = pd.read_csv(os.path.join(text_feature_dir, 'df_test_Q10_org.csv'))

    # Prepare for merging
    txt_feat_train_df.set_index('id', inplace=True)
    txt_feat_dev_df.set_index('id', inplace=True)
    txt_feat_test_df.set_index('id', inplace=True)
    quest_train_df.set_index('id', inplace=True)
    quest_dev_df.set_index('id', inplace=True)
    quest_test_df.set_index('id', inplace=True)

    # 5. Combine all features
    print("Merging all features...")
    
    # Deberta features
    X_train_text = np.vstack(txt_feat_train_df['features_deproberta'].apply(str_to_np_array).values)
    X_dev_text = np.vstack(txt_feat_dev_df['features_deproberta'].apply(str_to_np_array).values)
    X_test_text = np.vstack(txt_feat_test_df['features_deproberta'].apply(str_to_np_array).values)
    
    # Questionnaire features
    q_cols = [f'Q{i}' for i in range(1, 12)]
    X_train_quest = quest_train_df.loc[txt_feat_train_df.index][q_cols].values
    X_dev_quest = quest_dev_df.loc[txt_feat_dev_df.index][q_cols].values
    X_test_quest = quest_test_df.loc[txt_feat_test_df.index][q_cols].values

    # Video features
    X_train_vid = vid_feat_train_df.loc[txt_feat_train_df.index].values
    X_dev_vid = vid_feat_dev_df.loc[txt_feat_dev_df.index].values
    X_test_vid = vid_feat_test_df.loc[txt_feat_test_df.index].values
    
    # Labels
    y_train = quest_train_df.loc[txt_feat_train_df.index]['PHQ_Score'].values
    y_dev = quest_dev_df.loc[txt_feat_dev_df.index]['PHQ_Score'].values
    y_test = quest_test_df.loc[txt_feat_test_df.index]['PHQ_Score'].values
    
    # Return features as a dictionary
    features = {
        'train': {'text': X_train_text, 'quest': X_train_quest, 'video': X_train_vid},
        'dev': {'text': X_dev_text, 'quest': X_dev_quest, 'video': X_dev_vid},
        'test': {'text': X_test_text, 'quest': X_test_quest, 'video': X_test_vid},
    }
    
    labels = {'train': y_train, 'dev': y_dev, 'test': y_test}
    
    print("Data loading complete.")
    return features, labels 