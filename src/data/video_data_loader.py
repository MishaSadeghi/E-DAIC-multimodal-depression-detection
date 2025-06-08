import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class VideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading video features from OpenFace CSVs.
    """
    def __init__(self, csv_file, data_folder, feature_type='all', normalize=True):
        """
        Args:
            csv_file (str): Path to the CSV file with participant IDs and labels.
            data_folder (str): Path to the folder containing the OpenFace feature files.
            feature_type (str): The type of features to use ('pose', 'gaze', 'au_r', 'all', etc.).
            normalize (bool): Whether to apply StandardScaler to the features.
        """
        self.data_info = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.feature_type = feature_type
        self.normalize = normalize
        if self.normalize:
            self.scaler = StandardScaler()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        participant_id = self.data_info.iloc[idx]['Participant_ID']
        label = self.data_info.iloc[idx]['PHQ_Score'] # Or 'PHQ_Binary' if you use that
        
        filepath = os.path.join(self.data_folder, f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        
        try:
            features_df = pd.read_csv(filepath)
            # Clean up column names by stripping whitespace
            features_df.columns = features_df.columns.str.strip()
        except FileNotFoundError:
            print(f"Warning: File not found for participant {participant_id}. Skipping.")
            # Return empty tensors if file not found
            return torch.empty(0, 0), torch.tensor(0.0, dtype=torch.float64)

        # Filter out low-confidence frames
        if 'confidence' in features_df.columns:
            features_df = features_df[features_df['confidence'] > 0.9].copy()

        # Select features based on the specified type
        selected_columns = self._get_feature_columns(features_df.columns)
        
        if not selected_columns:
             # If feature type is not specific, use all columns except metadata
            selected_columns = [col for col in features_df.columns if col not in ['frame', 'face_id', 'timestamp', 'confidence', 'success']]
            
        features = features_df[selected_columns].values

        if self.normalize:
            features = self.scaler.fit_transform(features)
            
        return torch.tensor(features, dtype=torch.float64), torch.tensor(label, dtype=torch.float64)

    def _get_feature_columns(self, all_columns):
        """Helper function to get list of columns based on feature_type."""
        column_map = {
            'pose': [col for col in all_columns if col.startswith('pose_')],
            'gaze': [col for col in all_columns if col.startswith('gaze_')],
            'au_r': [col for col in all_columns if col.endswith('_r')],
            'au_c': [col for col in all_columns if col.endswith('_c')],
        }
        
        if self.feature_type in column_map:
            return column_map[self.feature_type]
            
        if self.feature_type == 'all':
            return column_map['pose'] + column_map['gaze'] + column_map['au_r'] + column_map['au_c']
            
        # Add other combinations as needed, e.g., 'pose_gaze'
        if self.feature_type == 'pose_gaze':
            return column_map['pose'] + column_map['gaze']

        return [] # Return empty if no match, handled in __getitem__

def pad_collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    This is a required helper function for the DataLoader.
    """
    (sequences, labels) = zip(*batch)
    
    # Filter out empty sequences that might result from missing files
    sequences = [s for s in sequences if s.nelement() > 0]
    labels = [l for s, l in zip(sequences, labels) if s.nelement() > 0]
    
    if not sequences:
        return torch.empty(0,0,0), torch.empty(0)

    # Pad sequences with 0s
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    return sequences_padded, torch.tensor(labels, dtype=torch.float64)

def create_data_loaders(data_dir, label_dir, feature_type, batch_size=32):
    """
    Creates and returns train, validation, and test DataLoaders.
    """
    train_csv_path = os.path.join(label_dir, 'train_split.csv')
    dev_csv_path = os.path.join(label_dir, 'dev_split.csv')
    test_csv_path = os.path.join(label_dir, 'test_split.csv')
    
    open_face_train_dir = os.path.join(data_dir, 'DAIC_open_face_train')
    open_face_dev_dir = os.path.join(data_dir, 'DAIC_open_face_dev')
    open_face_test_dir = os.path.join(data_dir, 'DAIC_open_face_test')

    train_dataset = VideoDataset(csv_file=train_csv_path, data_folder=open_face_train_dir, feature_type=feature_type)
    val_dataset = VideoDataset(csv_file=dev_csv_path, data_folder=open_face_dev_dir, feature_type=feature_type)
    test_dataset = VideoDataset(csv_file=test_csv_path, data_folder=open_face_test_dir, feature_type=feature_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate_fn, shuffle=False)
    
    return train_loader, val_loader, test_loader 