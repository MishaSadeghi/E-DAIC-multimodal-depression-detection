import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define path to the datasets
# video_data_folder = '/home/hpc/empk/empk004h/depression-detection/data/DAIC_openface_features/'

open_face_train = '/home/hpc/empk/empk004h/depression-detection/data/DAIC_open_face_train'
open_face_dev = '/home/hpc/empk/empk004h/depression-detection/data/DAIC_open_face_dev'
open_face_test = '/home/hpc/empk/empk004h/depression-detection/data/DAIC_open_face_test'

train_csv_path = '/home/hpc/empk/empk004h/depression-detection/data/labels/train_split.csv'
dev_csv_path = '/home/hpc/empk/empk004h/depression-detection/data/labels/dev_split.csv'
test_csv_path = '/home/hpc/empk/empk004h/depression-detection/data/labels/test_split.csv'

# Define the custom dataset class
class PHQDataset(Dataset):
    def __init__(self, csv_file, data_folder, max_seq_length=None, feature_type=None):
        self.data_folder = data_folder
        self.data_info = pd.read_csv(csv_file)
        self.max_seq_length = max_seq_length
        self.feature_type = feature_type

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        participant_id = self.data_info.iloc[idx]['Participant_ID']
        phq_score = self.data_info.iloc[idx]['PHQ_Score']
        filepath = os.path.join(self.data_folder, f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        features_df = pd.read_csv(filepath)
        features_df = features_df.iloc[:, 2:]  # Remove the first two columns (frame and timestamp)
        # features = pd.read_csv(filepath).to_numpy()
        # features = features[:, 2:] # Remove the first two features (frame and timestamp)

        # Define column sets
        pose_columns = [col for col in features_df.columns if col.startswith('pose_')]
        gaze_columns = [col for col in features_df.columns if col.startswith('gaze_')]
        au_r_columns = [col for col in features_df.columns if col.endswith('_r')]
        au_c_columns = [col for col in features_df.columns if col.endswith('_c')]

        # Select columns based on feature type
        if self.feature_type == 'pose':
            selected_columns = pose_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'gaze':
            selected_columns = gaze_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'au_r':
            selected_columns = au_r_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'au_c':
            selected_columns = au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'confidence_success_pose':
            selected_columns = pose_columns + ['confidence', 'success']
        elif self.feature_type == 'confidence_success_gaze':
            selected_columns = gaze_columns + ['confidence', 'success']
        elif self.feature_type == 'confidence_success_AUintens':
            selected_columns = au_r_columns + ['confidence', 'success']
        elif self.feature_type == 'confidence_success_AUoccurr':
            selected_columns = au_c_columns + ['confidence', 'success']
        elif self.feature_type == 'confidence_success_AUoccurr_pose_gaze':
            selected_columns = au_c_columns + gaze_columns + pose_columns + ['confidence', 'success']
        elif self.feature_type == 'confidence_success_AUintens_pose_gaze':
            selected_columns = au_r_columns + gaze_columns + pose_columns + ['confidence', 'success']
        elif self.feature_type == 'pose_gaze':
            selected_columns = gaze_columns + pose_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'pose_au_r':
            selected_columns = pose_columns + au_r_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'pose_au_c':
            selected_columns = pose_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'gaze_au_r':
            selected_columns = gaze_columns + au_r_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'gaze_au_c':
            selected_columns = gaze_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'au_r_au_c':
            selected_columns = au_r_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'pose_gaze_au_r':
            selected_columns = pose_columns + gaze_columns + au_r_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'pose_gaze_au_c':
            selected_columns = pose_columns + gaze_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'pose_au_r_au_c':
            selected_columns = pose_columns + au_r_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'gaze_au_r_au_c':
            selected_columns = gaze_columns + au_r_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'all':
            selected_columns = pose_columns + gaze_columns + au_r_columns + au_c_columns
            # Filter by confidence
            indices_to_remove = features_df[features_df['confidence'] < 0.9].index
            features_df = features_df.drop(index=indices_to_remove)
        elif self.feature_type == 'confidence_success_all':
            selected_columns = None
        else:
            selected_columns = None  # Use all columns if feature_type is not recognized
        
        if selected_columns is not None:
            features_df = features_df[selected_columns]
            # Remove rows with NaN values
            features_df = features_df.dropna()
            features = features_df.values
        else:
            # Remove rows with NaN values
            features_df = features_df.dropna()
            features = features_df.values
                
        if self.max_seq_length is not None:
            padded_features = np.zeros((self.max_seq_length, features.shape[1]))
            padded_features[:features.shape[0], :features.shape[1]] = features
            # print('padded_features.shape: ', padded_features.shape)
            features = padded_features

        # Convert DataFrame to PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Remove NaN values from features tensor
        features_tensor = features_tensor[~torch.isnan(features_tensor).any(dim=1)]

        return features_tensor, torch.tensor(phq_score, dtype=torch.float32)
        # print('features.shape at the end: ', features.shape)
        # return torch.tensor(features, dtype=torch.float32), torch.tensor(phq_score, dtype=torch.float32)

# class Attention(nn.Module):
#     def __init__(self, feature_dim, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#         self.feature_dim = feature_dim
#         self.proj = nn.Linear(feature_dim, 64)
#         self.context_vector = nn.Linear(64, 1, bias=False)

#     def forward(self, x):
#         x_proj = torch.tanh(self.proj(x))
#         context_vector = self.context_vector(x_proj).squeeze(2)
#         attention_weights = torch.softmax(context_vector, dim=1)
#         weighted = torch.mul(x, attention_weights.unsqueeze(-1).expand_as(x))
#         return torch.sum(weighted, dim=1)

# class Attention(nn.Module):
#     def __init__(self, feature_dim, hidden_dim, num_layers=1, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#         self.feature_dim = feature_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         # Define additional layers for more complex attention mechanism
#         self.proj_layers = nn.ModuleList([
#             nn.Linear(feature_dim, hidden_dim) for _ in range(num_layers)
#         ])
#         self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

#     def forward(self, x):
#         # Apply additional layers for more complex projections
#         for layer in self.proj_layers:
#             x = torch.tanh(layer(x))
#         context_vector = self.context_vector(x).squeeze(2)
#         attention_weights = torch.softmax(context_vector, dim=1)
#         weighted = torch.mul(x, attention_weights.unsqueeze(-1).expand_as(x))
#         return torch.sum(weighted, dim=1)

class Attention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        energy = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.feature_dim) # scaled dot-product attention
        attention_weights = self.softmax(energy)
        value = self.value(x)
        weighted = torch.matmul(attention_weights, value)
        return weighted.sum(dim=1)

class EnhancedPHQLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_rate=0.5):
        super(EnhancedPHQLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        # final_output = self.fc(lstm_out[:, -1, :])  # Taking the last output of the sequence to check the performance without attention layer
        final_output = self.fc(attn_out)
        return final_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
# model = EnhancedPHQLSTM(input_size=51, hidden_size=63, num_layers=3, dropout_rate=0.3).to(device)

criterion = nn.MSELoss()
mae = nn.L1Loss()

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_stack = torch.stack(yy, dim=0)
    return xx_pad, yy_stack, x_lens

# Define different combinations of feature types
feature_combinations = ['all', 'pose','gaze','au_r','au_c', \
                        'pose_gaze', 'pose_au_r', 'pose_au_c', \
                        'gaze_au_r', 'gaze_au_c', \
                        'au_r_au_c', \
                        'pose_gaze_au_r', 'pose_gaze_au_c', 'pose_au_r_au_c', 'gaze_au_r_au_c', \
                        'confidence_success_all', \
                        'confidence_success_AUintens', 'confidence_success_AUoccurr', \
                        'confidence_success_pose','confidence_success_gaze', \
                        'confidence_success_AUintens_pose_gaze', \
                        'confidence_success_AUoccurr_pose_gaze']

train_losses_list = []
val_losses_list = []
val_maes_list = []
models = []

num_epochs = 20
for feature_type in feature_combinations:
    print('feature_type: ', feature_type)

    # Create datasets
    train_dataset = PHQDataset(train_csv_path, open_face_train, feature_type=feature_type)
    dev_dataset = PHQDataset(dev_csv_path, open_face_dev, feature_type=feature_type)
    test_dataset = PHQDataset(test_csv_path, open_face_test, feature_type=feature_type)

    # print('train_dataset tensor shape: ', train_dataset[0][0].shape)
    # print('dev_dataset tensor shape: ', dev_dataset[0][0].shape)
    # print('test_dataset tensor shape: ', test_dataset[0][0].shape)
    # print('train_dataset[0][0].size(-1): ', train_dataset[0][0].size(-1))

    # Initialize the model with the current feature types
    model = EnhancedPHQLSTM(train_dataset[0][0].size(-1), hidden_size=64, num_layers=3, dropout_rate=0.3).to(device)
    # model = EnhancedPHQLSTM(train_dataset[0][0].size(-1), hidden_size=64, attention_hidden_dim=32, attention_num_layers=2, num_layers=3, dropout_rate=0.3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    models.append(model)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)

    # Initialize lists for storing losses
    train_losses = []
    val_losses = []
    val_maes = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        for i, (features, phq_scores, _) in enumerate(train_loader):
            # print('i train: ', i)
            features, phq_scores = features.to(device), phq_scores.to(device)
            # print('features.shape: ', features.shape)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = torch.sqrt(criterion(outputs, phq_scores))  # RMSE
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss train: {loss.item():.4f}')

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            epoch_val_losses = []
            epoch_val_maes = []
            for i, (features, phq_scores, _) in enumerate(dev_loader):
                # print('i dev: ', i)
                features, phq_scores = features.to(device), phq_scores.to(device)
                outputs = model(features).squeeze()
                val_loss = torch.sqrt(criterion(outputs, phq_scores))
                val_mae = mae(outputs, phq_scores)
                epoch_val_losses.append(val_loss.item())
                epoch_val_maes.append(val_mae.item())

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        avg_val_mae = sum(epoch_val_maes) / len(epoch_val_maes)
        val_maes.append(avg_val_mae)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation RMSE Loss: {avg_val_loss:.4f}, Validation MAE Loss: {avg_val_mae:.4f}')

    train_losses_list.append(train_losses)
    val_losses_list.append(val_losses)
    val_maes_list.append(val_maes)
    # Test loop for each feature combination
    model.eval()
    with torch.no_grad():
        for i, (features, phq_scores, _) in enumerate(test_loader):
            # print('i test: ', i)
            features, phq_scores = features.to(device), phq_scores.to(device)
            outputs = model(features).squeeze()
            test_loss = torch.sqrt(criterion(outputs, phq_scores))
            test_mae = mae(outputs, phq_scores)
        print(f'Test RMSE Loss ({feature_type}): {test_loss.item()}, Test MAE Loss ({feature_type}): {test_mae.item()}')

    print('---------------------------------')

plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plot the training and validation losses for each feature combination
for i, feature_type in enumerate(feature_combinations):
    plt.plot(train_losses_list[i], label=f'Training Loss ({feature_type})')
    plt.plot(val_losses_list[i], label=f'Validation Loss ({feature_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'loss_plot_{feature_type}.png'))
    plt.close()