# Install SHAP and Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap

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
    def __init__(self, csv_file, data_folder, max_seq_length=None):
        self.data_folder = data_folder
        self.data_info = pd.read_csv(csv_file)
        self.max_seq_length = max_seq_length
        # print('csv_file: ', self.data_info)
        # print('data_folder: ', self.data_folder)

    def __len__(self):
        # print('len(self.data_info): ', len(self.data_info))
        return len(self.data_info)

    def __getitem__(self, idx):
        participant_id = self.data_info.iloc[idx]['Participant_ID']
        # print('participant_id: ', participant_id)

        phq_score = self.data_info.iloc[idx]['PHQ_Score']
        # print('phq_score: ', phq_score)

        filepath = os.path.join(self.data_folder, f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        # print('filepath: ', filepath)

        features = pd.read_csv(filepath).to_numpy()
        features = features[:, 2:] # Remove the first two features (frame and timestamps)
        # print('features shape: ', features.shape)
        
        if self.max_seq_length is not None:
            # Padding
            # print('features.shape[1]: ', features.shape[1])
            # print('features.shape[0]: ', features.shape[0])
            
            padded_features = np.zeros((self.max_seq_length, features.shape[1]))
            padded_features[:features.shape[0], :features.shape[1]] = features
            print('padded_features.shape: ', padded_features.shape)
            features = padded_features

        return torch.tensor(features, dtype=torch.float32), torch.tensor(phq_score, dtype=torch.float32)

class Attention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, 64)
        self.context_vector = nn.Linear(64, 1, bias=False)

    def forward(self, x):
        x_proj = torch.tanh(self.proj(x))
        context_vector = self.context_vector(x_proj).squeeze(2)
        attention_weights = torch.softmax(context_vector, dim=1)
        weighted = torch.mul(x, attention_weights.unsqueeze(-1).expand_as(x))
        return torch.sum(weighted, dim=1)
    
class EnhancedPHQLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_rate=0.5):
        super(EnhancedPHQLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # print('lstm_out.shape: ', lstm_out.shape)
        attn_out = self.attention(lstm_out)
        # print('attn_out.shape: ', attn_out.shape)
        # final_output = self.fc(lstm_out[:, -1, :])  # Taking the last output of the sequence to check the performance without attention layer
        final_output = self.fc(attn_out)
        return final_output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# confidence	success	pose_Tx	pose_Ty	pose_Tz	pose_Rx	pose_Ry	pose_Rz	gaze_0_x	gaze_0_y	gaze_0_z	gaze_1_x	gaze_1_y	gaze_1_z	gaze_angle_x	gaze_angle_y	AU01_r	AU02_r	AU04_r	AU05_r	AU06_r	AU07_r	AU09_r	AU10_r	AU12_r	AU14_r	AU15_r	AU17_r	AU20_r	AU23_r	AU25_r	AU26_r	AU45_r	AU01_c	AU02_c	AU04_c	AU05_c	AU06_c	AU07_c	AU09_c	AU10_c	AU12_c	AU14_c	AU15_c	AU17_c	AU20_c	AU23_c	AU25_c	AU26_c	AU28_c	AU45_c

# Initialize the model
model = EnhancedPHQLSTM(input_size=51, hidden_size=64, num_layers=3, dropout_rate=0.3).to(device)
criterion_mse = nn.MSELoss()
criterion = nn.L1Loss()  # Mean Absolute Error criterion
mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Create datasets and dataloaders
train_dataset = PHQDataset(train_csv_path, open_face_train)
# print('train_dataset: ', train_dataset)

dev_dataset = PHQDataset(dev_csv_path, open_face_dev)
# print('dev_dataset: ', dev_dataset)

test_dataset = PHQDataset(test_csv_path, open_face_test)
# print('test_dataset: ', test_dataset)

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    yy_stack = torch.stack(yy, dim=0)

    return xx_pad, yy_stack, x_lens

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=pad_collate)

train_losses = []
val_losses = []
val_maes = []
val_rmses = []

num_epochs = 20

for epoch in range(num_epochs):
    # Training loop
    model.train()
    epoch_train_losses = []
    for i, (features, phq_scores, _) in enumerate(train_loader):
        features, phq_scores = features.to(device), phq_scores.to(device)
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        # loss = torch.sqrt(criterion(outputs, phq_scores)) #RMSE
        loss = criterion(outputs, phq_scores)  # MAE
        loss.backward()
        optimizer.step()
        epoch_train_losses.append(loss.item())

        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}') # Check the progress

    # Training loss (average) for the epoch
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    with torch.no_grad():
        epoch_val_losses = []
        epoch_val_maes = []
        epoch_val_rmses = []
        for i, (features, phq_scores, _) in enumerate(dev_loader):
            features, phq_scores = features.to(device), phq_scores.to(device)
            outputs = model(features).squeeze()
            # val_loss = torch.sqrt(criterion(outputs, phq_scores))
            val_loss = criterion(outputs, phq_scores)  # MAE
            # val_mae = mae(outputs, phq_scores)
            val_rmse = torch.sqrt(criterion_mse(outputs, phq_scores))
            epoch_val_losses.append(val_loss.item())
            # epoch_val_maes.append(val_mae.item())
            epoch_val_rmses.append(val_rmse.item())

    # Validation loss (RMSE and MAE) for the epoch
    avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
    val_losses.append(avg_val_loss)
    # avg_val_mae = sum(epoch_val_maes) / len(epoch_val_maes)
    # val_maes.append(avg_val_mae)
    avg_val_rmse = sum(epoch_val_rmses) / len(epoch_val_rmses)
    val_rmses.append(avg_val_rmse)

    # print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation RMSE Loss: {avg_val_loss:.4f}, Validation MAE Loss: {avg_val_mae:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation MAE Loss: {avg_val_loss:.4f}, Validation RMSE Loss: {avg_val_rmse:.4f}')

# Plot the training and validation losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')

# plt.show()

# Test loop
model.eval()
with torch.no_grad():
    for i, (features, phq_scores, _) in enumerate(test_loader):
        features, phq_scores = features.to(device), phq_scores.to(device)
        outputs = model(features).squeeze()
        test_loss = torch.sqrt(criterion_mse(outputs, phq_scores))
        test_mae = mae(outputs, phq_scores)
    # print(f'Test RMSE Loss: {test_loss.item()}, Test MAE Loss: {test_mae.item()}')
    print(f'Test MAE Loss: {test_mae.item()}, Test RMSE Loss: {test_loss.item()}')


# # Save and load the model
# torch.save(model, '/content/drive/MyDrive/Data/model_e10_dr3.pth')
# loaded_model = torch.load('/content/drive/MyDrive/Data/model_e10_dr3.pth')

# # Calculate and visualize the feature importances

# model.eval()

# test_features, test_labels, _ = next(iter(test_loader))
# test_features = test_features.to(device)

# torch.backends.cudnn.enabled = False

# # Check for NaN values in test_features
# nan_check = torch.isnan(test_features)
# if nan_check.any():
#     print("NaN values found in test_features. Locations with NaN:")
#     print(torch.nonzero(nan_check))
# else:
#     print("No NaN values found in test_features.")

# print('test_features: ', test_features)

# shap_kwargs=dict(check_additivity=False)

# # explainer = shap.DeepExplainer(model, test_features[:5])  # Restricted to 5 due to memory limit
# explainer = shap.DeepExplainer(model, test_features, shap_kwargs=shap_kwargs)
# # shap_values = explainer.shap_values(test_features[:5])
# shap_values = explainer.shap_values(test_features)

# print("SHAP Values:", shap_values)
# print("Model Output:", model(test_features).cpu().detach().numpy())

# torch.backends.cudnn.enabled = True

# # SHAP values and test features averages over time steps for visualization
# avg_shap_values = shap_values.mean(axis=1)
# avg_test_features = test_features.mean(axis=1).cpu().numpy()

# shap.initjs()
# # shap.summary_plot(avg_shap_values, avg_test_features[:5])
# shap.summary_plot(avg_shap_values, avg_test_features)

# plt.savefig('shap_summary_plot.png')