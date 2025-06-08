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
from torchviz import make_dot

# Directory to save models
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Directory to save models' visualizations
visualizations_dir = "model_visualizations"
os.makedirs(visualizations_dir, exist_ok=True)

# Directory to save models' SHAP values
shaps_dir = "shaps"
os.makedirs(shaps_dir, exist_ok=True)

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
        self.features_df = None  # Initialize features_df attribute

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

        # Store features_df
        self.features_df = features_df
        
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

    @property
    def get_features_df(self):
        """Getter method to retrieve the features_df attribute."""
        return self.features_df

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

    print('train_dataset[0][0]: ', train_dataset[0][0])

    # Access features_df after dataset instantiation
    train_features_df = train_dataset[0][0].features_df
    dev_features_df = dev_dataset[0][0].features_df
    test_features_df = test_dataset[0][0].features_df

    print('test_features_df.columns: ', test_features_df.columns)
    print('train_features_df.columns: ', train_features_df.columns)

    # print('train_dataset tensor shape: ', train_dataset[0][0].shape)
    # print('dev_dataset tensor shape: ', dev_dataset[0][0].shape)
    # print('test_dataset tensor shape: ', test_dataset[0][0].shape)
    # print('train_dataset[0][0].size(-1): ', train_dataset[0][0].size(-1))

    num_features = train_dataset[0][0].size(-1)

    # Initialize the model with the current feature types
    model = EnhancedPHQLSTM(train_dataset[0][0].size(-1), hidden_size=64, num_layers=3, dropout_rate=0.3).to(device)

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

    # Save models with their respective feature type
    model_save_path = os.path.join(model_dir, f"model_{feature_type}.pt")
    torch.save(model, model_save_path)

    # # Dummy input to the model for visualization
    # dummy_input = torch.randn(8, 10, num_features)

    # # Generate a visualization of the model
    # dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))

    # # Save the visualization
    # visualization_path = os.path.join(visualizations_dir, f"model_architecture_{feature_type}.png")
    # dot.render(visualization_path, format="png")

    test_features, test_labels, _ = next(iter(test_loader))
    test_features = test_features.to(device)
    # Check for NaN values in test_features
    nan_check = torch.isnan(test_features)
    if nan_check.any():
        print("NaN values found in test_features. Locations with NaN:")
        print(torch.nonzero(nan_check))
    else:
        print("No NaN values found in test_features.")

    ######
    test_features = np.nan_to_num(test_features)
    test_features =pd.DataFrame(data=test_features,columns=test_features_df.columns)

    # Initialize SHAP explainer
    explainer = shap.DeepExplainer(model, test_features)

    # Calculate SHAP values
    shap_values = explainer.shap_values(test_features)

    # # SHAP values and test features averages over time steps for visualization
    # avg_shap_values = shap_values.mean(axis=1)
    # avg_test_features = test_features.mean(axis=1).cpu().numpy()
    # shap.initjs()
    # # shap.summary_plot(avg_shap_values, avg_test_features[:5])
    # shap.summary_plot(avg_shap_values, avg_test_features)

    shap.summary_plot(shap_values, test_features, plot_type="bar")
    plt.savefig(os.path.join(shaps_dir, f'shap_summary_plot_{feature_type}.png'))
    plt.close()


# Create a directory called "plots" if it doesn't exist
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



# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import permutation_importance
# import shap
# from matplotlib import pyplot as plt
# rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X_train, y_train)
# rf.feature_importances_
# ######
# X_test= np.nan_to_num(X_test)
# X_test=pd.DataFrame(data=X_test,columns=X_normalized.columns)
# perm_importance = permutation_importance(rf, X_test, y_test)
# ######
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X_test)
# fig, ax = plt.subplots(figsize=(20,20))
# shap.summary_plot(shap_values, X_test, plot_type=“bar”,show=False)
# fig=shap.summary_plot(shap_values, X_test,show=False)