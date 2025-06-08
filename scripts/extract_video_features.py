import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.video_data_loader import create_data_loaders
from src.models.video_model import VideoLSTMModel

class FeatureExtractor(nn.Module):
    """
    Wrapper around the VideoLSTMModel to extract features from the attention layer.
    """
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # We use all layers of the original model except the final regressor
        self.lstm = original_model.lstm
        self.attention = original_model.attention

    def forward(self, x):
        mask = (torch.sum(x, dim=2) != 0)
        lstm_out, _ = self.lstm(x)
        # The context_vector is the feature we want to extract
        context_vector, _ = self.attention(lstm_out, mask=mask)
        return context_vector

def extract_features(model, data_loader, device):
    """
    Extracts features and labels for a given dataset.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc="Extracting features"):
            sequences = sequences.to(device)
            features = model(sequences)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.vstack(all_features), np.array(all_labels)

def main(args):
    """Main function to load model, extract features, and save them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        feature_type=args.feature_type,
        batch_size=args.batch_size
    )

    # Load the trained model
    # We need input_size to initialize the model structure before loading weights
    sample_batch, _ = next(iter(train_loader))
    input_size = sample_batch.shape[2]

    trained_model = VideoLSTMModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=0 # Dropout is not needed for inference
    ).to(device)

    model_path = os.path.join(args.model_load_dir, f"best_video_model_{args.feature_type}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
        
    trained_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained model from {model_path}")

    # Create the feature extractor
    feature_extractor = FeatureExtractor(trained_model).to(device)

    # Create output directory
    os.makedirs(args.feature_save_dir, exist_ok=True)

    # Extract and save features for each data split
    for split_name, data_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"\nProcessing {split_name} set...")
        features, labels = extract_features(feature_extractor, data_loader, device)
        
        feature_path = os.path.join(args.feature_save_dir, f"video_features_{split_name}.npy")
        label_path = os.path.join(args.feature_save_dir, f"video_labels_{split_name}.npy")
        
        np.save(feature_path, features)
        np.save(label_path, labels)
        
        print(f"Saved {split_name} features to {feature_path} (shape: {features.shape})")
        print(f"Saved {split_name} labels to {label_path} (shape: {labels.shape})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features from a trained Video LSTM model.")
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='/home/hpc/empk/empk004h/depression-detection/data', help='Directory with DAIC_open_face_* folders.')
    parser.add_argument('--label_dir', type=str, default='/home/hpc/empk/empk004h/depression-detection/data/labels', help='Directory with train/dev/test split CSVs.')
    parser.add_argument('--model_load_dir', type=str, default='./models', help='Directory where the trained model is saved.')
    parser.add_argument('--feature_save_dir', type=str, default='./data/processed_features/video', help='Directory to save the extracted features.')

    # Model parameters (must match the trained model)
    parser.add_argument('--feature_type', type=str, default='all', help='Feature set used for the trained model.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in the trained LSTM.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the trained LSTM.')
    
    # Data Loader parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction.')

    args = parser.parse_args()
    main(args) 