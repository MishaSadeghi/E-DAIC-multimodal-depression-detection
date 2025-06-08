import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.video_data_loader import create_data_loaders
from src.models.video_model import VideoLSTMModel

def train_epoch(model, data_loader, optimizer, device):
    """A single training epoch."""
    model.train()
    total_loss = 0
    for sequences, labels in data_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions, _ = model(sequences)
        loss = nn.MSELoss()(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Evaluate the model on the validation or test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            predictions, _ = model(sequences)
            loss = nn.MSELoss()(predictions, labels)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
    r2 = r2_score(all_labels, all_predictions)
    
    return avg_loss, rmse, r2

def main(args):
    """Main training and evaluation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        feature_type=args.feature_type,
        batch_size=args.batch_size
    )

    # Initialize model, optimizer
    # We need to get the input size from the first batch of data
    sample_batch, _ = next(iter(train_loader))
    input_size = sample_batch.shape[2]
    
    model = VideoLSTMModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, val_r2 = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R^2: {val_r2:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.model_save_dir, exist_ok=True)
            model_path = os.path.join(args.model_save_dir, f"best_video_model_{args.feature_type}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Final evaluation on the test set
    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_rmse, test_r2 = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f} | Test R^2: {test_r2:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM model on video features.")
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='/home/hpc/empk/empk004h/depression-detection/data', help='Directory with DAIC_open_face_* folders.')
    parser.add_argument('--label_dir', type=str, default='/home/hpc/empk/empk004h/depression-detection/data/labels', help='Directory with train/dev/test split CSVs.')
    parser.add_argument('--model_save_dir', type=str, default='./models', help='Directory to save the best model.')

    # Model parameters
    parser.add_argument('--feature_type', type=str, default='all', help='Feature set to use (e.g., all, pose, gaze).')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in LSTM.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate.')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')

    args = parser.parse_args()
    main(args) 