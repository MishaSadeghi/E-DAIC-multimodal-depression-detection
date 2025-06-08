import os
import argparse
import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.text_data_loader import load_and_extract_text_features

def find_best_svr_params(X_train, y_train, X_dev, y_dev):
    """
    Performs a grid search to find the best SVR hyperparameters.
    """
    print("\n--- Finding best SVR parameters ---")
    C_range = [1, 2, 3.5, 10, 20, 100]
    gamma_range = [0.001, 0.01, 0.1, 1, 10]
    
    best_mae = float('inf')
    best_params = {}

    for c in C_range:
        for gamma in gamma_range:
            svr = SVR(kernel='rbf', C=c, gamma=gamma)
            svr.fit(X_train, y_train)
            y_pred_dev = svr.predict(X_dev)
            mae = mean_absolute_error(y_dev, y_pred_dev)
            
            print(f"Testing C={c}, gamma={gamma} -> MAE: {mae:.4f}")
            
            if mae < best_mae:
                best_mae = mae
                best_params = {'C': c, 'gamma': gamma}

    print(f"\nBest SVR params found: {best_params} with MAE: {best_mae:.4f}")
    return best_params

def main(args):
    """Main function to train and evaluate the SVR model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and extract features
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_and_extract_text_features(
        label_dir=args.label_dir,
        text_data_path=args.text_data_path,
        model_name=args.model_name,
        device=device
    )
    print("\nFeature extraction complete.")
    print(f"Train features shape: {X_train.shape}")
    print(f"Dev features shape: {X_dev.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Find best SVR parameters using the validation set
    best_params = find_best_svr_params(X_train, y_train, X_dev, y_dev)
    
    # Train final model on the full training data with best params
    print("\n--- Training final SVR model ---")
    final_svr = SVR(kernel='rbf', **best_params)
    final_svr.fit(X_train, y_train)
    
    # Evaluate final model
    print("\n--- Final Model Evaluation ---")
    sets = {'Train': (X_train, y_train), 'Dev': (X_dev, y_dev), 'Test': (X_test, y_test)}
    
    for set_name, (X, y) in sets.items():
        y_pred = final_svr.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        print(f"Results for {set_name} set:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SVR on DepRoBERTa text features.")
    
    # Paths
    parser.add_argument('--label_dir', type=str, default='/home/hpc/empk/empk004h/depression-detection/data/DAIC/labels', help='Directory with train/dev/test split CSVs.')
    parser.add_argument('--text_data_path', type=str, default='/home/hpc/empk/empk004h/depression-detection/data/DAIC/DAIC_Chatgpt_text_1.xlsx', help='Path to the Excel file with ChatGPT processed text.')
    parser.add_argument('--model_name', type=str, default='rafalposwiata/deproberta-large-depression', help='Name of the Hugging Face model for feature extraction.')

    args = parser.parse_args()
    main(args) 