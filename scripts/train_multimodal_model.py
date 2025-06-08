import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import os

from src.data.multimodal_data_loader import load_multimodal_data

def main(video_feature_dir, text_feature_dir, video_model_path, output_dir):
    """
    Main function to train and evaluate the multimodal SVR model.
    """
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    features, labels = load_multimodal_data(
        video_feature_dir=video_feature_dir,
        text_feature_dir=text_feature_dir,
        video_model_path=video_model_path,
        device=device
    )

    # Apply PCA on video features, as done in the notebook
    pca_video = PCA(n_components=10)
    
    # Fit on training data and transform all sets
    video_train_pca = pca_video.fit_transform(features['train']['video'])
    video_dev_pca = pca_video.transform(features['dev']['video'])
    video_test_pca = pca_video.transform(features['test']['video'])
    
    # Combine all features for each set
    X_train = np.concatenate((features['train']['text'], features['train']['quest'], video_train_pca), axis=1)
    X_dev = np.concatenate((features['dev']['text'], features['dev']['quest'], video_dev_pca), axis=1)
    X_test = np.concatenate((features['test']['text'], features['test']['quest'], video_test_pca), axis=1)

    y_train = labels['train']
    y_dev = labels['dev']
    y_test = labels['test']
    
    print(f"Shape of final training features: {X_train.shape}")
    print(f"Shape of final dev features: {X_dev.shape}")
    print(f"Shape of final test features: {X_test.shape}")

    # Train SVR model with GridSearchCV
    print("\nTraining SVR model with GridSearchCV...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5, 1]
    }
    
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    
    # The notebook trains on train+dev, but for a cleaner pipeline, let's train on train and validate on dev.
    # We can train on train+dev before final testing if desired.
    # For now, following standard practice:
    grid_search.fit(X_train, y_train)
    
    best_svr = grid_search.best_estimator_
    print(f"Best SVR parameters found: {grid_search.best_params_}")

    # Evaluate on dev set
    y_pred_dev = best_svr.predict(X_dev)
    rmse_dev = np.sqrt(mean_squared_error(y_dev, y_pred_dev))
    mae_dev = mean_absolute_error(y_dev, y_pred_dev)
    print(f"\nEvaluation on Dev Set:")
    print(f"RMSE: {rmse_dev:.4f}")
    print(f"MAE: {mae_dev:.4f}")
    
    # Evaluate on test set
    y_pred_test = best_svr.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"\nEvaluation on Test Set:")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")

    # Save the model and results
    print(f"\nSaving model and results to {output_dir}")
    model_path = os.path.join(output_dir, 'multimodal_svr_model.joblib')
    joblib.dump(best_svr, model_path)
    
    results_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE'],
        'Dev': [rmse_dev, mae_dev],
        'Test': [rmse_test, mae_test]
    })
    results_path = os.path.join(output_dir, 'multimodal_model_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\nMultimodal model training and evaluation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the multimodal SVR model.")
    parser.add_argument('--video_feature_dir', type=str, required=True, help='Path to the directory containing video-related features (labels, OpenFace).')
    parser.add_argument('--text_feature_dir', type=str, required=True, help='Path to the directory containing text and questionnaire features.')
    parser.add_argument('--video_model_path', type=str, required=True, help='Path to the saved .pth file for the video LSTM model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model and results.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args.video_feature_dir, args.text_feature_dir, args.video_model_path, args.output_dir) 