import os
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.text_data_loader import create_text_data_loaders
from src.models.text_model import build_deproberta_model

def train_epoch(model, data_loader, optimizer, device):
    """A single training epoch for the text model."""
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['ids'].to(device)
        attention_mask = batch['mask'].to(device)
        labels = batch['targets'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Evaluate the text model on the validation or test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['ids'].to(device)
            attention_mask = batch['mask'].to(device)
            labels = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def main(args):
    """Main training and evaluation loop for the text model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model, _ = build_deproberta_model(num_labels=args.num_labels)
    model.to(device)

    # Create data loaders
    train_loader, val_loader, test_loader = create_text_data_loaders(
        data_dir=args.data_dir,
        prompt_name=args.prompt,
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.model_save_dir, exist_ok=True)
            model_path = os.path.join(args.model_save_dir, f"best_text_model_{args.prompt}.bin")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    # Final evaluation
    print("\n--- Final Evaluation on Test Set ---")
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune DepRoBERTa model for depression classification.")
    
    # Paths and identifiers
    parser.add_argument('--data_dir', type=str, default='/home/hpc/empk/empk004h/depression-detection/data/revised_transcripts_completions/', help='Directory containing the prompt folders.')
    parser.add_argument('--prompt', type=str, required=True, help='Name of the prompt folder to process.')
    parser.add_argument('--model_save_dir', type=str, default='./models', help='Directory to save the best model.')
    parser.add_argument('--model_name', type=str, default='rafalposwiata/deproberta-large-depression', help='Name of the Hugging Face model.')

    # Model and training parameters
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length for tokenizer.')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of output labels.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=5e-7, help='Learning rate.')

    args = parser.parse_args()
    main(args) 