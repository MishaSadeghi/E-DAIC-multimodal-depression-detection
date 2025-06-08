import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """
    A temporal attention mechanism to weigh the importance of different time steps,
    while accounting for padded sequences.
    """
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.feature_dim = feature_dim
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, lstm_output, mask):
        """
        Args:
            lstm_output (torch.Tensor): The output from the LSTM layer.
                                        Shape: (batch_size, seq_len, hidden_size)
            mask (torch.Tensor): A boolean mask indicating non-padded elements.
                                 Shape: (batch_size, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention_net(lstm_output).squeeze(2) # Shape: (batch_size, seq_len)
        
        # Mask out padded time steps by setting their scores to a very low number
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1) # Shape: (batch_size, seq_len)
        
        # Calculate context vector
        context_vector = torch.sum(lstm_output * attention_weights.unsqueeze(2), dim=1)
        # Shape: (batch_size, hidden_size)
        
        return context_vector, attention_weights

class VideoLSTMModel(nn.Module):
    """
    An LSTM-based regression model for video feature sequences, using temporal attention.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(VideoLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            # Apply dropout only if there are multiple LSTM layers
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.attention = TemporalAttention(hidden_size)
        self.regressor = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of video features.
                              Shape: (batch_size, seq_len, input_size)
        """
        # Create a mask for padded sequences. Assumes padding is all zeros.
        # Shape: (batch_size, seq_len)
        mask = (torch.sum(x, dim=2) != 0)
        
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(lstm_out, mask=mask)
        
        # Final regression layer
        output = self.regressor(context_vector)
        
        return output.squeeze(1), attention_weights 

    def get_feature_representation(self, x):
        """
        Extracts the feature representation from the model before the final regression layer.
        Args:
            x (torch.Tensor): Input tensor of video features.
                              Shape: (batch_size, seq_len, input_size)
        """
        mask = (torch.sum(x, dim=2) != 0)
        lstm_out, _ = self.lstm(x)
        context_vector, _ = self.attention(lstm_out, mask=mask)
        return context_vector 