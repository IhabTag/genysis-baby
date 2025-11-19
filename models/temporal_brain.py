import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal


class TemporalBrain(nn.Module):
    """
    Maintains working memory over sequences of latent states.
    
    Processes sequences of (z_t, a_t) pairs to maintain a hidden state h_t
    that represents the agent's "working memory" of recent events.
    
    Input: sequence of (z_t, a_t) pairs
    State: h_t (hidden working memory)
    Output: c_t (context vector for decision making)
    
    Example:
        >>> brain = TemporalBrain(latent_dim=256, action_dim=6)
        >>> z_t = torch.randn(1, 256)
        >>> a_t = torch.randn(1, 6)
        >>> c_t, h_t = brain(z_t, a_t)
        >>> print(c_t.shape)  # (1, 128)
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 6,
        hidden_dim: int = 256,
        context_dim: int = 128,
        num_layers: int = 2,
        model_type: Literal["gru", "lstm", "transformer"] = "gru",
        dropout: float = 0.1
    ):
        """
        Initialize temporal brain.
        
        Args:
            latent_dim: Dimension of latent state z_t from world model
            action_dim: Dimension of action vector a_t
            hidden_dim: Dimension of hidden state h_t
            context_dim: Dimension of output context vector c_t
            num_layers: Number of recurrent layers
            model_type: Type of sequence model ("gru", "lstm", or "transformer")
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        input_dim = latent_dim + action_dim
        
        # Create sequence model
        if model_type == "gru":
            self.rnn = nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif model_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif model_type == "transformer":
            # For transformer, we need input_dim to be divisible by num_heads
            # Project input to a dimension divisible by num_heads
            self.num_heads = 4
            # Find nearest dimension divisible by num_heads
            projected_dim = ((input_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
            
            self.input_projection = nn.Linear(input_dim, projected_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=projected_dim,
                nhead=self.num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # Project transformer output to hidden_dim
            self.transformer_proj = nn.Linear(projected_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Project hidden state to context vector
        self.context_head = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim)
        )
        
        # Initialize hidden state (will be set during forward)
        self.hidden = None
    
    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """
        Reset working memory at episode start.
        
        Args:
            batch_size: Batch size for hidden state
            device: Device to place hidden state on
        """
        self.hidden = None
    
    def forward(
        self, 
        z_t: torch.Tensor, 
        a_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Process one timestep and update working memory.
        
        Args:
            z_t: (batch, latent_dim) - current latent state
            a_t: (batch, action_dim) - current action
        
        Returns:
            c_t: (batch, context_dim) - context vector for decision making
            h_t: hidden state (for next step) - format depends on model_type
        """
        # Concatenate latent and action
        x = torch.cat([z_t, a_t], dim=-1)  # (batch, latent_dim + action_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim) - sequence length 1
        
        if self.model_type in ["gru", "lstm"]:
            # Initialize hidden state if needed
            if self.hidden is None:
                batch_size = x.size(0)
                device = x.device
                
                if self.model_type == "gru":
                    self.hidden = torch.zeros(
                        self.num_layers, batch_size, self.hidden_dim,
                        device=device
                    )
                else:  # lstm
                    h0 = torch.zeros(
                        self.num_layers, batch_size, self.hidden_dim, 
                        device=device
                    )
                    c0 = torch.zeros(
                        self.num_layers, batch_size, self.hidden_dim, 
                        device=device
                    )
                    self.hidden = (h0, c0)
            
            # Process sequence
            out, self.hidden = self.rnn(x, self.hidden)
            out = out[:, -1, :]  # Take last timestep (batch, hidden_dim)
        
        else:  # transformer
            # Transformers don't maintain hidden state in the same way
            # Project input first
            x_proj = self.input_projection(x)  # (batch, 1, projected_dim)
            out = self.rnn(x_proj)  # (batch, 1, projected_dim)
            out = out[:, -1, :]  # (batch, projected_dim)
            out = self.transformer_proj(out)  # (batch, hidden_dim)
        
        # Generate context vector
        c_t = self.context_head(out)  # (batch, context_dim)
        
        return c_t, self.hidden
    
    def forward_sequence(
        self, 
        z_seq: torch.Tensor, 
        a_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Process entire sequence at once (for training or planning).
        
        This is more efficient than calling forward() repeatedly.
        
        Args:
            z_seq: (batch, seq_len, latent_dim) - sequence of latent states
            a_seq: (batch, seq_len, action_dim) - sequence of actions
        
        Returns:
            c_seq: (batch, seq_len, context_dim) - context vectors for each timestep
        """
        # Concatenate latents and actions
        x = torch.cat([z_seq, a_seq], dim=-1)  # (batch, seq_len, input_dim)
        
        if self.model_type in ["gru", "lstm"]:
            out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
        else:  # transformer
            x_proj = self.input_projection(x)  # (batch, seq_len, projected_dim)
            out = self.rnn(x_proj)  # (batch, seq_len, projected_dim)
            out = self.transformer_proj(out)  # (batch, seq_len, hidden_dim)
        
        # Generate context vectors for all timesteps
        batch_size, seq_len, _ = out.shape
        out_flat = out.reshape(batch_size * seq_len, self.hidden_dim)
        c_flat = self.context_head(out_flat)  # (batch * seq_len, context_dim)
        c_seq = c_flat.reshape(batch_size, seq_len, self.context_dim)
        
        return c_seq
    
    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get current hidden state (for saving/debugging).
        
        Returns:
            Current hidden state, or None if not initialized
        """
        return self.hidden
    
    def set_hidden_state(self, hidden: Optional[Tuple[torch.Tensor, ...]]):
        """
        Set hidden state (for loading from checkpoint).
        
        Args:
            hidden: Hidden state to restore
        """
        self.hidden = hidden
    
    def save_state(self, path: str):
        """
        Save model weights and current hidden state.
        
        Args:
            path: Path to save checkpoint
        """
        state = {
            'model_state_dict': self.state_dict(),
            'hidden_state': self.hidden,
            'config': {
                'latent_dim': self.latent_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'context_dim': self.context_dim,
                'num_layers': self.num_layers,
                'model_type': self.model_type
            }
        }
        torch.save(state, path)
    
    @classmethod
    def load_state(cls, path: str, device: str = "cpu") -> "TemporalBrain":
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
        
        Returns:
            Loaded TemporalBrain instance
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create instance with saved config
        config = checkpoint['config']
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore hidden state
        model.hidden = checkpoint.get('hidden_state')
        
        return model.to(device)
