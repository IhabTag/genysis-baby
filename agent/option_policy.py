"""
Option Policies: Micro-policies for executing specific behavioral skills.

Each option policy is a small neural network that:
1. Takes current state (z_t, c_t) as input
2. Outputs low-level action parameters
3. Decides when to terminate

Example options:
- scroll_down: Execute sustained downward scrolling
- move_to_top: Move mouse to top of screen
- click_text: Move to and click on text element
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import os
import json


class OptionPolicy(nn.Module):
    """
    Micro-policy for executing a specific option/skill.
    
    Input: z_t (latent state), optional c_t (temporal context)
    Output: action parameters + termination probability
    """
    
    def __init__(
        self,
        option_name: str,
        latent_dim: int = 256,
        context_dim: int = 128,
        action_dim: int = 6,
        hidden_dim: int = 128,
        use_context: bool = True
    ):
        """
        Initialize option policy.
        
        Args:
            option_name: Name of this option (e.g., "scroll_down")
            latent_dim: Dimension of latent state z_t
            context_dim: Dimension of temporal context c_t
            action_dim: Dimension of action output
            hidden_dim: Hidden layer size
            use_context: Whether to use temporal context
        """
        super().__init__()
        
        self.option_name = option_name
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.use_context = use_context
        
        input_dim = latent_dim
        if use_context:
            input_dim += context_dim
        
        # Policy network: state → action parameters
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Termination network: state → terminate probability
        self.termination = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        z_t: torch.Tensor, 
        c_t: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action and termination probability.
        
        Args:
            z_t: (batch, latent_dim) - latent state
            c_t: (batch, context_dim) - temporal context (optional)
        
        Returns:
            action_params: (batch, action_dim) - raw action parameters
            terminate_prob: (batch, 1) - probability of terminating option
        """
        # Concatenate inputs
        if self.use_context and c_t is not None:
            x = torch.cat([z_t, c_t], dim=-1)
        else:
            x = z_t
        
        # Compute outputs
        action_params = self.policy(x)
        terminate_prob = self.termination(x)
        
        return action_params, terminate_prob
    
    def decode_action(
        self, 
        action_params: torch.Tensor,
        screen_width: int = 1024,
        screen_height: int = 768
    ) -> Dict[str, Any]:
        """
        Convert raw parameters to executable action dict.
        
        The decoding depends on the option type.
        
        Args:
            action_params: (action_dim,) raw parameters
            screen_width, screen_height: Screen dimensions
        
        Returns:
            Executable action dictionary
        """
        params = action_params.detach().cpu().numpy()
        
        # Decode based on option type
        if "scroll" in self.option_name:
            # Scroll options: use first param for amount
            if "down" in self.option_name:
                amount = int(params[0] * 200 + 150)  # 150-350
            elif "up" in self.option_name:
                amount = int(params[0] * 200 - 150)  # -350 to -150
            else:
                amount = int(params[0] * 400 - 200)  # -200 to 200
            
            return {"type": "SCROLL", "amount": amount}
        
        elif "move" in self.option_name:
            # Movement options: use first two params for x, y
            x = int((params[0] * 0.5 + 0.5) * screen_width)  # Normalize to [0, 1] then scale
            y = int((params[1] * 0.5 + 0.5) * screen_height)
            
            # Clamp to screen bounds
            x = max(0, min(screen_width - 1, x))
            y = max(0, min(screen_height - 1, y))
            
            return {"type": "MOVE_MOUSE", "x": x, "y": y}
        
        elif "click" in self.option_name:
            # Click options: alternate between move and click
            # Use param to decide if we should click now
            should_click = params[0] > 0
            
            if should_click:
                if "right" in self.option_name:
                    return {"type": "RIGHT_CLICK"}
                else:
                    return {"type": "LEFT_CLICK"}
            else:
                x = int((params[1] * 0.5 + 0.5) * screen_width)
                y = int((params[2] * 0.5 + 0.5) * screen_height)
                x = max(0, min(screen_width - 1, x))
                y = max(0, min(screen_height - 1, y))
                return {"type": "MOVE_MOUSE", "x": x, "y": y}
        
        elif "type" in self.option_name or "typing" in self.option_name:
            # Typing options: for now, return a simple type action
            return {"type": "TYPE_TEXT", "text": "test"}
            
        elif "idle" in self.option_name:
            return {"type": "NOOP"}
        
        else:
            # Default: mouse movement
            x = int((params[0] * 0.5 + 0.5) * screen_width)
            y = int((params[1] * 0.5 + 0.5) * screen_height)
            x = max(0, min(screen_width - 1, x))
            y = max(0, min(screen_height - 1, y))
            return {"type": "MOVE_MOUSE", "x": x, "y": y}


class OptionLibrary:
    """
    Manages collection of learned option policies.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize option library.
        
        Args:
            device: Device to run policies on
        """
        self.options: Dict[str, OptionPolicy] = {}
        self.device = device
        self.metadata: Dict[str, Dict] = {}  # Store metadata per option
    
    def add_option(
        self, 
        name: str, 
        policy: OptionPolicy,
        metadata: Optional[Dict] = None
    ):
        """
        Add option policy to library.
        
        Args:
            name: Option name
            policy: OptionPolicy instance
            metadata: Optional metadata (training info, etc.)
        """
        self.options[name] = policy.to(self.device)
        self.options[name].eval()  # Set to eval mode by default
        
        if metadata:
            self.metadata[name] = metadata
    
    def get_option(self, name: str) -> Optional[OptionPolicy]:
        """
        Get option policy by name.
        
        Args:
            name: Option name
        
        Returns:
            OptionPolicy or None if not found
        """
        return self.options.get(name)
    
    def list_options(self) -> List[str]:
        """
        List all available option names.
        
        Returns:
            List of option names
        """
        return list(self.options.keys())
    
    def remove_option(self, name: str):
        """Remove option from library."""
        if name in self.options:
            del self.options[name]
        if name in self.metadata:
            del self.metadata[name]
    
    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for an option."""
        return self.metadata.get(name)
    
    def save(self, path: str):
        """
        Save all option policies to disk.
        
        Args:
            path: Path to save checkpoint
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        checkpoint = {
            'options': {
                name: {
                    'state_dict': policy.state_dict(),
                    'config': {
                        'option_name': policy.option_name,
                        'latent_dim': policy.latent_dim,
                        'context_dim': policy.context_dim,
                        'action_dim': policy.action_dim,
                        'use_context': policy.use_context
                    }
                }
                for name, policy in self.options.items()
            },
            'metadata': self.metadata
        }
        
        torch.save(checkpoint, path)
        print(f"Saved {len(self.options)} options to {path}")
    
    def load(self, path: str):
        """
        Load option policies from disk.
        
        Args:
            path: Path to checkpoint
        """
        if not os.path.exists(path):
            print(f"Option library not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        for name, opt_data in checkpoint['options'].items():
            config = opt_data['config']
            
            # Create policy
            policy = OptionPolicy(**config)
            policy.load_state_dict(opt_data['state_dict'])
            
            # Add to library
            self.add_option(name, policy)
        
        # Load metadata
        self.metadata = checkpoint.get('metadata', {})
        
        print(f"Loaded {len(self.options)} options from {path}")
    
    def __len__(self) -> int:
        """Number of options in library."""
        return len(self.options)
    
    def __contains__(self, name: str) -> bool:
        """Check if option exists."""
        return name in self.options


def create_default_options(
    latent_dim: int = 256,
    context_dim: int = 128,
    device: str = "cpu"
) -> OptionLibrary:
    """
    Create option library with default untrained policies.
    
    These can be trained later via behavioral cloning.
    
    Args:
        latent_dim: Latent dimension
        context_dim: Context dimension
        device: Device
    
    Returns:
        OptionLibrary with default options
    """
    library = OptionLibrary(device=device)
    
    # Define common option types
    option_types = [
        "scroll_down",
        "scroll_up",
        "scroll_small",
        "move_to_top",
        "move_to_left",
        "move_large",
        "move_local",
        "move_mouse",
        "click_sequence",
        "left_click",
        "right_click",
        "typing",
        "idle"
    ]
    
    for opt_type in option_types:
        policy = OptionPolicy(
            option_name=opt_type,
            latent_dim=latent_dim,
            context_dim=context_dim,
            use_context=False  # Don't require context by default
        )
        
        # --- Heuristic Initialization ---
        # Initialize weights to produce desired behavior immediately
        # This solves the "cold start" problem where all options act randomly
        
        # 1. Set termination bias to negative (encourage sustained action)
        # Sigmoid(-3.0) ~= 0.05 probability of termination per step
        policy.termination[-2].bias.data.fill_(-3.0)
        policy.termination[-2].weight.data.fill_(0.0)
        
        # 2. Set policy output bias based on option type
        # Output layer is Linear(hidden, action_dim)
        # We zero the weights so input doesn't affect output (initially)
        # And set bias to target values
        policy.policy[-1].weight.data.fill_(0.0)
        bias = policy.policy[-1].bias.data
        bias.fill_(0.0)  # Reset all to 0
        
        if opt_type == "scroll_down":
            # Param 0 -> 0.5 (Amount ~250)
            bias[0] = 0.5
            
        elif opt_type == "scroll_up":
            # Param 0 -> 0.5 (Amount ~-250)
            bias[0] = 0.5
            
        elif opt_type == "move_to_top":
            # x=0.0 (center), y=-1.0 (top)
            bias[0] = 0.0
            bias[1] = -1.0
            
        elif opt_type == "move_to_left":
            # x=-1.0 (left), y=0.0 (center)
            bias[0] = -1.0
            bias[1] = 0.0
            
        elif opt_type == "move_large":
            # Random weights for large moves (exploration)
            # Don't zero weights, let it be random
            nn.init.xavier_uniform_(policy.policy[-1].weight)
            
        elif opt_type == "move_local":
            # Small random weights
            nn.init.normal_(policy.policy[-1].weight, std=0.01)
            
        elif opt_type in ["left_click", "right_click"]:
            # Param 0 > 0 means click
            bias[0] = 1.0
            # Terminate faster?
            policy.termination[-2].bias.data.fill_(0.0) # 50% termination chance
            
        # Add to library
        library.add_option(
            opt_type,
            policy,
            metadata={'trained': False, 'instances': 0, 'heuristic': True}
        )
    
    print(f"Created default option library with {len(library)} options (heuristic init)")
    
    return library


if __name__ == "__main__":
    # Example usage
    print("Creating default option library...")
    library = create_default_options()
    
    print(f"\nAvailable options: {library.list_options()}")
    
    # Test a policy
    scroll_policy = library.get_option("scroll_down")
    if scroll_policy:
        z_t = torch.randn(1, 256)
        c_t = torch.randn(1, 128)
        
        action_params, term_prob = scroll_policy(z_t, c_t)
        action = scroll_policy.decode_action(action_params[0])
        
        print(f"\nTest scroll_down policy:")
        print(f"  Action: {action}")
        print(f"  Terminate prob: {term_prob.item():.3f}")
    
    # Save and load
    library.save("test_option_library.pt")
    
    library2 = OptionLibrary()
    library2.load("test_option_library.pt")
    
    print(f"\nLoaded library has {len(library2)} options")
    
    # Cleanup
    if os.path.exists("test_option_library.pt"):
        os.remove("test_option_library.pt")
