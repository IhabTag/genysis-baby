import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.world_model import WorldModel
from agent.option_policy import create_default_options
from models.utils.preprocessing import encode_action

def debug_curiosity():
    print("Initializing Debug Session...")
    
    # 1. Setup
    device = "cpu"
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64, action_dim=7)
    world_model.to(device)
    world_model.eval()
    
    option_library = create_default_options(device=device)
    
    # 2. Create dummy state
    print("\nCreating dummy state...")
    z_t = torch.randn(1, 256).to(device)
    p_t = world_model.project(z_t)
    c_t = torch.randn(1, 128).to(device) # Dummy context
    
    # 3. Test each option
    print("\nTesting Options Curiosity...")
    print(f"{'Option':<15} | {'Action Type':<12} | {'Latent Change':<15} | {'Action Vec Norm':<15}")
    print("-" * 70)
    
    for option_name in option_library.list_options():
        policy = option_library.get_option(option_name)
        
        # Get action params
        action_params, _ = policy(z_t, c_t)
        
        # Decode action
        action_dict = policy.decode_action(action_params[0])
        
        # Encode action vector
        action_vec_np = encode_action(action_dict, screen_width=1024, screen_height=768)
        action_vec = torch.tensor(action_vec_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Predict next state
        z_next_pred = world_model.predict_latent(z_t, action_vec)
        p_next_pred = world_model.project(z_next_pred)
        
        # Calculate latent change
        latent_change = torch.mean((p_next_pred - p_t) ** 2).item()
        
        # Check action vector magnitude
        vec_norm = np.linalg.norm(action_vec_np)
        
        print(f"{option_name:<15} | {action_dict['type']:<12} | {latent_change:.8f}      | {vec_norm:.4f}")
        
        # Detailed debug for scroll_down
        if option_name == "scroll_down":
            print(f"  -> scroll_down action: {action_dict}")
            print(f"  -> scroll_down vector: {action_vec_np}")

if __name__ == "__main__":
    debug_curiosity()
