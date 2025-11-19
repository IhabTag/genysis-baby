"""
Hierarchical Agent: Selects and executes high-level options instead of raw actions.

Instead of choosing individual mouse clicks and scrolls, the agent:
1. Selects an option (e.g., "scroll_down", "move_to_top")
2. Executes the option's micro-policy until termination
3. Selects next option

This results in more coherent, sustained behaviors.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque

from models.world_model import WorldModel
from models.temporal_brain import TemporalBrain
from models.utils.preprocessing import preprocess_frame, encode_action
from memory.episodic_buffer import EpisodicBuffer
from memory.episodic_retrieval import EpisodicRetrieval
from agent.option_policy import OptionLibrary, OptionPolicy


class HierarchicalAgent:
    """
    Agent that selects high-level options, not raw actions.
    
    Decision flow:
    1. Observe current state
    2. If no active option, select new option based on curiosity
    3. Execute option policy to get low-level action
    4. Check if option should terminate
    5. Repeat
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        option_library: OptionLibrary,
        temporal_brain: Optional[TemporalBrain] = None,
        device: str = "cpu",
        max_option_steps: int = 50,
        epsilon: float = 0.2,  # Increased for more exploration
        curiosity_weight: float = 2.0,  # Increased weight
        novelty_weight: float = 1.5,  # Increased weight
        diversity_bonus: float = 0.5,  # NEW: bonus for trying different options
        exploration_boost: float = 1.5,  # NEW: boost early exploration
        max_memory: int = 2000,
        proj_dim: int = 64
    ):
        """
        Initialize hierarchical agent.
        
        Args:
            world_model: WorldModel for prediction
            option_library: Library of trained option policies
            temporal_brain: Optional temporal brain for context
            device: Device
            max_option_steps: Force termination after this many steps
            epsilon: Epsilon-greedy exploration
            curiosity_weight: Weight for latent change curiosity
            novelty_weight: Weight for novelty curiosity
            diversity_bonus: Bonus for trying different options
            exploration_boost: Multiplier for early exploration
            max_memory: Episodic memory size
            proj_dim: Projection dimension
        """
        self.world_model = world_model.to(device)
        self.world_model.eval()
        
        self.option_library = option_library
        self.temporal_brain = temporal_brain
        if self.temporal_brain is not None:
            self.temporal_brain.to(device)
            self.temporal_brain.eval()
        
        self.device = device
        self.max_option_steps = max_option_steps
        self.epsilon = epsilon
        self.curiosity_weight = curiosity_weight
        self.novelty_weight = novelty_weight
        self.diversity_bonus = diversity_bonus
        self.exploration_boost = exploration_boost
        
        # Episodic memory
        self.memory = EpisodicBuffer(
            max_size=max_memory,
            device=device,
            proj_dim=proj_dim
        )
        
        # Episodic retrieval
        self.episodic_retrieval = EpisodicRetrieval(
            self.memory,
            retrieval_k=5,
            context_dim=128,
            use_learned_aggregation=False
        )
        
        # Current option state
        self.current_option: Optional[str] = None
        self.current_policy: Optional[OptionPolicy] = None
        self.option_step: int = 0
        
        # Temporal state
        self.last_z: Optional[torch.Tensor] = None
        self.last_action_vec: Optional[torch.Tensor] = None
        
        # Statistics
        self.options_executed: int = 0
        self.option_history: deque = deque(maxlen=100)
        self.option_counts: Dict[str, int] = {}  # Track option usage
        self.total_steps: int = 0
        
        # Curiosity statistics for normalization
        self.running_curiosity_mean = 0.0
        self.running_curiosity_std = 1.0
        self.boredom_counter = 0
        self.boredom_threshold = 0.01  # If score < this, we are bored
    
    @torch.no_grad()
    def select_option(
        self, 
        z_t: torch.Tensor, 
        p_t: torch.Tensor,
        c_t: Optional[torch.Tensor] = None
    ) -> str:
        """
        Choose which option to execute based on predicted curiosity.
        
        Args:
            z_t: Current latent state
            p_t: Current projection
            c_t: Optional temporal context
        
        Returns:
            Selected option name
        """
        available_options = self.option_library.list_options()
        
        if not available_options:
            raise ValueError("No options available in library")
        
        # Epsilon-greedy: sometimes choose random option
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_options)
        
        # Score each option by predicted curiosity
        option_scores = {}
        
        mem_tensor = self.memory.get_memory_tensor(sample_size=512)
        
        for option_name in available_options:
            policy = self.option_library.get_option(option_name)
            
            # Predict what action this option would take
            action_params, _ = policy(z_t, c_t)
            
            # Decode to action vector for world model
            action_dict = policy.decode_action(action_params[0])
            action_vec_np = encode_action(action_dict, screen_width=1024, screen_height=768)
            action_vec = torch.tensor(
                action_vec_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            
            # Predict next state
            z_next_pred = self.world_model.predict_latent(z_t, action_vec)
            p_next_pred = self.world_model.project(z_next_pred)
            
            # Curiosity: latent change (amplified)
            latent_change = torch.mean((p_next_pred - p_t) ** 2).item()
            
            # Novelty: distance to memory (amplified)
            novelty = 1.0  # Default high novelty
            min_dist = 1.0
            if mem_tensor is not None and mem_tensor.size(0) > 0:
                diff_mem = mem_tensor - p_next_pred
                dists = torch.mean(diff_mem ** 2, dim=1)
                min_dist = torch.min(dists).item()
                # Use exponential scaling for novelty to make small differences matter
                # If dist is 0 (seen before), novelty is 0
                # If dist is large, novelty approaches 1
                novelty = 1.0 - np.exp(-10.0 * min_dist)
            
            # Diversity bonus: prefer less-used options
            usage_count = self.option_counts.get(option_name, 0)
            diversity = 1.0 / (1.0 + usage_count * 0.1)  # Decay with usage
            
            # Exploration boost for early steps
            exploration_factor = self.exploration_boost if self.total_steps < 1000 else 1.0
            
            # Raw score components
            raw_score = (
                self.curiosity_weight * latent_change
                + self.novelty_weight * novelty
                + self.diversity_bonus * diversity
            )
            
            # Apply exploration boost
            score = exploration_factor * raw_score
            
            option_scores[option_name] = {
                'score': score,
                'raw_components': {
                    'latent': latent_change,
                    'novelty': novelty,
                    'dist': min_dist,
                    'diversity': diversity
                }
            }
        
        # Select best option
        best_option = max(option_scores, key=lambda k: option_scores[k]['score'])
        best_score = option_scores[best_option]['score']
        best_components = option_scores[best_option]['raw_components']
        
        # Update running stats
        self.running_curiosity_mean = 0.95 * self.running_curiosity_mean + 0.05 * best_score
        
        # Boredom Breaker
        is_bored = best_score < self.boredom_threshold
        if is_bored:
            self.boredom_counter += 1
        else:
            self.boredom_counter = 0
            
        # If bored for too long, force random exploration
        if self.boredom_counter > 3:
            print(f"  [Boredom] Score {best_score:.4f} too low for {self.boredom_counter} steps. Forcing random option.")
            best_option = np.random.choice(available_options)
            self.boredom_counter = 0
        
        # Print scores for debugging (top 3)
        if self.total_steps % 1 == 0:  # Print more often
            sorted_options = sorted(option_scores.items(), key=lambda x: -x[1]['score'])[:3]
            print(f"\n  [Curiosity Debug] Step {self.total_steps}")
            for name, data in sorted_options:
                comps = data['raw_components']
                print(f"    {name:15s}: Score={data['score']:.4f} "
                      f"(Latent={comps['latent']:.4f}, Nov={comps['novelty']:.4f}, Div={comps['diversity']:.2f})")
        
        return best_option
    
    @torch.no_grad()
    def step(
        self, 
        frame: np.ndarray,
        screen_width: int = 1024,
        screen_height: int = 768
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute one step of hierarchical control.
        
        Args:
            frame: Current RGB frame (H, W, 3)
            screen_width, screen_height: Screen dimensions
        
        Returns:
            action: Executable action dict
            info: Information dict with option name, step, etc.
        """
        # Encode current frame
        img_np = preprocess_frame(frame)
        img_t = torch.tensor(
            img_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        z_t = self.world_model.encode(img_t)
        p_t = self.world_model.project(z_t)
        
        # Get temporal context
        c_t = None
        if self.temporal_brain is not None and self.last_z is not None and self.last_action_vec is not None:
            c_t, _ = self.temporal_brain(self.last_z, self.last_action_vec)
        
        # Remember state
        p_vec = p_t.squeeze(0).cpu().numpy()
        self.memory.add(p_vec)
        
        # If no active option or option should terminate, select new one
        if self.current_option is None or self.option_step >= self.max_option_steps:
            # Select new option
            self.current_option = self.select_option(z_t, p_t, c_t)
            self.current_policy = self.option_library.get_option(self.current_option)
            self.option_step = 0
            self.options_executed += 1
            self.option_history.append(self.current_option)
            
            # Track option usage
            self.option_counts[self.current_option] = self.option_counts.get(self.current_option, 0) + 1
            
            print(f"[Hierarchical] Selected option: {self.current_option} (usage: {self.option_counts[self.current_option]}x)")
        
        # Execute current option policy
        action_params, terminate_prob = self.current_policy(z_t, c_t)
        
        # Decode to executable action
        action = self.current_policy.decode_action(
            action_params[0],
            screen_width=screen_width,
            screen_height=screen_height
        )
        
        # Check termination
        should_terminate = terminate_prob.item() > 0.5
        
        if should_terminate:
            print(f"[Hierarchical] Option {self.current_option} terminated after {self.option_step} steps")
            self.current_option = None
            self.current_policy = None
        else:
            self.option_step += 1
        
        # Update temporal state
        if self.temporal_brain is not None:
            self.last_z = z_t.detach()
            action_vec_np = encode_action(action, screen_width=screen_width, screen_height=screen_height)
            self.last_action_vec = torch.tensor(
                action_vec_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        
        self.total_steps += 1
        
        # Info dict
        info = {
            'option': self.current_option,
            'option_step': self.option_step,
            'terminate_prob': terminate_prob.item(),
            'options_executed': self.options_executed,
            'total_steps': self.total_steps
        }
        
        return action, info
    
    def reset(self):
        """Reset agent state for new episode."""
        self.current_option = None
        self.current_policy = None
        self.option_step = 0
        self.last_z = None
        self.last_action_vec = None
        self.boredom_counter = 0
        
        if self.temporal_brain is not None:
            self.temporal_brain.reset_hidden()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        # Count option usage
        option_counts = {}
        for opt in self.option_history:
            option_counts[opt] = option_counts.get(opt, 0) + 1
        
        return {
            'options_executed': self.options_executed,
            'total_steps': self.total_steps,
            'avg_option_length': self.total_steps / max(1, self.options_executed),
            'option_counts': option_counts,
            'memory_size': self.memory.size
        }
    
    def save_state(self, state_dir: str):
        """Save agent state."""
        import os
        os.makedirs(state_dir, exist_ok=True)
        
        # Save episodic memory
        self.memory.save(os.path.join(state_dir, "episodic_memory.npz"))
        
        # Save statistics
        import json
        stats = self.get_statistics()
        with open(os.path.join(state_dir, "hierarchical_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved hierarchical agent state to {state_dir}")
    
    def load_state(self, state_dir: str):
        """Load agent state."""
        import os
        
        # Load episodic memory
        mem_path = os.path.join(state_dir, "episodic_memory.npz")
        if os.path.exists(mem_path):
            self.memory.load(mem_path)
        
        print(f"Loaded hierarchical agent state from {state_dir}")


if __name__ == "__main__":
    # Example usage
    from agent.option_policy import create_default_options
    
    print("Creating hierarchical agent...")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    option_library = create_default_options()
    
    agent = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library,
        device="cpu"
    )
    
    print(f"Agent created with {len(option_library)} options")
    
    # Simulate a few steps
    for i in range(5):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        action, info = agent.step(frame)
        
        print(f"Step {i}: {action['type']} (option: {info['option']}, step: {info['option_step']})")
    
    # Show statistics
    stats = agent.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
