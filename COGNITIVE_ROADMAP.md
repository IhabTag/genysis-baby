# GENYSIS-BABY: Cognitive Development Roadmap

## Overview

This document outlines the next phases of cognitive development for the GENYSIS-BABY AGI system. These phases progressively add higher-level cognitive capabilities: temporal reasoning, hierarchical skills, language grounding, planning, and meta-learning.

**Current State**: The baby has basic perception, world modeling, curiosity-driven exploration, and episodic memory.

**Goal**: Transform from reactive frame-by-frame behavior to intelligent, goal-directed, language-capable cognition.

---

## 🧠 Phase 4 — Temporal Brain & Working Memory

### Problem
Currently the agent operates mostly on **snapshots** (frame → action) with shallow history (episodic buffer, boredom tracking). It lacks:
- Sequential understanding of events
- Working memory for ongoing context
- Ability to reason about "what just happened"

### Solution: Add Temporal Context Model

#### 4.1 Sequence Model on Top of Latents

**Architecture**:
```
Current: frame → encoder → z_t → projection → p_t
                                ↓
                          world model: (z_t, a_t) → z_{t+1}

New:     [z_t, a_t] → Temporal Model (GRU/LSTM/Transformer) → h_t, c_t
                                                                  ↓
                                                    working memory context
```

**Implementation**:

```python
# models/temporal_brain.py

import torch
import torch.nn as nn

class TemporalBrain(nn.Module):
    """
    Maintains working memory over sequences of latent states.
    
    Input: sequence of (z_t, a_t) pairs
    State: h_t (hidden working memory)
    Output: c_t (context vector for decision making)
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 6,
        hidden_dim: int = 256,
        context_dim: int = 128,
        num_layers: int = 2,
        model_type: str = "gru"  # "gru", "lstm", or "transformer"
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.model_type = model_type
        
        input_dim = latent_dim + action_dim
        
        if model_type == "gru":
            self.rnn = nn.GRU(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True
            )
        elif model_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project hidden state to context vector
        self.context_head = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # Initialize hidden state
        self.hidden = None
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset working memory at episode start."""
        self.hidden = None
    
    def forward(self, z_t, a_t):
        """
        Process one timestep.
        
        Args:
            z_t: (batch, latent_dim) - current latent state
            a_t: (batch, action_dim) - current action
        
        Returns:
            c_t: (batch, context_dim) - context vector
            h_t: hidden state (for next step)
        """
        # Concatenate latent and action
        x = torch.cat([z_t, a_t], dim=-1)  # (batch, latent_dim + action_dim)
        x = x.unsqueeze(1)  # (batch, 1, input_dim) - sequence length 1
        
        if self.model_type in ["gru", "lstm"]:
            if self.hidden is None:
                # Initialize hidden state
                num_layers = self.rnn.num_layers
                batch_size = x.size(0)
                
                if self.model_type == "gru":
                    self.hidden = torch.zeros(
                        num_layers, batch_size, self.hidden_dim,
                        device=x.device
                    )
                else:  # lstm
                    h0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=x.device)
                    c0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=x.device)
                    self.hidden = (h0, c0)
            
            # Process sequence
            out, self.hidden = self.rnn(x, self.hidden)
            out = out[:, -1, :]  # Take last timestep
        
        else:  # transformer
            out = self.rnn(x)
            out = out[:, -1, :]
        
        # Generate context vector
        c_t = self.context_head(out)
        
        return c_t, self.hidden
    
    def forward_sequence(self, z_seq, a_seq):
        """
        Process entire sequence (for training or planning).
        
        Args:
            z_seq: (batch, seq_len, latent_dim)
            a_seq: (batch, seq_len, action_dim)
        
        Returns:
            c_seq: (batch, seq_len, context_dim)
        """
        x = torch.cat([z_seq, a_seq], dim=-1)
        
        if self.model_type in ["gru", "lstm"]:
            out, _ = self.rnn(x)
        else:
            out = self.rnn(x)
        
        c_seq = self.context_head(out)
        return c_seq
```

**Integration with CuriousAgent**:

```python
# agent/curious_agent.py modifications

class CuriousAgent:
    def __init__(self, ..., temporal_brain=None):
        # ... existing code ...
        self.temporal_brain = temporal_brain
        self.last_z = None
        self.last_action = None
    
    def select_action(self, frame):
        # Encode current frame
        z_t = self.world_model.encode(preprocess_frame(frame))
        
        # Get temporal context if available
        c_t = None
        if self.temporal_brain is not None and self.last_z is not None:
            a_vec = encode_action(self.last_action, ...)
            c_t, _ = self.temporal_brain(self.last_z, a_vec)
        
        # Generate candidates
        candidates = self._generate_candidate_actions(...)
        
        # Score candidates using [p_t, A_t, c_t]
        for candidate in candidates:
            # ... existing curiosity scoring ...
            
            # Add temporal context bonus
            if c_t is not None:
                temporal_score = self._compute_temporal_relevance(candidate, c_t)
                curiosity += self.temporal_weight * temporal_score
        
        # Select best action
        best_action = max(candidates, key=lambda x: x['curiosity'])
        
        # Remember for next step
        self.last_z = z_t
        self.last_action = best_action
        
        return best_action
```

#### 4.2 Episodic Recall into Working Memory

**Current**: Episodic memory only used for novelty detection.

**New**: Retrieve relevant past experiences and inject into working memory.

```python
# memory/episodic_retrieval.py

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

class EpisodicRetrieval:
    """
    Retrieves relevant past experiences based on current state.
    """
    
    def __init__(
        self,
        episodic_buffer,
        retrieval_k: int = 5,
        context_dim: int = 128
    ):
        self.buffer = episodic_buffer
        self.k = retrieval_k
        self.context_dim = context_dim
        
        # Learnable aggregation of retrieved memories
        self.aggregator = torch.nn.Sequential(
            torch.nn.Linear(context_dim * retrieval_k, context_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(context_dim, context_dim)
        )
    
    def retrieve(self, p_t: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Retrieve k nearest neighbors from episodic memory.
        
        Args:
            p_t: (1, proj_dim) - current projection
        
        Returns:
            r_t: (1, context_dim) - retrieved context vector
        """
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None or memory_tensor.size(0) < self.k:
            return None
        
        # Compute distances
        dists = torch.cdist(p_t, memory_tensor)  # (1, N)
        
        # Get k nearest
        topk_vals, topk_idx = torch.topk(dists, k=self.k, largest=False, dim=1)
        
        # Retrieve corresponding states
        retrieved = memory_tensor[topk_idx[0]]  # (k, proj_dim)
        
        # Aggregate into context
        retrieved_flat = retrieved.flatten().unsqueeze(0)  # (1, k * proj_dim)
        r_t = self.aggregator(retrieved_flat)
        
        return r_t
    
    def retrieve_with_metadata(self, p_t: torch.Tensor) -> List[dict]:
        """
        Retrieve memories with full metadata (for debugging/visualization).
        """
        # Similar to above but returns full episode info
        pass
```

**Integration**:

```python
# In CuriousAgent.select_action():

# Retrieve relevant memories
r_t = self.episodic_retrieval.retrieve(p_t)

# Combine with temporal context
if r_t is not None and c_t is not None:
    combined_context = torch.cat([c_t, r_t], dim=-1)
else:
    combined_context = c_t if c_t is not None else r_t

# Use combined_context for action scoring
```

**Benefits**:
- "I've seen something like this before; last time I scrolled and a new thing appeared"
- Faster learning from past experiences
- Better generalization across similar situations

---

## 🎯 Phase 5 — Skills / Options (Hierarchical Actions)

### Problem
Currently the baby operates at **raw motor ticks**: individual clicks, scrolls, mouse moves. This is:
- Inefficient (too many decisions per second)
- Incoherent (no sustained behaviors)
- Hard to plan with (action space too large)

### Solution: Hierarchical Options Framework

#### 5.1 Segment Behavior into Chunks (Options Discovery)

**Approach**: Analyze existing logs to find natural behavioral segments.

```python
# scripts/discover_options.py

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from models.utils.preprocessing import preprocess_frame_diff
from models.utils.ocr import run_ocr_on_frame

class OptionDiscovery:
    """
    Discovers reusable behavioral chunks from logged episodes.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        min_option_length: int = 3,
        max_option_length: int = 30,
        change_threshold: float = 0.02
    ):
        self.log_dir = log_dir
        self.min_len = min_option_length
        self.max_len = max_option_length
        self.threshold = change_threshold
    
    def find_change_points(self, episode_dir: str) -> List[int]:
        """
        Find timesteps where significant changes occur.
        
        Change indicators:
        - Layout signature change (pixel-level)
        - Attention map shift
        - OCR text change
        - Action type change
        """
        change_points = [0]
        
        frames = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
        
        for i in range(1, len(frames)):
            prev_frame = cv2.imread(os.path.join(episode_dir, frames[i-1]))
            curr_frame = cv2.imread(os.path.join(episode_dir, frames[i]))
            
            # Compute change metrics
            pixel_diff = preprocess_frame_diff(prev_frame, curr_frame)
            
            # Layout change
            layout_change = self._layout_change(prev_frame, curr_frame)
            
            # Combined change score
            total_change = pixel_diff + layout_change
            
            if total_change > self.threshold:
                change_points.append(i)
        
        change_points.append(len(frames))
        return change_points
    
    def segment_episode(self, episode_dir: str) -> List[Dict]:
        """
        Segment episode into candidate options.
        
        Returns:
            List of option segments with metadata
        """
        change_points = self.find_change_points(episode_dir)
        
        options = []
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            
            length = end - start
            if self.min_len <= length <= self.max_len:
                option = {
                    'start': start,
                    'end': end,
                    'length': length,
                    'episode': episode_dir,
                    'type': self._classify_option_type(episode_dir, start, end)
                }
                options.append(option)
        
        return options
    
    def _classify_option_type(self, episode_dir: str, start: int, end: int) -> str:
        """
        Classify option based on dominant action type.
        """
        # Load actions from metadata
        # Return: "scroll_down", "move_to_target", "click_sequence", "typing", etc.
        pass
    
    def cluster_options(self, all_options: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Cluster similar options together.
        
        Returns:
            Dictionary mapping option_type → list of instances
        """
        clusters = {}
        for opt in all_options:
            opt_type = opt['type']
            if opt_type not in clusters:
                clusters[opt_type] = []
            clusters[opt_type].append(opt)
        
        return clusters
```

**Option Types** (examples):
- `scroll_down_page`: Sustained downward scrolling
- `move_to_corner`: Mouse movement to screen corner
- `click_text_element`: Move + click on OCR-detected text
- `type_phrase`: Keyboard typing sequence
- `explore_region`: Random movements in local area

#### 5.2 Train Micro-Policies per Option

```python
# agent/option_policy.py

import torch
import torch.nn as nn
from typing import Dict, Any

class OptionPolicy(nn.Module):
    """
    Micro-policy for executing a specific option/skill.
    
    Input: z_t (latent state), optional context
    Output: low-level action at each step
    Termination: learned termination condition
    """
    
    def __init__(
        self,
        option_name: str,
        latent_dim: int = 256,
        context_dim: int = 128,
        action_dim: int = 6,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.option_name = option_name
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Termination network (binary: continue or stop)
        self.termination = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z_t, c_t=None):
        """
        Compute action and termination probability.
        
        Returns:
            action_params: raw action parameters
            terminate_prob: probability of terminating option
        """
        if c_t is not None:
            x = torch.cat([z_t, c_t], dim=-1)
        else:
            x = z_t
        
        action_params = self.policy(x)
        terminate_prob = self.termination(x)
        
        return action_params, terminate_prob
    
    def decode_action(self, action_params, screen_width, screen_height):
        """
        Convert raw parameters to executable action dict.
        """
        # Depends on option type
        # For example, scroll_down might ignore most params and just return scroll action
        pass


class OptionLibrary:
    """
    Manages collection of learned option policies.
    """
    
    def __init__(self):
        self.options = {}  # option_name → OptionPolicy
    
    def add_option(self, name: str, policy: OptionPolicy):
        self.options[name] = policy
    
    def get_option(self, name: str) -> OptionPolicy:
        return self.options.get(name)
    
    def list_options(self) -> List[str]:
        return list(self.options.keys())
    
    def save(self, path: str):
        torch.save({
            name: policy.state_dict() 
            for name, policy in self.options.items()
        }, path)
    
    def load(self, path: str):
        state_dicts = torch.load(path)
        for name, sd in state_dicts.items():
            if name in self.options:
                self.options[name].load_state_dict(sd)
```

**Training Options**:

```python
# scripts/train_options.py

def train_option_policy(
    option_name: str,
    option_instances: List[Dict],
    world_model,
    device: str = "cpu"
):
    """
    Train a micro-policy for a specific option type.
    
    Uses behavioral cloning on discovered option segments.
    """
    policy = OptionPolicy(option_name).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for instance in option_instances:
            # Load episode data
            episode_dir = instance['episode']
            start = instance['start']
            end = instance['end']
            
            # Get sequence of (z_t, a_t)
            states, actions = load_episode_segment(episode_dir, start, end)
            
            # Behavioral cloning loss
            for z_t, a_true in zip(states, actions):
                a_pred, term_prob = policy(z_t)
                
                # Action loss
                action_loss = F.mse_loss(a_pred, a_true)
                
                # Termination loss (last step should terminate)
                is_last = (t == end - 1)
                term_target = torch.tensor([1.0 if is_last else 0.0])
                term_loss = F.binary_cross_entropy(term_prob, term_target)
                
                total_loss = action_loss + 0.1 * term_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
    
    return policy
```

**High-Level Agent with Options**:

```python
# agent/hierarchical_agent.py

class HierarchicalAgent:
    """
    Agent that selects high-level options, not raw actions.
    """
    
    def __init__(
        self,
        world_model,
        temporal_brain,
        option_library: OptionLibrary,
        device: str = "cpu"
    ):
        self.world_model = world_model
        self.temporal_brain = temporal_brain
        self.options = option_library
        self.device = device
        
        self.current_option = None
        self.option_step = 0
    
    def select_option(self, z_t, c_t) -> str:
        """
        Choose which option to execute based on curiosity.
        """
        option_scores = {}
        
        for option_name in self.options.list_options():
            # Predict outcome of executing this option
            predicted_curiosity = self._predict_option_curiosity(
                option_name, z_t, c_t
            )
            option_scores[option_name] = predicted_curiosity
        
        # Select most curious option
        best_option = max(option_scores, key=option_scores.get)
        return best_option
    
    def step(self, frame):
        """
        Execute one step of hierarchical control.
        """
        z_t = self.world_model.encode(preprocess_frame(frame))
        c_t, _ = self.temporal_brain(z_t, last_action)
        
        # If no active option, select new one
        if self.current_option is None:
            self.current_option = self.select_option(z_t, c_t)
            self.option_step = 0
        
        # Execute current option
        policy = self.options.get_option(self.current_option)
        action_params, terminate_prob = policy(z_t, c_t)
        
        # Decode to executable action
        action = policy.decode_action(action_params, screen_w, screen_h)
        
        # Check termination
        if terminate_prob > 0.5 or self.option_step > max_option_steps:
            self.current_option = None
        else:
            self.option_step += 1
        
        return action
```

**Benefits**:
- More coherent behavior (sustained actions)
- Fewer decisions → cheaper computation
- Natural abstraction for planning
- Reusable skills across tasks

---

## 💬 Phase 6 — Instruction-Following & Language Grounding

### Problem
Currently the baby explores autonomously but cannot:
- Understand natural language instructions
- Respond to user requests
- Communicate its observations

### Solution: Language as Control Signal

#### 6.1 Represent Instructions as Goal Embeddings

```python
# agent/language_grounding.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class InstructionEncoder:
    """
    Maps natural language instructions to goal embeddings.
    
    Can use:
    - LLM API (GPT-4, Claude) for rich understanding
    - Local small model (e.g., sentence-transformers)
    - Learned embedding lookup for common instructions
    """
    
    def __init__(
        self,
        mode: str = "learned",  # "learned", "llm", "sentence_transformer"
        embedding_dim: int = 128
    ):
        self.mode = mode
        self.embedding_dim = embedding_dim
        
        if mode == "learned":
            # Fixed vocabulary of common instructions
            self.vocab = {
                "scroll_down": 0,
                "scroll_up": 1,
                "click_button": 2,
                "type_text": 3,
                "move_mouse": 4,
                "open_application": 5,
                # ... more
            }
            self.embeddings = nn.Embedding(len(self.vocab), embedding_dim)
        
        elif mode == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        elif mode == "llm":
            # Use LLM to parse instruction into structured format
            pass
    
    def encode(self, instruction: str) -> torch.Tensor:
        """
        Convert instruction to goal embedding.
        
        Args:
            instruction: "scroll down to find the button"
        
        Returns:
            g: (1, embedding_dim) goal vector
        """
        if self.mode == "learned":
            # Simple keyword matching
            for keyword, idx in self.vocab.items():
                if keyword.replace("_", " ") in instruction.lower():
                    return self.embeddings(torch.tensor([idx]))
            
            # Default: explore
            return self.embeddings(torch.tensor([0]))
        
        elif self.mode == "sentence_transformer":
            embedding = self.model.encode([instruction])
            return torch.tensor(embedding)
        
        elif self.mode == "llm":
            # Call LLM to parse instruction
            structured = self._llm_parse(instruction)
            return self._structured_to_embedding(structured)
    
    def _llm_parse(self, instruction: str) -> Dict:
        """
        Use LLM to parse instruction into structured format.
        
        Example output:
        {
            "intent": "FIND_AND_CLICK",
            "target": "submit button",
            "constraints": {"region": "bottom_half"},
            "sequence": ["scroll_down", "find_text:submit", "click"]
        }
        """
        # Call LLM API
        prompt = f"""
        Parse this instruction into a structured action plan:
        "{instruction}"
        
        Return JSON with: intent, target, constraints, sequence
        """
        # response = call_llm(prompt)
        # return json.loads(response)
        pass


class GoalConditionedAgent:
    """
    Agent that conditions behavior on language goals.
    """
    
    def __init__(
        self,
        base_agent,  # HierarchicalAgent or CuriousAgent
        instruction_encoder: InstructionEncoder
    ):
        self.agent = base_agent
        self.encoder = instruction_encoder
        self.current_goal = None
    
    def set_instruction(self, instruction: str):
        """
        Set new language instruction.
        """
        self.current_goal = self.encoder.encode(instruction)
        print(f"Goal set: {instruction}")
    
    def select_action(self, frame):
        """
        Select action conditioned on current goal.
        """
        # Get base action scores
        candidates = self.agent._generate_candidate_actions(...)
        
        # Re-score with goal alignment
        if self.current_goal is not None:
            for candidate in candidates:
                # Compute goal alignment
                goal_score = self._compute_goal_alignment(
                    candidate, self.current_goal
                )
                candidate['curiosity'] += self.goal_weight * goal_score
        
        return max(candidates, key=lambda x: x['curiosity'])
    
    def _compute_goal_alignment(self, action, goal_embedding):
        """
        Measure how well action aligns with goal.
        """
        # Learn a goal-action compatibility function
        # For now, simple heuristic based on action type
        pass
```

#### 6.2 Close the Loop in Text Editor

**Scenario**: User types in VNC text editor, baby reads and responds.

```python
# agent/text_editor_interface.py

import time
from typing import Optional
from models.utils.ocr import run_ocr_on_frame

class TextEditorInterface:
    """
    Enables baby to read from and write to a text editor.
    """
    
    def __init__(
        self,
        env,
        editor_region: Optional[tuple] = None  # (x, y, w, h)
    ):
        self.env = env
        self.editor_region = editor_region
        self.last_text = ""
    
    def read_editor(self, frame) -> str:
        """
        Extract text from editor region via OCR.
        """
        if self.editor_region:
            x, y, w, h = self.editor_region
            editor_frame = frame[y:y+h, x:x+w]
        else:
            editor_frame = frame
        
        text_elements = run_ocr_on_frame(editor_frame)
        full_text = "\n".join(text_elements)
        
        return full_text
    
    def detect_new_input(self, frame) -> Optional[str]:
        """
        Check if user has typed new text.
        """
        current_text = self.read_editor(frame)
        
        if current_text != self.last_text:
            new_text = current_text[len(self.last_text):]
            self.last_text = current_text
            return new_text.strip()
        
        return None
    
    def write_response(self, response_text: str):
        """
        Type response into editor.
        """
        # Move to end of document
        self.env.step({"type": "KEY", "key": "ctrl+end"})
        time.sleep(0.1)
        
        # Add newline
        self.env.step({"type": "KEY", "key": "return"})
        time.sleep(0.05)
        
        # Type response slowly (character by character for reliability)
        for char in response_text:
            self.env.step({"type": "TYPE", "text": char})
            time.sleep(0.02)  # Small delay between characters
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate response to user input.
        
        Can use:
        - LLM API for rich responses
        - Template-based responses
        - Learned response model
        """
        # Simple template for now
        if "hello" in user_input.lower():
            return "Hello! I'm learning to interact with this environment."
        
        elif "what" in user_input.lower() and "see" in user_input.lower():
            # Describe what baby sees
            return "I see text on the screen. I'm still learning to understand it."
        
        else:
            # Call LLM for general response
            return self._llm_response(user_input)
    
    def _llm_response(self, user_input: str) -> str:
        """
        Use LLM to generate contextual response.
        """
        prompt = f"""
        You are an AI baby learning to interact with a computer.
        The user said: "{user_input}"
        
        Respond briefly and honestly about your learning process.
        Keep it under 50 words.
        """
        # response = call_llm(prompt)
        # return response
        return "I'm still learning. Thank you for your patience!"


# Main interaction loop
def run_text_editor_interaction():
    env = ComputerEnv(...)
    agent = GoalConditionedAgent(...)
    editor = TextEditorInterface(env)
    
    obs = env.reset()
    
    while True:
        frame = obs['image']
        
        # Check for new user input
        new_input = editor.detect_new_input(frame)
        
        if new_input:
            print(f"User: {new_input}")
            
            # Generate response
            response = editor.generate_response(new_input)
            print(f"Baby: {response}")
            
            # Write response
            editor.write_response(response)
        
        else:
            # Normal exploration
            action = agent.select_action(frame)
            obs, _, _, _ = env.step(action)
        
        time.sleep(0.1)
```

**Benefits**:
- Direct communication with user
- Language-grounded learning
- Ability to explain observations
- Foundation for instruction following

---

## 🎯 Phase 7 — Model-Based Planning

### Problem
Currently the baby scores actions **one step ahead**. It cannot:
- Think before acting
- Anticipate multi-step consequences
- Plan toward distant goals

### Solution: Latent Imagination & Rollouts

```python
# agent/planner.py

import torch
from typing import List, Tuple, Dict

class LatentPlanner:
    """
    Plans multi-step action sequences in latent space.
    """
    
    def __init__(
        self,
        world_model,
        temporal_brain,
        option_library,
        horizon: int = 5,
        num_candidates: int = 8,
        device: str = "cpu"
    ):
        self.world_model = world_model
        self.temporal_brain = temporal_brain
        self.options = option_library
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.device = device
    
    def plan(
        self,
        z_t: torch.Tensor,
        c_t: torch.Tensor,
        goal: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Plan sequence of options to maximize predicted curiosity/reward.
        
        Args:
            z_t: current latent state
            c_t: current temporal context
            goal: optional goal embedding
        
        Returns:
            best_sequence: list of option names
        """
        # Generate candidate sequences
        sequences = self._generate_candidate_sequences()
        
        # Evaluate each sequence via imagination
        sequence_scores = []
        
        for seq in sequences:
            score = self._evaluate_sequence(seq, z_t, c_t, goal)
            sequence_scores.append((seq, score))
        
        # Select best sequence
        best_seq, best_score = max(sequence_scores, key=lambda x: x[1])
        
        return best_seq
    
    def _generate_candidate_sequences(self) -> List[List[str]]:
        """
        Generate diverse candidate action sequences.
        """
        sequences = []
        option_names = self.options.list_options()
        
        # Random sampling
        for _ in range(self.num_candidates):
            seq = [
                random.choice(option_names) 
                for _ in range(self.horizon)
            ]
            sequences.append(seq)
        
        # Could add: beam search, MCTS, etc.
        
        return sequences
    
    def _evaluate_sequence(
        self,
        sequence: List[str],
        z_t: torch.Tensor,
        c_t: torch.Tensor,
        goal: Optional[torch.Tensor]
    ) -> float:
        """
        Imagine executing sequence and compute expected value.
        """
        total_score = 0.0
        current_z = z_t
        current_c = c_t
        
        for option_name in sequence:
            # Get option policy
            policy = self.options.get_option(option_name)
            
            # Predict action from policy
            action_params, _ = policy(current_z, current_c)
            
            # Convert to action vector
            action_vec = self._params_to_vector(action_params)
            
            # Imagine next state via world model
            z_next = self.world_model.predict_latent(current_z, action_vec)
            
            # Update temporal context
            c_next, _ = self.temporal_brain(current_z, action_vec)
            
            # Compute predicted curiosity/reward
            step_score = self._compute_step_value(
                current_z, z_next, current_c, c_next, goal
            )
            
            total_score += step_score
            
            # Advance
            current_z = z_next
            current_c = c_next
        
        return total_score
    
    def _compute_step_value(
        self,
        z_t, z_next, c_t, c_next, goal
    ) -> float:
        """
        Compute value of transitioning from (z_t, c_t) to (z_next, c_next).
        """
        # Prediction error (curiosity)
        p_t = self.world_model.project(z_t)
        p_next = self.world_model.project(z_next)
        curiosity = torch.mean((p_next - p_t) ** 2).item()
        
        # Goal alignment (if goal provided)
        goal_score = 0.0
        if goal is not None:
            goal_score = self._goal_similarity(z_next, goal)
        
        return curiosity + self.goal_weight * goal_score


class PlanningAgent:
    """
    Agent that plans before acting.
    """
    
    def __init__(self, planner: LatentPlanner, replan_interval: int = 5):
        self.planner = planner
        self.replan_interval = replan_interval
        
        self.current_plan = []
        self.plan_step = 0
    
    def select_action(self, frame, goal=None):
        """
        Select action using planning.
        """
        z_t = self.planner.world_model.encode(preprocess_frame(frame))
        c_t, _ = self.planner.temporal_brain(z_t, last_action)
        
        # Replan if needed
        if self.plan_step == 0 or self.plan_step >= self.replan_interval:
            self.current_plan = self.planner.plan(z_t, c_t, goal)
            self.plan_step = 0
            print(f"New plan: {self.current_plan}")
        
        # Execute current step of plan
        option_name = self.current_plan[self.plan_step]
        policy = self.planner.options.get_option(option_name)
        
        action_params, _ = policy(z_t, c_t)
        action = policy.decode_action(action_params, screen_w, screen_h)
        
        self.plan_step += 1
        
        return action
```

**Benefits**:
- More purposeful behavior
- Less getting stuck in local loops
- Ability to aim for anticipated novelty
- Foundation for goal-directed behavior

---

## 🧬 Phase 8 — Meta-Curiosity & Homeostasis

### Problem
Currently all hyperparameters (curiosity weights, boredom thresholds) are **hand-tuned**. The baby cannot:
- Adapt to changing environments
- Self-regulate learning difficulty
- Optimize its own learning process

### Solution: Self-Tuning Brain

#### 8.1 Learn Curiosity Weights Online

```python
# agent/meta_learner.py

import torch
import numpy as np
from collections import deque

class HomeostasisController:
    """
    Maintains optimal learning challenge via self-tuning.
    
    Principle: Keep prediction error in a "Goldilocks zone"
    - Too low → increase exploration (novelty, goal weights)
    - Too high → increase exploitation (attention, latent weights)
    """
    
    def __init__(
        self,
        target_prediction_error: float = 0.05,
        target_curiosity: float = 0.3,
        adaptation_rate: float = 0.01,
        window_size: int = 100
    ):
        self.target_pred_error = target_prediction_error
        self.target_curiosity = target_curiosity
        self.adapt_rate = adaptation_rate
        
        # Running statistics
        self.pred_errors = deque(maxlen=window_size)
        self.curiosity_scores = deque(maxlen=window_size)
        self.loopiness_scores = deque(maxlen=window_size)
        
        # Current weights (will be adjusted)
        self.weights = {
            'latent': 1.0,
            'novelty': 0.5,
            'attention': 0.3,
            'goal': 0.8
        }
    
    def update(
        self,
        prediction_error: float,
        curiosity: float,
        loopiness: float
    ):
        """
        Update statistics and adjust weights.
        """
        self.pred_errors.append(prediction_error)
        self.curiosity_scores.append(curiosity)
        self.loopiness_scores.append(loopiness)
        
        if len(self.pred_errors) < 50:
            return  # Not enough data yet
        
        # Compute running averages
        avg_pred_error = np.mean(self.pred_errors)
        avg_curiosity = np.mean(self.curiosity_scores)
        avg_loopiness = np.mean(self.loopiness_scores)
        
        # Adjust weights based on deviations from targets
        
        # If prediction error too low → environment too predictable → explore more
        if avg_pred_error < self.target_pred_error * 0.8:
            self.weights['novelty'] += self.adapt_rate
            self.weights['goal'] += self.adapt_rate
            print("[homeostasis] Too predictable, increasing exploration")
        
        # If prediction error too high → environment too chaotic → focus more
        elif avg_pred_error > self.target_pred_error * 1.2:
            self.weights['latent'] += self.adapt_rate
            self.weights['attention'] += self.adapt_rate
            self.weights['novelty'] -= self.adapt_rate
            print("[homeostasis] Too chaotic, increasing focus")
        
        # If too loopy → stuck in repetitive behavior → boost novelty
        if avg_loopiness > 0.7:
            self.weights['novelty'] += self.adapt_rate * 2
            print("[homeostasis] Stuck in loop, boosting novelty")
        
        # Normalize weights
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current curiosity weights.
        """
        return self.weights.copy()
    
    def compute_loopiness(self, recent_states: List[torch.Tensor]) -> float:
        """
        Measure how repetitive recent behavior is.
        """
        if len(recent_states) < 10:
            return 0.0
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(recent_states) - 1):
            for j in range(i + 1, len(recent_states)):
                sim = torch.cosine_similarity(
                    recent_states[i], recent_states[j], dim=0
                )
                similarities.append(sim.item())
        
        # High average similarity → high loopiness
        return np.mean(similarities)
```

#### 8.2 Sleep / Consolidation Phases

```python
# agent/sleep_mode.py

import torch
import random
from typing import List

class SleepConsolidation:
    """
    Offline learning phase that:
    - Replays past experiences
    - Trains world model & policies
    - Compresses memories
    - Reduces catastrophic forgetting
    """
    
    def __init__(
        self,
        world_model,
        option_library,
        episodic_buffer,
        text_memory
    ):
        self.world_model = world_model
        self.options = option_library
        self.episodic = episodic_buffer
        self.text_mem = text_memory
    
    def sleep(
        self,
        duration_steps: int = 1000,
        replay_batch_size: int = 32
    ):
        """
        Enter sleep mode for consolidation.
        """
        print(f"\n💤 Entering sleep mode for {duration_steps} consolidation steps...")
        
        # 1. Replay experiences for world model
        self._replay_world_model(duration_steps, replay_batch_size)
        
        # 2. Consolidate option policies
        self._consolidate_options(duration_steps // 2)
        
        # 3. Compress text memory
        self._compress_text_memory()
        
        # 4. Prune episodic buffer (keep most important)
        self._prune_episodic_buffer()
        
        print("✅ Sleep consolidation complete\n")
    
    def _replay_world_model(self, steps: int, batch_size: int):
        """
        Sample past experiences and train world model.
        """
        optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-4)
        
        # Load past episodes from logs
        episodes = self._load_past_episodes()
        
        for step in range(steps):
            # Sample batch of transitions
            batch = self._sample_transitions(episodes, batch_size)
            
            # Train on batch
            loss = self._train_world_model_batch(batch, optimizer)
            
            if step % 100 == 0:
                print(f"  [sleep] World model replay step {step}, loss={loss:.4f}")
    
    def _consolidate_options(self, steps: int):
        """
        Fine-tune option policies on recent successful executions.
        """
        for option_name in self.options.list_options():
            policy = self.options.get_option(option_name)
            
            # Find recent successful uses of this option
            successful_instances = self._find_successful_option_uses(option_name)
            
            if len(successful_instances) < 10:
                continue
            
            # Fine-tune policy
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
            
            for _ in range(steps // len(self.options.list_options())):
                instance = random.choice(successful_instances)
                loss = self._train_option_policy(policy, instance, optimizer)
            
            print(f"  [sleep] Consolidated option: {option_name}")
    
    def _compress_text_memory(self):
        """
        Use LLM to summarize and compress text memories.
        """
        # Get all observed text
        all_text = self.text_mem.get_all_text()
        
        if len(all_text) < 100:
            return
        
        # Group by similarity/topic
        clusters = self._cluster_text(all_text)
        
        # Summarize each cluster
        for cluster in clusters:
            summary = self._summarize_cluster(cluster)
            # Store summary, remove individual items
        
        print(f"  [sleep] Compressed {len(all_text)} text items into {len(clusters)} summaries")
    
    def _prune_episodic_buffer(self):
        """
        Keep most important episodic memories, discard redundant ones.
        """
        # Importance criteria:
        # - High novelty at the time
        # - Frequently retrieved
        # - Associated with significant events
        
        memories = self.episodic.get_all_memories()
        
        # Compute importance scores
        scored_memories = [
            (mem, self._compute_memory_importance(mem))
            for mem in memories
        ]
        
        # Keep top 50%
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        keep_count = len(scored_memories) // 2
        
        important_memories = [mem for mem, score in scored_memories[:keep_count]]
        
        # Replace buffer
        self.episodic.clear()
        for mem in important_memories:
            self.episodic.add(mem)
        
        print(f"  [sleep] Pruned episodic buffer: {len(memories)} → {len(important_memories)}")


# Integration with lifelong learning

def lifelong_learning_with_sleep():
    # ... existing setup ...
    
    sleep_controller = SleepConsolidation(world_model, options, episodic, text_mem)
    homeostasis = HomeostasisController()
    
    steps_since_sleep = 0
    sleep_interval = 5000  # Sleep every 5000 steps
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        while not done:
            # Normal step
            action = agent.select_action(frame)
            next_obs, _, done, _ = env.step(action)
            
            # Update homeostasis
            homeostasis.update(pred_error, curiosity, loopiness)
            weights = homeostasis.get_weights()
            agent.update_curiosity_weights(weights)
            
            steps_since_sleep += 1
            
            # Sleep if needed
            if steps_since_sleep >= sleep_interval:
                sleep_controller.sleep(duration_steps=1000)
                steps_since_sleep = 0
```

**Benefits**:
- Self-regulating learning difficulty
- Reduced catastrophic forgetting
- Automatic hyperparameter tuning
- More robust long-term learning

---

## 🚀 Recommended Implementation Order

Based on impact and dependencies:

### **Most Impactful: Phase 4 + 5 Combined**

**Why**: This gives the biggest immediate improvement in behavior quality.

**Implementation Steps**:

1. **Add Temporal Brain** (Phase 4.1)
   - Create `models/temporal_brain.py`
   - Integrate GRU into `CuriousAgent`
   - Train on existing logs

2. **Discover Options** (Phase 5.1)
   - Create `scripts/discover_options.py`
   - Analyze existing episode logs
   - Identify 5-10 common behavioral patterns

3. **Train Option Policies** (Phase 5.2)
   - Create `agent/option_policy.py`
   - Behavioral cloning on discovered options
   - Save to option library

4. **Hierarchical Agent** (Phase 5.2)
   - Create `agent/hierarchical_agent.py`
   - Select options instead of raw actions
   - Test on exploration tasks

**Expected Outcome**:
- More coherent, sustained behaviors
- Faster learning (fewer decisions)
- Foundation for language and planning

### **Next: Phase 6 (Language)**

Once hierarchical control is stable:
- Add instruction encoder
- Implement text editor interface
- Enable basic instruction following

### **Then: Phase 7 (Planning)**

With options and language:
- Add latent imagination
- Short-horizon planning (H=3-5)
- Goal-directed behavior

### **Finally: Phase 8 (Meta-Learning)**

For long-term robustness:
- Homeostatic weight tuning
- Sleep consolidation
- Self-optimization

---

## 📊 Success Metrics

For each phase:

**Phase 4 (Temporal)**:
- ✅ Context vector changes meaningfully over episodes
- ✅ Agent remembers recent events (test with "return to previous location")
- ✅ Reduced repetitive behavior

**Phase 5 (Options)**:
- ✅ 5-10 distinct options discovered
- ✅ Option policies achieve >80% success rate
- ✅ Hierarchical agent completes tasks faster than baseline

**Phase 6 (Language)**:
- ✅ Correctly follows 80% of simple instructions
- ✅ Responds coherently in text editor
- ✅ Goal-conditioned exploration reaches targets

**Phase 7 (Planning)**:
- ✅ Completes multi-step tasks without getting stuck
- ✅ Anticipates consequences 3-5 steps ahead
- ✅ Reaches distant goals more efficiently

**Phase 8 (Meta)**:
- ✅ Curiosity weights adapt to environment changes
- ✅ Sleep consolidation improves performance
- ✅ Maintains stable learning over 100k+ steps

---

## 🎯 Next Steps

**Immediate Action**: Implement Phase 4 + 5 (Temporal Brain + Options)

**Files to Create**:
1. `models/temporal_brain.py` - GRU/LSTM working memory
2. `scripts/discover_options.py` - Behavioral segmentation
3. `agent/option_policy.py` - Micro-policies
4. `agent/hierarchical_agent.py` - High-level controller
5. `scripts/train_options.py` - Option training pipeline

**Testing Strategy**:
1. Train temporal brain on existing logs
2. Discover options from past episodes
3. Train option policies via behavioral cloning
4. Compare hierarchical vs. flat agent performance
5. Measure coherence, efficiency, learning speed

**Timeline Estimate**:
- Phase 4: 2-3 days
- Phase 5: 3-4 days
- Integration & testing: 2 days
- **Total: ~1-2 weeks for Phase 4+5**

---

This roadmap transforms GENYSIS-BABY from a reactive explorer into an intelligent, goal-directed, language-capable agent with planning and self-optimization abilities. Each phase builds naturally on the previous, creating a developmental progression toward AGI.
