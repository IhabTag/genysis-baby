# GENYSIS-BABY Usage Guide

## Running Online Lifelong Learning

The main training script is `scripts/online_lifelong_learning.py`. It now supports Phase 4 and Phase 5 features!

### Basic Usage (Standard Agent)

```bash
# Run with default settings (standard curious agent)
python scripts/online_lifelong_learning.py
```

### Phase 4: With Temporal Brain

```bash
# Enable temporal brain for working memory
python scripts/online_lifelong_learning.py --use-temporal
```

**Features enabled**:
- Working memory of recent events
- Episodic retrieval of similar past experiences
- Temporal context in action selection

### Phase 5: With Hierarchical Agent

```bash
# Enable hierarchical agent with default options
python scripts/online_lifelong_learning.py --use-hierarchical

# Use trained option library
python scripts/online_lifelong_learning.py \
  --use-hierarchical \
  --option-library checkpoints/option_library.pt
```

**Features enabled**:
- High-level option selection
- Sustained, coherent behaviors
- Reusable skills

### Phase 4 + 5: Combined

```bash
# Enable both temporal brain and hierarchical agent
python scripts/online_lifelong_learning.py \
  --use-temporal \
  --use-hierarchical \
  --option-library checkpoints/option_library.pt
```

**Features enabled**:
- Working memory + episodic retrieval
- Hierarchical option-based control
- Temporal context for option selection

### Additional Options

```bash
# Full command with all options
python scripts/online_lifelong_learning.py \
  --device cpu \
  --use-temporal \
  --use-hierarchical \
  --option-library checkpoints/option_library.pt \
  --episodes 100 \
  --fast-mode
```

**Arguments**:
- `--device`: Device to use (cpu/cuda), default: cpu
- `--use-temporal`: Enable temporal brain
- `--use-hierarchical`: Enable hierarchical agent
- `--option-library`: Path to trained option library
- `--episodes`: Number of episodes to run, default: 999999
- `--fast-mode`: Skip OCR on predicted frames (faster), default: True

---

## Training Option Policies

To use hierarchical agent with trained options, you need to:

### 1. Collect Episode Data

Run the agent to collect episodes:

```bash
python scripts/online_lifelong_learning.py --episodes 50
```

Episodes are logged to `logs/` directory.

### 2. Discover Options

Analyze logs to find behavioral patterns:

```bash
python scripts/discover_options.py \
  --log-dir logs \
  --output datasets/discovered_options.json \
  --min-length 3 \
  --max-length 30
```

### 3. Train Option Policies

Train micro-policies via behavioral cloning:

```bash
python scripts/train_options.py \
  --options datasets/discovered_options.json \
  --world-model checkpoints/world_model_contrastive.pt \
  --output checkpoints/option_library.pt \
  --epochs 50 \
  --batch-size 16
```

### 4. Use Trained Options

Run with trained option library:

```bash
python scripts/online_lifelong_learning.py \
  --use-hierarchical \
  --option-library checkpoints/option_library.pt
```

---

## Testing

### Test Individual Components

```bash
# Test temporal brain
python scripts/debug/test_temporal_brain.py

# Test episodic retrieval
python scripts/debug/test_episodic_retrieval.py

# Test curious agent integration
python scripts/debug/test_curious_agent_integration.py

# Test hierarchical options
python scripts/debug/test_hierarchical_options.py
```

### Quick Smoke Test

```bash
# Run 5 episodes with all features
python scripts/online_lifelong_learning.py \
  --use-temporal \
  --use-hierarchical \
  --episodes 5
```

---

## State Persistence

All agent state is saved to `state/` directory:

**Standard Agent**:
- `episodic_memory.npz` - Episodic buffer
- `text_memory.json` - Text/OCR memory
- `goal_memory.json` - Goal curiosity memory
- `agent_meta.json` - Action counts, scores, etc.
- `age.json` - Total episodes and steps

**With Temporal Brain**:
- `temporal_brain.pt` - Temporal brain weights + hidden state

**Hierarchical Agent**:
- `hierarchical_stats.json` - Option usage statistics

The agent automatically loads and saves state between runs, enabling **true lifelong learning**!

---

## Performance Tips

### Fast Mode (Default)
```bash
python scripts/online_lifelong_learning.py --fast-mode
```
- Skips OCR on predicted frames
- ~2-3x faster
- Recommended for initial exploration

### Fewer Candidates
Edit `online_lifelong_learning.py`:
```python
n_candidates=4  # Default, try 2-3 for faster
```

### Smaller Memory
```python
max_memory=1000  # Default: 1500
mem_sample_size=256  # Default: 512
```

### GPU Acceleration
```bash
python scripts/online_lifelong_learning.py --device cuda
```

---

## Monitoring Progress

The script prints:
- Episode number
- Step count
- World model loss
- Curiosity score
- Cognitive age estimate

**Example output**:
```
=== EPISODE 42 ===
[ep 42] step=10 loss=0.0234 curiosity=0.1234 (global_steps=5420)
[ep 42] step=20 loss=0.0198 curiosity=0.0987 (global_steps=5430)
  [Hierarchical] Option: scroll_down, step: 5

Episode 42 finished. Total episodes=42, total steps=5450, 
estimated cognitive age ≈ 0.11 months (~0.01 years). State + age saved.
```

---

## Troubleshooting

### "No checkpoint found"
- Normal on first run
- Agent starts from scratch
- Checkpoint saved after first episode

### "No option library found"
- Use `--option-library` to specify path
- Or let it create default options
- Train options for better performance

### Memory issues
- Reduce `max_memory`
- Reduce `mem_sample_size`
- Use `--fast-mode`

### Slow performance
- Enable `--fast-mode`
- Reduce `n_candidates`
- Use GPU with `--device cuda`

---

## Example Workflows

### Workflow 1: Standard Exploration
```bash
# Start with basic agent
python scripts/online_lifelong_learning.py --episodes 20

# Add temporal brain
python scripts/online_lifelong_learning.py --use-temporal --episodes 20
```

### Workflow 2: Hierarchical Learning
```bash
# Collect data
python scripts/online_lifelong_learning.py --episodes 50

# Discover options
python scripts/discover_options.py --log-dir logs --output datasets/options.json

# Train options
python scripts/train_options.py \
  --options datasets/options.json \
  --output checkpoints/option_library.pt

# Run with trained options
python scripts/online_lifelong_learning.py \
  --use-hierarchical \
  --option-library checkpoints/option_library.pt
```

### Workflow 3: Full System
```bash
# Run with all features
python scripts/online_lifelong_learning.py \
  --use-temporal \
  --use-hierarchical \
  --option-library checkpoints/option_library.pt \
  --device cuda
```

---


## Exploration & Behavior

The agent now includes aggressive exploration features enabled by default:

1.  **Heuristic Initialization**: Default options (like `scroll_down`, `move_to_top`) are pre-initialized to perform their named actions immediately, avoiding the "cold start" problem where the agent does nothing.
2.  **Curiosity Boost**: The agent prioritizes actions that cause visual changes or lead to novel states.
3.  **Boredom Breaker**: If the agent gets stuck, it will force a random action to break out of loops.

To adjust exploration aggressiveness, you can modify the parameters in `agent/hierarchical_agent.py` or pass them via `online_lifelong_learning.py` (requires code modification to expose all params).

See `EXPLORATION_ENHANCEMENTS.md` for full details on the math and parameters.

## Next Steps

After running the system:

1. **Analyze behavior**: Check `logs/` for episode recordings
2. **Inspect memory**: Look at `state/` for agent state
3. **Train options**: Use discovered patterns to train skills
4. **Iterate**: Improve option library, tune hyperparameters

Enjoy watching your AGI baby grow! 🎉
