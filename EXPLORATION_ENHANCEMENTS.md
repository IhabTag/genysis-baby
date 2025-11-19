# Exploration Enhancements

This document details the improvements made to the GENYSIS-BABY agent to address the issue of low curiosity and limited exploration.

## Problem
The initial implementation of the `HierarchicalAgent` exhibited:
1.  **Low Curiosity Scores**: Curiosity values were consistently near zero (< 0.001), leading to random or stagnant behavior.
2.  **Cold Start Issue**: Default options (randomly initialized networks) produced similar, meaningless actions (small jitters), causing the world model to predict "no change" for all of them. This resulted in zero curiosity and no incentive to explore.
3.  **Lack of Diversity**: The agent would often get stuck repeating the same option or switching between two similar ones.

## Solutions

### 1. Aggressive Curiosity Formula
We updated the `select_option` method in `HierarchicalAgent` with a more aggressive scoring formula:

```python
score = exploration_factor * (
    curiosity_weight * latent_change
    + novelty_weight * novelty
    + diversity_bonus * diversity
)
```

*   **Latent Change**: Measures how much the world state changes (`||z_{t+1} - z_t||^2`).
*   **Novelty**: Exponentially scaled distance to nearest memory (`1.0 - exp(-10 * min_dist)`). This ensures even small deviations from known states register as novel.
*   **Diversity Bonus**: `1.0 / (1.0 + usage_count * 0.1)`. Encourages trying options that haven't been used much.
*   **Exploration Boost**: Multiplier (1.5x) applied during the first 1000 steps to jump-start exploration.

### 2. Heuristic Initialization (Cold Start Fix)
We modified `agent/option_policy.py` to initialize default options with heuristic behaviors instead of random weights. This ensures that from step 0, the agent has distinct tools available:

*   **`scroll_down`**: Initialized to output a strong positive value for the scroll parameter.
*   **`scroll_up`**: Initialized to output a strong negative value for the scroll parameter.
*   **`move_to_top`**: Initialized to move mouse to y=-1.0.
*   **`move_to_left`**: Initialized to move mouse to x=-1.0.
*   **`move_large`**: Initialized with Xavier uniform weights for random large movements.
*   **Termination Bias**: Set to -3.0 (sigmoid ≈ 0.05) to encourage options to run for at least ~20 steps before terminating, preventing rapid-fire switching.

### 3. Boredom Breaker
A safety mechanism was added to `HierarchicalAgent`:
*   If the best option's score is below `boredom_threshold` (0.01) for 3 consecutive steps, the agent forces a **random** option selection.
*   This prevents getting stuck in local minima where all known options seem "boring".

## Tuning Parameters

The following parameters in `HierarchicalAgent` control this behavior:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `curiosity_weight` | 2.0 | Importance of causing state changes. |
| `novelty_weight` | 1.5 | Importance of reaching new states. |
| `diversity_bonus` | 0.5 | Bonus for rarely used options. |
| `exploration_boost` | 1.5 | Multiplier for first 1000 steps. |
| `epsilon` | 0.2 | Probability of random action (epsilon-greedy). |
| `boredom_threshold` | 0.01 | Minimum score before "boredom" triggers. |

## Expected Behavior
With these changes, you should observe:
1.  **Higher Curiosity Scores**: Values in the 0.01 - 0.1 range (instead of 0.0001).
2.  **Distinct Actions**: The agent will clearly scroll, move to corners, and type, rather than just jittering.
3.  **Phase Shifts**: The agent will try an option (e.g., scroll down) for a while, then switch to another (e.g., move to top) as the first one becomes predictable/boring.
