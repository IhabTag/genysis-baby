# GENYSIS-BABY: AGI Infant Development System

## 🎯 Project Overview

**GENYSIS-BABY** is an ambitious artificial general intelligence (AGI) research project that simulates the cognitive development of a newborn AI agent. The project implements a curiosity-driven learning system that perceives, learns, and interacts with a computer desktop environment through a biologically-inspired developmental approach.

This is the core developmental module of the broader Genysis AGI project, focusing on foundational learning mechanisms that mirror early human cognitive development.

## 🧠 Core Concept

The project simulates an "AGI infant" that:
- **Perceives** its world through visual observation (desktop screen capture)
- **Acts** through motor control (mouse and keyboard interactions)
- **Learns** through self-supervised world modeling and curiosity-driven exploration
- **Remembers** through episodic and semantic memory systems
- **Grows** through continuous online learning and self-improvement

## 🏗️ System Architecture

### 1. **Environment Layer** (`env/`)
The sensorimotor interface between the agent and the computer desktop environment.

- **`computer_env.py`**: Gym-like environment wrapper providing:
  - Observation space: RGB screen captures (H×W×3)
  - Action space: Mouse movements, clicks, keyboard input, scrolling
  - Episode management and logging
  - Configurable screen resolution (default: 1280×1024)

- **`obs.py`**: Screen capture using MSS (Multiple Screen Shots)
  - Real-time desktop frame acquisition
  - Configurable resolution and region

- **`actions.py`**: Action execution via PyAutoGUI
  - Mouse movements (absolute and relative)
  - Left/right clicks
  - Keyboard typing
  - Scroll actions

- **`logging.py`**: Episode logging and data persistence
  - Saves observations, actions, and metadata
  - Structured episode directories

### 2. **World Model** (`models/`)
Neural network-based predictive model of the environment dynamics.

**Core Architecture** (`world_model.py`):
- **Encoder**: Convolutional neural network that compresses visual observations into latent representations (z)
- **Projection Head**: Maps latent states to contrastive learning space (p) for similarity comparisons
- **Dynamics Model**: Predicts next latent state given current state and action: `f(z_t, a_t) → z_{t+1}`
- **Decoder**: Reconstructs predicted next frame from predicted latent state

**Model Utilities** (`models/utils/`):
- **`encoder_blocks.py`**: Convolutional encoder/decoder architectures
- **`preprocessing.py`**: Frame normalization, action encoding, and data preparation
- **`attention.py`**: Visual attention mechanism for saliency detection
- **`ocr.py`**: Optical character recognition for text extraction (Tesseract-based)
- **`patch_embeddings.py`**: Patch-based visual feature extraction
- **`projection_head.py`**: Contrastive learning projection layers
- **`dynamics.py`**: State transition prediction networks

**Learning Objectives**:
1. **Reconstruction Loss**: Minimize pixel-level prediction error
2. **Latent Consistency**: Ensure predicted latent states match true encoded states
3. **Contrastive Learning**: Learn meaningful representations through similarity comparisons

### 3. **Memory Systems** (`memory/`)

**`episodic_buffer.py`**: Short-term episodic memory
- Stores recent state projections (p_t) in contrastive space
- Fixed-size circular buffer (default: 1500 entries)
- Used for novelty detection and state comparison

**`text_memory.py`**: Semantic text memory
- Tracks observed text elements from OCR
- Maintains frequency counts and recency information
- Enables text-based novelty detection

**`replay_buffer.py`**: Experience replay storage
- Stores (state, action, next_state) transitions
- Supports offline training and batch learning

**`curiosity.py`**: Intrinsic motivation module
Computes multi-factor curiosity scores:
1. **Latent Curiosity**: `||p_pred - p_true||²` (prediction error in contrastive space)
2. **Novelty Curiosity**: Distance to nearest memory embedding
3. **Attention Curiosity**: `||A_pred - A_true||²` (attention map changes)

Combined formula:
```
curiosity = w_latent × latent_error + w_novelty × novelty + w_attention × attention_error
```

### 4. **Agent Systems** (`agent/`)

**`curious_agent.py`**: Main curiosity-driven agent (559 lines)
- **Action Selection**: Generates and evaluates candidate actions based on curiosity
- **Multi-Factor Curiosity**: Combines latent change, text novelty, layout change, and goal alignment
- **Fast Mode**: Optimized inference path that skips OCR on predicted frames
- **Persistent Memory**: Maintains episodic, text, and goal memories across sessions
- **State Persistence**: Saves/loads complete agent state for lifelong learning

**Key Features**:
- Candidate action generation (mouse moves, clicks, scrolls, text-targeted actions)
- Screen interpretation via OCR
- Text signature and layout signature for state comparison
- Boredom mechanism to avoid repetitive actions
- Goal-based curiosity for task-oriented exploration

**`random_agent.py`**: Baseline random exploration agent
- Generates random actions for comparison and fallback
- Used for initial data collection

**`instruction_agent.py`**: Natural language instruction executor
- Parses simple text commands
- Maps commands to action sequences

**`text_actions.py`**: Text-aware action agent
- Targets interactive elements detected via OCR
- Clicks on buttons, links, and text elements

**`task_planner.py`**: Multi-step task planning and verification
- Decomposes complex instructions into substeps
- Executes with verification (OCR change, screen change, pixel difference)
- Retry logic for failed steps

**`screen_interpreter.py`**: Screen understanding module
- OCR-based element detection
- Screen layout analysis

**`goal_head.py`**: Goal-oriented curiosity head
- Learns goal representations
- Computes goal alignment scores

**`command_parser.py`**: Natural language command parsing
- Extracts action parameters from text instructions

### 5. **Training Scripts** (`scripts/`)

**Data Collection**:
- **`collect_random_dataset.py`**: Generate offline exploration dataset with random actions
  - Creates structured episode directories
  - Saves observations and actions for supervised training

**World Model Training**:
- **`train_world_model_contrastive.py`**: Offline contrastive learning
  - Trains on collected datasets
  - Combines reconstruction and contrastive losses
  - Saves checkpoints to `checkpoints/world_model_contrastive.pt`

**Inference & Testing**:
- **`run_world_model_inference.py`**: Test forward pass and predictions
- **`test_curious_agent.py`**: Validate curiosity-driven behavior
- **`run_curious_training.py`**: Online curiosity-driven exploration loop

**Lifelong Learning**:
- **`online_lifelong_learning.py`**: Continuous learning system
  - Persistent agent state across sessions
  - Tracks "cognitive age" (total episodes and steps)
  - Online world model updates during exploration
  - Saves state, memory, and age metrics
  - Runs indefinitely for developmental progression

**Debug Tools** (`scripts/debug/`):
- `test_obs.py`: Verify screen capture
- `test_actions.py`: Test mouse/keyboard control
- `test_attn.py`: Validate attention maps
- `print_shapes.py`: Check data tensor shapes

## 🔬 Key Innovations

### 1. **Curiosity-Driven Learning**
Unlike traditional RL agents that require external rewards, GENYSIS-BABY is intrinsically motivated by:
- Prediction errors (what surprises the model)
- Novelty (what hasn't been seen before)
- Attention changes (what's visually salient)
- Goal alignment (what matches learned objectives)

### 2. **Developmental Approach**
The system tracks "cognitive age" based on total experience:
- ~50,000 steps ≈ 1 human cognitive month (heuristic)
- Persistent memory across sessions
- Continuous learning without catastrophic forgetting

### 3. **Multi-Modal Perception**
Combines:
- Raw pixel observations
- Learned latent representations
- OCR text extraction
- Visual attention maps
- Layout signatures

### 4. **Fast Mode Optimization**
Intelligent inference path that:
- Skips expensive OCR on predicted frames
- Samples episodic memory for efficiency
- Reduces candidate actions for speed
- Maintains exploration quality

### 5. **Hierarchical Action Selection**
- Low-level: Random motor babbling
- Mid-level: Curiosity-driven exploration
- High-level: Instruction following and task planning

## 🛠️ Technology Stack

**Core ML/AI**:
- PyTorch 2.1.0 (neural networks and training)
- TorchVision 0.16.0 (vision utilities)

**Computer Vision**:
- OpenCV 4.8.1 (image processing)
- Pillow 10.0.1 (image I/O)
- Matplotlib 3.8.0 (visualization)

**Environment Control**:
- MSS 9.0.1 (screen capture)
- PyAutoGUI 0.9.54 (mouse/keyboard control)
- pynput 1.7.6 (input monitoring)
- python-xlib 0.33 (X11 interface for Linux)

**OCR**:
- Tesseract OCR (external dependency)
- pytesseract (Python wrapper)

**Utilities**:
- NumPy 1.25.2 (numerical computing)
- SciPy 1.11.3 (scientific computing)
- tqdm 4.66.1 (progress bars)
- psutil 5.9.5 (system monitoring)
- PyYAML 6.0.1 (configuration)

**Infrastructure**:
- Docker (containerized environment)
- VNC (remote desktop access)
- Xvfb (virtual framebuffer)
- Openbox (lightweight window manager)

## 🐳 Deployment Architecture

The system runs in a sandboxed Docker container (`genysis-sandbox` - separate repository):
- Full Linux desktop environment
- VNC server on port 5900
- Isolated from host system
- Pre-configured GUI libraries and dependencies

**Workflow**:
1. Build and run `genysis-sandbox` Docker container
2. Connect via VNC to view the "baby's world"
3. Inside container: clone `genysis-baby` repository
4. Install Python dependencies in virtualenv
5. Run training/exploration scripts
6. Observe autonomous learning behavior

## 📊 Data Flow

```
Screen Capture → Preprocessing → Encoder → Latent State (z)
                                              ↓
                                         Projection (p)
                                              ↓
                                    Episodic Memory Storage
                                              ↓
Action Candidates ← Curiosity Scoring ← Memory Comparison
        ↓
Action Execution → Environment Change → Next Observation
        ↓
World Model Training (online or offline)
```

## 🎓 Learning Pipeline

### Phase 1: Bootstrap (Offline)
1. Collect random exploration dataset
2. Train world model on collected data
3. Learn basic visual representations

### Phase 2: Curious Exploration (Online)
1. Agent generates candidate actions
2. World model predicts outcomes
3. Curiosity module scores each candidate
4. Execute most curious action
5. Update memory with new experience
6. Incrementally train world model

### Phase 3: Instruction Following
1. Parse natural language commands
2. Decompose into substeps
3. Execute with verification
4. Retry failed steps

### Phase 4: Lifelong Learning
1. Persistent state across sessions
2. Continuous memory accumulation
3. Progressive skill development
4. Age tracking and developmental milestones

## 📁 Project Structure

```
genysis-baby/
├── env/                    # Environment interface
│   ├── obs.py             # Screen capture
│   ├── actions.py         # Action execution
│   ├── logging.py         # Episode logging
│   └── computer_env.py    # Main environment
│
├── models/                 # Neural network models
│   ├── world_model.py     # Unified world model
│   └── utils/             # Model components
│       ├── encoder_blocks.py
│       ├── preprocessing.py
│       ├── attention.py
│       ├── ocr.py
│       ├── dynamics.py
│       ├── projection_head.py
│       └── patch_embeddings.py
│
├── memory/                 # Memory systems
│   ├── curiosity.py       # Intrinsic motivation
│   ├── episodic_buffer.py # Short-term memory
│   ├── text_memory.py     # Semantic text memory
│   └── replay_buffer.py   # Experience replay
│
├── agent/                  # Agent implementations
│   ├── curious_agent.py   # Main curiosity agent
│   ├── random_agent.py    # Baseline agent
│   ├── instruction_agent.py
│   ├── text_actions.py
│   ├── task_planner.py
│   ├── screen_interpreter.py
│   ├── goal_head.py
│   └── command_parser.py
│
├── scripts/                # Training & testing
│   ├── collect_random_dataset.py
│   ├── train_world_model_contrastive.py
│   ├── run_world_model_inference.py
│   ├── test_curious_agent.py
│   ├── run_curious_training.py
│   ├── online_lifelong_learning.py
│   └── debug/             # Debug utilities
│
├── datasets/               # Collected data
├── checkpoints/            # Model weights
├── state/                  # Agent persistent state
├── logs/                   # Episode logs
├── text/                   # Text data
├── requirements.txt        # Python dependencies
└── README.MD              # Setup instructions
```

## 🎯 Research Goals

1. **Developmental AI**: Study how artificial agents can learn through curiosity-driven exploration, similar to infant development
2. **Self-Supervised Learning**: Minimize reliance on external rewards and labels
3. **Continual Learning**: Enable lifelong learning without catastrophic forgetting
4. **Emergent Behavior**: Observe complex behaviors emerging from simple curiosity mechanisms
5. **World Modeling**: Learn predictive models of environment dynamics
6. **Transfer Learning**: Develop skills that transfer across tasks

## 🚀 Future Directions

- **Language Models**: Integrate LLMs for better instruction understanding
- **Multi-Modal Fusion**: Combine vision, audio, and text modalities
- **Hierarchical Planning**: More sophisticated task decomposition
- **Social Learning**: Learn from demonstrations and human feedback
- **Meta-Learning**: Learn how to learn more efficiently
- **Sim-to-Real**: Transfer learned behaviors to real-world robotics

## 🔍 Verification & Testing

The project includes comprehensive smoke tests:
1. ✅ Screen capture functionality
2. ✅ Mouse/keyboard action execution
3. ✅ Environment reset/step cycle
4. ✅ Dataset collection pipeline
5. ✅ World model forward pass
6. ✅ World model training loop
7. ✅ Attention map generation
8. ✅ Curious agent behavior
9. ✅ Online training loop
10. ✅ Lifelong learning persistence

## 📝 Key Metrics

- **Cognitive Age**: Total steps / 50,000 ≈ months of development
- **Curiosity Score**: Multi-factor intrinsic motivation value
- **Prediction Error**: World model accuracy (MSE)
- **Memory Size**: Number of unique states remembered
- **Text Novelty**: Unique text elements discovered
- **Episode Length**: Steps before termination
- **Exploration Coverage**: Unique screen states visited

## 🎓 Scientific Foundations

The project draws inspiration from:
- **Developmental Psychology**: Infant learning and exploration
- **Neuroscience**: Predictive coding and curiosity in the brain
- **Reinforcement Learning**: Intrinsic motivation and exploration bonuses
- **Self-Supervised Learning**: Contrastive learning and world models
- **Cognitive Science**: Episodic memory and attention mechanisms

## 🏆 Project Status

**Current Capabilities**:
- ✅ Autonomous desktop exploration
- ✅ Curiosity-driven action selection
- ✅ Online world model learning
- ✅ Persistent memory across sessions
- ✅ OCR-based screen understanding
- ✅ Basic instruction following
- ✅ Lifelong learning with age tracking

**In Development**:
- 🔄 Advanced task planning
- 🔄 Goal-oriented behavior
- 🔄 Language model integration
- 🔄 Hierarchical skill learning

## 👥 Target Audience

- **AI Researchers**: Studying developmental AI and curiosity-driven learning
- **ML Engineers**: Implementing self-supervised learning systems
- **Cognitive Scientists**: Modeling infant cognitive development
- **Roboticists**: Developing autonomous exploration systems
- **Students**: Learning about AGI, world models, and intrinsic motivation

## 📄 License & Attribution

This is a research project exploring artificial general intelligence through developmental approaches. The codebase demonstrates a complete implementation of curiosity-driven learning in a computer environment.

---

**GENYSIS-BABY** represents a step toward artificial general intelligence through biologically-inspired developmental learning. By simulating the curiosity and exploration of a newborn, the system aims to discover fundamental learning mechanisms that could scale to more complex cognitive abilities.
