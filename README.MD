🌱 **GENYSIS-BABY**
===================

### The AGI Infant: Perception → World Model → Curiosity → Action

genysis-baby is the _developmental core_ of the Genysis AGI project.

It simulates an **AGI newborn** with:

*   **Visual perception** (desktop screen)
    
*   **Motor control** (mouse/keyboard)
    
*   **World model learning** (contrastive + predictive)
    
*   **Attention saliency**
    
*   **Episodic memory**
    
*   **Curiosity-driven action selection**
    
*   **Online self-training**
    

This repository contains all baby logic (environment, world-model, agent, memory).The **Docker runtime** (GUI + VNC + dependencies) lives in the separate repo:

📦 genysis-sandbox → https://github.com/your-org/genysis-sandbox

📦 1 — Installation
===================

1.1 Get into the sandbox environment
------------------------------------

Clone and build the sandbox:

L`   git clone https://github.com/your-org/genysis-sandbox.git  cd genysis-sandbox  docker build -t genysis-sandbox -f Dockerfile.sandbox .   `

Run the sandbox:

L`   docker run --rm -it \    --shm-size=2g \    --name genysis \    genysis-sandbox   `

This opens a full Linux desktop accessible via VNC.

1.2 Inside sandbox: clone & install genysis-baby
------------------------------------------------

Inside the container shell:

L`   cd /home/agent/  git clone https://github.com/your-org/genysis-baby.git  cd genysis-baby   `

Install Python deps inside the pre-created virtualenv:

L`   source venv/bin/activate  pip install -r requirements.txt   `

That’s it.The container already includes:

*   Xvfb
    
*   Openbox
    
*   VNC server
    
*   PyAutoGUI deps
    
*   System GUI libraries
    

🖥 2 — Open VNC
===============

Use any VNC client:

L`   Host: localhost  Port: 5900  Password: (if configured in genysis-sandbox)   `

You should see the Openbox desktop—this is the AGI baby’s “world”.

🔎 3 — Smoke Test Checklist
===========================

These are the **sanity tests** to confirm the entire pipeline works end-to-end.

Run all tests _inside the container + venv_:

L`   source /home/agent/genysis-baby/venv/bin/activate  cd /home/agent/genysis-baby   `

✅ **Phase 1 — Environment Layer**
=================================

### 1.1 Test screen capture

L`   python3 scripts/debug/test_obs.py   `

Expected →

L`   Captured frame shape: (768,1024,3)   `

### 1.2 Test mouse/keyboard actions

L`   python3 scripts/debug/test_actions.py   `

Expected → mouse moves, clicks, types, scrolls in VNC.

### 1.3 Test full environment

L`   python3 - << 'EOF'  from env.computer_env import ComputerEnv  env = ComputerEnv()  obs = env.reset()  print(obs["image"].shape)  obs2,_,_,_ = env.step({"type":"LEFT_CLICK"})  print(obs2["image"].shape)  EOF   `

✅ **Phase 2 — Dataset Collection**
==================================

Generate offline dataset:

L`   python3 scripts/collect_random_dataset.py   `

Expect:

L`   datasets/experience/ep_000001/  datasets/experience/ep_000002/  ...   `

Verify shapes:

L`   python3 scripts/debug/print_shapes.py   `

✅ **Phase 3 — World Model Inference**
=====================================

Try model forward pass (random weights):

L`   python3 scripts/run_world_model_inference.py   `

You should see:

*   img\_t
    
*   img\_next
    
*   pred\_next (random noise)
    

✅ **Phase 4 — World Model Training**
====================================

Train 1–10 epochs:

L`   python3 scripts/train_world_model_contrastive.py   `

Expected:

*   Epoch loss printed
    
*   Checkpoint saved at:checkpoints/world\_model\_contrastive.pt
    

✅ **Phase 5 — Attention Map Test**
==================================

L`   python3 scripts/debug/test_attn.py   `

Expected:

*   Attention map (H,W)
    
*   Values between 0–1
    
*   Top saliency regions printed
    

✅ **Phase 6 — Curious Agent Basic Test**
========================================

L`   python3 scripts/test_curious_agent.py   `

Expect:

*   Actions printed per step
    
*   Curiosity values > 0
    
*   Baby moves mouse, scrolls, clicks, types
    

⚡ Optional: Full Online Training Loop
=====================================

Runs curiosity → saves transitions → updates world model → repeats.

L`   python3 scripts/run_curious_training.py   `

You’ll see:

*   Agent acts autonomously
    
*   Online experience saved under datasets/online\_experience/
    
*   Model gets updated each episode
    
*   Curiosity behavior becomes slightly less random over time
    

✔️ If all phases pass
=====================

🎉 **Your AGI baby is fully operational.**

You now have:

*   Perception
    
*   Motor control
    
*   World model
    
*   Attention
    
*   Episodic memory
    
*   Curiosity policy
    
*   Online self-learning
    

Everything working inside a controlled VM-like sandbox.

📚 Structure Reference
======================

L`   genysis-baby/    env/      obs.py      actions.py      logging.py      computer_env.py    models/      world_model.py      utils/        preprocessing.py        encoder_blocks.py        projection_head.py        dynamics.py        attention.py    memory/      curiosity.py      episodic_buffer.py    agent/      random_agent.py      curious_agent.py    scripts/      collect_random_dataset.py      train_world_model_contrastive.py      run_world_model_inference.py      run_curious_training.py      test_curious_agent.py      debug/        test_obs.py        test_actions.py        test_attn.py        print_shapes.py    requirements.txt    README.md   `