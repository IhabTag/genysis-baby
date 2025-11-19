import os
import json
import numpy as np
import torch
import cv2

from collections import deque
from typing import Tuple, Dict, Any, List, Optional

from models.utils.preprocessing import preprocess_frame, encode_action
from models.utils.attention import SaliencyAttention
from models.utils.ocr import run_ocr_with_boxes
from memory.episodic_buffer import EpisodicBuffer
from memory.text_memory import TextMemory
from memory.episodic_retrieval import EpisodicRetrieval
from agent.screen_interpreter import ScreenInterpreter
from agent.goal_head import GoalCuriosityHead


class CuriousAgent:
    """
    Curiosity-driven agent with persistent memory and a fast-mode path.

    Curiosity score per candidate action is a mixture of:
      1) Latent change       : ||p_{t+1} - p_t||^2
      2) Episodic novelty    : distance to nearest memory state
      3) Attention change    : ||A_pred - A_t||^2
      4) Text novelty (OCR)  : new tokens predicted vs current
      5) Structural novelty  : coarse layout difference
      6) Goal curiosity      : discovering new 'goal states'

    Additional mechanisms:
      - Structured action proposals (regions, scroll, text-targeted clicks)
      - Boredom detection and escape actions
      - Diversity penalty for repeating same action type

    New:
      - save_state / load_state for:
          * episodic memory
          * text memory
          * goal memory
          * action stats
      - fast_mode to skip expensive OCR/goal evaluations on predicted frames
      - mem_sample_size to keep novelty computation cheap
    """

    def __init__(
        self,
        world_model,
        action_generator=None,       # optional fallback () -> action_dict
        n_candidates: int = 8,
        device: str = "cpu",
        max_memory: int = 2000,
        latent_weight: float = 1.0,
        novelty_weight: float = 0.7,
        attention_weight: float = 0.5,
        text_weight: float = 0.8,
        layout_weight: float = 0.3,
        goal_weight: float = 0.8,
        temporal_weight: float = 0.3,  # NEW: weight for temporal context
        epsilon: float = 0.05,
        boredom_window: int = 50,
        boredom_factor: float = 0.5,
        proj_dim: int = 64,
        fast_mode: bool = True,
        mem_sample_size: int = 512,
        temporal_brain=None,           # NEW: TemporalBrain instance
        use_episodic_retrieval: bool = True,  # NEW: enable episodic retrieval
    ):
        """
        Args:
          world_model: WorldModel instance
          action_generator: optional fallback function () -> action_dict
          n_candidates: number of candidate actions per step
          device: "cpu" or "cuda"
          max_memory: max episodic memory size
          *_weight: weights for curiosity components
          epsilon: epsilon-greedy random exploration
          boredom_window: how many past curiosity scores to consider
          boredom_factor: threshold ratio for boredom
          proj_dim: projection dimension for episodic memory
          fast_mode: if True, skip OCR/goal eval on predicted frames
          mem_sample_size: max number of memory states used for novelty
        """
        self.world_model = world_model.to(device)
        self.world_model.eval()

        self.action_generator = action_generator
        self.n_candidates = n_candidates
        self.device = device

        # Episodic memory on projection space
        self.memory = EpisodicBuffer(
            max_size=max_memory,
            device=device,
            proj_dim=proj_dim,
        )

        # Visual attention
        self.att_module = SaliencyAttention()

        # Text-level perception and memory
        self.text_memory = TextMemory()
        self.screen_interpreter = ScreenInterpreter()

        # Curiosity weights
        self.latent_weight = latent_weight
        self.novelty_weight = novelty_weight
        self.attention_weight = attention_weight
        self.text_weight = text_weight
        self.layout_weight = layout_weight
        self.goal_weight = goal_weight
        self.temporal_weight = temporal_weight  # NEW

        # Exploration
        self.epsilon = epsilon

        # Action diversity tracking
        self.action_counts: Dict[str, int] = {}
        self.last_action_type: Optional[str] = None

        # Boredom tracking
        self.recent_scores: deque = deque(maxlen=boredom_window)
        self.boredom_factor = boredom_factor

        # Goal curiosity head
        self.goal_head = GoalCuriosityHead()

        # Speed / performance knobs
        self.fast_mode = fast_mode
        self.mem_sample_size = mem_sample_size

        # NEW: Temporal brain for working memory
        self.temporal_brain = temporal_brain
        if self.temporal_brain is not None:
            self.temporal_brain.to(device)
            self.temporal_brain.eval()

        # NEW: Episodic retrieval for historical context
        self.episodic_retrieval = None
        if use_episodic_retrieval:
            self.episodic_retrieval = EpisodicRetrieval(
                self.memory,
                retrieval_k=5,
                context_dim=128,
                use_learned_aggregation=False  # Start with simple averaging
            )

        # NEW: Track last state for temporal continuity
        self.last_z: Optional[torch.Tensor] = None
        self.last_action_vec: Optional[torch.Tensor] = None
        self.last_context: Optional[torch.Tensor] = None

    # --------------------------------------------------------
    # Utility: text signatures
    # --------------------------------------------------------
    def _text_signature(self, elems: List[Dict[str, Any]]) -> set:
        """
        Build a set of lowercase tokens from OCR / screen elements.
        """
        tokens = set()
        for e in elems:
            t = e.get("text", "")
            for tok in t.split():
                tok = tok.strip().lower()
                if tok:
                    tokens.add(tok)
        return tokens

    # --------------------------------------------------------
    # Utility: coarse layout fingerprint
    # --------------------------------------------------------
    def _layout_signature(self, frame: np.ndarray, grid: int = 8) -> np.ndarray:
        """
        Downscale to grid x grid grayscale and flatten.
        Rough global structure signature.
        """
        small = cv2.resize(frame, (grid, grid), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        return gray.flatten()  # (grid*grid,)

    # --------------------------------------------------------
    # Memory: remember visited projection states
    # --------------------------------------------------------
    @torch.no_grad()
    def remember_state(self, frame: np.ndarray) -> None:
        """
        Encodes the given frame, projects to contrastive space, and stores p_t.
        """
        img_np = preprocess_frame(frame)  # (3,H,W), float32
        img_t = torch.tensor(
            img_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        z_t = self.world_model.encode(img_t)
        p_t = self.world_model.project(z_t)  # (1,proj_dim)
        p_vec = p_t.squeeze(0).cpu().numpy()

        self.memory.add(p_vec)

    # --------------------------------------------------------
    # Structured action proposal
    # --------------------------------------------------------
    def _generate_candidate_actions(
        self,
        width: int,
        height: int,
        frame: np.ndarray,
        screen_elems: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate structured candidate actions:
          - random big/small mouse moves
          - scroll up/down
          - clicks near detected text elements
          - fallback from external generator
        """
        import random

        candidates: List[Dict[str, Any]] = []

        # 1) Click near text elements (if any)
        random.shuffle(screen_elems)
        for e in screen_elems[:3]:  # up to 3 elements
            cx, cy = e["center"]
            candidates.append({"type": "MOVE_MOUSE", "x": int(cx), "y": int(cy)})
            candidates.append({"type": "LEFT_CLICK"})

        # 2) Global random moves (large exploration)
        for _ in range(2):
            candidates.append(
                {
                    "type": "MOVE_MOUSE",
                    "x": random.randint(0, width - 1),
                    "y": random.randint(0, height - 1),
                }
            )

        # 3) Region-focused moves (top bar, left panel, center)
        candidates.append(
            {  # Top bar
                "type": "MOVE_MOUSE",
                "x": random.randint(0, width - 1),
                "y": random.randint(0, int(height * 0.15)),
            }
        )
        candidates.append(
            {  # Left panel
                "type": "MOVE_MOUSE",
                "x": random.randint(0, int(width * 0.2)),
                "y": random.randint(0, height - 1),
            }
        )
        candidates.append(
            {  # Center
                "type": "MOVE_MOUSE",
                "x": random.randint(int(width * 0.3), int(width * 0.7)),
                "y": random.randint(int(height * 0.3), int(height * 0.7)),
            }
        )

        # 4) Scrolls
        candidates.append({"type": "SCROLL", "amount": random.randint(-300, -120)})
        candidates.append({"type": "SCROLL", "amount": random.randint(120, 300)})

        # 5) Occasionally type something small
        if random.random() < 0.2:
            txt = random.choice(["hello", "test", "ai", "genysis", "hi"])
            candidates.append({"type": "TYPE_TEXT", "text": txt})

        # 6) Fallback from external action generator
        if self.action_generator is not None:
            try:
                candidates.append(self.action_generator())
            except Exception:
                pass

        # Deduplicate
        unique: List[Dict[str, Any]] = []
        seen = set()
        for c in candidates:
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # Limit to n_candidates
        if len(unique) > self.n_candidates:
            unique = random.sample(unique, self.n_candidates)

        return unique

    # --------------------------------------------------------
    # Core: select action based on redesigned curiosity + goals
    # --------------------------------------------------------
    @torch.no_grad()
    def select_action(self, frame: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Given current RGB frame (H,W,3), chooses the most curious action.

        Also:
          - Runs OCR via ScreenInterpreter on current frame
          - Updates TextMemory with visible text segments
          - Computes multi-factor curiosity for each candidate

        In fast_mode:
          - OCR and goal curiosity on predicted frames are skipped
          - Text novelty and goal curiosity are effectively 0 for predictions.
        """
        assert isinstance(frame, np.ndarray), "Expected numpy array frame"
        assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3"

        H, W, _ = frame.shape

        # ----------------------------------------------------
        # 0) Interpret screen & update text memory (current frame only)
        # ----------------------------------------------------
        screen_elems = self.screen_interpreter.interpret(frame)
        for elem in screen_elems:
            self.text_memory.add(
                text=elem["text"],
                bbox=elem["bbox"],
                center=elem["center"],
                score=elem["score"],
                embedding=None,
            )

        # Text + layout signatures (for current frame)
        text_sig_t = self._text_signature(screen_elems)
        layout_sig_t = self._layout_signature(frame)

        # Simple 'num windows' heuristic (can replace with window detector later)
        num_windows_t = len(screen_elems)

        # Placeholder cursor mode (can be refined later)
        cursor_mode_t = "arrow"

        # ----------------------------------------------------
        # 1) Preprocess current frame & compute baseline features
        # ----------------------------------------------------
        img_np = preprocess_frame(frame)  # (3,H,W)
        img_t = torch.tensor(
            img_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Current attention map
        A_t, _ = self.att_module.compute(frame, prev_frame=None)
        A_t_tensor = torch.tensor(A_t, dtype=torch.float32, device=self.device)

        # Current latent projection
        z_t = self.world_model.encode(img_t)        # (1,latent)
        p_t = self.world_model.project(z_t)         # (1,proj_dim)

        # NEW: Get temporal context from working memory
        c_t = None
        if self.temporal_brain is not None and self.last_z is not None and self.last_action_vec is not None:
            c_t, _ = self.temporal_brain(self.last_z, self.last_action_vec)
            self.last_context = c_t

        # NEW: Retrieve relevant historical experiences
        r_t = None
        if self.episodic_retrieval is not None:
            r_t = self.episodic_retrieval.retrieve(p_t)

        # Episodic memory tensor (possibly sampled)
        mem_tensor = self.memory.get_memory_tensor(sample_size=self.mem_sample_size)

        # Remember current state in episodic memory
        self.remember_state(frame)

        best_action: Optional[Dict[str, Any]] = None
        best_score = -float("inf")

        # Generate structured candidate actions
        candidates = self._generate_candidate_actions(W, H, frame, screen_elems)
        if not candidates:
            return {"type": "NOOP"}, 0.0

        # ----------------------------------------------------
        # 2) Evaluate candidates
        # ----------------------------------------------------
        for action in candidates:
            # Encode action
            act_vec_np = encode_action(action, screen_width=W, screen_height=H)
            act_vec = torch.tensor(
                act_vec_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Predict next latent and projection
            z_next_pred = self.world_model.predict_latent(z_t, act_vec)
            p_next_pred = self.world_model.project(z_next_pred)  # (1,D)

            # 1) Latent curiosity
            latent_change = torch.mean((p_next_pred - p_t) ** 2).item()

            # 2) Episodic novelty (farther from memory = higher)
            novelty = 0.0
            if mem_tensor is not None:
                diff_mem = mem_tensor - p_next_pred  # (N,D)
                dists = torch.mean(diff_mem ** 2, dim=1)  # (N,)
                novelty = torch.min(dists).item()

            # 3) Predict next frame & attention change
            pred_img_next, _ = self.world_model(img_t, act_vec)
            pred_img_np = (
                pred_img_next.squeeze(0).permute(1, 2, 0).cpu().numpy()
            )
            pred_img_np = np.clip(pred_img_np * 255.0, 0, 255).astype(np.uint8)

            pred_img_resized = cv2.resize(
                pred_img_np, (W, H), interpolation=cv2.INTER_AREA
            )
            A_pred, _ = self.att_module.compute(pred_img_resized, prev_frame=frame)
            A_pred_tensor = torch.tensor(
                A_pred, dtype=torch.float32, device=self.device
            )
            att_change = torch.mean(
                (A_pred_tensor - A_t_tensor) ** 2
            ).item()

            # Defaults for fast_mode
            text_novelty = 0.0
            layout_diff = float(np.mean(
                (self._layout_signature(pred_img_resized) - layout_sig_t) ** 2
            ))
            goal_curiosity = 0.0

            # 4 & 6) Only do heavy OCR & goal curiosity on predictions if NOT in fast_mode
            if not self.fast_mode:
                try:
                    ocr_next = run_ocr_with_boxes(pred_img_resized)
                    text_sig_next = self._text_signature(ocr_next)
                    if text_sig_t or text_sig_next:
                        new_tokens = text_sig_next - text_sig_t
                        text_novelty = len(new_tokens) / max(1, len(text_sig_next))
                    else:
                        text_novelty = 0.0
                except Exception:
                    ocr_next = []
                    text_sig_next = set()
                    text_novelty = 0.0

                num_windows_next = len(ocr_next)
                cursor_mode_next = cursor_mode_t

                features_next = self.goal_head.extract_features(
                    ocr_tokens=list(text_sig_next),
                    layout_vec=self._layout_signature(pred_img_resized),
                    num_windows=num_windows_next,
                    cursor_mode=cursor_mode_next,
                    screen_elems=ocr_next,
                )
                goal_curiosity = self.goal_head.compute_goal_curiosity(features_next)

            # NEW: Temporal relevance (if temporal brain available)
            temporal_score = 0.0
            if c_t is not None:
                # Measure how well this action aligns with recent context
                # Simple heuristic: prefer actions that lead to states different from recent context
                # This encourages exploration of new temporal patterns
                temporal_score = latent_change  # Can be refined with learned metric

            # Combined curiosity
            score = (
                self.latent_weight * latent_change
                + self.novelty_weight * novelty
                + self.attention_weight * att_change
                + self.text_weight * text_novelty
                + self.layout_weight * layout_diff
                + self.goal_weight * goal_curiosity
                + self.temporal_weight * temporal_score
            )

            # Penalize repeating the same action type
            a_type = action.get("type")
            if a_type == self.last_action_type:
                score *= 0.7

            # Normalize by usage frequency to encourage diversity
            freq = self.action_counts.get(a_type, 1)
            score /= (freq ** 0.5)

            if score > best_score:
                best_score = score
                best_action = action

        # ----------------------------------------------------
        # 3) Update diversity stats and apply boredom logic
        # ----------------------------------------------------
        if best_action is None:
            best_action = {"type": "NOOP"}

        best_type = best_action.get("type")
        self.action_counts[best_type] = self.action_counts.get(best_type, 1) + 1
        self.last_action_type = best_type

        # Record score history
        self.recent_scores.append(best_score)

        # Boredom detection: if current score << recent average, escape
        if len(self.recent_scores) >= max(2, self.recent_scores.maxlen // 2):
            avg_recent = float(np.mean(self.recent_scores))
            if avg_recent > 0 and best_score < self.boredom_factor * avg_recent:
                import random
                # Override with an exploration-heavy action
                if random.random() < 0.5:
                    explore_action: Dict[str, Any] = {
                        "type": "MOVE_MOUSE",
                        "x": np.random.randint(0, W),
                        "y": np.random.randint(0, H),
                    }
                else:
                    explore_action = {
                        "type": "SCROLL",
                        "amount": random.choice([-400, -250, 250, 400]),
                    }
                best_action = explore_action
                best_score = 0.0

        # Epsilon-greedy random exploration (if external generator exists)
        if self.action_generator is not None and np.random.rand() < self.epsilon:
            try:
                from agent.random_agent import random_action
                random_act = random_action(width=W, height=H)
                return random_act, 0.0
            except Exception:
                pass

        # NEW: Update temporal state for next step
        if self.temporal_brain is not None:
            self.last_z = z_t.detach()
            # Encode the selected action
            best_act_vec_np = encode_action(best_action, screen_width=W, screen_height=H)
            self.last_action_vec = torch.tensor(
                best_act_vec_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        return best_action, best_score

    # --------------------------------------------------------
    # Helper: get visible text
    # --------------------------------------------------------
    @torch.no_grad()
    def get_visible_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns OCR/interpreter elements for debugging or higher-level reasoning.
        """
        return self.screen_interpreter.interpret(frame)

    # --------------------------------------------------------
    # Persistence API (full agent state)
    # --------------------------------------------------------
    def save_state(self, state_dir: str):
        """
        Save episodic memory, text memory, goal memory, and agent meta.
        """
        os.makedirs(state_dir, exist_ok=True)

        # 1) Episodic memory
        epi_path = os.path.join(state_dir, "episodic_memory.npz")
        self.memory.save(epi_path)

        # 2) Text memory
        text_path = os.path.join(state_dir, "text_memory.json")
        self.text_memory.save(text_path)

        # 3) Goal memory
        goal_path = os.path.join(state_dir, "goal_memory.json")
        self.goal_head.save(goal_path)

        # 4) Agent meta
        meta = {
            "action_counts": self.action_counts,
            "last_action_type": self.last_action_type,
            "recent_scores": list(self.recent_scores),
            "fast_mode": self.fast_mode,
            "mem_sample_size": self.mem_sample_size,
        }
        meta_path = os.path.join(state_dir, "agent_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # NEW: 5) Temporal brain state
        if self.temporal_brain is not None:
            temporal_path = os.path.join(state_dir, "temporal_brain.pt")
            self.temporal_brain.save_state(temporal_path)

    def load_state(self, state_dir: str):
        """
        Load episodic memory, text memory, goal memory, and agent meta
        if present. Safe if files missing.
        """
        # 1) Episodic memory
        epi_path = os.path.join(state_dir, "episodic_memory.npz")
        self.memory.load(epi_path)

        # 2) Text memory
        text_path = os.path.join(state_dir, "text_memory.json")
        self.text_memory.load(text_path)

        # 3) Goal memory
        goal_path = os.path.join(state_dir, "goal_memory.json")
        self.goal_head.load(goal_path)

        # 4) Agent meta
        meta_path = os.path.join(state_dir, "agent_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.action_counts = meta.get("action_counts", {})
            self.last_action_type = meta.get("last_action_type", None)
            self.recent_scores.clear()
            for s in meta.get("recent_scores", []):
                self.recent_scores.append(float(s))
            # Respect persisted fast_mode / mem_sample_size if present
            self.fast_mode = bool(meta.get("fast_mode", self.fast_mode))
            self.mem_sample_size = int(meta.get("mem_sample_size", self.mem_sample_size))

        # NEW: 5) Temporal brain state
        if self.temporal_brain is not None:
            temporal_path = os.path.join(state_dir, "temporal_brain.pt")
            if os.path.exists(temporal_path):
                from models.temporal_brain import TemporalBrain
                self.temporal_brain = TemporalBrain.load_state(temporal_path, device=self.device)
                self.temporal_brain.eval()
