import numpy as np
import torch
import cv2

from typing import Tuple, Dict, Any, Optional, List

from models.utils.preprocessing import preprocess_frame, encode_action
from models.utils.attention import SaliencyAttention
from memory.episodic_buffer import EpisodicBuffer
from memory.text_memory import TextMemory
from agent.screen_interpreter import ScreenInterpreter
from agent.text_actions import TextActionAgent  # Step 15: semantic actions


class CuriousAgent:
    """
    Curiosity-driven agent operating in:
      - contrastive projection space (p_t)
      - attention space (A_t)
      - high-level text space (via TextMemory)

    Curiosity score per candidate action is a combination of:
      1) Local latent change:       ||p_{t+1} - p_t||^2
      2) Novelty vs episodic memory (min distance in projection space)
      3) Attention change:          ||A_pred - A_t||^2

    TextMemory is updated every time select_action() is called
    using the ScreenInterpreter (OCR over salient patches).

    Step 15 extension:
      - The agent can optionally be wired to a TextActionAgent
        to perform semantic actions like:
          * find_and_click("Search")
          * type_into("Search", "hello world")
    """

    def __init__(
        self,
        world_model,
        action_generator,
        n_candidates: int = 8,
        device: str = "cpu",
        max_memory: int = 1000,
        local_weight: float = 1.0,
        novelty_weight: float = 0.5,
        attention_weight: float = 0.4,
        epsilon: float = 0.1,
        text_action_agent: Optional[TextActionAgent] = None,
    ):
        """
        Args:
          world_model: WorldModel instance (with encode, project, predict_latent, forward)
          action_generator: function () -> action_dict
          n_candidates: number of candidate actions per step
          device: "cpu" or "cuda"
          max_memory: max episodic states in projection memory
          local_weight: weight for local latent change
          novelty_weight: weight for novelty
          attention_weight: weight for attention change
          epsilon: epsilon-greedy exploration probability
          text_action_agent: optional TextActionAgent instance for semantic actions
        """
        self.world_model = world_model.to(device)
        self.world_model.eval()

        self.action_generator = action_generator
        self.n_candidates = n_candidates
        self.device = device

        # We store projections p_t in episodic memory
        self.memory = EpisodicBuffer(max_size=max_memory, device=device)

        # Visual attention
        self.att_module = SaliencyAttention()

        # Text-level perception and memory
        self.text_memory = TextMemory()
        self.screen_interpreter = ScreenInterpreter()

        # Optional semantic-action layer (Step 15)
        self.text_action_agent: Optional[TextActionAgent] = text_action_agent

        # Curiosity weights
        self.local_weight = local_weight
        self.novelty_weight = novelty_weight
        self.attention_weight = attention_weight

        # Exploration
        self.epsilon = epsilon

        # Action diversity tracking
        self.action_counts: Dict[str, int] = {}
        self.last_action_type: Any = None

    # --------------------------------------------------------
    # Wiring / integration helpers
    # --------------------------------------------------------
    def attach_text_action_agent(self, taa: TextActionAgent) -> None:
        """
        Attach a TextActionAgent after construction.

        Typical pattern:
            taa = TextActionAgent(executor, screen_interpreter, text_memory)
            agent.attach_text_action_agent(taa)
        """
        self.text_action_agent = taa

    # --------------------------------------------------------
    # Memory: remember visited projection states
    # --------------------------------------------------------
    @torch.no_grad()
    def remember_state(self, frame: np.ndarray) -> None:
        """
        Encodes the given frame, projects to contrastive space, and stores p_t.
        frame: RGB uint8 (H,W,3)
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
    # Internal helper: update TextMemory from current frame
    # --------------------------------------------------------
    def _update_text_memory(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs OCR+attention via ScreenInterpreter and pushes all
        visible text segments into TextMemory.

        Returns:
          list of screen elements (same structure as ScreenInterpreter.interpret)
        """
        screen_elems = self.screen_interpreter.interpret(frame)
        for elem in screen_elems:
            self.text_memory.add(
                text=elem["text"],
                bbox=elem["bbox"],
                center=elem["center"],
                score=elem["score"],
                embedding=None,  # can be wired to patch embeddings later
            )
        return screen_elems

    # --------------------------------------------------------
    # Core: select action based on curiosity
    # --------------------------------------------------------
    @torch.no_grad()
    def select_action(self, frame: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Given current RGB frame (H,W,3), chooses the most curious action.

        Also:
          - Runs OCR on salient patches
          - Updates TextMemory with visible text segments

        Returns:
          best_action: dict
          curiosity_score: float
        """
        assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3 RGB"

        H, W, _ = frame.shape

        # ----------------------------------------------------
        # 0) OCR + text memory (proto-reading)
        # ----------------------------------------------------
        self._update_text_memory(frame)

        # ----------------------------------------------------
        # 1) Preprocess current frame once
        # ----------------------------------------------------
        img_np = preprocess_frame(frame)  # (3,H,W)
        img_t = torch.tensor(
            img_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Current attention map
        A_t, _ = self.att_module.compute(frame, prev_frame=None)

        # Current latent projection
        z_t = self.world_model.encode(img_t)
        p_t = self.world_model.project(z_t)  # (1,D)

        # Episodic memory tensor
        mem_tensor = self.memory.get_memory_tensor()

        A_t_tensor = torch.tensor(
            A_t, dtype=torch.float32, device=self.device
        )

        best_action = None
        best_score = -float("inf")

        # ----------------------------------------------------
        # 2) Try multiple candidate actions (motor-space search)
        # ----------------------------------------------------
        for _ in range(self.n_candidates):
            action = self.action_generator()

            act_vec_np = encode_action(action, screen_width=W, screen_height=H)
            act_vec = torch.tensor(
                act_vec_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Predict next latent
            z_next_pred = self.world_model.predict_latent(z_t, act_vec)
            p_next_pred = self.world_model.project(z_next_pred)  # (1,D)

            # 1) Local curiosity: change in projection
            local_change = torch.mean((p_next_pred - p_t) ** 2).item()

            # 2) Novelty: min distance from memory
            novelty = 0.0
            if mem_tensor is not None:
                diff_mem = mem_tensor - p_next_pred  # (N,D)
                dists = torch.mean(diff_mem ** 2, dim=1)  # (N,)
                novelty = torch.min(dists).item()

            # 3) Attention change: difference between predicted attention and current
            #    We use the decoder to approximate next frame.
            pred_img_next, _ = self.world_model(img_t, act_vec)
            pred_img_np = (
                pred_img_next.squeeze(0).permute(1, 2, 0).cpu().numpy()
            )
            pred_img_np = np.clip(pred_img_np * 255.0, 0, 255).astype(np.uint8)

            # resize to match original frame size
            pred_img_resized = cv2.resize(
                pred_img_np, (W, H), interpolation=cv2.INTER_AREA
            )

            A_pred, _ = self.att_module.compute(
                pred_img_resized, prev_frame=frame
            )
            A_pred_tensor = torch.tensor(
                A_pred, dtype=torch.float32, device=self.device
            )

            att_change = torch.mean(
                (A_pred_tensor - A_t_tensor) ** 2
            ).item()

            # Combined curiosity
            score = (
                self.local_weight * local_change
                + self.novelty_weight * novelty
                + self.attention_weight * att_change
            )

            # Penalize repeating the same action type
            if action.get("type") == self.last_action_type:
                score *= 0.6

            # Normalize by usage frequency to encourage diversity
            freq = self.action_counts.get(action.get("type"), 1)
            score /= (freq ** 0.5)

            if score > best_score:
                best_score = score
                best_action = action

        # ----------------------------------------------------
        # 3) Update diversity stats and epsilon-greedy
        # ----------------------------------------------------
        if best_action is None:
            best_action = {"type": "NOOP"}

        best_type = best_action.get("type")
        self.action_counts[best_type] = self.action_counts.get(best_type, 0) + 1
        self.last_action_type = best_type

        # Epsilon-greedy random exploration
        if np.random.rand() < self.epsilon:
            from agent.random_agent import random_action
            random_act = random_action(width=W, height=H)
            return random_act, 0.0

        return best_action, best_score

    # --------------------------------------------------------
    # Step 15: Semantic / Text-guided helpers
    # --------------------------------------------------------
    def get_visible_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns structured list of all visible text elements
        based on ScreenInterpreter + TextMemory update.
        """
        elems = self._update_text_memory(frame)
        return elems

    def semantic_find_and_click(self, frame: np.ndarray, text: str) -> bool:
        """
        Use attached TextActionAgent (if present) to:
          - re-run OCR on frame
          - find closest text match
          - emit the appropriate actions (inside TextActionAgent)

        Returns True if something was clicked, False otherwise.
        """
        if self.text_action_agent is None:
            raise RuntimeError(
                "No TextActionAgent attached. "
                "Call agent.attach_text_action_agent(...) first."
            )
        return self.text_action_agent.find_and_click(frame, text)

    def semantic_type_into(
        self,
        frame: np.ndarray,
        target_text: str,
        content: str,
    ) -> bool:
        """
        Use attached TextActionAgent to:
          - click UI element matching target_text
          - type the provided content
        """
        if self.text_action_agent is None:
            raise RuntimeError(
                "No TextActionAgent attached. "
                "Call agent.attach_text_action_agent(...) first."
            )
        return self.text_action_agent.type_into(frame, target_text, content)
