"""
Option Discovery: Segment logged episodes into reusable behavioral chunks.

Analyzes episode logs to find natural behavioral patterns (options/skills) by:
1. Detecting change points (layout, attention, OCR, action type)
2. Segmenting episodes into coherent chunks
3. Classifying option types
4. Clustering similar options

Example discovered options:
- scroll_down: Sustained downward scrolling
- move_to_corner: Mouse movement to screen corner
- click_text: Move + click on OCR-detected text
- explore_region: Random movements in local area
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class OptionDiscovery:
    """
    Discovers reusable behavioral chunks from logged episodes.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        min_option_length: int = 3,
        max_option_length: int = 30,
        change_threshold: float = 0.02,
        layout_grid: int = 8
    ):
        """
        Initialize option discovery.
        
        Args:
            log_dir: Directory containing episode logs
            min_option_length: Minimum steps for an option
            max_option_length: Maximum steps for an option
            change_threshold: Threshold for detecting significant changes
            layout_grid: Grid size for layout signature
        """
        self.log_dir = log_dir
        self.min_len = min_option_length
        self.max_len = max_option_length
        self.threshold = change_threshold
        self.layout_grid = layout_grid
    
    def _layout_signature(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute coarse layout fingerprint.
        
        Args:
            frame: RGB frame (H, W, 3)
        
        Returns:
            Flattened grayscale grid (grid*grid,)
        """
        small = cv2.resize(frame, (self.layout_grid, self.layout_grid), 
                          interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        return gray.flatten()
    
    def _layout_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute layout change between two frames.
        
        Args:
            frame1, frame2: RGB frames
        
        Returns:
            Layout difference score
        """
        sig1 = self._layout_signature(frame1)
        sig2 = self._layout_signature(frame2)
        return float(np.mean((sig1 - sig2) ** 2))
    
    def _pixel_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute pixel-level difference.
        
        Args:
            frame1, frame2: RGB frames
        
        Returns:
            Normalized pixel difference
        """
        # Resize to standard size for comparison
        h, w = 128, 128
        f1 = cv2.resize(frame1, (w, h))
        f2 = cv2.resize(frame2, (w, h))
        
        # Convert to float and normalize
        f1 = f1.astype(np.float32) / 255.0
        f2 = f2.astype(np.float32) / 255.0
        
        diff = np.mean((f1 - f2) ** 2)
        return float(diff)
    
    def find_change_points(self, episode_dir: str) -> List[int]:
        """
        Find timesteps where significant changes occur.
        
        Change indicators:
        - Layout signature change
        - Pixel-level change
        - Action type change (from metadata)
        
        Args:
            episode_dir: Path to episode directory
        
        Returns:
            List of change point indices
        """
        change_points = [0]  # Always start at 0
        
        # Load episode metadata
        meta_path = os.path.join(episode_dir, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"Warning: No metadata found in {episode_dir}")
            return change_points
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        steps = metadata.get('steps', [])
        if len(steps) < 2:
            return change_points
        
        # Analyze each step transition
        prev_action_type = None
        prev_frame = None
        
        for i, step in enumerate(steps):
            # Load frame if available
            frame_path = os.path.join(episode_dir, f"frame_{i:06d}.png")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = None
            
            # Check for changes
            is_change = False
            
            # 1. Action type change
            action = step.get('action', {})
            action_type = action.get('type', 'NOOP')
            
            if prev_action_type is not None and action_type != prev_action_type:
                # Significant action type change
                if self._is_significant_action_change(prev_action_type, action_type):
                    is_change = True
            
            # 2. Visual change (if frames available)
            if frame is not None and prev_frame is not None:
                pixel_change = self._pixel_diff(prev_frame, frame)
                layout_change = self._layout_change(prev_frame, frame)
                
                total_change = pixel_change + layout_change
                
                if total_change > self.threshold:
                    is_change = True
            
            if is_change and i > 0:
                change_points.append(i)
            
            prev_action_type = action_type
            prev_frame = frame
        
        # Always end at last step
        if len(steps) not in change_points:
            change_points.append(len(steps))
        
        return change_points
    
    def _is_significant_action_change(self, type1: str, type2: str) -> bool:
        """
        Check if action type change is significant.
        
        Some transitions are natural (e.g., MOVE_MOUSE -> LEFT_CLICK),
        others indicate a new behavior (e.g., SCROLL -> TYPE_TEXT).
        """
        # Natural sequences (not significant changes)
        natural_sequences = [
            ('MOVE_MOUSE', 'LEFT_CLICK'),
            ('MOVE_MOUSE', 'RIGHT_CLICK'),
            ('LEFT_CLICK', 'LEFT_CLICK'),  # Double click
        ]
        
        if (type1, type2) in natural_sequences:
            return False
        
        # Different action categories = significant
        action_categories = {
            'MOVE_MOUSE': 'movement',
            'LEFT_CLICK': 'click',
            'RIGHT_CLICK': 'click',
            'SCROLL': 'scroll',
            'TYPE_TEXT': 'typing',
            'KEY': 'typing',
            'NOOP': 'idle'
        }
        
        cat1 = action_categories.get(type1, 'other')
        cat2 = action_categories.get(type2, 'other')
        
        return cat1 != cat2
    
    def segment_episode(self, episode_dir: str) -> List[Dict]:
        """
        Segment episode into candidate options.
        
        Args:
            episode_dir: Path to episode directory
        
        Returns:
            List of option segments with metadata
        """
        change_points = self.find_change_points(episode_dir)
        
        options = []
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            
            length = end - start
            
            # Filter by length
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
        
        Args:
            episode_dir: Episode directory
            start, end: Option boundaries
        
        Returns:
            Option type string
        """
        # Load metadata
        meta_path = os.path.join(episode_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return "unknown"
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        steps = metadata.get('steps', [])
        
        # Count action types in this segment
        action_counts = defaultdict(int)
        for i in range(start, min(end, len(steps))):
            step = steps[i]
            action = step.get('action', {})
            action_type = action.get('type', 'NOOP')
            action_counts[action_type] += 1
        
        if not action_counts:
            return "unknown"
        
        # Dominant action type
        dominant_type = max(action_counts, key=action_counts.get)
        
        # Map to option names
        option_mapping = {
            'SCROLL': self._classify_scroll_option(episode_dir, start, end, steps),
            'MOVE_MOUSE': self._classify_movement_option(episode_dir, start, end, steps),
            'LEFT_CLICK': 'click_sequence',
            'TYPE_TEXT': 'typing',
            'KEY': 'keyboard_input',
            'NOOP': 'idle'
        }
        
        return option_mapping.get(dominant_type, f"{dominant_type.lower()}_option")
    
    def _classify_scroll_option(self, episode_dir: str, start: int, end: int, steps: List) -> str:
        """Classify scroll direction."""
        total_amount = 0
        for i in range(start, min(end, len(steps))):
            action = steps[i].get('action', {})
            if action.get('type') == 'SCROLL':
                amount = action.get('amount', 0)
                total_amount += amount
        
        if total_amount < -100:
            return "scroll_up"
        elif total_amount > 100:
            return "scroll_down"
        else:
            return "scroll_small"
    
    def _classify_movement_option(self, episode_dir: str, start: int, end: int, steps: List) -> str:
        """Classify mouse movement pattern."""
        positions = []
        for i in range(start, min(end, len(steps))):
            action = steps[i].get('action', {})
            if action.get('type') == 'MOVE_MOUSE':
                x = action.get('x', 0)
                y = action.get('y', 0)
                positions.append((x, y))
        
        if len(positions) < 2:
            return "move_mouse"
        
        # Analyze movement pattern
        start_pos = positions[0]
        end_pos = positions[-1]
        
        # Distance traveled
        dist = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Movement to corner/edge?
        screen_w, screen_h = 1024, 768  # Assume default
        
        if end_pos[1] < screen_h * 0.15:
            return "move_to_top"
        elif end_pos[0] < screen_w * 0.15:
            return "move_to_left"
        elif dist > 500:
            return "move_large"
        elif dist < 100:
            return "move_local"
        else:
            return "move_mouse"
    
    def discover_all_options(self, max_episodes: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Discover options from all episodes in log directory.
        
        Args:
            max_episodes: Maximum number of episodes to process (None = all)
        
        Returns:
            Dictionary mapping option_type → list of instances
        """
        if not os.path.exists(self.log_dir):
            print(f"Log directory not found: {self.log_dir}")
            return {}
        
        # Find all episode directories
        episode_dirs = []
        for item in os.listdir(self.log_dir):
            item_path = os.path.join(self.log_dir, item)
            if os.path.isdir(item_path):
                # Check if it has metadata
                if os.path.exists(os.path.join(item_path, "metadata.json")):
                    episode_dirs.append(item_path)
        
        if max_episodes:
            episode_dirs = episode_dirs[:max_episodes]
        
        print(f"Found {len(episode_dirs)} episodes to analyze")
        
        # Discover options from each episode
        all_options = defaultdict(list)
        
        for ep_dir in episode_dirs:
            try:
                options = self.segment_episode(ep_dir)
                for opt in options:
                    opt_type = opt['type']
                    all_options[opt_type].append(opt)
            except Exception as e:
                print(f"Error processing {ep_dir}: {e}")
                continue
        
        # Print summary
        print(f"\nDiscovered {len(all_options)} option types:")
        for opt_type, instances in sorted(all_options.items(), key=lambda x: -len(x[1])):
            print(f"  {opt_type}: {len(instances)} instances")
        
        return dict(all_options)
    
    def save_discovered_options(self, options: Dict[str, List[Dict]], output_path: str):
        """
        Save discovered options to JSON file.
        
        Args:
            options: Dictionary of discovered options
            output_path: Path to save JSON
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(options, f, indent=2)
        
        print(f"Saved discovered options to {output_path}")
    
    @staticmethod
    def load_discovered_options(input_path: str) -> Dict[str, List[Dict]]:
        """
        Load discovered options from JSON file.
        
        Args:
            input_path: Path to JSON file
        
        Returns:
            Dictionary of options
        """
        with open(input_path, 'r') as f:
            options = json.load(f)
        
        return options


def main():
    """
    Example usage: discover options from logged episodes.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover behavioral options from episode logs")
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory containing episode logs')
    parser.add_argument('--output', type=str, default='datasets/discovered_options.json', help='Output JSON file')
    parser.add_argument('--max-episodes', type=int, default=None, help='Maximum episodes to process')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum option length')
    parser.add_argument('--max-length', type=int, default=30, help='Maximum option length')
    parser.add_argument('--threshold', type=float, default=0.02, help='Change detection threshold')
    
    args = parser.parse_args()
    
    # Create discovery instance
    discovery = OptionDiscovery(
        log_dir=args.log_dir,
        min_option_length=args.min_length,
        max_option_length=args.max_length,
        change_threshold=args.threshold
    )
    
    # Discover options
    print(f"Discovering options from {args.log_dir}...")
    options = discovery.discover_all_options(max_episodes=args.max_episodes)
    
    # Save results
    discovery.save_discovered_options(options, args.output)
    
    print(f"\nDone! Found {sum(len(v) for v in options.values())} total option instances")


if __name__ == "__main__":
    main()
