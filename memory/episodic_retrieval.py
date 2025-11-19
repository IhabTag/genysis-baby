import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class EpisodicRetrieval(nn.Module):
    """
    Retrieves relevant past experiences based on current state.
    
    Uses k-nearest neighbor search in the episodic buffer to find
    similar past states, then aggregates them into a retrieved context vector.
    
    This enables the agent to reference past experiences:
    "I've seen something like this before; last time I scrolled and a new thing appeared."
    
    Example:
        >>> buffer = EpisodicBuffer(max_size=1000)
        >>> retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
        >>> p_t = torch.randn(1, 64)  # Current projection
        >>> r_t = retrieval.retrieve(p_t)  # Retrieved context
        >>> print(r_t.shape)  # (1, 128)
    """
    
    def __init__(
        self,
        episodic_buffer,
        retrieval_k: int = 5,
        context_dim: int = 128,
        use_learned_aggregation: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize episodic retrieval system.
        
        Args:
            episodic_buffer: EpisodicBuffer instance to retrieve from
            retrieval_k: Number of nearest neighbors to retrieve
            context_dim: Dimension of output context vector
            use_learned_aggregation: If True, use neural network to aggregate.
                                    If False, use simple averaging.
            temperature: Temperature for attention-based aggregation
        """
        super().__init__()
        
        self.buffer = episodic_buffer
        self.k = retrieval_k
        self.context_dim = context_dim
        self.use_learned_aggregation = use_learned_aggregation
        self.temperature = temperature
        
        # Get projection dimension from buffer
        # Assume buffer stores projections of shape (proj_dim,)
        self.proj_dim = None  # Will be set on first retrieval
        
        # Initialize as None, will be created lazily when we know proj_dim
        self.aggregator = None
        self.attention = None
        
    def _init_aggregator(self, proj_dim: int):
        """
        Initialize aggregation network once we know projection dimension.
        
        Args:
            proj_dim: Dimension of projection vectors
        """
        if self.aggregator is not None:
            return  # Already initialized
        
        self.proj_dim = proj_dim
        
        if self.use_learned_aggregation:
            # Attention-based aggregation
            # Query: current state, Keys/Values: retrieved memories
            self.attention = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=4,
                batch_first=True
            )
            
            # Final projection to context dimension
            self.aggregator = nn.Sequential(
                nn.Linear(proj_dim, self.context_dim),
                nn.ReLU(),
                nn.Linear(self.context_dim, self.context_dim)
            )
    
    def retrieve(
        self, 
        p_t: torch.Tensor,
        return_distances: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Retrieve k nearest neighbors from episodic memory and aggregate.
        
        Args:
            p_t: (batch, proj_dim) - current projection in contrastive space
            return_distances: If True, also return distances to retrieved memories
        
        Returns:
            r_t: (batch, context_dim) - retrieved context vector, or None if buffer empty
            distances: (batch, k) - distances to retrieved memories (if return_distances=True)
        """
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None or memory_tensor.size(0) < self.k:
            # Not enough memories yet
            if return_distances:
                return None, None
            return None
        
        # Initialize aggregator if needed
        proj_dim = p_t.size(-1)
        self._init_aggregator(proj_dim)
        
        # Compute distances to all memories
        # memory_tensor: (N, proj_dim)
        # p_t: (batch, proj_dim)
        dists = torch.cdist(p_t, memory_tensor)  # (batch, N)
        
        # Get k nearest neighbors
        topk_vals, topk_idx = torch.topk(dists, k=min(self.k, memory_tensor.size(0)), 
                                         largest=False, dim=1)
        
        # Retrieve corresponding states
        batch_size = p_t.size(0)
        retrieved = []
        for b in range(batch_size):
            retrieved_batch = memory_tensor[topk_idx[b]]  # (k, proj_dim)
            retrieved.append(retrieved_batch)
        
        retrieved = torch.stack(retrieved, dim=0)  # (batch, k, proj_dim)
        
        # Aggregate into context vector
        if self.use_learned_aggregation:
            # Attention-based aggregation
            # Use current state as query, retrieved memories as keys/values
            query = p_t.unsqueeze(1)  # (batch, 1, proj_dim)
            
            # Attend to retrieved memories
            attended, _ = self.attention(
                query, retrieved, retrieved
            )  # (batch, 1, proj_dim)
            
            attended = attended.squeeze(1)  # (batch, proj_dim)
            
            # Project to context dimension
            r_t = self.aggregator(attended)  # (batch, context_dim)
        else:
            # Simple averaging
            avg_retrieved = retrieved.mean(dim=1)  # (batch, proj_dim)
            
            # If context_dim != proj_dim, we need a projection
            if self.context_dim != proj_dim:
                if self.aggregator is None:
                    self.aggregator = nn.Linear(proj_dim, self.context_dim).to(p_t.device)
                r_t = self.aggregator(avg_retrieved)
            else:
                r_t = avg_retrieved
        
        if return_distances:
            return r_t, topk_vals
        return r_t
    
    def retrieve_with_metadata(
        self, 
        p_t: torch.Tensor,
        max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories with full metadata for debugging/visualization.
        
        Args:
            p_t: (1, proj_dim) - current projection
            max_results: Maximum number of results to return (default: self.k)
        
        Returns:
            List of dicts with keys:
                - 'projection': retrieved projection vector
                - 'distance': distance to current state
                - 'index': index in episodic buffer
                - 'metadata': any metadata stored with memory
        """
        if max_results is None:
            max_results = self.k
        
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None or memory_tensor.size(0) == 0:
            return []
        
        # Compute distances
        dists = torch.cdist(p_t, memory_tensor)  # (1, N)
        dists = dists.squeeze(0)  # (N,)
        
        # Get top k
        k = min(max_results, memory_tensor.size(0))
        topk_vals, topk_idx = torch.topk(dists, k=k, largest=False)
        
        # Build result list
        results = []
        for i in range(k):
            idx = topk_idx[i].item()
            dist = topk_vals[i].item()
            
            result = {
                'projection': memory_tensor[idx].cpu().numpy(),
                'distance': dist,
                'index': idx,
                'metadata': self.buffer.get_metadata(idx) if hasattr(self.buffer, 'get_metadata') else None
            }
            results.append(result)
        
        return results
    
    def compute_novelty(self, p_t: torch.Tensor) -> float:
        """
        Compute novelty score as distance to nearest memory.
        
        This is similar to the novelty curiosity in CuriosityModule,
        but can be used independently.
        
        Args:
            p_t: (1, proj_dim) - current projection
        
        Returns:
            novelty: scalar novelty score (higher = more novel)
        """
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None or memory_tensor.size(0) == 0:
            return 1.0  # Maximum novelty if no memories
        
        # Compute distances
        dists = torch.cdist(p_t, memory_tensor)  # (1, N)
        
        # Minimum distance = novelty
        min_dist = torch.min(dists).item()
        
        return min_dist
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the episodic buffer and retrieval.
        
        Returns:
            Dictionary with statistics
        """
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None:
            return {
                'num_memories': 0,
                'memory_capacity': self.buffer.max_size if hasattr(self.buffer, 'max_size') else None,
                'retrieval_k': self.k
            }
        
        num_memories = memory_tensor.size(0)
        
        # Compute diversity (average pairwise distance)
        if num_memories > 1:
            sample_size = min(100, num_memories)
            sample_idx = torch.randperm(num_memories)[:sample_size]
            sample = memory_tensor[sample_idx]
            
            pairwise_dists = torch.cdist(sample, sample)
            # Exclude diagonal
            mask = ~torch.eye(sample_size, dtype=bool, device=pairwise_dists.device)
            diversity = pairwise_dists[mask].mean().item()
        else:
            diversity = 0.0
        
        return {
            'num_memories': num_memories,
            'memory_capacity': self.buffer.max_size if hasattr(self.buffer, 'max_size') else None,
            'retrieval_k': self.k,
            'memory_diversity': diversity,
            'proj_dim': self.proj_dim
        }


class EpisodicRetrievalSimple:
    """
    Simplified episodic retrieval without learned aggregation.
    
    This is a lightweight alternative that doesn't require training.
    Useful for initial testing or when you want pure k-NN retrieval.
    """
    
    def __init__(
        self,
        episodic_buffer,
        retrieval_k: int = 5
    ):
        """
        Initialize simple episodic retrieval.
        
        Args:
            episodic_buffer: EpisodicBuffer instance
            retrieval_k: Number of nearest neighbors
        """
        self.buffer = episodic_buffer
        self.k = retrieval_k
    
    def retrieve(self, p_t: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Retrieve k nearest neighbors and return their average.
        
        Args:
            p_t: (batch, proj_dim) - current projection
        
        Returns:
            r_t: (batch, proj_dim) - averaged retrieved projections
        """
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None or memory_tensor.size(0) < self.k:
            return None
        
        # Compute distances
        dists = torch.cdist(p_t, memory_tensor)  # (batch, N)
        
        # Get k nearest
        _, topk_idx = torch.topk(dists, k=self.k, largest=False, dim=1)
        
        # Retrieve and average
        batch_size = p_t.size(0)
        retrieved_avg = []
        
        for b in range(batch_size):
            retrieved_batch = memory_tensor[topk_idx[b]]  # (k, proj_dim)
            avg = retrieved_batch.mean(dim=0)  # (proj_dim,)
            retrieved_avg.append(avg)
        
        r_t = torch.stack(retrieved_avg, dim=0)  # (batch, proj_dim)
        
        return r_t
    
    def compute_novelty(self, p_t: torch.Tensor) -> float:
        """Compute novelty as distance to nearest memory."""
        memory_tensor = self.buffer.get_memory_tensor()
        
        if memory_tensor is None or memory_tensor.size(0) == 0:
            return 1.0
        
        dists = torch.cdist(p_t, memory_tensor)
        return torch.min(dists).item()
