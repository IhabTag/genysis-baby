"""
Test script for EpisodicRetrieval module.

Verifies:
1. Basic retrieval
2. Novelty computation
3. Integration with episodic buffer
4. Retrieval statistics
"""

import os
import sys
import torch

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from memory.episodic_buffer import EpisodicBuffer
from memory.episodic_retrieval import EpisodicRetrieval, EpisodicRetrievalSimple


def test_basic_retrieval():
    """Test basic k-NN retrieval."""
    print("\n=== Test 1: Basic Retrieval ===")
    
    # Create buffer and populate
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(50):
        p = torch.randn(64)
        buffer.add(p)
    
    print(f"Buffer size: {buffer.size}")
    
    # Create retrieval system
    retrieval = EpisodicRetrieval(
        buffer,
        retrieval_k=5,
        context_dim=128,
        use_learned_aggregation=False  # Simple averaging for test
    )
    
    # Retrieve for current state
    p_t = torch.randn(1, 64)
    r_t = retrieval.retrieve(p_t)
    
    print(f"Query shape: {p_t.shape}")
    print(f"Retrieved context shape: {r_t.shape}")
    
    assert r_t is not None, "Should retrieve context"
    assert r_t.shape == (1, 128), f"Expected shape (1, 128), got {r_t.shape}"
    
    print("✓ Basic retrieval successful")


def test_retrieval_with_learned_aggregation():
    """Test retrieval with attention-based aggregation."""
    print("\n=== Test 2: Learned Aggregation ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(50):
        p = torch.randn(64)
        buffer.add(p)
    
    # Create retrieval with learned aggregation
    retrieval = EpisodicRetrieval(
        buffer,
        retrieval_k=5,
        context_dim=128,
        use_learned_aggregation=True
    )
    
    p_t = torch.randn(1, 64)
    r_t = retrieval.retrieve(p_t)
    
    print(f"Retrieved context shape: {r_t.shape}")
    assert r_t.shape == (1, 128), "Learned aggregation failed"
    
    print("✓ Learned aggregation successful")


def test_empty_buffer():
    """Test retrieval from empty buffer."""
    print("\n=== Test 3: Empty Buffer ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
    
    p_t = torch.randn(1, 64)
    r_t = retrieval.retrieve(p_t)
    
    assert r_t is None, "Should return None for empty buffer"
    print("✓ Empty buffer handling correct")


def test_insufficient_memories():
    """Test retrieval when buffer has fewer than k memories."""
    print("\n=== Test 4: Insufficient Memories ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    # Add only 3 memories
    for i in range(3):
        buffer.add(torch.randn(64))
    
    # Request 5 neighbors
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
    
    p_t = torch.randn(1, 64)
    r_t = retrieval.retrieve(p_t)
    
    # Should return None (not enough memories)
    assert r_t is None, "Should return None when k > buffer size"
    print("✓ Insufficient memories handled correctly")


def test_novelty_computation():
    """Test novelty score computation."""
    print("\n=== Test 5: Novelty Computation ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    # Add some memories
    for i in range(20):
        p = torch.randn(64)
        buffer.add(p)
    
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
    
    # Test with similar state
    p_similar = buffer.get_memory_tensor()[0].unsqueeze(0)  # Exact match
    novelty_low = retrieval.compute_novelty(p_similar)
    
    # Test with very different state
    p_novel = torch.randn(1, 64) * 10  # Very different
    novelty_high = retrieval.compute_novelty(p_novel)
    
    print(f"Novelty (similar): {novelty_low:.4f}")
    print(f"Novelty (novel): {novelty_high:.4f}")
    
    assert novelty_low < novelty_high, "Novel state should have higher novelty"
    assert novelty_low < 0.1, "Similar state should have low novelty"
    
    print("✓ Novelty computation correct")


def test_retrieval_with_distances():
    """Test retrieval with distance information."""
    print("\n=== Test 6: Retrieval with Distances ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(30):
        buffer.add(torch.randn(64))
    
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
    
    p_t = torch.randn(1, 64)
    r_t, distances = retrieval.retrieve(p_t, return_distances=True)
    
    print(f"Retrieved context: {r_t.shape}")
    print(f"Distances: {distances.shape}")
    print(f"Distance values: {distances[0].tolist()}")
    
    assert distances.shape == (1, 5), "Should return 5 distances"
    
    # Distances should be sorted (nearest first)
    dists = distances[0]
    for i in range(len(dists) - 1):
        assert dists[i] <= dists[i+1], "Distances should be sorted"
    
    print("✓ Distance retrieval correct")


def test_retrieval_with_metadata():
    """Test retrieval with metadata."""
    print("\n=== Test 7: Retrieval with Metadata ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(20):
        buffer.add(torch.randn(64))
    
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
    
    p_t = torch.randn(1, 64)
    results = retrieval.retrieve_with_metadata(p_t)
    
    print(f"Retrieved {len(results)} memories")
    
    assert len(results) == 5, "Should retrieve 5 memories"
    
    for i, result in enumerate(results):
        print(f"  Memory {i}: distance={result['distance']:.4f}, index={result['index']}")
        assert 'projection' in result
        assert 'distance' in result
        assert 'index' in result
    
    print("✓ Metadata retrieval correct")


def test_retrieval_stats():
    """Test retrieval statistics."""
    print("\n=== Test 8: Retrieval Statistics ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(40):
        buffer.add(torch.randn(64))
    
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5)
    
    # Trigger initialization
    p_t = torch.randn(1, 64)
    retrieval.retrieve(p_t)
    
    stats = retrieval.get_retrieval_stats()
    
    print("Retrieval statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    assert stats['num_memories'] == 40
    assert stats['retrieval_k'] == 5
    assert stats['memory_diversity'] > 0
    
    print("✓ Statistics correct")


def test_simple_retrieval():
    """Test simplified retrieval without learned aggregation."""
    print("\n=== Test 9: Simple Retrieval ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(30):
        buffer.add(torch.randn(64))
    
    retrieval = EpisodicRetrievalSimple(buffer, retrieval_k=5)
    
    p_t = torch.randn(1, 64)
    r_t = retrieval.retrieve(p_t)
    
    print(f"Simple retrieval output: {r_t.shape}")
    
    assert r_t.shape == (1, 64), "Simple retrieval should return averaged projections"
    
    # Test novelty
    novelty = retrieval.compute_novelty(p_t)
    print(f"Novelty: {novelty:.4f}")
    
    assert novelty > 0, "Novelty should be positive"
    
    print("✓ Simple retrieval successful")


def test_batch_retrieval():
    """Test retrieval with batch of queries."""
    print("\n=== Test 10: Batch Retrieval ===")
    
    buffer = EpisodicBuffer(max_size=100, proj_dim=64)
    
    for i in range(50):
        buffer.add(torch.randn(64))
    
    retrieval = EpisodicRetrieval(buffer, retrieval_k=5, use_learned_aggregation=False)
    
    # Batch of queries
    batch_size = 4
    p_batch = torch.randn(batch_size, 64)
    
    r_batch = retrieval.retrieve(p_batch)
    
    print(f"Batch query shape: {p_batch.shape}")
    print(f"Batch retrieval shape: {r_batch.shape}")
    
    assert r_batch.shape == (batch_size, 128), "Batch retrieval failed"
    
    print("✓ Batch retrieval successful")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing EpisodicRetrieval Module")
    print("=" * 60)
    
    try:
        test_basic_retrieval()
        test_retrieval_with_learned_aggregation()
        test_empty_buffer()
        test_insufficient_memories()
        test_novelty_computation()
        test_retrieval_with_distances()
        test_retrieval_with_metadata()
        test_retrieval_stats()
        test_simple_retrieval()
        test_batch_retrieval()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
