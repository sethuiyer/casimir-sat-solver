from casimir_sat import SATInstance, CasimirSolver
import numpy as np

def test_spectral_partitioning():
    print("================================================================================")
    print("TEST: Spectral Partitioning on Disconnected Clusters")
    print("================================================================================")
    
    # Create two disconnected clusters of variables
    # Cluster A: vars 1, 2, 3 (indices 0, 1, 2)
    # Cluster B: vars 4, 5, 6 (indices 3, 4, 5)
    
    clauses = [
        # Cluster A constraints
        [1, 2], [-1, 3], [2, -3],
        # Cluster B constraints
        [4, 5], [-4, 6], [5, -6]
    ]
    
    instance = SATInstance(num_vars=6, clauses=clauses)
    solver = CasimirSolver(instance, correlation_length=1.0) # Small correlation length to emphasize locality
    
    print(f"Instance: {instance.num_vars} variables, {len(instance.clauses)} clauses")
    print("Structure: Two disconnected clusters {1,2,3} and {4,5,6}")
    
    # Run partitioning
    cluster_a_idx, cluster_b_idx, fiedler_val = solver.spectral_partitioning()
    
    # Convert to 1-based variable names for display
    cluster_a_vars = sorted([i + 1 for i in cluster_a_idx])
    cluster_b_vars = sorted([i + 1 for i in cluster_b_idx])
    
    print(f"\nFiedler Value (should be close to 0 for disconnected): {fiedler_val:.6f}")
    
    # Debug: Print the Fiedler vector
    # We need to access the internal vector, but the method returns indices.
    # Let's modify the solver to return the vector or just inspect it here if we could.
    # Since we can't easily change the return signature without breaking other things, 
    # let's just copy the logic here for debugging or modify the solver to be more verbose if needed.
    # Actually, let's just modify the solver to print debug info if a flag is set, or just trust me to fix it.
    
    # Wait, I can't access the vector from here.
    # I will modify the test to manually compute it using the solver's helper methods.
    
    adj = solver.build_adjacency_matrix()
    dists = solver.compute_graph_distances()
    grads = solver.compute_gradients()
    grad_mags = np.abs(grads)
    weights = np.exp(-dists / solver.correlation_length)
    grad_term = 1.0 + grad_mags.reshape(-1, 1) + grad_mags.reshape(1, -1)
    weights *= grad_term
    np.fill_diagonal(weights, 0.0)
    degrees = np.sum(weights, axis=1)
    L = np.diag(degrees) - weights
    eigenvals, eigenvecs = np.linalg.eigh(L)
    idx = np.argsort(eigenvals)
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    fiedler_vector = eigenvecs[:, 1]
    
    print(f"Eigenvalues: {eigenvals[:4]}")
    print(f"Fiedler Vector: {fiedler_vector}")
    
    print(f"Partition A: {cluster_a_vars}")
    print(f"Partition B: {cluster_b_vars}")
    
    # Verification
    # One partition should be {1,2,3} and the other {4,5,6}
    set_a = set(cluster_a_vars)
    set_b = set(cluster_b_vars)
    
    expected_1 = {1, 2, 3}
    expected_2 = {4, 5, 6}
    
    if (set_a == expected_1 and set_b == expected_2) or \
       (set_a == expected_2 and set_b == expected_1):
        print("\n*** SUCCESS: Spectral partitioning correctly identified the clusters! ***")
    else:
        print("\n*** FAILURE: Partitioning failed to identify clusters. ***")

if __name__ == "__main__":
    test_spectral_partitioning()
