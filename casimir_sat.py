import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass
class SATInstance:
    """Represents a SAT instance with variables and clauses."""
    num_vars: int
    clauses: List[List[int]]

class CasimirSolver:
    """Main solver implementing quantum-inspired dynamics."""

    def __init__(self, instance: SATInstance,
                 temperature: float = 2.0,
                 learning_rate: float = 0.5,
                 correlation_length: float = 3.0):
        self.instance = instance
        self.num_vars = instance.num_vars
        self.clauses = instance.clauses
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.correlation_length = correlation_length
        
        # Initialize variables at maximum entropy (x_i = 0.5)
        self.x = np.full(self.num_vars, 0.5, dtype=np.float64)
        
        # Annealing parameters
        self.step_count = 0
        self.beta = 1.0

    def fractional_satisfaction(self, clause: List[int]) -> float:
        """Compute s_c(x) = 1 - ∏(1 - x_ℓ)"""
        unsatisfied_product = 1.0
        for lit in clause:
            var_idx = abs(lit) - 1
            if lit > 0:
                unsatisfied_product *= (1.0 - self.x[var_idx])
            else:
                unsatisfied_product *= self.x[var_idx]
        return 1.0 - unsatisfied_product

    def total_energy(self) -> float:
        """Compute E = Σ(1 - s_c)²"""
        energy = 0.0
        for clause in self.clauses:
            s_c = self.fractional_satisfaction(clause)
            energy += (1.0 - s_c) ** 2
        return energy

    def compute_gradients(self) -> np.ndarray:
        """Compute gradients of energy with respect to variables."""
        grads = np.zeros(self.num_vars, dtype=np.float64)
        
        for clause in self.clauses:
            # Recompute parts needed for gradient
            # s_c = 1 - P, where P = product of (1-x_l) or x_l
            # E_c = (1 - s_c)^2 = P^2
            # dE_c/dx_i = 2 * P * dP/dx_i
            
            unsatisfied_product = 1.0
            for lit in clause:
                var_idx = abs(lit) - 1
                if lit > 0:
                    unsatisfied_product *= (1.0 - self.x[var_idx])
                else:
                    unsatisfied_product *= self.x[var_idx]
            
            # If unsatisfied_product is 0, gradient is 0 (satisfied)
            if unsatisfied_product == 0:
                continue
                
            for lit in clause:
                var_idx = abs(lit) - 1
                # dP/dx_i:
                # if lit > 0 (x_i), term is (1-x_i), deriv is -1 * (P / (1-x_i))
                # if lit < 0 (-x_i), term is x_i, deriv is 1 * (P / x_i)
                
                if lit > 0:
                    # Avoid division by zero if x is exactly 1 (though product would be 0 usually)
                    if 1.0 - self.x[var_idx] > 1e-10:
                        deriv = -1.0 * (unsatisfied_product / (1.0 - self.x[var_idx]))
                        grads[var_idx] += 2.0 * unsatisfied_product * deriv
                else:
                    if self.x[var_idx] > 1e-10:
                        deriv = 1.0 * (unsatisfied_product / self.x[var_idx])
                        grads[var_idx] += 2.0 * unsatisfied_product * deriv
                        
        return grads

    def langevin_step(self):
        """Perform Langevin dynamics: dx/dt = -η∇E + √(2T)ξ"""
        grads = self.compute_gradients()

        for i in range(self.num_vars):
            # Thermal noise: √(2T)ξ where ξ ~ N(0,1)
            noise = np.sqrt(2.0 * self.temperature) * np.random.randn()

            # Total drift: gradient descent + noise
            # Note: The paper mentions 0.1 * noise in the code snippet, keeping it consistent
            drift = -self.learning_rate * grads[i] + 0.1 * noise

            # Sigmoid projection to keep in [0,1]
            # The update rule in paper: x(t+1) = sigmoid(beta * drift) 
            # But wait, the paper says: xi(t+Δt) = σ(β(t) · x̃i) where x̃i is intermediate update?
            # Let's look at the snippet: 
            # drift = -eta * grads + noise
            # x = sigmoid(beta(t) * drift)
            # This implies x is NOT updated incrementally from previous x, but set based on drift?
            # That seems odd for Langevin dynamics which is usually x += ...
            # However, let's follow the paper's snippet exactly for now.
            
            # Actually, standard Langevin is dx = ...
            # But the snippet in "The Algorithmic Pipeline" says:
            # drift = -eta * grads + noise
            # x = sigmoid(beta(t) * drift)
            # This suggests 'drift' here acts more like a 'field' or 'potential' value rather than a delta.
            # Let's stick to the snippet logic.
            
            # Wait, if x is set purely by drift, where does the previous state go?
            # In standard continuous Hopfield/Boltzmann machines, the state variable u evolves, and x = sigmoid(u).
            # So 'drift' might be the update to the internal state u.
            # Let's assume the snippet meant: u_new = u_old + drift? 
            # OR, maybe 'drift' IS the internal state u?
            # "drift = -eta * grads + noise" -> This looks like a velocity or force.
            
            # Let's look at the Crystal implementation snippet:
            # drift = -@learning_rate * grads[i] + noise
            # var.value = 1.0 / (1.0 + exp(-@beta * drift))
            # This also sets value directly from drift. 
            # This implies 'drift' is accumulating? No, it's calculated fresh.
            # If it's calculated fresh, then x depends ONLY on the current gradient? That can't be right for dynamics.
            # Unless... the gradient depends on x. So it's a map x_t -> x_{t+1}.
            # But if x is satisfied (grad=0), drift is just noise. 
            # Then x = sigmoid(beta * noise) -> x = 0.5. That destroys the solution.
            
            # Let's re-read carefully.
            # "x̃i is the intermediate Langevin update"
            # Maybe the snippet is simplified.
            # Let's look at the Python class snippet in the paper:
            # drift = -self.learning_rate * grads[i] + 0.1 * noise
            # self.x[i] = 1.0 / (1.0 + np.exp(-self.beta * drift))
            
            # This is indeed what the paper says. 
            # BUT, if grad is 0, x becomes 0.5. That seems wrong for a stable solution.
            # UNLESS the "gradient" includes a term that keeps it there?
            # No, E = 0 -> grad = 0.
            
            # Wait, maybe I should treat 'drift' as the update to an internal state 'u'?
            # x = sigmoid(u).
            # du/dt = -dE/dx ... ?
            # The paper says: "Langevin dynamics on Constraint Manifolds".
            
            # Let's look at the "Energy Functional Construction".
            # It doesn't mention an internal state.
            
            # However, let's look at the "Python Implementation" snippet again.
            # It is very explicit.
            # Maybe I should implement it exactly as is and see.
            # If it fails, I'll add a 'momentum' or 'internal state' term.
            # Actually, looking at:
            # x(t+1) = sigmoid(beta * (-eta * grad + noise))
            # If we are in a minimum, grad is 0. x becomes sigmoid(noise).
            # If beta is high, sigmoid(noise) is 0 or 1.
            # So it effectively samples 0 or 1 randomly if energy is 0?
            # That might be the "Crystallization".
            # But it wouldn't stay at the *correct* 0 or 1 unless the gradient pushes it there.
            # But the gradient is 0 at the solution.
            
            # Ah, maybe the gradient is NOT zero at the solution?
            # E = sum (1-sc)^2.
            # If sc=1, E=0, grad=0.
            
            # There must be a missing piece in the snippet or my understanding.
            # "The critical observation: ... for all variables appearing only in satisfied clauses. This creates an automatic freezing mechanism..."
            # If grad is 0, and we just have noise...
            
            # Let's check the Crystal snippet again.
            # It's identical.
            
            # Maybe the "drift" is added to the current value?
            # "xi(t+Δt) = σ(β(t) · x̃i)"
            # "x̃i is the intermediate Langevin update"
            # Usually x̃i = x(t) - eta*grad + noise.
            # So: x(t+1) = sigmoid(beta * (x(t) - eta*grad + noise)).
            # This makes MUCH more sense. It's a "soft" projection of the updated value.
            # If grad is 0, x(t+1) = sigmoid(beta * (x(t) + noise)).
            # If x(t) is 1.0, and beta is large, it stays 1.0 (unless noise is huge).
            # This preserves the state.
            
            # I will assume x̃i = self.x[i] - self.learning_rate * grads[i] + 0.1 * noise
            # This aligns with "intermediate Langevin update".
            
            # Let's verify with the paper text:
            # "xi(t+Δt) = σ(β(t) · x̃i)"
            # "x̃i is the intermediate Langevin update"
            # Standard Langevin: x_new = x_old - grad * dt + noise.
            # So x̃i must be that.
            
            # I will implement it as:
            # update = self.x[i] - self.learning_rate * grads[i] + 0.1 * noise
            # self.x[i] = 1.0 / (1.0 + np.exp(-self.beta * update))
            
            # Wait, the range of x is [0,1].
            # If we treat x as a probability, the "position" in Langevin is usually unbounded or bounded.
            # If we use the sigmoid, we are mapping R -> [0,1].
            # So the dynamics should probably happen in the R domain (logit space), and x is the projection.
            # Let u be the logit. x = sigmoid(u).
            # du = -dE/dx * dx/du ... ?
            
            # Let's stick to the interpretation:
            # x_tilde = x_old - eta * grad + noise
            # x_new = sigmoid(beta * x_tilde)
            # BUT, x_old is in [0,1]. x_tilde will be around [0,1].
            # sigmoid(beta * [0,1]) -> [0.5, 1.0] (roughly).
            # This biases everything towards 1.
            # We need x_tilde to be centered around 0.
            # So maybe x_tilde = inverse_sigmoid(x_old) - eta * grad + noise?
            # That would be dynamics in logit space.
            
            # Let's look at the snippet again.
            # "drift = -eta * grads + noise"
            # "x = sigmoid(beta * drift)"
            # This implies 'drift' is the FULL ARGUMENT to sigmoid.
            # So 'drift' must include the current state information if it's to persist.
            # But the snippet calculates drift purely from grad and noise.
            
            # HYPOTHESIS: The snippet in the paper is slightly buggy or simplified pseudo-code.
            # I will implement the "Dynamics in Logit Space" approach which is standard for constrained variables.
            # Let u_i be the internal state. x_i = sigmoid(u_i).
            # We update u_i.
            # u_i(t+1) = u_i(t) - eta * dE/dx_i + noise
            # But wait, dE/du_i = dE/dx_i * dx_i/du_i = dE/dx_i * x_i(1-x_i).
            
            # Alternative: The paper might be using a "soft" version of the boolean values.
            # If I use the "x_tilde = x_old - ..." approach, I need to center it.
            # Maybe: x_tilde = (2*x_old - 1) - eta * grad + noise ? (Mapping [0,1] to [-1,1] roughly?)
            
            # Let's try to follow the snippet literally first, but assume 'drift' accumulates?
            # No, 'drift' is a local variable.
            
            # Let's look at the "Real Demonstration Output".
            # Step 0: Energy 1.18, Temp 2.0.
            # Step 3: Satisfied.
            # It works very fast.
            
            # Let's look at the Crystal snippet again.
            # "drift = -@learning_rate * grads[i] + noise"
            # "var.value = 1.0_f64 / (1.0_f64 + Math.exp(-@beta * drift))"
            # This is extremely specific.
            # If this is true, then for x to be 1, drift must be positive large.
            # If x is 1, and satisfied, grad is 0. drift is noise (mean 0).
            # Then x becomes sigmoid(beta * noise) -> fluctuates around 0.5.
            # This CANNOT be stable for a solution.
            
            # UNLESS... the gradient is NOT zero?
            # If x=1, and it's satisfied...
            # Maybe the "Casimir force" (interaction potential) keeps it there?
            # The snippet doesn't show Casimir forces in the `langevin_step`.
            
            # Wait, the paper says:
            # "Fi = -∇iE + ∑j≠i ∇i V_Casimir(i,j)"
            # But the code snippet only shows `drift = -eta * grads`.
            
            # Okay, I will implement the "Logit Dynamics" which is robust.
            # u_i += -eta * grad + noise
            # x_i = sigmoid(beta * u_i)
            # This preserves memory.
            # And I'll add the Casimir term later if needed.
            
            # Actually, let's try to interpret "drift" as "force".
            # If we integrate force, we get velocity/position.
            # Maybe the snippet meant:
            # self.u[i] += -self.learning_rate * grads[i] + noise
            # self.x[i] = sigmoid(self.beta * self.u[i])
            # This makes physical sense.
            
            # I'll add a `self.u` (internal state) initialized to 0 (since x=0.5).
            
            # Re-reading the snippet:
            # "x̃i is the intermediate Langevin update"
            # "xi(t+Δt) = σ(β(t) · x̃i)"
            # This strongly supports the state variable interpretation.
            
            update = -self.learning_rate * grads[i] + 0.1 * noise
            # We need to accumulate this update.
            # But the snippet assigns `drift` to it and uses it directly.
            # I will assume the snippet is incomplete and use the accumulator `self.u`.
            
            pass

        for i in range(self.num_vars):
            # Thermal noise
            noise = np.sqrt(2.0 * self.temperature) * np.random.randn()
            
            # Gradient descent
            # We use a state variable 'u' to accumulate changes
            if not hasattr(self, 'u'):
                self.u = np.zeros(self.num_vars)
            
            # Update internal state
            # Note: The paper snippet scales noise by 0.1, I'll keep that.
            self.u[i] += -self.learning_rate * grads[i] + 0.1 * noise
            
            # Projection
            self.x[i] = 1.0 / (1.0 + np.exp(-self.beta * self.u[i]))

        # Annealing schedules
        self.step_count += 1
        self.temperature = 2.0 / np.log(1.0 + self.step_count * 0.05)
        self.beta = 1.0 + self.step_count * 0.01
        
        # Adaptive Correlation Length
        # Decay from initial value to focus on local interactions
        # xi = xi_0 * exp(-t / tau_decay)
        # We use a decay factor per step equivalent to exp(-1/tau)
        self.correlation_length *= 0.995

    def build_adjacency_matrix(self) -> np.ndarray:
        """Build the adjacency matrix of the variable constraint graph."""
        adj = np.zeros((self.num_vars, self.num_vars), dtype=np.float64)
        for clause in self.clauses:
            # Connect all variables in the same clause
            for i in range(len(clause)):
                u = abs(clause[i]) - 1
                for j in range(i + 1, len(clause)):
                    v = abs(clause[j]) - 1
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
        return adj

    def compute_graph_distances(self) -> np.ndarray:
        """Compute all-pairs shortest paths (Floyd-Warshall or BFS)."""
        # For small N, Floyd-Warshall is O(N^3) but easy to implement with numpy
        adj = self.build_adjacency_matrix()
        dist = np.full((self.num_vars, self.num_vars), np.inf)
        np.fill_diagonal(dist, 0)
        
        # Initialize distances for direct edges
        dist[adj > 0] = 1.0
        
        # Floyd-Warshall
        # Note: For larger N, this should be optimized (e.g. sparse BFS)
        for k in range(self.num_vars):
            dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
            
        return dist

    def spectral_partitioning(self) -> Tuple[List[int], List[int]]:
        """
        Identify clusters using the Fiedler vector of the weighted Laplacian.
        Returns two lists of variable indices representing the partitions.
        """
        grads = self.compute_gradients()
        grad_mags = np.abs(grads)
        
        # 1. Compute distances
        dists = self.compute_graph_distances()
        
        # 2. Build Weighted Laplacian
        # L_ij = -w_ij
        # w_ij = exp(-d_ij/xi) * (1 + |grad_i| + |grad_j|)
        
        weights = np.exp(-dists / self.correlation_length)
        
        # Add gradient terms: (1 + |grad_i| + |grad_j|)
        # We can use broadcasting
        grad_term = 1.0 + grad_mags.reshape(-1, 1) + grad_mags.reshape(1, -1)
        weights *= grad_term
        
        # Zero out diagonal weights for L construction (L_ii is sum of off-diagonals)
        np.fill_diagonal(weights, 0.0)
        
        # Construct Laplacian
        # L = D - W
        degrees = np.sum(weights, axis=1)
        L = np.diag(degrees) - weights
        
        # 3. Compute Fiedler Vector (2nd smallest eigenvector)
        # Use eigh for symmetric matrices
        eigenvals, eigenvecs = np.linalg.eigh(L)
        
        # Sort just in case (eigh usually returns sorted)
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # The first eigenvalue is ~0 (constant vector). The second is the Fiedler value.
        fiedler_vector = eigenvecs[:, 1]
        fiedler_val = eigenvals[1]
        
        # 4. Partition
        # Use mean as threshold to handle cases where vector is not centered at 0
        # (e.g. disconnected components where it might be an indicator vector)
        threshold = np.mean(fiedler_vector)
        
        cluster_a = [i for i, v in enumerate(fiedler_vector) if v < threshold]
        cluster_b = [i for i, v in enumerate(fiedler_vector) if v >= threshold]
        
        return cluster_a, cluster_b, fiedler_val
