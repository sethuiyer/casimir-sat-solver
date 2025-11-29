from casimir_sat import SATInstance, CasimirSolver
import random
import time
import numpy as np
from typing import List, Tuple

def generate_random_3sat(num_vars: int, clause_ratio: float) -> SATInstance:
    num_clauses = int(num_vars * clause_ratio)
    clauses = []
    
    for _ in range(num_clauses):
        # Pick 3 distinct variables
        vars_idx = random.sample(range(1, num_vars + 1), 3)
        # Randomly negate
        literals = [v if random.random() > 0.5 else -v for v in vars_idx]
        clauses.append(literals)
        
    return SATInstance(num_vars, clauses)

def run_benchmark():
    print(f"{'Instance Type':<20} | {'Vars':<5} | {'Clauses':<8} | {'Success':<8} | {'Steps':<6} | {'Time (ms)':<10}")
    print("-" * 75)
    
    configs = [
        ("Random 10-SAT", 10, 3.5),
        ("Random 20-SAT", 20, 3.5),
        ("Random 30-SAT", 30, 3.5),
        ("Random 50-SAT", 50, 3.5)
    ]
    
    for name, n, ratio in configs:
        total_time = 0
        success_count = 0
        total_steps = 0
        num_runs = 5
        
        for _ in range(num_runs):
            instance = generate_random_3sat(n, ratio)
            solver = CasimirSolver(instance, temperature=2.0, learning_rate=0.5)
            
            start_time = time.time()
            steps = 0
            max_steps = 1000
            solved = False
            
            for t in range(max_steps):
                if solver.total_energy() < 1e-3:
                    # Verify boolean-ness
                    if all(x > 0.9 or x < 0.1 for x in solver.x):
                        solved = True
                        steps = t
                        break
                solver.langevin_step()
            
            end_time = time.time()
            
            if solved:
                success_count += 1
                total_steps += steps
                total_time += (end_time - start_time) * 1000
        
        avg_time = total_time / success_count if success_count > 0 else 0
        avg_steps = total_steps / success_count if success_count > 0 else 0
        success_rate = f"{success_count}/{num_runs}"
        
        print(f"{name:<20} | {n:<5} | {int(n*ratio):<8} | {success_rate:<8} | {int(avg_steps):<6} | {avg_time:<10.2f}")

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    run_benchmark()
