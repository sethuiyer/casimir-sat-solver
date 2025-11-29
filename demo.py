from casimir_sat import SATInstance, CasimirSolver
import numpy as np

def run_demo():
    print("================================================================================")
    print("DEMO 1: Easy Instance with Forced Assignments")
    print("================================================================================")
    
    # Instance from paper: 5 variables, 7 clauses
    # 1: (x1)
    # 2: (¬x2 ∨ x3)
    # 3: (x2)
    # 4: (¬x3 ∨ x4)
    # 5: (x3)
    # 6: (¬x4 ∨ x5)
    # 7: (x4)
    
    # Note: Variables are 1-indexed in paper, 0-indexed in list but 1-based in clause representation
    clauses = [
        [1],
        [-2, 3],
        [2],
        [-3, 4],
        [3],
        [-4, 5],
        [4]
    ]
    
    instance = SATInstance(num_vars=5, clauses=clauses)
    solver = CasimirSolver(instance)
    
    print(f"Instance: {instance.num_vars} variables, {len(instance.clauses)} clauses")
    print("\nStarting Casimir dynamics...\n")
    print(f"{'Step':<6} | {'Energy':<10} | {'Temp':<10} | {'Xi':<10} | {'Status'}")
    print("-" * 55)
    
    max_steps = 100
    for t in range(max_steps):
        energy = solver.total_energy()
        
        # Check for convergence (energy close to 0 and variables crystallized)
        is_satisfied = energy < 1e-3
        status = "SATISFIED" if is_satisfied else "EVOLVING"
        
        print(f"{t:<6} | {energy:<10.6f} | {solver.temperature:<10.4f} | {solver.correlation_length:<10.3f} | {status}")
        
        if is_satisfied:
            # Check if variables are boolean-like
            if all(x > 0.9 or x < 0.1 for x in solver.x):
                print(f"\n*** SOLUTION FOUND in {t} steps ***")
                assignment = [f"x{i+1}={x > 0.5}" for i, x in enumerate(solver.x)]
                print(f"Assignment: {', '.join(assignment)}")
                return
        
        solver.langevin_step()

    print("\nMax steps reached.")

if __name__ == "__main__":
    run_demo()
