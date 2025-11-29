from casimir_sat import SATInstance, CasimirSolver
import random
import time
import numpy as np

def generate_random_3sat(num_vars: int, num_clauses: int) -> SATInstance:
    clauses = []
    for _ in range(num_clauses):
        vars_idx = random.sample(range(1, num_vars + 1), 3)
        literals = [v if random.random() > 0.5 else -v for v in vars_idx]
        clauses.append(literals)
    return SATInstance(num_vars, clauses)

def run_large_scale_test():
    print("================================================================================")
    print("LARGE SCALE TEST: N=1000, M=4300 (alpha = 4.3)")
    print("================================================================================")
    print("Note: alpha=4.3 is above the theoretical SAT threshold (4.26).")
    print("This instance is likely UNSATISFIABLE or extremely hard.")
    print("We expect the solver to struggle or the spectral gap to close.")
    print("================================================================================")

    # Generate instance
    print("Generating instance...")
    instance = generate_random_3sat(1000, 4300)
    
    # Initialize solver
    # We increase correlation length slightly for larger graph
    solver = CasimirSolver(instance, temperature=2.0, learning_rate=0.5, correlation_length=10.0)
    
    print(f"Instance: {instance.num_vars} variables, {len(instance.clauses)} clauses")
    print("\nStarting Casimir dynamics (max 500 steps)...")
    print(f"{'Step':<6} | {'Energy':<10} | {'Temp':<10} | {'Xi':<10} | {'Fiedler Val'}")
    print("-" * 65)
    
    start_time = time.time()
    
    for t in range(500):
        energy = solver.total_energy()
        
        # Run spectral analysis every 50 steps (expensive for N=1000)
        fiedler_val_str = "N/A"
        if t % 50 == 0:
            try:
                # This might be slow
                _, _, fiedler_val = solver.spectral_partitioning()
                fiedler_val_str = f"{fiedler_val:.6f}"
            except Exception as e:
                fiedler_val_str = "Error"
        
        if t % 10 == 0:
            print(f"{t:<6} | {energy:<10.4f} | {solver.temperature:<10.4f} | {solver.correlation_length:<10.3f} | {fiedler_val_str}")
        
        if energy < 1e-3:
             if all(x > 0.9 or x < 0.1 for x in solver.x):
                print(f"\n*** SURPRISE: SOLUTION FOUND in {t} steps! ***")
                break
        
        solver.langevin_step()
        
    end_time = time.time()
    print(f"\nTest completed in {end_time - start_time:.2f} seconds.")
    
    # MAX-SAT Analysis
    unsatisfied_count = 0
    for clause in instance.clauses:
        satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1
            val = solver.x[var_idx]
            # Check if literal is satisfied (using 0.5 threshold for boolean interpretation)
            if (lit > 0 and val > 0.5) or (lit < 0 and val < 0.5):
                satisfied = True
                break
        if not satisfied:
            unsatisfied_count += 1
            
    print(f"Unsatisfied Clauses: {unsatisfied_count} / {len(instance.clauses)}")
    print(f"Satisfaction Rate: {(1 - unsatisfied_count/len(instance.clauses))*100:.2f}%")
    
    print("If no solution was found, check the Fiedler value.")
    print("A small Fiedler value (< 1e-4) confirms the instance is structurally hard/fragmented.")

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    run_large_scale_test()
