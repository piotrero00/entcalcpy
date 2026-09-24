#This script was created in entcalcpy version 0.1.3


from pathlib import Path
from multiprocessing import freeze_support
from benchmark_fresh import run_benchmark

N = 50
DIM = [3, 3]
RUN_GR = True
RUN_K = True  # gekppt, k=3
RUN_UPPER = False
OUTPUT_DIR = Path(__file__).resolve().parent / "fresh_process_results"

STATE_FILE = None

if __name__ == "__main__":
    freeze_support()
    run_benchmark(
        dim=DIM, prefix='2qutrit_', n=N,
        run_gr=RUN_GR, run_k=RUN_K, run_upper=RUN_UPPER,
        output_dir=OUTPUT_DIR, state_file=STATE_FILE,
        seed_start=1000, algorithm_seed=2000,
        upper_iteramax=5000, solver="MOSEK", accuracy="high",
        sample_interval=0.01,
    )
