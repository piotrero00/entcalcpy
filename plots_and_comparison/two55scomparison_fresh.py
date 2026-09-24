#This script was created in entcalcpy version 0.1.3


from pathlib import Path
from multiprocessing import freeze_support
from benchmark_fresh import run_benchmark

N = 50
DIM = [5, 5]
RUN_GR = False  # Wlacz True, aby uwzglednic ge_mixed_gr.
RUN_K = False  # gekppt, k=3
RUN_UPPER = False
OUTPUT_DIR = Path(__file__).resolve().parent / "fresh_process_results"
# Opcjonalnie: sciezka do istniejacego pliku .npy ze stanami (N, D, D).
# Po rozpoczeciu serii skrypt zawsze korzysta z jej zapisanej kopii stanow.
STATE_FILE = None

if __name__ == "__main__":
    freeze_support()
    run_benchmark(
        dim=DIM, prefix='55_', n=N,
        run_gr=RUN_GR, run_k=RUN_K, run_upper=RUN_UPPER,
        output_dir=OUTPUT_DIR, state_file=STATE_FILE,
        seed_start=1000, algorithm_seed=2000,
        upper_iteramax=5000, solver="MOSEK", accuracy="high",
        sample_interval=0.01,
    )
