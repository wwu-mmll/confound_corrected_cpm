import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold

from cccpm import CPMRegression
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cccpm.simulation.simulate_simple import simulate_confounded_data_chyzhyk


sample_sizes = [250, 500, 1000, 1500, 2000]
feature_sizes = [1225, 4950, 19900]

n_repetitions = 3

base_dir = "./benchmarks"
os.makedirs(base_dir, exist_ok=True)

random_state = 42


edge_selection = UnivariateEdgeSelection(
    edge_statistic="pearson",
    edge_selection=[
        PThreshold(threshold=[0.05], correction=[None])
    ],
    t_test_filter=False
)


cv = RepeatedKFold(
    n_splits=10,
    n_repeats=10,
    random_state=random_state
)


results = []

for n_samples in sample_sizes:
    for n_features in feature_sizes:

        print(f"\n=== Benchmark: N={n_samples}, P={n_features} ===")

        X, y, _ = simulate_confounded_data_chyzhyk(
            n_samples=n_samples,
            n_features=n_features,
            link_type="direct_link"
        )

        covariates = np.zeros((n_samples, 1))

        run_times = []

        for run_idx in range(n_repetitions):
            print(f"  Run {run_idx + 1}/{n_repetitions}")

            run_name = f"N{n_samples}_P{n_features}_GPU_run{run_idx + 1}"
            run_dir = os.path.join(base_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            cpm = CPMRegression(
                results_directory=run_dir,
                cv=cv,
                edge_selection=edge_selection,
                inner_cv=None,
                n_permutations=1000,
                select_stable_edges=False,
                device='cuda'
            )

            start_time = time.perf_counter()

            try:
                cpm.run(X=X, y=y, covariates=covariates)
            except Exception as e:
                print("  Exception: ", e)

            elapsed = time.perf_counter() - start_time
            run_times.append(elapsed)

            print(f"    Zeit: {elapsed:.2f} s")

        run_times = np.array(run_times)

        mean_time = run_times.mean()
        std_time = run_times.std(ddof=1)  # sample std

        print(f"  Mittelwert (1 Perm): {mean_time:.2f} ± {std_time:.2f} s")
        print(f"  Hochgerechnet (1000): {mean_time * 1000 / 60:.2f} min")

        results.append({
            "N": n_samples,
            "P": n_features,
            "n_runs": n_repetitions,
            "mean_time_1perm_sec": mean_time,
            "std_time_1perm_sec": std_time,
            "mean_time_1000perm_sec": mean_time * 1000,
            "std_time_1000perm_sec": std_time * 1000
        })


df = pd.DataFrame(results)
df.to_csv(os.path.join(base_dir, "benchmark_times_mean_std.csv"), index=False)
