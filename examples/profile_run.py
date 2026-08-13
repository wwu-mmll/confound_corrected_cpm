import time
import torch.cuda.nvtx as nvtx
from sklearn.model_selection import ShuffleSplit, RepeatedKFold
from cccpm import CPMAnalysis
from cccpm.simulation.simulate_sem import simulate_data_given_kappa
from cccpm.edge_selection import PThreshold, UnivariateEdgeSelection


sim = simulate_data_given_kappa(
    R2_X_y=0.4,
    kappa=0.3,
    n_features=8001, # 180*179/2
    n_features_informative=60,
    n_pure_signal_features=30,
    n_confound_only_features=30,
    n_confounds=2,
    n_samples=2000,
    random_state=42,
)
X, y, covariates = sim["X"], sim["y"], sim["Z"]

univariate_edge_selection = UnivariateEdgeSelection(
    edge_statistic='pearson',
    edge_selection=[PThreshold(threshold=[0.05, 0.01], correction=['bonferroni'])]
)

DEVICE = 'cuda'

cpm = CPMAnalysis(
    results_directory='./tmp/example_simulated_data',
    cv=RepeatedKFold(n_splits=10, n_repeats=10, random_state=42),
    edge_selection=univariate_edge_selection,
    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
    n_permutations=1000,
    select_stable_edges=False,
    device=DEVICE,
)

_start = time.perf_counter()
#nvtx.range_push("cpm_run")
cpm.run(X=X, y=y, covariates=covariates)
#nvtx.range_pop()
wall_time_s = time.perf_counter() - _start

print(f"Wall time: {wall_time_s:.2f}s")
