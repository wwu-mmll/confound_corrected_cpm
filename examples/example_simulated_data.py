import torch
from sklearn.model_selection import RepeatedKFold, ShuffleSplit
from cpm.edge_selection import PThreshold, UnivariateEdgeSelection
from cpm import CPMRegression
from cpm.simulate_data import simulate_regression_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda available: ", torch.cuda.is_available())

X_np, y_np, cov_np = simulate_regression_data(
    n_features=1225,
    n_informative_features=50,
    covariate_effect_size=0.2,
    feature_effect_size=100,
    noise_level=0.1
)

X = torch.from_numpy(X_np).float().to(device)
y = torch.from_numpy(y_np).float().to(device)
covariates = torch.from_numpy(cov_np).float().to(device)

edge_selector = UnivariateEdgeSelection(
    edge_statistic='pearson',
    t_test_filter=False,
    edge_selection=PThreshold(
        threshold=[0.05, 0.01],
        correction=None
    ),
    device=device
)

cpm = CPMRegression(
    results_directory='./tmp/example_simulated_data',
    cv=RepeatedKFold(n_splits=10, n_repeats=1, random_state=42),
    edge_selection=edge_selector,
    inner_cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
    n_permutations=2,
    atlas_labels='atlas_labels.csv',
    select_stable_edges=False
)

import time

start = time.time()
cpm.run(X=X, y=y, covariates=covariates)
end = time.time()
print("Time to run CPM: ", end - start, " s")
cpm.generate_html_report()
