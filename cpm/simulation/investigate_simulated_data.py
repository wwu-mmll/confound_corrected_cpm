from cpm.simulation.simulate_data import simulate_confounded_data_chyzhyk, plot_confounding_gridspec
import numpy as np


if __name__ == '__main__':
    X, y, z = simulate_confounded_data_chyzhyk(link_type='weak_link', n_samples=200, n_features=105)
    summary_df = plot_confounding_gridspec(X, y, np.stack([z, z], axis=1))
    summary_df.to_csv('weak_link.csv', index=False)

    X, y, z = simulate_confounded_data_chyzhyk(link_type='no_link', n_samples=200, n_features=105)
    summary_df = plot_confounding_gridspec(X, y, np.stack([z, z], axis=1))
    summary_df.to_csv('no_link.csv', index=False)

    X, y, z = simulate_confounded_data_chyzhyk(link_type='direct_link', n_samples=200, n_features=105)
    summary_df = plot_confounding_gridspec(X, y, np.stack([z, z], axis=1))
    summary_df.to_csv('direct_link.csv', index=False)


