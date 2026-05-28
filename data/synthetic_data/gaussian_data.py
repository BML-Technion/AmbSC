import numpy as np

def generate_gaussian_mixture_data(
    n_samples,
    mus_pos,
    covs_pos,
    mus_neg,
    covs_neg,
    weights_pos=None,
    weights_neg=None,
    class_priors=(0.5, 0.5),
    seed=None
):
    """
    Samples data from two Gaussian mixtures.
    """

    if seed is not None:
        np.random.seed(seed)

    mus_pos = list(mus_pos)
    covs_pos = list(covs_pos)
    mus_neg = list(mus_neg)
    covs_neg = list(covs_neg)

    d = len(mus_pos[0])

    if weights_pos is None:
        weights_pos = np.ones(len(mus_pos)) / len(mus_pos)
    if weights_neg is None:
        weights_neg = np.ones(len(mus_neg)) / len(mus_neg)

    weights_pos = np.asarray(weights_pos)
    weights_neg = np.asarray(weights_neg)

    pi_pos, pi_neg = class_priors
    n_pos = int(n_samples * pi_pos)
    n_neg = n_samples - n_pos

    def sample_from_mixture(n, mus, covs, weights):
        comps = np.random.choice(len(weights), size=n, p=weights)
        X = np.zeros((n, d))
        for i, c in enumerate(comps):
            X[i] = np.random.multivariate_normal(mus[c], covs[c])
        return X

    X_pos = sample_from_mixture(n_pos, mus_pos, covs_pos, weights_pos)
    X_neg = sample_from_mixture(n_neg, mus_neg, covs_neg, weights_neg)

    y_pos = np.ones(n_pos)
    y_neg = -np.ones(n_neg)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]

    return X, y

def get_gaussian_params(d):
    """
    Generates Gaussian mixture parameters for a given dimension d.
    
    Rules:
    - Means: Index 0 is fixed (1.0 or -0.5), all other indices are 0.0.
    - Covariances: Diagonal matrices.
        - Positive: Base [0.5, 0.3]. Extra dimensions use variance 0.3.
        - Negative: Base [2.0, 8.0]. Extra dimensions use variance 2.0.
    """

    mu_pos_val = np.zeros(d)
    mu_neg_val = np.zeros(d)
    
    mu_pos_val[0] = 1.0
    mu_neg_val[0] = -0.5
    
    mus_pos = [mu_pos_val]
    mus_neg = [mu_neg_val]

    diag_pos = [0.5, 0.3]
    diag_neg = [2.0, 8.0]
    
    if d > 2:
        extra_dims = d - 2
        diag_pos.extend([0.3] * extra_dims)
        diag_neg.extend([2.0] * extra_dims)
        
    cov_pos_val = np.diag(diag_pos)
    cov_neg_val = np.diag(diag_neg)

    covs_pos = [cov_pos_val]
    covs_neg = [cov_neg_val]

    return mus_pos, covs_pos, mus_neg, covs_neg