"""Match genotypes at k sites."""

import jax.numpy as np


def get_indices_with_particular_states(
    genotypes: np.ndarray, sites: np.ndarray, states: np.ndarray
):
    """Get the indices of the genotypes that have desired states at k sites.

    :param genotypes: A collection of genotypes.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param sites: The site at which the genotypes should be matched.
        Should be of shape (k,).
    :param states: The genotype states that should be matched.
        Should be of shape (k, n_genotype_states)
        and should be a one-hot encoding vector.
    :return: The indices of the genotypes that satisfy the condition.
    """
    has_genotypes = np.all(genotypes[:, sites, :] == states, axis=(1, 2))
    indices = np.where(has_genotypes)[0]
    return indices
