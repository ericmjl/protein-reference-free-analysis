"""Match genotypes at k sites."""

import jax.numpy as np


def all_sites_match_states(states: np.ndarray, sites: np.ndarray):
    """Generate a condition function for matching k genotype states at k sites.

    :param states: The genotype states.
        Should be of shape (k, n_genotype_states)
        and should be a one-hot encoding vector.
    :param sites: The site indices.
        Should be of shape (k,).
    :return: A condition function that checks for the presence of
        k genotype states at k sites.
    """
    k = len(states)

    def condition(genotype: np.ndarray):
        """Check if the genotype matches the condition.

        :param genotype: The genotype.
            Should be of shape (n_genotype_states, n_sites).
        :raises IndexError: If the genotype does not have enough sites
            to match the states.
        :return: True if all genotypes at k sites matches the designated states.
        """
        for i in range(k):
            # Raise error if sites[i] is greater than the genotype's length.
            if len(genotype) <= sites[i]:
                raise IndexError(
                    "The genotype does not have enough sites to match the states."
                )
            if (genotype[sites[i]] != states[i]).any():
                return False
        return True

    return condition


def get_indices_with_particular_states(
    genotypes: np.ndarray, states: np.ndarray, sites: np.ndarray
):
    """Get the indices of the genotypes that have desired states at k sites.

    :param states: The genotype states.
        Should be of shape (k, n_genotype_states)
        and should be a one-hot encoding vector.
    :param sites: The site indices.
        Should be of shape (k,).


    :param genotypes: A collection of genotypes.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param states: The genotype states that should be matched.
    :param sites: The sites at which the genotypes should be matched.
    :return: The indices of the genotypes that satisfy the condition.
    """

    condition_func = all_sites_match_states(states, sites)
    idxs = []
    for i, genotype in enumerate(genotypes):
        if condition_func(genotype):
            idxs.append(i)
    return np.array(idxs)
