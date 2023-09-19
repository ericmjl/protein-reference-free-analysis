"""Functions that generate a genotype matrix."""

import jax.numpy as np
from itertools import product


def make_comprehensive_genotypes(num_states: int, num_positions: int) -> np.ndarray:
    """Make a comprehensive genotype matrix.

    :param num_states: The number of genotype states desired.
    :param num_positions: The number of genotype positions desired.
    :return: A comprehensive genotype matrix of all possible genotypes.
    """
    genotypes_oh = np.eye(num_states, dtype=np.int8)
    genotypes = []
    for genotype in product(range(num_states), repeat=num_positions):
        genotype = np.array([genotypes_oh[position] for position in genotype])
        genotypes.append(genotype)
    genotypes = np.array(genotypes)

    # Test:

    return genotypes
