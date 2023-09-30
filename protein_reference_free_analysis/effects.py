"""Implementation of the main effects.

This file implements the zeroth, first, and second order effects.
"""
import jax.numpy as np


def zeroth_order_effects(genotypes: np.ndarray, phenotypes: np.ndarray):
    """Calculate zeroth order effects.

    This is calculated by taking the mean of the phenotypes.

    :param genotypes: The genotype matrix.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param phenotypes: The phenotype vector.
        Should be of shape (num_genotypes,)
    :return: The zeroth order effects.
    """
    return np.mean(phenotypes)
