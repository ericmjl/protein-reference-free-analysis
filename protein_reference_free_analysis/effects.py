"""Implementation of the main effects.

This file implements the zeroth, first, and second order effects.
"""
import jax.numpy as np

from .matching import get_indices_with_particular_states


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


def calculate_single_genotype_averages(genotypes: np.ndarray, phenotypes: np.ndarray):
    """
    Calculates the average phenotype for each genotype.

    :param genotypes: The one-hot genotype matrix.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param phenotypes: The continuous phenotype vector.
        Should be of shape (num_genotypes,)
    :returns: The average phenotype for each genotype.
        It will be of shape (num_sites, num_states).
    """
    _, num_sites, num_states = genotypes.shape
    states = np.eye(num_states, dtype=int)

    genotype_averages = np.zeros((num_sites, num_states))

    for site_idx in range(num_sites):
        for state_idx, state in enumerate(states):
            site = np.array([site_idx])
            indices = get_indices_with_particular_states(genotypes, site, state)
            genotype_averages = genotype_averages.at[site_idx, state_idx].set(
                np.mean(phenotypes[indices])
            )
    return genotype_averages


def first_order_effects(genotypes: np.ndarray, phenotypes: np.ndarray):
    """Calculate the first order effects.

    :param genotypes: The genotype matrix.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param phenotypes: The phenotype vector.
        Should be of shape (num_genotypes,)
    :returns: The first order effects.
        It will be of shape (num_states, num_sites).
    """
    e_0 = zeroth_order_effects(genotypes, phenotypes)
    single_genotype_averages = calculate_single_genotype_averages(genotypes, phenotypes)
    return single_genotype_averages - e_0
