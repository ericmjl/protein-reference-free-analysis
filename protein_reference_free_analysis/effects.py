"""Implementation of the main effects.

This file implements the zeroth, first, and second order effects.
"""
from itertools import combinations, product

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


def calculate_double_genotype_averages(genotypes: np.ndarray, phenotypes: np.ndarray):
    """Calculate double-genotype average phenotype.

    :param genotypes: The one-hot genotype matrix.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param phenotypes: The continuous phenotype vector.
        Should be of shape (num_genotypes,)
    :returns: The double-genotype average phenotype.
        It is of shape (num_sites, num_states, num_sites, num_states).
    """
    num_genotypes, num_sites, num_states = genotypes.shape
    states = np.eye(num_states, dtype=int)
    genotype_averages = np.zeros(shape=(num_sites, num_states, num_sites, num_states))

    for site1_idx, site2_idx in combinations(range(num_sites), 2):
        for state1_idx, state2_idx in product(range(num_states), repeat=2):
            site = np.array([site1_idx, site2_idx])
            state1 = states.at[state1_idx].get()
            state2 = states.at[state2_idx].get()
            state = np.stack([state1, state2])
            indices = get_indices_with_particular_states(genotypes, site, state)
            genotype_averages = genotype_averages.at[
                site1_idx, state1_idx, site2_idx, state2_idx
            ].set(np.mean(phenotypes[indices]))
    return genotype_averages


def second_order_effects(genotypes: np.ndarray, phenotypes: np.ndarray):
    """Calculate second-order effects.

    :param genotypes: The one-hot genotype matrix.
        Should be of shape (num_genotypes, num_sites, num_states).
    :param phenotypes: The continuous phenotype vector.
        Should be of shape (num_genotypes,)
    :returns: The second-order effects.
        It is of shape (num_sites, num_states, num_sites, num_states).
    """
    e_0 = zeroth_order_effects(genotypes, phenotypes)
    e_1 = first_order_effects(genotypes, phenotypes)
    double_genotype_averages = calculate_double_genotype_averages(genotypes, phenotypes)

    num_genotypes, num_sites, num_states = genotypes.shape
    effects = np.zeros_like(double_genotype_averages)
    for site1, site2 in combinations(range(num_sites), 2):
        for state1, state2 in product(range(num_states), repeat=2):
            phenotype_average = double_genotype_averages.at[
                site1, state1, site2, state2
            ].get()
            effect = phenotype_average - (
                e_0 + e_1.at[site1, state1].get() + e_1.at[site2, state2].get()
            )
            effects = effects.at[site1, state1, site2, state2].set(effect)
    return effects
