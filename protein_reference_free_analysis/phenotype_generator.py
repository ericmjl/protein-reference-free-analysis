"""Phenotype generators from genotype data."""
import jax.numpy as np


def count_kth_genotype(genotype, k=0):
    """Count the number of kth genotypes.

    :param genotype: The genotype to count.
    :param k: The kth genotype.
    :return: The number of kth genotypes.
    """
    return np.sum(genotype, axis=0)[k]
