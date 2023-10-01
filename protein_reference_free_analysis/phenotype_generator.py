"""Phenotype generators from genotype data."""
import jax.numpy as np
from jax import random


def count_kth_genotype(genotype: np.ndarray, k=0):
    """Count the number of kth genotypes.

    Usage example:

    ```python
    from jax import vmap
    from functools import partial

    genotypes = ...
    k = 1
    phenotypes = vmap(partial(count_kth_genotype, k=k))(genotypes)
    ```

    :param genotype: The genotype to count.
    :param k: The kth genotype.
    :return: The number of kth genotypes.
    """
    return np.sum(genotype, axis=0)[k]


def random_phenotype(genotype: np.ndarray, key: random.PRNGKey) -> np.ndarray:
    """Generate a random phenotype drawn from a Gaussian.

    Usage example:

    ```python
    from jax import vmap

    genotypes = ...
    key = random.PRNGKey(22)
    keys = random.split(key, len(genotypes))
    phenotypes = vmap(random_phenotype)(genotypes, keys)
    ```

    :param genotype: The genotype.
    :param key: The key to use for random number generation.
    :returns: A random phenotype.
    """
    return random.normal(key)
