"""Integration test for the overall model."""
from functools import partial

import pytest
from jax import numpy as np
from jax import random, vmap

from protein_reference_free_analysis.effects import (
    first_order_effects,
    get_first_order_effect,
    get_second_order_effect,
    second_order_effects,
    zeroth_order_effects,
)
from protein_reference_free_analysis.genotype_generator import (
    make_comprehensive_genotypes,
)
from protein_reference_free_analysis.phenotype_generator import (
    count_kth_genotype,
    random_phenotype,
)


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("num_states", [3, 4])
@pytest.mark.parametrize("num_sites", [2, 3])
def test_overall_model_count_kth_genotype(k, num_states, num_sites):
    """Test that the overall model works as expected.

    This test tests that after calculating zeroth, first, and second order effects,
    they can be recalculated correctly.

    :param k: The index of the k-th genotype to calculate.
    :param num_states: The number of states per site.
    :param num_sites: The number of sites.
    """
    genotypes = make_comprehensive_genotypes(num_states=num_states, num_sites=num_sites)
    count_geno = partial(count_kth_genotype, k=k)
    phenotypes = vmap(count_geno)(genotypes)
    e_0 = zeroth_order_effects(genotypes, phenotypes)
    e_1 = first_order_effects(genotypes, phenotypes)
    e_2 = second_order_effects(genotypes, phenotypes)

    for idx, genotype in enumerate(genotypes):
        recalculated_phenotype = (
            e_0
            + get_first_order_effect(e_1, genotype)
            + get_second_order_effect(e_2, genotype)
        )
        assert np.allclose(recalculated_phenotype, phenotypes[idx], atol=1e-5)


@pytest.mark.parametrize("seed", [0, 10, 20])
@pytest.mark.parametrize("num_states", [3, 4])
@pytest.mark.parametrize("num_sites", [2])  # only works with 2 sites, but not any more
def test_overall_model_random_genotype(seed, num_states, num_sites):
    """Test that the overall model works as expected with random genotypes.

    This test tests that after calculating zeroth, first, and second order effects,
    they can be recalculated correctly.

    :param seed: Seed number for JAX's PRNG.
    :param num_states: The number of states per site.
    :param num_sites: The number of sites.
    """
    genotypes = make_comprehensive_genotypes(num_states=num_states, num_sites=num_sites)

    key = random.PRNGKey(seed)
    keys = random.split(key, len(genotypes))
    phenotypes = vmap(random_phenotype)(genotypes, keys)
    e_0 = zeroth_order_effects(genotypes, phenotypes)
    e_1 = first_order_effects(genotypes, phenotypes)
    e_2 = second_order_effects(genotypes, phenotypes)

    for idx, genotype in enumerate(genotypes):
        recalculated_phenotype = (
            e_0
            + get_first_order_effect(e_1, genotype)
            + get_second_order_effect(e_2, genotype)
        )
        assert np.allclose(recalculated_phenotype, phenotypes[idx], atol=1e-5)
