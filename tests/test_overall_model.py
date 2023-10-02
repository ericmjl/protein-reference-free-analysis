"""Integration test for the overall model."""
from functools import partial

import pytest
from jax import numpy as np
from jax import random, vmap

from protein_reference_free_analysis.effects import (
    calculate_phenotypes,
    first_order_effects,
    get_first_order_effect,
    get_second_order_effect,
    random_first_order_effects,
    random_second_order_effects,
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
    genotypes = make_comprehensive_genotypes(num_sites=num_sites, num_states=num_states)
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


@pytest.mark.parametrize("seed", [0, 42, 99])
@pytest.mark.parametrize("num_states", [3, 5])
@pytest.mark.parametrize("num_sites", [2, 4])
def test_overall_model_random_genotype(seed, num_states, num_sites):
    """Test that the overall model works as expected with random genotypes.

    This test tests that after calculating zeroth, first, and second order effects,
    the re-calculated phenotypes are correlated with the original phenotypes.
    This kind of shows that the model is capable of learning something
    even when the genotypes are random, possibly indicating a limitation of the model.

    :param seed: Seed number for JAX's PRNG.
    :param num_states: The number of states per site.
    :param num_sites: The number of sites.
    """
    genotypes = make_comprehensive_genotypes(num_sites=num_sites, num_states=num_states)

    key = random.PRNGKey(seed)
    keys = random.split(key, len(genotypes))
    phenotypes = vmap(random_phenotype)(genotypes, keys)
    e_0 = zeroth_order_effects(genotypes, phenotypes)
    e_1 = first_order_effects(genotypes, phenotypes)
    e_2 = second_order_effects(genotypes, phenotypes)

    recalculated_phenotypes = []
    for idx, genotype in enumerate(genotypes):
        recalculated_phenotype = (
            e_0
            + get_first_order_effect(e_1, genotype)
            + get_second_order_effect(e_2, genotype)
        )
        recalculated_phenotypes.append(recalculated_phenotype)
    recalculated_phenotypes = np.array(recalculated_phenotypes)
    assert np.corrcoef(recalculated_phenotypes, phenotypes)[0, 1] > 0


def test_overall_model_with_random_effects():
    """Test that the overall model works as expected with random effects.

    This test asserts that even when the inferred effects
    are different from the randomly-generated effects,
    we still calculate the correct phenotypes.
    """
    genotypes = make_comprehensive_genotypes(num_sites=4, num_states=3)
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)

    e_0 = random.normal(k1)

    e_1 = random_first_order_effects(genotypes, k2)
    e_2 = random_second_order_effects(genotypes, k3)

    phenotypes_true = calculate_phenotypes(e_0, e_1, e_2, genotypes)

    # Now, infer the effects from genotype-phenotype.
    e_0_est = zeroth_order_effects(genotypes, phenotypes_true)
    e_1_est = first_order_effects(genotypes, phenotypes_true)
    e_2_est = second_order_effects(genotypes, phenotypes_true)

    # Finally, re-calculate phenotypes
    phenotypes_est = calculate_phenotypes(e_0_est, e_1_est, e_2_est, genotypes)

    assert np.allclose(phenotypes_est, phenotypes_true, atol=1e-5)
