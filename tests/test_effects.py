"""Tests for Nth order effects."""
import jax.numpy as np
import pytest
from jax import random

from protein_reference_free_analysis.effects import (
    calculate_single_genotype_averages,
    first_order_effects,
    second_order_effects,
    zeroth_order_effects,
)
from protein_reference_free_analysis.genotype_generator import (
    make_comprehensive_genotypes,
)


@pytest.fixture
def genotypes():
    """Genotypes fixture.

    :returns: a comprehensive set of genotypes.
    """
    return make_comprehensive_genotypes(num_states=2, num_positions=3)


@pytest.mark.parametrize("seed", [0, 10, 20, 30])
def test_zeroth_order_effects(genotypes, seed):
    """Test that zeroth order effects are correct.

    :param genotypes: The genotypes to test. Comes from the genotypes() fixture.
    :param seed: The seed to use.
    """
    _, num_sites, num_states = genotypes.shape

    key = random.PRNGKey(seed)
    phenotypes = random.normal(key, (num_states**num_sites,))
    expected = np.mean(phenotypes)

    result = zeroth_order_effects(genotypes, phenotypes)

    assert np.allclose(result, expected)


@pytest.mark.parametrize("seed", [0, 10, 20, 30])
def test_calculate_single_genotype_averages(genotypes, seed):
    """Test that the shape of the single genotype averages is correct.

    :param genotypes: The genotypes to test. Comes from the genotypes() fixture.
    :param seed: The seed to use.
    """
    _, num_sites, num_states = genotypes.shape
    key = random.PRNGKey(seed)
    phenotypes = random.normal(key, (num_states**num_sites,))
    expected_shape = (num_sites, num_states)

    single_genotype_averages = calculate_single_genotype_averages(genotypes, phenotypes)
    assert single_genotype_averages.shape == expected_shape


@pytest.mark.parametrize("seed", [0, 10, 20, 30])
def test_first_order_effects(genotypes, seed):
    """Test that the shape of the first-order effects is correct.

    :param genotypes: The genotypes to test. Comes from the genotypes() fixture.
    :param seed: The seed to use.
    """
    _, num_sites, num_states = genotypes.shape
    key = random.PRNGKey(seed)
    phenotypes = random.normal(key, (num_states**num_sites,))
    expected_shape = (num_sites, num_states)
    first_order_fx = first_order_effects(genotypes, phenotypes)
    assert first_order_fx.shape == expected_shape
    assert np.isclose(first_order_fx.sum(), 0, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 10, 20, 30])
def test_second_order_effects(genotypes, seed):
    """Test that the shape of the second-order effects is correct.

    :param genotypes: The genotypes to test. Comes from the genotypes() fixture.
    :param seed: The seed to use.
    """
    _, num_sites, num_states = genotypes.shape
    key = random.PRNGKey(seed)
    phenotypes = random.normal(key, (num_states**num_sites,))
    expected_shape = (num_sites, num_states, num_sites, num_states)
    second_order_fx = second_order_effects(genotypes, phenotypes)
    assert second_order_fx.shape == expected_shape
    assert np.isclose(second_order_fx.sum(), 0, atol=1e-6)
