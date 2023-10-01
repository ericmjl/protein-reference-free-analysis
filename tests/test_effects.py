"""Tests for Nth order effects."""
import jax.numpy as np
import pytest
from jax import random

from protein_reference_free_analysis.effects import zeroth_order_effects
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
    _, num_states, num_positions = genotypes.shape

    key = random.PRNGKey(seed)
    phenotypes = random.normal(key, (num_states**num_positions,))
    expected = np.mean(phenotypes)

    result = zeroth_order_effects(genotypes, phenotypes)

    assert np.allclose(result, expected)
