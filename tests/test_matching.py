"""Tests for the matching submodule."""
import pytest
import jax.numpy as np
from jax import random

from protein_reference_free_analysis.matching import (
    all_sites_match_states,
    get_indices_with_particular_states,
)
from protein_reference_free_analysis.genotype_generator import (
    make_comprehensive_genotypes,
)

from hypothesis import given, strategies as st, settings

# Define the test data as a list of tuples.
test_data_match_two = [
    # Test 1: Matching two genotype states at positions 0 and 1.
    (
        np.array([0, 1]),  # sites to check for match
        np.array([[0, 1, 0], [1, 0, 0]]),  # states to check
        np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # genotype to check
        True,  # expected result
    ),
    # Test 2: Not matching two genotype states at positions 0 and 1.
    (
        np.array([0, 1]),  # sites to check for match
        np.array([[0, 1, 0], [1, 0, 0]]),  # states to check
        np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # genotype to check
        False,  # expected result
    ),
    # Test 3: Matching two different genotype states at positions 2 and 0.
    (
        np.array([2, 0]),  # sites to check for match
        np.array([[0, 0, 1], [0, 1, 0]]),  # states to check
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]),  # genotype to check
        True,  # expected result
    ),
    # Test 4: Matching two genotype states when one of the sites is out of range.
    (
        np.array([0, 1]),  # sites to check for match
        np.array([[0, 1, 0], [1, 0, 0]]),  # states to check
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]),  # genotype to check
        True,  # expected result
    ),
    # Test 5: Matching one genotype state at position 0.
    (
        np.array([0]),  # sites to check for match
        np.array([[0, 1, 0]]),  # states to check
        np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]),  # genotype to check
        True,  # expected result
    ),
    # Test 6: Not matching one genotype state at position 0.
    (
        np.array([0]),  # sites to check for match
        np.array([[0, 1, 0]]),  # states to check
        np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # genotype to check
        False,  # expected result
    ),
]


@pytest.mark.parametrize(
    "sites, states, genotype, expected_result", test_data_match_two
)
def test_all_sites_match_states(sites, states, genotype, expected_result):
    """Test that all_sites_match_states returns the expected result.

    :param sites: The site indices.
    :param states: The genotype states.
    :param genotype: The genotype.
    :param expected_result: The expected result.
    """
    condition_func = all_sites_match_states(states, sites)
    assert condition_func(genotype) == expected_result


def test_all_sites_match_states_raise_error():
    """Test that IndexError is raised when a site is out of range."""
    sites = np.array([0, 9])  # position 9 is out of range
    states = np.array([[0, 1, 0], [1, 0, 0]])
    genotype = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(IndexError):
        condition_func = all_sites_match_states(states, sites)
        condition_func(genotype)


@given(
    num_states=st.integers(min_value=2, max_value=4),
    num_positions=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=100),
)
@settings(deadline=None)
def test_get_indices_with_particular_states(
    num_states: int, num_positions: int, seed: int
):
    """Test get_indices_with_particular_states function.

    This test tests that get_indices_with_particular_states returns a list of indices.
    It must contain at least one element that is a valid index.

    :param num_states: The number of states to generate.
    :param num_positions: The number of positions to generate.
    :param seed: The seed for the random number generator.
    """
    genotypes = make_comprehensive_genotypes(
        num_states=num_states, num_positions=num_positions
    )
    # randomly pick a genotype, and then randomly pick a site within that genotype.
    key = random.PRNGKey(seed)
    k1, k2 = random.split(key, 2)
    genotype_idx = random.randint(k1, shape=(), minval=0, maxval=len(genotypes))
    genotype = genotypes[genotype_idx]
    site = random.randint(k2, shape=(), minval=0, maxval=len(genotype))
    state = genotype[site]
    indices = get_indices_with_particular_states(
        genotypes, states=np.expand_dims(state, 0), sites=np.expand_dims(site, 0)
    )

    # There should always be >=1 indices returned,
    # because we generated genotypes from a comprehensive set of genotypes.
    assert len(indices) >= 1

    assert genotype_idx in indices

    # Each genotype within the genotypes set should have the exact state
    # drawn randomly from the genotypes
    for idx in indices:
        assert (genotypes[idx, site] == state).all()
