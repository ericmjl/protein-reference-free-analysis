"""Tests for the matching submodule."""
import pytest
import jax.numpy as np

from protein_reference_free_analysis.matching import (
    all_sites_match_states,
)

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
