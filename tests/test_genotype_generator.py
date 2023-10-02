"""Tests for genotype_generator.py"""
from hypothesis import given, settings
from hypothesis import strategies as st

from protein_reference_free_analysis.genotype_generator import (
    make_comprehensive_genotypes,
)


@given(
    num_sites=st.integers(min_value=1, max_value=5),
    num_states=st.integers(min_value=1, max_value=3),
)
@settings(deadline=None)
def test_make_comprehensive_genotypes(num_sites, num_states):
    """Test that make_comprehensive_genotypes returns a comprehensive genotype matrix.

    This test tests that the comprehensive genotype matrix is of the correct shape.

    :param num_states: The number of states in the comprehensive genotype matrix.
    :param num_sites: The number of positions in the comprehensive genotype matrix.
    """
    genotypes = make_comprehensive_genotypes(num_sites=num_sites, num_states=num_states)
    assert genotypes.shape == (num_states**num_sites, num_sites, num_states)
