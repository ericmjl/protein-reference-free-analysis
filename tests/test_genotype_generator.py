"""Tests for genotype_generator.py"""
from protein_reference_free_analysis.genotype_generator import (
    make_comprehensive_genotypes,
)
from hypothesis import strategies as st
from hypothesis import given, settings


@given(
    num_states=st.integers(min_value=1, max_value=3),
    num_positions=st.integers(min_value=1, max_value=5),
)
@settings(deadline=None)
def test_make_comprehensive_genotypes(num_states, num_positions):
    """Test that make_comprehensive_genotypes returns a comprehensive genotype matrix.

    This test tests that the comprehensive genotype matrix is of the correct shape.

    :param num_states: The number of states in the comprehensive genotype matrix.
    :param num_positions: The number of positions in the comprehensive genotype matrix.
    """
    genotypes = make_comprehensive_genotypes(num_states, num_positions)
    assert genotypes.shape == (num_states**num_positions, num_positions, num_states)
