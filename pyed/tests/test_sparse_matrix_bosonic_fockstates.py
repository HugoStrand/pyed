""" Tests for bosonic sparse matrix operator representation.

Author: Hugo U. R. Strand (2026), hugo.strand@gmail.com """


import numpy as np


from pyed.SparseMatrixFockStates import SparseMatrixBosonicCreationOperators

from test_sparse_matrix_fockstates import compare_sparse_matrices


def test_single_bosonic_operator_relations():

    ops = SparseMatrixBosonicCreationOperators(N=1, Nmax=5)

    # -- Test density operator n = diag(0, 1, 2, ..., Nmax)

    n = ops.b_dag[0] * ops.b_dag[0].getH()
    np.testing.assert_array_almost_equal(n.diagonal(), np.arange(ops.Nmax))

    # -- Test commutator [b, b^+] = 1

    comm = ops.b_dag[0].getH() * ops.b_dag[0] - ops.b_dag[0] * ops.b_dag[0].getH()
    comm[-1, -1] = 1. # Last element in commutator is "wrong" due to truncated number of bosons
    compare_sparse_matrices(comm, ops.I.tocsr())


if __name__ == '__main__':

    test_single_bosonic_operator_relations()
