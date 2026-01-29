""" Tests for bosonic sparse matrix operator representation.

Author: Hugo U. R. Strand (2026), hugo.strand@gmail.com """


import numpy as np


from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators

from test_sparse_matrix_fockstates import compare_sparse_matrices


def test_single_fermionic_and_bosonic_operator_relations():

    ops = SparseMatrixFermiBoseCreationOperators(Nf=1, Nb=1, Nb_max=5)

    # -- Test bosonic density operator n = I_f x diag(0, 1, 2, ..., Nmax)

    n = ops.b_dag[0] * ops.b_dag[0].getH()
    np.testing.assert_array_almost_equal(n.diagonal(), np.arange(ops.Nb_max).tolist()*2)

    # -- Test bosonic commutator [b, b^+] = 1

    comm = ops.b_dag[0].getH() * ops.b_dag[0] - ops.b_dag[0] * ops.b_dag[0].getH()
    comm[ops.bops.Ngamma-1, ops.bops.Ngamma-1] = 1. # Last element in commutator is "wrong" due to truncated number of bosons
    comm[-1, -1] = 1. # Last element in commutator is "wrong" due to truncated number of bosons
    compare_sparse_matrices(comm, ops.I.tocsr())

    # -- Test fermionic density operator n = diag(0, 1) x I_b

    nf = ops.c_dag[0] * ops.c_dag[0].getH()
    nf_ref = np.kron(np.diag([0, 1]), np.eye(ops.bops.Ngamma))
    np.testing.assert_array_almost_equal(nf.diagonal(), np.diag(nf_ref))

    # -- Test fermionic commutator {c, c^+} = 1
    
    comm = ops.c_dag[0].getH() * ops.c_dag[0] + ops.c_dag[0] * ops.c_dag[0].getH()
    compare_sparse_matrices(comm, ops.I.tocsr())


if __name__ == '__main__':

    test_single_fermionic_and_bosonic_operator_relations()
