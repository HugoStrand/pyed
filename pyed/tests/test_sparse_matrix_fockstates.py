# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from pytriqs.operators import c, c_dag

# ----------------------------------------------------------------------

from pyed.SparseMatrixFockStates import SparseMatrixRepresentation
from pyed.FockStates import ManyBodyOperatorFactory

# ----------------------------------------------------------------------
def compare_sparse_matrices(A, B):

    A.eliminate_zeros()
    B.eliminate_zeros()
    A = A.tocoo()
    B = B.tocoo()

    np.testing.assert_array_almost_equal(A.data, B.data)
    np.testing.assert_array_almost_equal(A.row, B.row)
    np.testing.assert_array_almost_equal(A.col, B.col)
    
# ----------------------------------------------------------------------
def test_sparse_matrix_representation():
    
    op = ManyBodyOperatorFactory(norbs=2)

    up, do = 0, 1
    fundamental_operators = [c(up,0), c(do,0)]
    
    rep = SparseMatrixRepresentation(fundamental_operators)

    # -- Test an operator
    O_mat = rep.sparse_matrix(c(up, 0))
    O_ref = op.c[0]
    compare_sparse_matrices(O_mat, O_ref)

    # -- Test
    O_mat = rep.sparse_matrix(c(do, 0))
    O_ref = op.c[1]
    compare_sparse_matrices(O_mat, O_ref)

    # -- Test expression
    H_expr = c(up, 0) * c(do, 0) * c_dag(up, 0) * c_dag(do, 0)
    H_mat = rep.sparse_matrix(H_expr)
    H_ref = op.c[0] * op.c[1] * op.cdagger[0] * op.cdagger[1]
    compare_sparse_matrices(H_mat, H_ref)

# ----------------------------------------------------------------------
def test_trimer_hamiltonian():

    # ------------------------------------------------------------------
    # -- Hubbard atom with two bath sites, Hamiltonian
    
    beta = 2.0
    V1 = 2.0
    V2 = 5.0
    epsilon1 = 0.00
    epsilon2 = 4.00
    mu = 2.0
    U = 1.0

    # -- construction using triqs operators
    
    up, do = 0, 1
    docc = c_dag(up,0) * c(up,0) * c_dag(do,0) * c(do,0)
    nA = c_dag(up,0) * c(up,0) + c_dag(do,0) * c(do,0)
    nB = c_dag(up,1) * c(up,1) + c_dag(do,1) * c(do,1)
    nC = c_dag(up,2) * c(up,2) + c_dag(do,2) * c(do,2)

    H_expr = -mu * nA + epsilon1 * nB + epsilon2 * nC + U * docc + \
        V1 * (c_dag(up,0)*c(up,1) + c_dag(up,1)*c(up,0) + \
              c_dag(do,0)*c(do,1) + c_dag(do,1)*c(do,0) ) + \
        V2 * (c_dag(up,0)*c(up,2) + c_dag(up,2)*c(up,0) + \
              c_dag(do,0)*c(do,2) + c_dag(do,2)*c(do,0) )    

    # ------------------------------------------------------------------
    fundamental_operators = [
        c(up,0), c(do,0), c(up,1), c(do,1), c(up,2), c(do,2)]
    
    rep = SparseMatrixRepresentation(fundamental_operators)
    H_mat = rep.sparse_matrix(H_expr)
    
    # -- explicit construction
    
    op = ManyBodyOperatorFactory(norbs=6)
    
    H_ref = -mu * (op.n[0] + op.n[1]) + \
        epsilon1 * (op.n[2] + op.n[3]) + \
        epsilon2 * (op.n[4] + op.n[5]) + \
        U * op.n[0] * op.n[1] + \
        V1 * (op.cdagger[0] * op.c[2] + op.cdagger[2] * op.c[0] + \
              op.cdagger[1] * op.c[3] + op.cdagger[3] * op.c[1] ) + \
        V2 * (op.cdagger[0] * op.c[4] + op.cdagger[4] * op.c[0] + \
              op.cdagger[1] * op.c[5] + op.cdagger[5] * op.c[1] )    

    # ------------------------------------------------------------------
    # -- compare

    compare_sparse_matrices(H_mat, H_ref)
    
#----------------------------------------------------------------------
if __name__ == '__main__':

    test_sparse_matrix_representation()
    test_trimer_hamiltonian()
