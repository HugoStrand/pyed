
"""
Exact diagonalization and single- and two-particle Green's function calculator for Triqs operator expressions.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
        Yaroslav Zhumagulov, yaroslav.zhumagulov@gmail.com
"""

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixRepresentation

# ----------------------------------------------------------------------
class TriqsExactDiagonalization(object):

    """ Exact diagonalization for Triqs operator expressions. """

    # ------------------------------------------------------------------
    def __init__(self, H, fundamental_operators, beta,nstates=None):

        self.beta = beta
        self.rep = SparseMatrixRepresentation(fundamental_operators)
        self.ed = SparseExactDiagonalization(self.rep.sparse_matrix(H),self.rep.indexes_blocks, beta,nstates=nstates)

    # ------------------------------------------------------------------
    def get_expectation_value(self, op):
        return self.ed.get_expectation_value(self.rep.sparse_matrix(op))

    # ------------------------------------------------------------------
    def get_free_energy(self):
        return self.ed.get_free_energy()
    def get_partition_function(self):
        return self.ed.get_partition_function()
    def get_density_matrix(self):
        return self.ed.get_density_matrix()
    def get_ground_state_energy(self):
        return self.ed.get_ground_state_energy()


    def set_g2_w(self, g_w, op1, op2,eta=0.1):

        op1_mat = self.rep.sparse_matrix(op1)
        op2_mat = self.rep.sparse_matrix(op2)

        w = np.array([w for w in g_w.mesh])

        g_w.data[:, 0, 0] = \
            self.ed.get_real_frequency_greens_function_component(
                w, op1_mat, op2_mat, eta)
    
