"""
General routines for exact diagonalization using sparse matrices

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
        Yaroslav Zhumagulov (2017), yaroslav.zhumagulov@gmail.com
"""

# ----------------------------------------------------------------------

import time
import itertools
import progressbar
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
# ----------------------------------------------------------------------

import scipy.sparse as sparse
from scipy.sparse.linalg import eigs as eigs_sparse
from scipy.sparse.linalg import eigsh as eigsh_sparse
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from scipy.linalg import expm
# ----------------------------------------------------------------------

def gf(M,E,eta,x):
    return np.sum(M/(x+1j*eta-E))

# ----------------------------------------------------------------------
class SparseExactDiagonalization(object):

    """ Exact diagonalization and one- and two- particle Green's
    function calculator. """

    # ------------------------------------------------------------------
    def __init__(self, H,blocks, beta,
                 nstates, hermitian=True,
                 v0=None, tol=0):

        self.v0 = v0
        self.tol = tol
        self.nstates=nstates
        self.hermitian = hermitian
        self.H = H
        self.blocks=blocks
        self.beta = beta
        self._diagonalize_hamiltonian()
        self._number_of_states_reduction()
        self._calculate_partition_function()
        self._calculate_density_matrix()
    # ------------------------------------------------------------------
    def _diagonalize_hamiltonian(self):
        self.U=csr_matrix(self.H.shape,dtype=np.float)
        self.E=np.zeros(self.H.shape[0])
        print 'Hamiltonian diagonalization:'
        bar = progressbar.ProgressBar()
        for i in bar(range(len(self.blocks))):
            block=self.blocks[i]
            E,U=np.linalg.eigh(self.H[block][:,block].todense())
            self.E[block]=E
            for i,n in enumerate(block):
                self.U[n,block]=U[i]
        self.E=np.array(self.E)
        self.E0 = np.min(self.E)
        self.E = self.E-self.E0
     # ------------------------------------------------------------------
    def _number_of_states_reduction(self):
        if self.nstates is not None:
            indexes=np.argsort(self.E)[:self.nstates]
            self.E=self.E[indexes]
            self.U=self.U[:,indexes]
    # ------------------------------------------------------------------
    def _calculate_partition_function(self):
        self.Z = np.sum(np.exp(-self.beta*self.E))

    # ------------------------------------------------------------------
    def _calculate_density_matrix(self):
        self.rho=csr_matrix(self.H.shape,dtype=np.float)
        print 'Density matrix calculation:'
        bar = progressbar.ProgressBar()
        for i in bar(range(len(self.blocks))):
            block=self.blocks[i]
            X,Y=np.meshgrid(block,block)
            exp_bE = np.exp(-self.beta * self.E[block]) / self.Z
            self.rho[X,Y]= np.einsum('ij,j,jk->ik', self.U[X,Y].todense(), exp_bE, self.U[X,Y].H.todense())

    # ------------------------------------------------------------------
    def _operators_to_eigenbasis(self, op_vec):

        dop_vec = []
        for op in op_vec:
            dop=self.U.getH()*op*self.U
            dop_vec.append(dop)
        return dop_vec

    # ------------------------------------------------------------------
    def get_expectation_value(self, operator):

        exp_val = 0.0
        for idx in xrange(self.E.size):
            vec = self.U[:, idx]
            dot_prod = np.dot(vec.H, operator * vec)[0,0] # <n|O|n>
            exp_val += np.exp(-self.beta * self.E[idx]) * dot_prod

        exp_val /= self.Z

        return exp_val


    # ------------------------------------------------------------------
    def get_free_energy(self):

        r""" Free energy using ground state energy shift

        Z = \sum_n e^{-\beta E_n}
        \Omega = -1/\beta \ln Z

        Z = e^{-\beta E_0} x \sum_n e^{-\beta (E_n - E_0)} = e^{-beta E_0} Z'
        \Omega = -1/\beta ( \ln Z' - \beta E_0 ) """

        Omega = -1./self.beta * (np.log(self.Z) - self.beta * self.E0)
        return Omega

    # ------------------------------------------------------------------
    def get_partition_function(self):
        return self.Z

    # ------------------------------------------------------------------
    def get_density_matrix(self):
        return self.rho

    # ------------------------------------------------------------------
    def get_eigen_values(self):
        return self.E

    # ------------------------------------------------------------------
    def get_eigen_vectors(self):
        return self.U

    # ------------------------------------------------------------------
    def get_ground_state_energy(self):
        return self.E0

    def get_grand_potential(self):
        return self.E0-np.log(np.sum(np.exp(-self.beta*self.E)))/self.beta

    def get_real_frequency_greens_function_component(self, w, op1, op2,eta):
        r"""
        Returns:
        G^{(2)}(i\omega_n) = -1/Z < O_1(i\omega_n) O_2(-i\omega_n) >
        """
        op1_eig, op2_eig = self._operators_to_eigenbasis([op1, op2])
        Q=(op1_eig.getH().multiply(op2_eig)).tocoo()
        M=(np.exp(-self.beta*self.E[Q.row])+np.exp(-self.beta*self.E[Q.col]))*Q.data
        E=(self.E[Q.row]-self.E[Q.col])
        G = np.zeros((len(w)), dtype=np.complex)
        G = Parallel(n_jobs=12)(delayed(gf)(M,E,eta,x) for x in w)
        G /= self.Z
        return G

