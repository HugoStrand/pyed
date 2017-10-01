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
from CubeTetras import CubeTetras
# ----------------------------------------------------------------------

def gf(M,E,x):
    return np.sum(M/(x-E))

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
        # self._calculate_density_matrix()
    # ------------------------------------------------------------------
    def _diagonalize_hamiltonian(self):
        self.U=csr_matrix(self.H.shape,dtype=np.float)
        self.E=np.zeros(self.H.shape[0])
        print 'Hamiltonian diagonalization:'
        bar = progressbar.ProgressBar()
        for i in bar(range(len(self.blocks))):
            block=self.blocks[i]
            X,Y=np.meshgrid(block,block)
            E,U=np.linalg.eigh(self.H[X,Y].todense())
            self.E[block]=E
            self.U[Y,X]=U
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
        exp_bE=csr_matrix(self.H.shape,dtype=np.float)
        exp_bE[range(self.E.size),range(self.E.size)]=np.exp(-self.beta * self.E) / self.Z
        self.rho=self.U.getH()*exp_bE*self.U

    # ------------------------------------------------------------------
    def _operators_to_eigenbasis(self, op_vec):

        dop_vec = []
        for op in op_vec:
            dop=self.U.getH()*op*self.U
            dop_vec.append(dop)
        return dop_vec

    # ------------------------------------------------------------------
    def get_expectation_value(self, operator):

        if not hasattr(self, 'rho'): self._calculate_density_matrix()
        return np.sum((operator * self.rho).diagonal())


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

    def get_real_frequency_greens_function_component(self, w, op1, op2,eta,xi):
        r"""
        Returns:
        G^{(2)}(i\omega_n) = -1/Z < O_1(i\omega_n) O_2(-i\omega_n) >
        """
        op1_eig, op2_eig = self._operators_to_eigenbasis([op1, op2])
        Q=(op1_eig.getH().multiply(op2_eig)).tocoo()
        M=(np.exp(-self.beta*self.E[Q.row])-xi*np.exp(-self.beta*self.E[Q.col]))*Q.data
        E=(self.E[Q.row]-self.E[Q.col])
        G = np.zeros((len(w)), dtype=np.complex)
        G = Parallel(n_jobs=4)(delayed(gf)(M,E-1j*eta,x) for x in w)
        G /= self.Z
        return G

    # ------------------------------------------------------------------
    def get_frequency_greens_function_component(self, iwn, op1, op2, xi):

        r"""
        Returns:
        G^{(2)}(i\omega_n) = -1/Z < O_1(i\omega_n) O_2(-i\omega_n) >
        """

        op1_eig, op2_eig = self._operators_to_eigenbasis([op1, op2])
        Q=(op1_eig.getH().multiply(op2_eig)).tocoo()
        M=(np.exp(-self.beta*self.E[Q.row])-xi*np.exp(-self.beta*self.E[Q.col]))*Q.data
        E=(self.E[Q.row]-self.E[Q.col])
        G = np.zeros((len(iwn)), dtype=np.complex)
        G = Parallel(n_jobs=4)(delayed(gf)(M,E,x) for x in iwn)
        G /= self.Z

        return G

    # ------------------------------------------------------------------
    def get_high_frequency_tail_coeff_component(
            self, op1, op2, xi, Norder=3):

        r""" The high frequency tail corrections can be derived
        directly from the imaginary time expression for the Green's function

        G(t) = -1/Z Tr[e^{-\beta H} e^{tH} b e^{-tH} b^+]

        and the observation that the high frequency components of the
        Matsubara Green's function G(i\omega_n) can be obtained by partial
        integration in

        G(i\omega_n) = \int_0^\beta dt e^{i\omega_n t} G(t)
                     = \sum_k=0^\infty (-1)^k
                       (\xi G^{(k)}(\beta^-) - G^{(k)}(0^+))/(i\omega_n)^(k+1)
                     = \sum_{k=1} c_k / (i\omega_n)^{k}

        where the n:th order derivative G^{(n)}(t) can be expressed as

        G^{(k)}(t) = - < [[ H , b(t) ]]^{(k)} b^+ >

        where [[H, b]]^{(k)} = [H, [H, [H, ... [H, b] ... ]]] is the k:th order
        left side commutator of H with b.

        Using this the high frequency coefficients c_k takes the form

        c_k = (-1)^(k-1) (\xi G^{(k-1)}(\beta^-) - G^{(k-1)}(0^+))
            = (-1)^k < [ [[ H , b ]]^{(k-1)} , b^+ ]_{-\xi} >

        """

        def xi_commutator(A, B, xi):
            return A * B - xi * B * A

        def commutator(A, B):
            return A * B - B * A

        H = self.H

        Gc = np.zeros((Norder), dtype=np.complex)
        ba, bc = op1, op2

        Hba = ba
        for order in xrange(Norder):
            tail_op = xi_commutator(Hba, bc, xi)
            Gc[order] = (-1.)**(order) * \
                        self.get_expectation_value(tail_op)
            Hba = commutator(H, Hba)

        return Gc

    # ------------------------------------------------------------------
    def get_high_frequency_tail(self, iwn, Gc, start_order=-1):

        """ from the high frequency coefficients Gc calculate the
        Matsubara Green's function tail

        G(i\omega_n) = \sum_k Gc[k] / (i\omega_n)^k """

        Nop = Gc.shape[-1]
        Nw = len(iwn)
        G = np.zeros((Nw, Nop, Nop), dtype=np.complex)
        iwn_idx = np.nonzero(iwn)[0] # -- Only eval for non-zero freq.
        for idx, gc in enumerate(Gc):
            G[iwn_idx, :, :] += \
                iwn[iwn_idx, None, None]**(-idx+start_order) * gc[None, :, :]

        return G

    # ------------------------------------------------------------------

# ----------------------------------------------------------------------
