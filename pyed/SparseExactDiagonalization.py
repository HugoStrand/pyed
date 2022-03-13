"""
General routines for exact diagonalization using sparse matrices

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
"""

# ----------------------------------------------------------------------

import time
import itertools
import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------------

from scipy.sparse.linalg import eigs as eigs_sparse
from scipy.sparse.linalg import eigsh as eigsh_sparse

# ----------------------------------------------------------------------

from .CubeTetras import CubeTetras

# ----------------------------------------------------------------------
class SparseExactDiagonalization(object):

    """ Exact diagonalization and one- and two- particle Green's
    function calculator. """

    # ------------------------------------------------------------------
    def __init__(self, H, beta,
                 nstates=None, hermitian=True,
                 v0=None, tol=0):

        self.v0 = v0
        self.tol = tol

        self.nstates = nstates
        self.hermitian = hermitian

        self.H = H
        self.beta = beta

        self._diagonalize_hamiltonian()
        self._calculate_partition_function()
        self._calculate_density_matrix()

    # ------------------------------------------------------------------
    def _diagonalize_hamiltonian(self):

        if self.nstates is None:
            if self.hermitian:
                self.E, self.U = np.linalg.eigh(self.H.todense())
            else:
                self.E, self.U = np.linalg.eig(self.H.todense())
        else:
            if self.hermitian:
                t = time.time()
                self.E, self.U = eigsh_sparse(
                    self.H, k=self.nstates, which='SA',
                    v0=self.v0, tol=self.tol, ncv=self.nstates*8+1)
                print('ED:', time.time() - t, ' s')
            else:
                self.E, self.U = eigs_sparse(
                    self.H, k=self.nstates, which='SR',
                    v0=self.v0, tol=self.tol)

        self.U = np.mat(self.U)
        self.E0 = np.min(self.E)
        self.E = self.E - self.E0

    # ------------------------------------------------------------------
    def _calculate_partition_function(self):

        exp_bE = np.exp(-self.beta * self.E)
        self.Z = np.sum(exp_bE)

    # ------------------------------------------------------------------
    def _calculate_density_matrix(self):

        exp_bE = np.exp(-self.beta * self.E) / self.Z
        self.rho = np.einsum('ij,j,jk->ik', self.U, exp_bE, self.U.H)

    # ------------------------------------------------------------------
    def _operators_to_eigenbasis(self, op_vec):

        dop_vec = []
        for op in op_vec:
            dop = np.mat(self.U).H * op.todense() * np.mat(self.U)
            dop_vec.append(dop)

        return dop_vec

    # ------------------------------------------------------------------
    def get_expectation_value_sparse(self, operator):

        exp_val = 0.0
        for idx in range(self.E.size):
            vec = self.U[:, idx]
            dot_prod = np.dot(vec.H, operator * vec)[0,0] # <n|O|n>
            exp_val += np.exp(-self.beta * self.E[idx]) * dot_prod

        exp_val /= self.Z

        return exp_val

    # ------------------------------------------------------------------
    def get_expectation_value_dense(self, operator):

        if not hasattr(self, 'rho'): self._calculate_density_matrix()
        return np.sum(np.diag(operator * self.rho))

    # ------------------------------------------------------------------
    def get_expectation_value(self, operator):

        if self.nstates is None:
            return self.get_expectation_value_dense(operator)
        else:
            return self.get_expectation_value_sparse(operator)

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

    # ------------------------------------------------------------------
    def get_g2_dissconnected_tau_tetra(self, tau, tau_g, g):

        g = np.squeeze(g) # fix for now throwing orb idx
        g = g.real

        N = len(tau)
        G4 = np.zeros((N, N, N), dtype=np.complex)

        def gint(t):
            sign = 1.0
            if (t < 0).any():
                assert( (t <= 0).all() )
                t = self.beta + t
                sign = -1.0

            return sign * np.interp(t, tau_g, g)

        for idx, taus, perm, perm_sign in CubeTetras(tau):
            t1, t2, t3 = taus
            G4[idx] = gint(t1-t2)*gint(t3) - gint(t1)*gint(t3-t2)

        return G4

    # ------------------------------------------------------------------
    def get_g2_dissconnected_tau(self, tau, tau_g, g):

        g = np.squeeze(g) # fix for now throwing orb idx
        g = g.real

        N = len(tau)
        G4 = np.zeros((N, N, N), dtype=np.complex)

        def gint(t_in):
            t = np.copy(t_in)
            sidx = (t < 0)
            sign = np.ones_like(t)
            sign[sidx] *= -1.
            t[sidx] = self.beta + t[sidx]
            return sign * np.interp(t, tau_g, g)

        t1, t2, t3 = np.meshgrid(tau, tau, tau, indexing='ij')
        G4 = gint(t1-t2)*gint(t3) - gint(t1)*gint(t3-t2)

        return G4

    # ------------------------------------------------------------------
    def get_g2_tau(self, tau, ops):

        N = len(tau)
        G4 = np.zeros((N, N, N), dtype=np.complex)
        ops = np.array(ops)

        for tidx, tetra in enumerate(CubeTetras(tau)):
            idx, taus, perm, perm_sign = tetra

            print('Tetra:', tidx)

            # do not permute the last operator
            ops_perm = ops[perm + [3]]
            taus_perm = taus[perm] # permute the times

            G4[idx] = self.get_timeordered_three_tau_greens_function(
                taus_perm, ops_perm) * perm_sign

        return G4

    # ------------------------------------------------------------------
    def get_timeordered_two_tau_greens_function(self, taus, ops):

        r"""
        taus = [t1, t2] (ordered beta>t1>t2>0)
        ops = [O1, O2, O3]

        Returns:
        G^{(4)}(t1, t2) = -1/Z < O1(t1) O2(t2) O3(0) >

        """

        Nop = 3

        assert( taus.shape[0] == 2 )
        assert( len(ops) == Nop )

        G = np.zeros((taus.shape[-1]), dtype=np.complex)

        E = self.E[None, :]

        t1, t2 = taus
        t1, t2 = t1[:, None], t2[:, None]

        assert( (t1 <= self.beta).all() )
        assert( (t1 >= t2).all() )
        assert( (t2 >= 0).all() )

        et_a = np.exp((-self.beta + t1)*E)
        et_b = np.exp((t2-t1)*E)
        et_c = np.exp((-t2)*E)

        dops = self._operators_to_eigenbasis(ops)
        op1, op2, op3 = dops

        G = np.einsum('ta,tb,tc,ab,bc,ca->t', et_a, et_b, et_c, op1, op2, op3)

        G /= self.Z
        return G

    # ------------------------------------------------------------------
    def get_timeordered_three_tau_greens_function(self, taus, ops):

        r"""
        taus = [t1, t2, t3] (ordered beta>t1>t2>t3>0)
        ops = [O1, O2, O3, O4]

        Returns:
        G^{(4)}(t1, t2, t3) = -1/Z < O1(t1) O2(t2) O3(t3) O4(0) >

        """

        assert( taus.shape[0] == 3 )
        assert( len(ops) == 4 )

        Nop = 4
        G = np.zeros((taus.shape[-1]), dtype=np.complex)

        E = self.E[None, :]

        t1, t2, t3 = taus
        t1, t2, t3 = t1[:, None], t2[:, None], t3[:, None]

        assert( (t1 <= self.beta).all() )
        assert( (t1 >= t2).all() )
        assert( (t2 >= t3).all() )
        assert( (t3 >= 0).all() )

        et_a = np.exp((-self.beta + t1)*E)
        et_b = np.exp((t2-t1)*E)
        et_c = np.exp((t3-t2)*E)
        et_d = np.exp((-t3)*E)

        dops = self._operators_to_eigenbasis(ops)
        op1, op2, op3, op4 = dops

        if True:
            q_tac = np.einsum('tb,ab,bc->tac', et_b, op1, op2)
            q_tca = np.einsum('td,cd,da->tca', et_d, op3, op4)
            G = np.einsum('ta,tc,tac,tca->t', et_a, et_c, q_tac, q_tca)
        else:
            # Not efficient...
            G = np.einsum(
                'ta,tb,tc,td,ab,bc,cd,da->t',
                et_a, et_b, et_c, et_d, op1, op2, op3, op4)

        G /= self.Z
        return G

    # ------------------------------------------------------------------
    def get_tau_greens_function_component(self, tau, op1, op2):

        r"""
        Returns:
        G^{(2)}(\tau) = -1/Z < O_1(\tau) O_2(0) >
        """

        G = np.zeros((len(tau)), dtype=np.complex)

        op1_eig, op2_eig = self._operators_to_eigenbasis([op1, op2])

        et_p = np.exp((-self.beta + tau[:,None])*self.E[None,:])
        et_m = np.exp(-tau[:,None]*self.E[None,:])

        G = -np.einsum('tn,tm,nm,mn->t', et_p, et_m, op1_eig, op2_eig)

        G /= self.Z
        return G

    # ------------------------------------------------------------------
    def get_frequency_greens_function_component(self, iwn, op1, op2, xi):

        r"""
        Returns:
        G^{(2)}(i\omega_n) = -1/Z < O_1(i\omega_n) O_2(-i\omega_n) >
        """

        # -- Components of the Lehman expression
        dE = - self.E[:, None] + self.E[None, :]
        exp_bE = np.exp(-self.beta * self.E)
        M = exp_bE[:, None] - xi * exp_bE[None, :]

        inv_freq = iwn[:, None, None] - dE[None, :, :]
        nonzero_idx = np.nonzero(inv_freq)
        # -- Only eval for non-zero values
        freq = np.zeros_like(inv_freq)
        freq[nonzero_idx] = inv_freq[nonzero_idx]**(-1)

        op1_eig, op2_eig = self._operators_to_eigenbasis([op1, op2])

        # -- Compute Lehman sum for all operator combinations
        G = np.zeros((len(iwn)), dtype=np.complex)
        G = np.einsum('nm,mn,nm,znm->z', op1_eig, op2_eig, M, freq)
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
        for order in range(Norder):
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
