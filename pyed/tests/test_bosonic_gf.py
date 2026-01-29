""" Tests for bosonic single particle Green's functions.

Author: Hugo U. R. Strand (2026), hugo.strand@gmail.com """


import numpy as np


from triqs.gf import MeshImFreq, MeshImTime
from triqs.gf import Gf, inverse, iOmega_n, make_gf_from_fourier


from pyed.SparseMatrixFockStates import SparseMatrixBosonicCreationOperators
from pyed.SparseExactDiagonalization import SparseExactDiagonalization


def test_non_interacting_bosonic_greens_functions_from_ed(verbose=False):

    beta = 1.
    omega = 1.0
    Nmax = 100 # need to converge in the Fock space truncation (for a given beta, omega)

    print(f'beta = {beta}')
    print(f'omega = {omega}')

    ops = SparseMatrixBosonicCreationOperators(N=1, Nmax=Nmax)

    b_dag = ops.b_dag[0]
    b = b_dag.getH()
    n = b_dag * b
    xi = 1.0

    H = omega * (n + 0.5 * ops.I)
    #print(f'H = \n{H.todense()}')

    ed = SparseExactDiagonalization(H, beta)

    print(f'Z = {ed.get_partition_function()}')

    n_exp = ed.get_expectation_value_sparse(n)
    print(f'n_exp = {n_exp}')

    mesh_tau = MeshImTime(beta, 'Boson', n_tau=1001)
    tau = np.array([float(t) for t in mesh_tau])

    mesh_iwn = MeshImFreq(beta, 'Boson', n_iw=101)
    iwn = np.array([complex(w) for w in mesh_iwn])

    G_tau = Gf(mesh=mesh_tau, target_shape=[])
    G_tau.data[:] = ed.get_tau_greens_function_component(tau, b, b_dag)

    G_iwn = Gf(mesh=mesh_iwn, target_shape=[])
    G_iwn.data[:] = ed.get_frequency_greens_function_component(iwn, b, b_dag, xi)

    # -- Analytic references for non-interacting boson

    G_iwn_ref = Gf(mesh=mesh_iwn, target_shape=[])
    G_iwn_ref << inverse(iOmega_n - omega)
    G_tau_ref = make_gf_from_fourier(G_iwn_ref, n_tau=len(mesh_tau))

    if verbose:

        from triqs.plot.mpl_interface import oplot, oploti, oplotr, plt

        plt.figure(figsize=(8, 8))

        subp = [3, 2, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplotr(G_iwn, 'x', label='ED') 
        oplotr(G_iwn_ref, '+', label='Analytic')
        plt.ylabel(r'Re[$G(i\omega_n)$]')
        plt.xlabel(r'$i\omega_n$')

        plt.subplot(*subp); subp[-1] += 1
        oploti(G_iwn, 'x', label='ED')
        oploti(G_iwn_ref, '+', label='Analytic')
        plt.ylabel(r'Im[$G(i\omega_n)$]')
        plt.xlabel(r'$i\omega_n$')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(G_iwn - G_iwn_ref, 'x') 
        plt.ylabel(r'Re[G-G_{anal}]')
        plt.xlabel(r'$i\omega_n$')

        plt.subplot(*subp); subp[-1] += 1
        oploti(G_iwn - G_iwn_ref, 'x')
        plt.ylabel(r'Im[G-G_{anal}]')
        plt.xlabel(r'$i\omega_n$')
        
        plt.subplot(*subp); subp[-1] += 1
        oplotr(G_tau, '-x', label='ED')
        oplotr(G_tau_ref, '+', label='Analytic')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(G_tau - G_tau_ref, '-')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G - G_{anal}$')

        plt.tight_layout()
        plt.show()

    # -- Compare

    np.testing.assert_array_almost_equal(G_iwn.data, G_iwn_ref.data)
    np.testing.assert_array_almost_equal(G_tau.data, G_tau_ref.data)


if __name__ == '__main__':

    test_non_interacting_bosonic_greens_functions_from_ed()
