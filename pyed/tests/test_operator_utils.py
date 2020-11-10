  
""" Tests for OperatorUtils

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

 """ 

# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from triqs.operators import n, c, c_dag, Operator, dagger

from triqs.operators.util.U_matrix import U_matrix_kanamori, U_matrix
from triqs.operators.util.hamiltonians import h_int_kanamori

from transform_kanamori import h_int_kanamori_transformed

# ----------------------------------------------------------------------

from pyed.OperatorUtils import fundamental_operators_from_gf_struct
from pyed.OperatorUtils import op_is_fundamental, op_serialize_fundamental

from pyed.OperatorUtils import get_quadratic_operator, \
    quadratic_matrix_from_operator, operator_single_particle_transform

from pyed.OperatorUtils import symmetrize_quartic_tensor
from pyed.OperatorUtils import quartic_tensor_from_operator
from pyed.OperatorUtils import operator_from_quartic_tensor

# ----------------------------------------------------------------------
def test_gf_struct():

    orb_idxs = [0, 1, 2]
    spin_idxs = ['up', 'do']
    gf_struct = [ [spin_idx, orb_idxs] for spin_idx in spin_idxs ]
    
    fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)

    fundamental_operators_ref = [
        c('up', 0), 
        c('up', 1), 
        c('up', 2), 
        c('do', 0), 
        c('do', 1), 
        c('do', 2),
        ]

    print(fundamental_operators)
    assert( fundamental_operators == fundamental_operators_ref )

# ----------------------------------------------------------------------
def test_fundamental():

    assert( op_is_fundamental(c(0, 0)) is True )
    assert( op_is_fundamental(c_dag(0, 0)) is True )
    assert( op_is_fundamental(c_dag(0, 0)*c(0, 0)) is False )
    assert( op_is_fundamental(Operator(1.0)) is False )

    assert( op_serialize_fundamental(c(0,0)) == (False, (0,0)) )
    assert( op_serialize_fundamental(c_dag(0,0)) == (True, (0,0)) )

    assert( op_serialize_fundamental(c(2,4)) == (False, (2,4)) )
    assert( op_serialize_fundamental(c_dag(4,3)) == (True, (4,3)) )

# ----------------------------------------------------------------------
def test_quadratic():

    n = 10
    
    h_loc = np.random.random((n, n))
    h_loc = 0.5 * (h_loc + h_loc.T)

    fund_op = [ c(0, idx) for idx in range(n) ]
    H_loc = get_quadratic_operator(h_loc, fund_op)
    h_loc_ref = quadratic_matrix_from_operator(H_loc, fund_op)
    
    np.testing.assert_array_almost_equal(h_loc, h_loc_ref)

# ----------------------------------------------------------------------
def test_quartic(verbose=False):

    if verbose:
        print('--> test_quartic')
        
    num_orbitals = 2
    num_spins = 2

    U, J = 1.0, 0.2

    up, do = 0, 1
    spin_names = [up, do]
    orb_names = list(range(num_orbitals))
    
    U_ab, UPrime_ab = U_matrix_kanamori(n_orb=2, U_int=U, J_hund=J)

    if verbose:
        print('U_ab =\n', U_ab)
        print('UPrime_ab =\n', UPrime_ab)

    T_ab = np.array([
        [1., 1.],
        [1., -1.],
        ]) / np.sqrt(2.)

    # -- Check hermitian
    np.testing.assert_array_almost_equal(np.mat(T_ab) * np.mat(T_ab).H, np.eye(2))
    
    I = np.eye(num_spins)
    T_ab_spin = np.kron(T_ab, I)
    
    H_int = h_int_kanamori(
        spin_names, orb_names, U_ab, UPrime_ab, J_hund=J,
        off_diag=True, map_operator_structure=None, H_dump=None)

    op_imp = [c(up,0), c(do,0), c(up,1), c(do,1)]
    Ht_int = operator_single_particle_transform(H_int, T_ab_spin, op_imp)

    if verbose:
        print('H_int =', H_int)
        print('Ht_int =', Ht_int)

    from transform_kanamori import h_int_kanamori_transformed

    Ht_int_ref = h_int_kanamori_transformed(
        [T_ab, T_ab], spin_names, orb_names, U_ab, UPrime_ab, J_hund=J,
        off_diag=True, map_operator_structure=None, H_dump=None)

    if verbose:
        print('Ht_int_ref =', Ht_int_ref)
    
    assert( (Ht_int_ref - Ht_int).is_zero() )

# ----------------------------------------------------------------------
def test_single_particle_transform(verbose=False):

    if verbose:
        print('--> test_single_particle_transform')
        
    h_loc = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
        ])

    op_imp = [c(0,0), c(0,1)]

    H_loc = get_quadratic_operator(h_loc, op_imp)
    H_loc_ref = c_dag(0,0) * c(0,0) - c_dag(0,1) * c(0,1)

    assert( (H_loc - H_loc_ref).is_zero() )

    h_loc_ref = quadratic_matrix_from_operator(H_loc, op_imp)
    np.testing.assert_array_almost_equal(h_loc, h_loc_ref)

    if verbose:
        print('h_loc =\n', h_loc)
        print('h_loc_ref =\n', h_loc_ref)
        print('H_loc =', H_loc)

    T_ab = np.array([
        [1., 1.],
        [1., -1.],
        ]) / np.sqrt(2.)

    Ht_loc = operator_single_particle_transform(H_loc, T_ab, op_imp)
    ht_loc = quadratic_matrix_from_operator(Ht_loc, op_imp)

    ht_loc_ref = np.array([
        [0., 1.],
        [1., 0.],
        ])

    Ht_loc_ref = c_dag(0, 0) * c(0, 1) + c_dag(0, 1) * c(0, 0)

    if verbose:
        print('ht_loc =\n', ht_loc)
        print('ht_loc_ref =\n', ht_loc_ref)
        print('Ht_loc =', Ht_loc)
        print('Ht_loc_ref =', Ht_loc_ref)
    
    assert( (Ht_loc - Ht_loc_ref).is_zero() )

# ----------------------------------------------------------------------
def test_quartic_tensor_from_operator(verbose=False):


    N = 3
    fundamental_operators = [ c(0, x) for x in range(N) ]
    shape = (N, N, N, N)
    
    U = np.random.random(shape) + 1.j * np.random.random(shape)
    U_sym = symmetrize_quartic_tensor(U)
    
    H = operator_from_quartic_tensor(U, fundamental_operators)
    U_ref = quartic_tensor_from_operator(H, fundamental_operators, perm_sym=True)

    np.testing.assert_array_almost_equal(U_ref, U_sym)

    if verbose:
        print('-'*72)
        import itertools
        for idxs in itertools.product(list(range(N)), repeat=4):
            print(idxs, U_ref[idxs] - U_sym[idxs], U[idxs], U_ref[idxs], U_sym[idxs])

# ----------------------------------------------------------------------
if __name__ == '__main__':

    test_fundamental()
    test_quadratic()
    test_quartic(verbose=True)
    test_single_particle_transform(verbose=True)
    test_quartic_tensor_from_operator(verbose=True)
