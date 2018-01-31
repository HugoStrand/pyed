  
""" Tests for OperatorUtils

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

 """ 

# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from pytriqs.operators import n, c, c_dag, Operator, dagger

# ----------------------------------------------------------------------

from pyed.OperatorUtils import op_is_fundamental, op_serialize_fundamental

from pyed.OperatorUtils import get_quadratic_operator, \
    quadratic_matrix_from_operator, operator_single_particle_transform

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

    fund_op = [ c(0, idx) for idx in xrange(n) ]
    H_loc = get_quadratic_operator(h_loc, fund_op)
    h_loc_ref = quadratic_matrix_from_operator(H_loc, fund_op)
    
    np.testing.assert_array_almost_equal(h_loc, h_loc_ref)

# ----------------------------------------------------------------------
def test_single_particle_transform(verbose=False):

    if verbose:
        print '--> test_single_particle_transform'
        
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
        print 'h_loc =\n', h_loc
        print 'h_loc_ref =\n', h_loc_ref
        print 'H_loc =', H_loc

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
        print 'ht_loc =\n', ht_loc
        print 'ht_loc_ref =\n', ht_loc_ref
        print 'Ht_loc =', Ht_loc
        print 'Ht_loc_ref =', Ht_loc_ref
    
    assert( (Ht_loc - Ht_loc_ref).is_zero() )
    
# ----------------------------------------------------------------------
if __name__ == '__main__':

    test_fundamental()
    test_quadratic()
    test_single_particle_transform(verbose=True)
