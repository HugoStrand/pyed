  
""" Utilities for working with Triqs second quantized operator
expressions.

Author: Hugo U.R. Strand (2018) hugo.strand@gmail.com """

# ----------------------------------------------------------------------

import numpy as np

from pytriqs.operators import c, c_dag, Operator, dagger

# ----------------------------------------------------------------------
def get_quadratic_operator(h, fundamental_operators):

    # -- Check Hermicity
    np.testing.assert_array_almost_equal(h, h.T.conj())
    
    H = Operator(0.)
    for idx1, o1 in enumerate(fundamental_operators):
        o1 = dagger(o1)
        for idx2, o2 in enumerate(fundamental_operators):
            H += h[idx1, idx2] * o1 * o2

    return H
            
# ----------------------------------------------------------------------
def op_is_fundamental(op):
    
    """ Check for single "fundamental" operator in operator expression. """

    monomial_sum = list(op)    
    if len(monomial_sum) != 1:
        return False

    monomial = monomial_sum[0]
    operator_list, prefactor = monomial

    if len(operator_list) != 1:
        return False

    return True

# ----------------------------------------------------------------------
def op_serialize_fundamental(op):
    
    """ Return a tuple specifying dagger and operator indices of a 
    fundamental (single) creation/annihilation operator. """

    assert( op_is_fundamental(op) )
    dag, idxs = list(op)[0][0][0]
    return dag, tuple(idxs)
    
# ----------------------------------------------------------------------
def get_operator_index_map(fundamental_operators):

    """ Construct list of tuples of orbital index pairs from list of
    fundamental operators. """

    op_idx_map = []
    for op in fundamental_operators:
        dag, idxs = op_serialize_fundamental(op)
        assert( dag is False ) # Only accept annihilation ops
        op_idx_map.append(tuple(idxs))

    return op_idx_map

# ----------------------------------------------------------------------
def quadratic_matrix_from_operator(op, fundamental_operators):

    """ Build matrix of quadratic operator terms spanned by the list
    of fundamental_operators in given operator expression op. """

    op_idx_map = get_operator_index_map(fundamental_operators)
    op_idx_set = set(op_idx_map)

    nop = len(fundamental_operators)
    h_quad = np.zeros((nop, nop), dtype=np.complex)
    
    for term in op:
        op_list, prefactor = term
        if len(op_list) == 2:
            (d1, t1), (d2, t2) = list(op_list)
            t1, t2 = tuple(t1), tuple(t2)
            assert( d1 is True and d2 is False ) # no anomalous terms, yet...
            if t1 in op_idx_set and t2 in op_idx_set:
                i, j = op_idx_map.index(t1), op_idx_map.index(t2)
                h_quad[i, j] = prefactor

    return h_quad
    
# ----------------------------------------------------------------------
def operator_single_particle_transform(op, U, fundamental_operators):
    
    """ Transform the operator op according to the single particle 
    transform matrix U defined in the subspace of operators listed in 
    fundamental_operators. """

    # -- Convert fundamental operators back and forth from index

    op_idx_map = get_operator_index_map(fundamental_operators)
    op_idx_set = set(op_idx_map)

    # -- Transformed creation operator

    def c_transf(s, i):
        if (s, i) not in op_idx_set:
            return c(s, i)

        k = op_idx_map.index((s, i))
        
        ret = Operator()
        for l in xrange(U.shape[0]):
            op_idx = op_idx_map[l]
            ret += U[k, l] * c(*op_idx)

        return ret

    # -- Precompute transformed operators
    
    op_trans_dict = {}
    for fop in fundamental_operators:
        dag, idxs = op_serialize_fundamental(fop)
        op_trans_dict[(dag, idxs)] = c_transf(*idxs)
        op_trans_dict[(not dag, idxs)] = dagger(c_transf(*idxs))
            
    # -- Loop over given operator and substitute operators
    # -- fundamental_operators with the transformed operators
    
    op_trans = Operator()
    
    for term in op:
        op_factor = Operator(1.)
        for factor in term:
            if type(factor) is list:
                for dag, idxs in factor:
                    op_factor *= op_trans_dict[(dag, tuple(idxs))]
                    
            else: # constant prefactor
                op_factor *= factor

        op_trans += op_factor

    return op_trans

# ----------------------------------------------------------------------
