"""
Sparse matrix representation of fermionic creation
and annihilation operators for a finite Fock space.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
"""

# ----------------------------------------------------------------------

import numpy as np

from scipy import sparse

# ----------------------------------------------------------------------
class SparseMatrixRepresentation(object):

    """ Generator for sparse matrix representations of
    Triqs operator expressions, given a set of fundamental
    creation operators. """

    # ------------------------------------------------------------------
    def __init__(self, fundamental_operators):

        self.fundamental_operators = fundamental_operators

        self.operator_labels = []
        for operator_expression in fundamental_operators:
            for term, coeff in operator_expression:
                assert( coeff == 1 )
                assert( len(term) == 1 )
                label = term[0]
                dag, idx = label
                label = (False, tuple(idx)) # only store annihilation operators
                self.operator_labels.append(label)

        # remove operator repetitions
        #self.operator_labels = list(set(self.operator_labels))
        # The set operator changes order!

        # Check for repeated operators
        operator_labels_set = list(set(self.operator_labels))

        assert len(operator_labels_set) == len(self.operator_labels), \
            "ERROR: Repeated operators in fundamental_operators!"

        self.operator_labels = [
            (dag, list(idx)) for dag, idx in self.operator_labels ]

        self.nfermions = len(self.operator_labels)
        self.sparse_operators = \
            SparseMatrixCreationOperators(self.nfermions)

    # ------------------------------------------------------------------
    def sparse_matrix(self, triqs_operator_expression):

        """ Convert a general Triqs operator expression to a sparse
        matrix representation. """

        matrix_rep = 0.0 * self.sparse_operators.I

        for term, coef in triqs_operator_expression:

            product = coef * self.sparse_operators.I

            for fact in term:

                dagger, idx = fact
                oidx = self.operator_labels.index((False, idx))

                op = self.sparse_operators.c_dag[oidx]
                if not dagger: op = op.getH()

                product = product * op

            matrix_rep = matrix_rep + product

        return matrix_rep

    # ------------------------------------------------------------------
    def sparse_sympy_matrix(self, triqs_operator_expression):

        Hsp = self.sparse_matrix(triqs_operator_expression)

        from sympy.matrices import SparseMatrix
        from sympy.simplify.simplify import nsimplify

        d = dict([ ((i, j), nsimplify(val)) \
                   for (i, j), val in Hsp.todok().items() ])

        H = SparseMatrix(Hsp.shape[0], Hsp.shape[1], d)

        return H

# ----------------------------------------------------------------------
class SparseMatrixCreationOperators:

    """ Generator of sparse matrix representation of fermionic
    creation operators, for finite number of fermions. """

    # ------------------------------------------------------------------
    def __init__(self, nfermions):

        self.nfermions = nfermions
        self.nstates = 2**nfermions

        self.c_dag = []
        for fidx in range(nfermions):
            c_dag_fidx = self._build_creation_operator(fidx)
            self.c_dag.append(c_dag_fidx)

        self.I = sparse.eye(
            self.nstates, self.nstates, dtype=np.float64, format='csr')

    # ------------------------------------------------------------------
    def _build_creation_operator(self, orbidx):
        nstates = self.nstates

        # -- Make python based fock states
        numbers = np.arange(nstates, dtype=np.uint32)
        tmp = numbers.flatten().view(np.uint8).reshape((nstates, 4))
        tmp = np.fliplr(tmp)
        states = np.unpackbits(tmp, axis=1)

        # -- Apply creation operator
        orbocc = states[:, -1 - orbidx]
        rightstates = states[:, -1 - orbidx:]

        states_new = np.copy(states)
        states_new[:, -1 - orbidx] = 1

        # -- collect sign
        sign = 1 - 2*np.array(
            np.mod(np.sum(rightstates[:, 1:], axis=1), 2),
            dtype=np.float64)

        # -- Transform back to uint16
        tmp_new = np.packbits(states_new, axis=1)
        tmp_new = np.fliplr(tmp_new).flatten()
        numbers_new = tmp_new.view(dtype=np.uint32)

        # -- Collect non-zero elements
        idx = orbocc == 0
        I = numbers_new[idx]
        J = numbers[idx]
        D = sign[idx]

        # -- Build sparse matrix repr.
        cdagger = sparse.coo_matrix((D,(I,J)), \
            shape=(nstates, nstates)).tocsr()

        return cdagger

# ----------------------------------------------------------------------
