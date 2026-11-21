"""Linear-operator Jacobian over disjoint row blocks."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .derivatives import right_singular_blocks


class BlockJacobian(LinearOperator):
    def __init__(self, diffraction_blocks, diffraction_rows, npar, restraints):
        self.diffraction_blocks = diffraction_blocks
        self.diffraction_rows = diffraction_rows
        self.restraints = restraints
        super().__init__(dtype=np.dtype('float64'), shape=(diffraction_rows + len(restraints), npar))

    def blocks(self):
        yield from self.diffraction_blocks()
        if len(self.restraints):
            yield slice(self.diffraction_rows, self.shape[0]), self.restraints

    def _matvec(self, v):
        return self._matmat(np.asarray(v).reshape(-1, 1))[:, 0]

    def _matmat(self, matrix):
        result = np.empty((self.shape[0], matrix.shape[1]))
        for rows, block in self.blocks():
            result[rows] = block @ matrix
        return result

    def _rmatvec(self, v):
        result = np.zeros(self.shape[1])
        v = np.asarray(v).reshape(-1)
        for rows, block in self.blocks():
            result += block.T @ v[rows]
        return result

    def column_scaling(self):
        norm = np.zeros(self.shape[1])
        for _, block in self.blocks():
            norm = np.hypot(norm, np.linalg.norm(block, axis=0))
        if not np.isfinite(norm).all():
            raise ValueError('Jacobian column scale is outside the supported numeric range')
        return np.divide(1., norm, out=np.ones_like(norm), where=norm > np.finfo(float).tiny)

    def right_basis(self, include_restraints=True):
        blocks = self.blocks() if include_restraints else self.diffraction_blocks()
        return right_singular_blocks((block for _, block in blocks), self.shape[1])
