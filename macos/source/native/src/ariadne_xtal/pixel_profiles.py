"""Pixel-integrated separable Gaussian profiles with analytic perturbations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class GaussianPixels:
    lower: np.ndarray
    upper: np.ndarray
    centres: np.ndarray
    sigma: np.ndarray

    def __post_init__(self):
        for name in ("lower", "upper", "centres", "sigma"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 2 or not value.size or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a nonempty, finite matrix")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (self.lower.shape != self.upper.shape or self.centres.shape != self.sigma.shape
                or self.lower.shape[1] != self.centres.shape[1] or not 1 <= self.ndim <= 3
                or np.any(self.upper <= self.lower) or np.any(self.sigma <= 0)):
            raise ValueError("pixel edges and positive peak widths must match one to three axes")

    @property
    def ndim(self):
        return self.lower.shape[1]

    def evaluate(self, parameters, *, derivatives=False):
        """Return the full-profile fraction and optional shift/log-width derivatives."""
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.ndim + 1,) or not np.isfinite(values).all():
            raise ValueError("expected a shift per axis and one finite log-width change")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            width = self.sigma * np.exp(values[-1])
            centre = self.centres + values[:-1]
            lower = (self.lower[:, None, :] - centre) / width
            upper = (self.upper[:, None, :] - centre) / width
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or not np.isfinite(width).all() or np.any(width <= 0)):
            raise ValueError("pixel profile is outside the supported numerical range")
        # Evaluate the far positive tail in the negative half-plane.
        reflect = lower > 0
        masses = ndtr(np.where(reflect, -lower, upper)) - ndtr(np.where(reflect, -upper, lower))
        fractions = masses.prod(axis=2)
        if not derivatives:
            return fractions
        normalizer = np.sqrt(2 * np.pi)
        low_density = np.exp(-.5 * lower**2) / normalizer
        high_density = np.exp(-.5 * upper**2) / normalizer
        jacobian = np.zeros((*fractions.shape, self.ndim + 1))
        for axis in range(self.ndim):
            other_axes = masses[:, :, :axis].prod(axis=2) * masses[:, :, axis+1:].prod(axis=2)
            low, high = low_density[:, :, axis], high_density[:, :, axis]
            jacobian[:, :, axis] = other_axes * (low - high) / width[:, axis]
            jacobian[:, :, -1] += other_axes * (lower[:, :, axis] * low - upper[:, :, axis] * high)
        return fractions, jacobian
