"""
Lentz soliton warp drive (2021).

Erik Lentz proposed a class of soliton-like warp drive solutions
that claim to satisfy energy conditions using hyperbolic relations
between metric components.

The key insight is to construct the metric so that the stress-energy
tensor has a specific form that can be sourced by known matter fields
(electromagnetic and plasma).

[ASSUMPTION] The exact parameterization in Lentz's paper is complex.
This implementation follows the general structure but specific
numerical coefficients may differ. Points marked [ASSUMPTION] indicate
where interpretation was necessary.

Reference:
    Lentz, E. W. (2021). "Breaking the warp barrier: hyper-fast solitons
    in Einstein-Maxwell-plasma theory." Classical and Quantum Gravity,
    38(7), 075015.
    arXiv:2006.07125
"""

import numpy as np
from typing import Callable, Dict, Tuple
from warpbubblesim.metrics.base import WarpMetric, tanh_shape


class LentzMetric(WarpMetric):
    """
    Lentz soliton warp drive.

    The metric is constructed using a soliton-like profile that
    maintains hyperbolic relations between components.

    From the paper, the metric ansatz uses:
    ds² = -N² dt² + (dx - β^x dt)² + (dy - β^y dt)² + (dz - β^z dt)²

    with specific relationships between N and β to satisfy
    energy conditions.

    Parameters
    ----------
    v0 : float
        Soliton velocity.
    R : float
        Soliton width parameter.
    sigma : float
        Transition sharpness.
    amplitude : float
        Metric perturbation amplitude.
    x0 : float
        Initial position.

    [ASSUMPTION] Parameters are our interpretation of Lentz's construction.
    """

    @property
    def name(self) -> str:
        return "Lentz Soliton Warp Drive"

    @property
    def citation(self) -> str:
        return "Lentz2021"

    def _default_params(self) -> dict:
        return {
            'v0': 0.5,           # Soliton velocity
            'R': 1.0,            # Width
            'sigma': 5.0,        # Sharpness
            'amplitude': 0.1,    # Perturbation amplitude [ASSUMPTION]
            'x0': 0.0,
            # Lentz-specific parameters
            'hyperbolic_param': 0.5,  # [ASSUMPTION] Controls hyperbolic relation
        }

    def soliton_profile(self, r: float) -> float:
        """
        Soliton-like profile function.

        Lentz uses sech-type profiles for the soliton.
        φ(r) = sech(r/R)

        Parameters
        ----------
        r : float
            Distance from soliton center.

        Returns
        -------
        float
            Profile value.
        """
        R = self.params['R']
        # sech(x) = 1/cosh(x)
        return 1.0 / np.cosh(r / R)

    def soliton_profile_derivative(self, r: float, h: float = 1e-6) -> float:
        """Compute dφ/dr numerically."""
        return (self.soliton_profile(r + h) - self.soliton_profile(r - h)) / (2 * h)

    def shape_function(self, r: float) -> float:
        """
        Shape function based on soliton profile.

        [ASSUMPTION] We use the soliton profile as the shape function.
        """
        return self.soliton_profile(r)

    def lapse_function(self, r: float) -> float:
        """
        Lapse function with soliton modification.

        From Lentz's construction, the lapse and shift are related
        hyperbolically to satisfy energy conditions.

        N² = 1 + ε φ(r) where ε controls the perturbation.

        [ASSUMPTION] Specific form adapted from paper's general structure.
        """
        amp = self.params['amplitude']
        v0 = self.params['v0']
        h_param = self.params['hyperbolic_param']

        phi = self.soliton_profile(r)

        # Hyperbolic modification from Lentz
        # N² = 1 + ε v² φ² cosh(h)
        N_sq = 1 + amp * v0**2 * phi**2 * np.cosh(h_param)

        return np.sqrt(max(N_sq, 0.01))  # Protect against numerical issues

    def lapse(self, t: float, x: float, y: float, z: float) -> float:
        """Lapse function at spacetime point."""
        r_s = self.r_from_center(t, x, y, z)
        return self.lapse_function(r_s)

    def shift_magnitude(self, r: float) -> float:
        """
        Shift vector magnitude.

        From Lentz, the shift is related to the soliton velocity
        and profile with hyperbolic scaling.

        |β| = v φ(r) * tanh(h) * correction

        [ASSUMPTION] Specific form adapted from paper.
        """
        v0 = self.params['v0']
        amp = self.params['amplitude']
        h_param = self.params['hyperbolic_param']

        phi = self.soliton_profile(r)

        # Hyperbolic shift relation
        beta_mag = v0 * phi * np.tanh(h_param) * (1 + amp * phi)

        return beta_mag

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Shift vector for Lentz soliton.

        The shift points in the direction of soliton motion (x-direction)
        with magnitude determined by the soliton profile.
        """
        r_s = self.r_from_center(t, x, y, z)
        beta_mag = self.shift_magnitude(r_s)

        # Shift points in -x direction for forward motion
        return np.array([-beta_mag, 0.0, 0.0])

    def metric(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        """
        Full 4-metric for Lentz soliton.

        ds² = -N² dt² + (dx - β^x dt)² + dy² + dz²

        [ASSUMPTION] Using flat spatial metric as base.
        """
        N = self.lapse(t, x, y, z)
        beta = self.shift(t, x, y, z)

        g = np.zeros((4, 4))

        # g_{00} = -N² + β_i β^i = -N² + |β|²
        beta_sq = np.dot(beta, beta)
        g[0, 0] = -N**2 + beta_sq

        # g_{0i} = β_i (for flat spatial metric, β_i = β^i)
        g[0, 1:] = beta
        g[1:, 0] = beta

        # g_{ij} = δ_{ij}
        g[1, 1] = 1.0
        g[2, 2] = 1.0
        g[3, 3] = 1.0

        return g

    def em_field_estimate(self, t: float, x: float, y: float, z: float) -> Dict:
        """
        Estimate the EM field configuration needed.

        Lentz proposes sourcing the metric with EM and plasma fields.
        This provides rough estimates of field strengths needed.

        [ASSUMPTION] This is a heuristic based on energy density requirements.

        Parameters
        ----------
        t, x, y, z : float
            Spacetime point.

        Returns
        -------
        dict
            Estimated EM field properties.
        """
        r_s = self.r_from_center(t, x, y, z)
        phi = self.soliton_profile(r_s)
        v0 = self.params['v0']
        amp = self.params['amplitude']

        # EM energy density scales as B²/(8π) + E²/(8π)
        # To source the metric perturbation, need ρ_EM ~ amp * v² * phi²

        rho_estimate = amp * v0**2 * phi**2 / (8 * np.pi)

        # B² ~ 8π ρ_EM
        B_sq_estimate = 8 * np.pi * abs(rho_estimate)
        B_estimate = np.sqrt(B_sq_estimate)

        return {
            'energy_density_required': rho_estimate,
            'magnetic_field_estimate': B_estimate,
            'soliton_profile': phi,
            'distance_from_center': r_s,
        }

    def plasma_properties(self) -> Dict:
        """
        Describe the plasma properties needed.

        [ASSUMPTION] Based on Lentz's discussion of Einstein-Maxwell-plasma.
        """
        v0 = self.params['v0']

        return {
            'description': (
                "Lentz's soliton requires a specific plasma configuration "
                "with EM fields to source the metric. The plasma must have "
                "properties that create the required stress-energy tensor."
            ),
            'velocity': v0,
            'assumption_note': (
                "[ASSUMPTION] Exact plasma parameters depend on detailed "
                "field configurations not fully specified in this implementation."
            ),
        }


class LentzSimplified(WarpMetric):
    """
    Simplified Lentz metric for pedagogical purposes.

    Uses the basic structure of Lentz's soliton but with simpler
    parameterization for clearer demonstration.

    [ASSUMPTION] This is a teaching simplification.
    """

    @property
    def name(self) -> str:
        return "Lentz Soliton (Simplified)"

    @property
    def citation(self) -> str:
        return "Lentz2021"

    def _default_params(self) -> dict:
        return {
            'v0': 0.3,
            'R': 2.0,
            'sigma': 3.0,
            'x0': 0.0,
        }

    def shape_function(self, r: float) -> float:
        """Simplified sech profile."""
        R = self.params['R']
        return 1.0 / np.cosh(r / R)

    def shift(self, t: float, x: float, y: float, z: float) -> np.ndarray:
        v_s = self.bubble_velocity(t)
        r_s = self.r_from_center(t, x, y, z)
        f = self.shape_function(r_s)
        return np.array([-v_s * f, 0.0, 0.0])


class LentzFieldSourcing:
    """
    Utility class for computing the field sources for Lentz soliton.

    Given a metric configuration, computes the required electromagnetic
    and plasma fields to source it via Einstein-Maxwell-plasma equations.

    [ASSUMPTION] This is interpretive based on Lentz's paper.
    """

    def __init__(self, metric: LentzMetric):
        """
        Initialize with a Lentz metric instance.

        Parameters
        ----------
        metric : LentzMetric
            The soliton metric to analyze.
        """
        self.metric = metric

    def compute_required_em_field(
        self,
        coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the EM field (E, B) needed at a point.

        This is approximate and based on matching the stress-energy tensor.

        Parameters
        ----------
        coords : np.ndarray
            Spacetime coordinates [t, x, y, z].

        Returns
        -------
        tuple
            (E_field, B_field) as 3-vectors.
        """
        t, x, y, z = coords

        em_info = self.metric.em_field_estimate(t, x, y, z)
        B_mag = em_info['magnetic_field_estimate']

        # [ASSUMPTION] Field is predominantly magnetic, along y-axis
        E = np.array([0.0, 0.0, 0.0])
        B = np.array([0.0, B_mag, 0.0])

        return E, B

    def compute_current_density(
        self,
        coords: np.ndarray
    ) -> np.ndarray:
        """
        Compute the current density j^μ needed for plasma sourcing.

        This satisfies ∇_ν F^{μν} = 4π j^μ (Maxwell equations with sources).

        [ASSUMPTION] Simplified estimation.

        Parameters
        ----------
        coords : np.ndarray
            Spacetime coordinates.

        Returns
        -------
        np.ndarray
            Current 4-vector j^μ.
        """
        # Simplified: assume small current for slowly varying fields
        return np.zeros(4)

    def energy_condition_analysis(
        self,
        n_points: int = 50
    ) -> Dict:
        """
        Analyze energy conditions for the Lentz configuration.

        Parameters
        ----------
        n_points : int
            Number of radial points to sample.

        Returns
        -------
        dict
            Analysis results.
        """
        R = self.metric.params['R']

        r_values = np.linspace(0, 5*R, n_points)

        em_energies = []
        for r in r_values:
            coords = np.array([0.0, r, 0.0, 0.0])
            em_info = self.metric.em_field_estimate(0.0, r, 0.0, 0.0)
            em_energies.append(em_info['energy_density_required'])

        em_energies = np.array(em_energies)

        return {
            'r_values': r_values,
            'em_energy_density': em_energies,
            'min_energy_density': np.min(em_energies),
            'max_energy_density': np.max(em_energies),
            'all_positive': np.all(em_energies >= -1e-15),
            'note': (
                "Lentz claims positive energy can be achieved through "
                "proper EM+plasma sourcing. This analysis shows the "
                "estimated required energy densities."
            ),
        }
