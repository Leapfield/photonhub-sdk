"""Full-vector eigenmode-expansion (EME) propagator — CPU, frequency-domain.

A device is represented by z-invariant :class:`Section` objects.  Each section
propagates its *raw* local eigenmodes diagonally, adjacent sections are matched by
independent tangential-E and tangential-H continuity equations, and rectangular
S-matrices are cascaded with the Redheffer star product.

The interface uses the unconjugated Lorentz/reaction product
``integral (E x H).z dA``.  It constructs reaction-adjoint test traces while
leaving the primal eigenvectors untouched, then solves a two-sided
Petrov-Galerkin system with a rank-revealing SVD.  Adjacent sections may have
different mode counts.  Truncation appears as a reported field-continuity
residual; losslessness is never fabricated by forcing a unitary matrix.

Validated full-box uniform engine-native Yee modes are matched without
transverse co-location: ``Ex``/``Hy`` and ``Ey``/``Hx`` already share their
respective Yee locations.  Generic Yee port, graded, and symmetry-reduced modes
do not carry the required reaction-mass provenance and are rejected.  Every
mode on both sides must use the same registration; mixing Yee-staggered and
node-collocated traces is rejected.

``VectorModeSolver.solve()`` remains the fast guided-only path.
``VectorModeSolver.solve_eme_basis()`` adds PML-discretised radiation and
evanescent channels from one common complex-coordinate operator.  Reaction
matching follows the complex contour, while user-visible power uses a separate
Hermitian Poynting metric over the non-PML physical region.  Consequently
``|amplitude|^2`` is not assumed to be power for evanescent or PML modes.
This continuum path is **experimental**: analytic operator checks pass, but the
current high-contrast all-bound-port device sweep is not stable under nested
continuum-shell refinement.  Per-interface nondegenerate reaction-orthogonality
and bidirectional guided-input residuals expose the present high-order trace
inconsistency.  Do not use it for quantitative radiation loss without
independent mesh/window/PML/basis and FDTD convergence.

EME uses ``exp(+i*omega*t - i*beta_eme*z)`` with
``beta_eme = k0*mode.n_eff_complex.conjugate()``.  Its propagation factor is
therefore ``exp(-i*k0*n_eff*L-k0*k_eff*L)``.  ``n_eff_complex`` itself remains
positive-loss engine/eigensolver metadata; inserting it directly into the EME
exponential would produce growth. Conjugate complex S amplitudes when comparing
phase to engine DFT data.

All modes at an interface must share the same transverse grid and complex PML
contour.  Cross-grid and cross-contour interpolation are intentionally rejected.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .vector_modes import VectorMode, VectorModeSolver

__all__ = [
    "Section",
    "EMEResult",
    "InterfaceDiagnostics",
    "EMEConvergenceReport",
    "interface_smatrix",
    "eme_convergence_report",
    "propagation_smatrix",
    "star_product",
    "cascade",
    "run_eme",
    "waveguide_section",
    "rectangular_base_section",
    "run_eme_band",
]

#: An S-matrix as the block 4-tuple ``(S11, S12, S21, S22)``.
SMatrix = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class Section:
    """One z-invariant cross-section of an EME device.

    Attributes
    ----------
    modes:
        The eigenmodes of this cross-section.  A guided-only tuple from
        :meth:`VectorModeSolver.solve` remains valid; a complete EME basis from
        :meth:`VectorModeSolver.solve_eme_basis` may additionally contain
        experimental radiation and evanescent modes.
    length_um:
        Physical length of the section along propagation (microns). ``0.0`` marks
        a **port / semi-infinite lead** — it contributes its interface with its
        neighbour but no propagation phase.
    """

    modes: Sequence[VectorMode]
    length_um: float = 0.0


@dataclass(frozen=True)
class InterfaceDiagnostics:
    """Numerical evidence for one generalized mode-matched interface.

    ``*_reaction_orthogonality_error`` is the largest self-normalized reaction
    overlap between modes with distinct propagation constants.  It complements
    Gram condition/asymmetry: a large *symmetric* off-diagonal overlap can expose
    an inaccurate high-order E/H reconstruction while leaving both of those
    older diagnostics deceptively modest. ``max_guided_input_residual`` filters
    the continuity residual to incident guided channels on either side, so a
    poor continuum test channel cannot hide the accuracy of the physical ports.
    """

    n_left: int
    n_right: int
    rank: int
    equations: int
    condition_number: float
    raw_residual_by_input: np.ndarray
    residual_by_input: np.ndarray
    raw_passivity_violation: float
    passivity_violation: float
    passivity_correction_applied: bool
    reciprocity_error: float
    left_gram_condition: float
    right_gram_condition: float
    left_gram_asymmetry: float
    right_gram_asymmetry: float
    left_reaction_orthogonality_error: float = 0.0
    right_reaction_orthogonality_error: float = 0.0
    max_guided_input_residual: float = 0.0


@dataclass
class EMEResult:
    """Device scattering matrix from :func:`run_eme`.

    Blocks may be rectangular: ``S21`` is ``N_right x N_left``.  Physical power
    is evaluated with Hermitian Poynting metrics, so evanescent/PML amplitudes
    are never mistaken for ``|a|^2`` power channels. ``n_modes`` retains its
    legacy meaning as the minimum section-basis count; use ``n_left_modes``,
    ``n_right_modes``, and ``section_mode_counts`` for the rectangular model.
    """

    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray
    n_modes: int
    n_left_modes: int = 0
    n_right_modes: int = 0
    section_mode_counts: Tuple[int, ...] = ()
    left_modes: Tuple[VectorMode, ...] = ()
    right_modes: Tuple[VectorMode, ...] = ()
    left_power_metric: Optional[np.ndarray] = None
    right_power_metric: Optional[np.ndarray] = None
    interface_diagnostics: Tuple[InterfaceDiagnostics, ...] = ()

    def __post_init__(self) -> None:
        if self.n_left_modes == 0:
            self.n_left_modes = int(self.s11.shape[0])
        if self.n_right_modes == 0:
            self.n_right_modes = int(self.s22.shape[0])
        if self.left_power_metric is None:
            self.left_power_metric = np.eye(self.n_left_modes, dtype=complex)
        if self.right_power_metric is None:
            self.right_power_metric = np.eye(self.n_right_modes, dtype=complex)

    @staticmethod
    def _metric_power(amplitudes: np.ndarray, metric: np.ndarray) -> float:
        value = float(np.real(np.vdot(amplitudes, metric @ amplitudes)))
        return 0.0 if -1e-12 < value < 0.0 else value

    def _incident_power(self, input_mode: int) -> float:
        assert self.left_power_metric is not None
        if not 0 <= input_mode < self.n_left_modes:
            raise IndexError("input_mode is outside the left port basis")
        if (
            len(self.left_modes) == self.n_left_modes
            and self.left_modes[input_mode].mode_type == "evanescent"
        ):
            raise ValueError(
                "the selected input is evanescent; a propagating power ratio "
                "is undefined even if a finite-box/PML flux metric is nonzero"
            )
        a = np.zeros(self.n_left_modes, dtype=complex)
        a[input_mode] = 1.0
        power = self._metric_power(a, self.left_power_metric)
        if power <= 1e-12:
            raise ValueError(
                "the selected input is evanescent/zero-flux; a power ratio is "
                "undefined"
            )
        return power

    @property
    def transmission(self) -> float:
        """Fundamental-to-fundamental physical power transmission.

        Index-based, so it follows ``n_eff`` ordering and NOT polarization —
        see :meth:`_fundamental_index`. Warns when the two ports disagree on
        the polarization at index 0. Prefer :meth:`transmission_of`
        (``result.transmission_of("TE")``) whenever the ports can reorder,
        which is the normal case for a taper.
        """
        self._warn_if_ports_reordered("transmission")
        assert self.right_power_metric is not None
        pin = self._incident_power(0)
        return float(
            np.abs(self.s21[0, 0]) ** 2
            * max(float(np.real(self.right_power_metric[0, 0])), 0.0)
            / pin
        )

    @property
    def reflection(self) -> float:
        """Fundamental-to-fundamental physical power reflection."""
        assert self.left_power_metric is not None
        pin = self._incident_power(0)
        return float(
            np.abs(self.s11[0, 0]) ** 2
            * max(float(np.real(self.left_power_metric[0, 0])), 0.0)
            / pin
        )

    def _fundamental_index(
        self, modes: Sequence[VectorMode], polarization: Optional[str]
    ) -> int:
        """Row/column of the fundamental mode of ``polarization`` in ``modes``.

        Index 0 is the highest-``n_eff`` mode of the port, which is NOT
        necessarily the polarization the device is meant to carry. Along a
        taper the ordering swaps: at a 160 nm tip mode 0 is TM (1.4704) and
        mode 1 is TE (1.4462), while at 450 nm mode 0 is TE. Reading
        ``S21[0, 0]`` across that swap measures TM->TE conversion and reports
        ~1e-7 (62 dB "loss") for a taper whose true TE->TE transmission is
        0.98-0.999.
        """
        if not modes or polarization is None:
            return 0
        want = str(polarization).upper()
        for i, m in enumerate(modes):
            if str(getattr(m, "polarization", "")).upper() == want:
                return i
        raise ValueError(
            f"no {want} mode in this EME port basis "
            f"(polarizations: "
            f"{[getattr(m, 'polarization', None) for m in modes]}); "
            "widen num_modes or drop the polarization filter"
        )

    def transmission_of(self, polarization: Optional[str] = None) -> float:
        """Power transmission between the fundamental ``polarization`` modes
        of the two ports, resolved by mode identity rather than by index.

        Metric-aware like :attr:`transmission` (physical power ratio);
        ``polarization=None`` reproduces it exactly.
        """
        i = self._fundamental_index(self.left_modes, polarization)
        j = self._fundamental_index(self.right_modes, polarization)
        assert self.right_power_metric is not None
        pin = self._incident_power(i)
        return float(
            np.abs(self.s21[j, i]) ** 2
            * max(float(np.real(self.right_power_metric[j, j])), 0.0)
            / pin
        )

    def _warn_if_ports_reordered(self, what: str) -> None:
        """Warn when index 0 means a different polarization at each port.

        That is exactly the condition under which an index-based read is
        silently wrong, and it is detectable from the port bases alone — so
        it is reported rather than left for the user to discover from an
        implausible number.
        """
        if not self.left_modes or not self.right_modes:
            return
        a = str(getattr(self.left_modes[0], "polarization", "") or "").upper()
        b = str(getattr(self.right_modes[0], "polarization", "") or "").upper()
        if a and b and a != b:
            warnings.warn(
                f"EME {what} reads S-matrix element [0, 0], but the ports' "
                f"fundamental modes have different polarizations "
                f"(input {a}, output {b}) — n_eff ordering swapped along the "
                f"device, so this measures {a}->{b} conversion, not "
                f"{a}->{a} throughput. Use "
                f"result.{what}_of('{a}') to resolve by mode identity.",
                RuntimeWarning,
                stacklevel=3,
            )

    def transmitted_power(self, input_mode: int = 0) -> float:
        """Total physical output-port power from ``input_mode``."""
        assert self.right_power_metric is not None
        return self._metric_power(
            self.s21[:, input_mode], self.right_power_metric
        ) / self._incident_power(input_mode)

    def reflected_power(self, input_mode: int = 0) -> float:
        """Total physical reflected power from ``input_mode``."""
        assert self.left_power_metric is not None
        return self._metric_power(
            self.s11[:, input_mode], self.left_power_metric
        ) / self._incident_power(input_mode)

    def energy_balance(self, input_mode: int = 0) -> float:
        """``T_total + R_total`` for ``input_mode``.

        Unlike the former forced-unitary interface, this is a physical flux
        diagnostic: radiation/PML attenuation or an incomplete trace basis may
        produce a deficit.  A value above one is a passivity/convergence failure.
        """
        return self.transmitted_power(input_mode) + self.reflected_power(input_mode)


@dataclass(frozen=True)
class EMEConvergenceReport:
    """Successive fixed-port stability evidence for EME basis/PML sweeps.

    ``smatrix_stable`` means that the requested number of consecutive,
    basis-aligned propagating-port S-matrix deltas and port-subspace errors are
    within ``tolerance``.  It deliberately does **not** mean that the underlying
    field expansion is physically converged.  Interface continuity,
    reciprocity, passivity/energy balance, mesh, window, and PML sweeps remain
    independent gates.
    """

    labels: Tuple[str, ...]
    section_mode_counts: Tuple[Tuple[int, ...], ...]
    port_smatrix_deltas: Tuple[float, ...]
    port_subspace_errors: Tuple[float, ...]
    energy_balances: Tuple[float, ...]
    max_interface_residuals: Tuple[float, ...]
    max_reciprocity_errors: Tuple[float, ...]
    tolerance: float
    stability_steps: int
    smatrix_stable: bool
    max_guided_interface_residuals: Tuple[float, ...] = ()
    max_reaction_orthogonality_errors: Tuple[float, ...] = ()


def eme_convergence_report(
    results: Sequence[EMEResult],
    *,
    labels: Optional[Sequence[str]] = None,
    port_left: int = 1,
    port_right: int = 1,
    tolerance: float = 1e-3,
    stability_steps: int = 2,
) -> EMEConvergenceReport:
    """Compare a nested sweep on one fixed propagating external-port subspace.

    Radiation eigenvectors reorder when the window or PML changes, so comparing
    their individual columns is meaningless.  Propagating eigenvectors can likewise
    phase-flip or rotate inside a degenerate subspace.  On a common external-port
    grid this report least-squares aligns each current port basis to the
    previous one before comparing the four complex S blocks.  A requested guided
    subset may not split a degenerate beta cluster.

    ``energy_balances`` is the worst-case generalized output/input power ratio
    over the selected left port subspace, not the coordinate-dependent result
    of exciting whichever degenerate vector happened to be returned first.
    Field-continuity (both all-channel and guided-input maxima), reaction
    orthogonality, and generalized reciprocity are exposed separately; they are
    not folded into ``smatrix_stable`` because useful thresholds depend on the
    problem and retained channel types.

    ``stability_steps`` prevents one accidental flat pair from being declared
    stable.  The default requires two consecutive sub-tolerance deltas, hence
    at least three results.  Use ``stability_steps=1`` only for an exact
    algebraic identity check, not a numerical convergence claim.
    """
    if len(results) < 2:
        raise ValueError("convergence reporting needs at least two EME results")
    if port_left < 1 or port_right < 1:
        raise ValueError("port_left and port_right must be >= 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be > 0")
    if int(stability_steps) != stability_steps or stability_steps < 1:
        raise ValueError("stability_steps must be an integer >= 1")
    stability_steps = int(stability_steps)
    if labels is None:
        label_tuple = tuple(f"level-{i}" for i in range(len(results)))
    else:
        if len(labels) != len(results):
            raise ValueError("labels must have one entry per result")
        label_tuple = tuple(str(x) for x in labels)

    def selected_port_modes(
        r: EMEResult, *, side: str, count: int,
    ) -> Tuple[VectorMode, ...]:
        modes = r.left_modes if side == "left" else r.right_modes
        available_count = r.n_left_modes if side == "left" else r.n_right_modes
        if available_count < count or len(modes) < count:
            raise ValueError(
                f"a result has fewer {side} external port modes than requested"
            )
        selected = tuple(modes[:count])
        if any(mode.mode_type == "evanescent" for mode in selected):
            raise ValueError(
                f"the selected {side} external-port modes must carry "
                "propagating power, not evanescent channels"
            )
        if (
            count < len(modes)
            and modes[count].mode_type == selected[-1].mode_type
        ):
            last = selected[-1].n_eff_complex
            next_beta = modes[count].n_eff_complex
            if abs(last - next_beta) <= 1e-6 * max(
                abs(last), abs(next_beta), 1.0
            ):
                raise ValueError(
                    f"port_{side}={count} splits a degenerate external-port "
                    "mode cluster; include the complete multiplet"
                )
        return selected

    def fixed_blocks(r: EMEResult) -> Tuple[np.ndarray, ...]:
        if r.n_left_modes < port_left or r.n_right_modes < port_right:
            raise ValueError("a result has fewer external port modes than requested")
        return (
            r.s11[:port_left, :port_left],
            r.s12[:port_left, :port_right],
            r.s21[:port_right, :port_left],
            r.s22[:port_right, :port_right],
        )

    def basis_map(
        reference: Tuple[VectorMode, ...],
        current: Tuple[VectorMode, ...],
        *,
        side: str,
    ) -> Tuple[np.ndarray, float]:
        """Return C with ``current_fields ~= reference_fields @ C``."""
        reference0, current0 = reference[0], current[0]
        reference_offset = (
            (0.0, 0.0)
            if reference0.center_offset_um is None
            else tuple(reference0.center_offset_um)
        )
        current_offset = (
            (0.0, 0.0)
            if current0.center_offset_um is None
            else tuple(current0.center_offset_um)
        )
        same_grid = (
            reference0.shape == current0.shape
            and np.isclose(reference0.dl_x_um, current0.dl_x_um)
            and np.isclose(reference0.dl_y_um, current0.dl_y_um)
            and np.isclose(
                reference0.wavelength_um,
                current0.wavelength_um,
                rtol=1e-12,
                atol=1e-15,
            )
            and np.allclose(
                reference_offset, current_offset, rtol=0.0, atol=1e-12
            )
        )
        if not same_grid:
            raise ValueError(
                f"convergence alignment needs one fixed {side} external-port "
                "grid and wavelength across all results"
            )

        def stacks(
            modes: Tuple[VectorMode, ...],
        ) -> Tuple[np.ndarray, np.ndarray]:
            # EMEResult amplitudes refer to the reaction-normalized primal
            # traces built by _basis_trace, not to the arbitrary raw scaling or
            # phase of the public eigenvectors.
            trace = _basis_trace(
                modes,
                modes[0].dl_x_um,
                modes[0].dl_y_um,
                rcond=1e-10,
            )
            electric = np.concatenate((trace.ex, trace.ey), axis=1).T
            magnetic = np.concatenate((trace.hx, trace.hy), axis=1).T
            return electric, magnetic

        reference_e, reference_h = stacks(reference)
        current_e, current_h = stacks(current)

        def fit(
            reference_fields: np.ndarray,
            current_fields: np.ndarray,
            *,
            field_name: str,
        ) -> Tuple[np.ndarray, float]:
            q_reference, r_reference = np.linalg.qr(
                reference_fields, mode="reduced"
            )
            q_current, r_current = np.linalg.qr(
                current_fields, mode="reduced"
            )
            scale_reference = max(
                float(np.linalg.norm(r_reference, ord=2)), 1e-300
            )
            scale_current = max(
                float(np.linalg.norm(r_current, ord=2)), 1e-300
            )
            if (
                np.min(np.abs(np.diag(r_reference)))
                <= 1e-10 * scale_reference
                or np.min(np.abs(np.diag(r_current)))
                <= 1e-10 * scale_current
            ):
                raise ValueError(
                    f"the selected {side} port {field_name} trace "
                    "is rank deficient"
                )
            error = float(
                max(
                    np.linalg.norm(
                        q_current
                        - q_reference
                        @ (q_reference.conj().T @ q_current),
                        ord=2,
                    ),
                    np.linalg.norm(
                        q_reference
                        - q_current @ (q_current.conj().T @ q_reference),
                        ord=2,
                    ),
                )
            )
            mapping = np.linalg.lstsq(
                reference_fields, current_fields, rcond=1e-10
            )[0]
            if np.linalg.matrix_rank(mapping, tol=1e-10) < mapping.shape[0]:
                raise ValueError(
                    f"the successive {side} port {field_name} "
                    "basis map is singular"
                )
            return mapping, error

        mapping_e, error_e = fit(
            reference_e, current_e, field_name="electric"
        )
        mapping_h, error_h = fit(
            reference_h, current_h, field_name="magnetic"
        )
        map_disagreement = float(
            np.linalg.norm(mapping_e - mapping_h)
            / max(
                np.linalg.norm(mapping_e),
                np.linalg.norm(mapping_h),
                1e-300,
            )
        )
        subspace_error = max(error_e, error_h, map_disagreement)
        if subspace_error > 0.25:
            raise ValueError(
                f"successive {side} E/H port traces do not represent "
                "the same fixed external port and impedance"
            )
        mapping = 0.5 * (mapping_e + mapping_h)
        return mapping, subspace_error

    deltas: List[float] = []
    port_errors: List[float] = []
    for previous, current in zip(results[:-1], results[1:]):
        previous_left = selected_port_modes(
            previous, side="left", count=port_left
        )
        current_left = selected_port_modes(
            current, side="left", count=port_left
        )
        previous_right = selected_port_modes(
            previous, side="right", count=port_right
        )
        current_right = selected_port_modes(
            current, side="right", count=port_right
        )
        map_left, error_left = basis_map(
            previous_left, current_left, side="left"
        )
        map_right, error_right = basis_map(
            previous_right, current_right, side="right"
        )
        inverse_left = np.linalg.solve(
            map_left, np.eye(port_left, dtype=complex)
        )
        inverse_right = np.linalg.solve(
            map_right, np.eye(port_right, dtype=complex)
        )
        c11, c12, c21, c22 = fixed_blocks(current)
        aligned_current = (
            map_left @ c11 @ inverse_left,
            map_left @ c12 @ inverse_right,
            map_right @ c21 @ inverse_left,
            map_right @ c22 @ inverse_right,
        )
        deltas.append(
            max(
                float(np.max(np.abs(previous_block - current_block)))
                for previous_block, current_block in zip(
                    fixed_blocks(previous), aligned_current, strict=True
                )
            )
        )
        port_errors.append(max(error_left, error_right))

    def worst_case_energy_balance(r: EMEResult) -> float:
        assert r.left_power_metric is not None
        assert r.right_power_metric is not None
        incident_metric = np.asarray(
            r.left_power_metric[:port_left, :port_left],
            dtype=complex,
        )
        reflected = r.s11[:, :port_left]
        transmitted = r.s21[:, :port_left]
        output_metric = (
            reflected.conj().T @ r.left_power_metric @ reflected
            + transmitted.conj().T @ r.right_power_metric @ transmitted
        )
        eig, vec = np.linalg.eigh(
            0.5 * (incident_metric + incident_metric.conj().T)
        )
        if eig.size < port_left or eig[-1] <= 0.0:
            raise ValueError("selected left port has no positive power")
        keep = eig > 1e-10 * eig[-1]
        if np.count_nonzero(keep) < port_left:
            raise ValueError(
                "selected left port power metric is rank deficient"
            )
        whitening = vec[:, keep] / np.sqrt(eig[keep])[None, :]
        ratio = whitening.conj().T @ output_metric @ whitening
        return float(
            np.max(np.linalg.eigvalsh(0.5 * (ratio + ratio.conj().T)))
        )

    balances = tuple(worst_case_energy_balance(r) for r in results)
    residuals = tuple(
        max(
            (
                float(np.max(d.residual_by_input))
                for d in r.interface_diagnostics
            ),
            default=0.0,
        )
        for r in results
    )
    guided_residuals = tuple(
        max(
            (d.max_guided_input_residual for d in r.interface_diagnostics),
            default=0.0,
        )
        for r in results
    )
    orthogonality_errors = tuple(
        max(
            (
                max(
                    d.left_reaction_orthogonality_error,
                    d.right_reaction_orthogonality_error,
                )
                for d in r.interface_diagnostics
            ),
            default=0.0,
        )
        for r in results
    )
    reciprocal = tuple(
        max((d.reciprocity_error for d in r.interface_diagnostics), default=0.0)
        for r in results
    )
    return EMEConvergenceReport(
        labels=label_tuple,
        section_mode_counts=tuple(r.section_mode_counts for r in results),
        port_smatrix_deltas=tuple(deltas),
        port_subspace_errors=tuple(port_errors),
        energy_balances=balances,
        max_interface_residuals=residuals,
        max_guided_interface_residuals=guided_residuals,
        max_reaction_orthogonality_errors=orthogonality_errors,
        max_reciprocity_errors=reciprocal,
        tolerance=float(tolerance),
        stability_steps=stability_steps,
        smatrix_stable=bool(
            len(deltas) >= stability_steps
            and all(
                delta <= tolerance
                for delta in deltas[-stability_steps:]
            )
            and all(
                error <= tolerance
                for error in port_errors[-stability_steps:]
            )
        ),
    )


@dataclass(frozen=True)
class _BasisTrace:
    ex: np.ndarray
    ey: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    dual_ex: np.ndarray
    dual_ey: np.ndarray
    dual_hx: np.ndarray
    dual_hy: np.ndarray
    reaction_gram: np.ndarray
    power_metric: np.ndarray
    weights: np.ndarray
    gram_condition: float
    gram_asymmetry: float
    reaction_orthogonality_error: float


def _svd_pinv(
    matrix: np.ndarray, rcond: float,
) -> Tuple[np.ndarray, int, float, np.ndarray]:
    """Rank-revealing Moore-Penrose solve primitive (never normal equations)."""
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0:
        return np.zeros(matrix.T.shape, dtype=complex), 0, np.inf, singular
    cutoff = rcond * singular[0]
    keep = singular > cutoff
    rank = int(np.count_nonzero(keep))
    pinv = (vh.conj().T[:, keep] / singular[keep]) @ u.conj().T[keep, :]
    condition = (
        float(singular[0] / singular[keep][-1]) if rank else np.inf
    )
    return pinv, rank, condition, singular


def _reaction_overlap(
    ex: np.ndarray,
    ey: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    weights: np.ndarray,
    dA: float,
) -> np.ndarray:
    """Rows are E modes, columns are H modes; no complex conjugation."""
    w = weights.ravel()[None, :]
    return (ex @ (hy * w).T - ey @ (hx * w).T) * dA


def _basis_trace(
    modes: Sequence[VectorMode],
    dl_x_um: float,
    dl_y_um: float,
    *,
    rcond: float,
) -> _BasisTrace:
    if not modes:
        raise ValueError("a modal basis must contain at least one mode")
    shape = modes[0].shape
    for m in modes:
        if m.shape != shape:
            raise ValueError("all modes in one basis must share a grid")
    first_weights = modes[0].overlap_weights
    weights = (
        np.ones(shape, dtype=complex)
        if first_weights is None
        else np.asarray(first_weights, dtype=complex)
    )
    if weights.shape != shape:
        raise ValueError("mode overlap_weights must match the field grid")
    for m in modes[1:]:
        mw = (
            np.ones(shape, dtype=complex)
            if m.overlap_weights is None
            else np.asarray(m.overlap_weights, dtype=complex)
        )
        if mw.shape != shape or not np.allclose(mw, weights, rtol=1e-11, atol=1e-13):
            raise ValueError(
                "all modes in a section must share one complex-coordinate contour"
            )

    ex = np.array([m.ex.ravel() for m in modes], dtype=complex)
    ey = np.array([m.ey.ravel() for m in modes], dtype=complex)
    hx = np.array([m.hx.ravel() for m in modes], dtype=complex)
    hy = np.array([m.hy.ravel() for m in modes], dtype=complex)
    dA = dl_x_um * dl_y_um
    gram = _reaction_overlap(ex, ey, hx, hy, weights, dA)
    diagonal = np.diag(gram)
    if np.any(np.abs(diagonal) < rcond * max(np.max(np.abs(gram)), 1e-300)):
        raise ValueError(
            "self-orthogonal/zero-reaction mode in basis; exclude the beta≈0 "
            "or exceptional mode, or add its generalized adjoint chain"
        )
    scale = (1.0 / np.sqrt(diagonal))[:, None]
    ex, ey, hx, hy = ex * scale, ey * scale, hx * scale, hy * scale
    gram = _reaction_overlap(ex, ey, hx, hy, weights, dA)
    gram_pinv, rank, gram_condition, _ = _svd_pinv(gram, rcond)
    if rank != len(modes):
        raise ValueError(
            f"rank-deficient reaction Gram ({rank}/{len(modes)}); add/refine "
            "modes or remove a self-orthogonal/duplicate channel"
        )

    # Construct reaction-adjoint test traces in the retained modal subspace.
    # Only the dual rows are transformed: the primal eigenmodes (and therefore
    # diagonal beta propagation) remain untouched.
    dual_ex = gram_pinv @ ex
    dual_ey = gram_pinv @ ey
    dual_hx = gram_pinv.T @ hx
    dual_hy = gram_pinv.T @ hy

    # Hermitian physical Poynting metric.  Never integrate physical power along
    # the complex PML contour; exclude its artificial cells instead.
    mask = (
        np.ones(shape, dtype=float)
        if modes[0].physical_mask is None
        else np.asarray(modes[0].physical_mask, dtype=float)
    )
    for m in modes[1:]:
        mm = (
            np.ones(shape, dtype=float)
            if m.physical_mask is None
            else np.asarray(m.physical_mask, dtype=float)
        )
        if mm.shape != shape or not np.array_equal(mm, mask):
            raise ValueError("all modes in a section must share one physical PML mask")
    c = (ex @ (hy.conj() * mask.ravel()[None, :]).T
         - ey @ (hx.conj() * mask.ravel()[None, :]).T) * dA
    power_metric = 0.5 * (c.T + c.conj())
    power_metric = 0.5 * (power_metric + power_metric.conj().T)
    asymmetry = float(
        np.linalg.norm(gram - gram.T)
        / max(np.linalg.norm(gram), 1e-300)
    )
    # Distinct eigenvalues of a reciprocal waveguide operator should be
    # orthogonal under the unconjugated Lorentz/reaction product.  Degenerate
    # multiplets are intentionally excluded: an arbitrary basis rotation inside
    # one eigenspace can have off-diagonal Gram entries without losing span.
    beta = np.asarray([m.n_eff_complex for m in modes], dtype=complex)
    diagonal_scale = np.sqrt(
        np.abs(np.diag(gram))[:, None] * np.abs(np.diag(gram))[None, :]
    )
    beta_scale = np.maximum(
        np.maximum(np.abs(beta)[:, None], np.abs(beta)[None, :]),
        1.0,
    )
    nondegenerate = np.abs(beta[:, None] - beta[None, :]) > 1e-6 * beta_scale
    normalized_offdiagonal = np.divide(
        np.abs(gram),
        diagonal_scale,
        out=np.zeros_like(np.abs(gram), dtype=float),
        where=diagonal_scale > 0.0,
    )
    reaction_orthogonality_error = float(
        np.max(normalized_offdiagonal[nondegenerate], initial=0.0)
    )
    return _BasisTrace(
        ex=ex, ey=ey, hx=hx, hy=hy,
        dual_ex=dual_ex, dual_ey=dual_ey,
        dual_hx=dual_hx, dual_hy=dual_hy,
        reaction_gram=gram,
        power_metric=power_metric,
        weights=weights,
        gram_condition=gram_condition,
        gram_asymmetry=asymmetry,
        reaction_orthogonality_error=reaction_orthogonality_error,
    )


def _positive_metric_passivity_projection(
    scattering: np.ndarray,
    metric: np.ndarray,
    *,
    rcond: float,
    enforce: bool,
) -> Tuple[np.ndarray, float, float]:
    """Clip only gain in the positive-flux subspace; never force unitarity."""
    eig, vec = np.linalg.eigh(0.5 * (metric + metric.conj().T))
    if eig.size == 0 or eig[-1] <= 0.0:
        return scattering, 0.0, 0.0
    keep = eig > rcond * eig[-1]
    if not np.any(keep):
        return scattering, 0.0, 0.0
    q = vec[:, keep]
    w = eig[keep]
    sqrt_w = np.sqrt(w)
    invsqrt_w = 1.0 / sqrt_w
    reduced = (
        sqrt_w[:, None]
        * (q.conj().T @ scattering @ q)
        * invsqrt_w[None, :]
    )
    u, singular, vh = np.linalg.svd(reduced, full_matrices=False)
    raw_violation = float(max(0.0, singular[0] ** 2 - 1.0))
    if not enforce or raw_violation <= 10.0 * np.finfo(float).eps:
        return scattering, raw_violation, raw_violation
    clipped = (u * np.minimum(singular, 1.0)) @ vh
    delta = (
        invsqrt_w[:, None] * (clipped - reduced) * sqrt_w[None, :]
    )
    corrected = scattering + q @ delta @ q.conj().T
    singular_after = np.linalg.svd(
        sqrt_w[:, None]
        * (q.conj().T @ corrected @ q)
        * invsqrt_w[None, :],
        compute_uv=False,
    )
    violation = float(max(0.0, singular_after[0] ** 2 - 1.0))
    return corrected, raw_violation, violation


def interface_smatrix(
    left_modes: Sequence[VectorMode],
    right_modes: Sequence[VectorMode],
    dl_x_um: float,
    dl_y_um: float,
    *,
    rcond: float = 1e-10,
    enforce_passivity: bool = True,
    return_diagnostics: bool = False,
) -> Union[SMatrix, Tuple[SMatrix, InterfaceDiagnostics]]:
    """Generalized two-sided Petrov-Galerkin match at a waveguide step.

    Independent E- and H-continuity projections are tested from both sides with
    reaction-adjoint modal traces.  The resulting tall system supports unequal
    mode counts and complex/PML bases.  Uniform node-collocated and validated
    full-box uniform Yee-staggered bases are both supported, provided every mode
    on both sides uses the same registration.  (On the Yee lattice the two
    reaction-product pairs ``Ex*Hy`` and ``Ey*Hx`` are natively co-located.)
    A rank-revealing SVD exposes truncation through continuity residuals; it
    never fabricates exact unitarity.
    """
    n_left = len(left_modes)
    n_right = len(right_modes)
    if n_left < 1 or n_right < 1:
        raise ValueError("both interface bases must contain at least one mode")
    if rcond <= 0.0:
        raise ValueError("rcond must be > 0")
    wavelength_um = left_modes[0].wavelength_um
    reference_offset = (
        (0.0, 0.0)
        if left_modes[0].center_offset_um is None
        else tuple(left_modes[0].center_offset_um)
    )
    for mode in (*left_modes, *right_modes):
        if not np.isclose(
            mode.wavelength_um, wavelength_um, rtol=1e-12, atol=1e-15
        ):
            raise ValueError(
                "all interface modes must be solved at one wavelength"
            )
        if not (
            np.isclose(mode.dl_x_um, dl_x_um)
            and np.isclose(mode.dl_y_um, dl_y_um)
        ):
            raise ValueError(
                "interface spacing arguments must match every mode's grid"
            )
        if mode.x_coords_um is not None or mode.y_coords_um is not None:
            raise ValueError(
                "EME interface matching currently requires a uniform transverse "
                "grid; coordinate-array/nonuniform modes are not supported"
            )
        offset = (
            (0.0, 0.0)
            if mode.center_offset_um is None
            else tuple(mode.center_offset_um)
        )
        if not np.allclose(offset, reference_offset, rtol=0.0, atol=1e-12):
            raise ValueError(
                "interface modes have different transverse center offsets; "
                "cross-grid translation/resampling is not implemented"
            )
    registrations = {
        bool(mode.yee_staggered) for mode in (*left_modes, *right_modes)
    }
    if len(registrations) != 1:
        raise ValueError(
            "both interface bases must use the same field registration; "
            "mixing Yee-staggered and node-collocated modes is not supported"
        )
    if True in registrations and any(
        not mode.yee_eme_compatible for mode in (*left_modes, *right_modes)
    ):
        raise ValueError(
            "Yee-staggered interface modes must come from an EME-compatible "
            "full-box basis validated by solve_yee_eme_basis; generic Yee "
            "port, graded, and symmetry-reduced modes are not supported"
        )
    if True in registrations:
        yee_registrations = {
            (
                mode.yee_eme_axis,
                None
                if mode.yee_eme_origin_um is None
                else tuple(mode.yee_eme_origin_um),
            )
            for mode in (*left_modes, *right_modes)
        }
        if any(axis is None or origin is None for axis, origin in yee_registrations):
            raise ValueError(
                "validated Yee interface modes must carry their propagation "
                "axis and absolute transverse window origin"
            )
        if len(yee_registrations) != 1:
            raise ValueError(
                "Yee interface bases have different absolute transverse "
                "registrations; cross-grid translation/resampling is not "
                "implemented"
            )
    left = _basis_trace(left_modes, dl_x_um, dl_y_um, rcond=rcond)
    right = _basis_trace(right_modes, dl_x_um, dl_y_um, rcond=rcond)
    if (
        left.weights.shape != right.weights.shape
        or not np.allclose(left.weights, right.weights, rtol=1e-11, atol=1e-13)
    ):
        raise ValueError(
            "interface sections use different PML/complex-coordinate contours; "
            "cross-contour mode matching is not implemented"
        )
    dA = dl_x_um * dl_y_um

    def ov(
        ex: np.ndarray, ey: np.ndarray,
        hx: np.ndarray, hy: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        return _reaction_overlap(ex, ey, hx, hy, weights, dA)

    # E continuity, tested with left/right dual H.
    p_l = ov(left.ex, left.ey, left.dual_hx, left.dual_hy, left.weights).T
    a_lr = ov(right.ex, right.ey, left.dual_hx, left.dual_hy,
              left.weights).T
    c_rl = ov(left.ex, left.ey, right.dual_hx, right.dual_hy,
              right.weights).T
    p_r = ov(right.ex, right.ey, right.dual_hx, right.dual_hy,
             right.weights).T
    # H continuity, independently tested with left/right dual E.
    q_l = ov(left.dual_ex, left.dual_ey, left.hx, left.hy, left.weights)
    d_lr = ov(left.dual_ex, left.dual_ey, right.hx, right.hy,
              left.weights)
    b_rl = ov(right.dual_ex, right.dual_ey, left.hx, left.hy,
              right.weights)
    q_r = ov(right.dual_ex, right.dual_ey, right.hx, right.hy,
             right.weights)

    k_out = np.vstack((
        np.hstack((p_l, -a_lr)),
        np.hstack((c_rl, -p_r)),
        np.hstack((q_l, d_lr)),
        np.hstack((b_rl, q_r)),
    ))
    k_in = np.vstack((
        np.hstack((-p_l, a_lr)),
        np.hstack((-c_rl, p_r)),
        np.hstack((q_l, d_lr)),
        np.hstack((b_rl, q_r)),
    ))
    # Equilibrate each weak equation before the SVD so a large-impedance field
    # component does not dominate solely because of units/scaling.
    row_scale = np.maximum(
        np.sqrt(np.sum(np.abs(k_out) ** 2, axis=1)
                + np.sum(np.abs(k_in) ** 2, axis=1)),
        1e-300,
    )
    kout_eq = k_out / row_scale[:, None]
    kin_eq = k_in / row_scale[:, None]
    pinv, rank, condition, _ = _svd_pinv(kout_eq, rcond)
    raw_scattering = pinv @ kin_eq
    raw_residual = np.linalg.norm(
        k_out @ raw_scattering - k_in, axis=0
    ) / np.maximum(np.linalg.norm(k_in, axis=0), 1e-300)

    metric = np.block([
        [left.power_metric, np.zeros((n_left, n_right), dtype=complex)],
        [np.zeros((n_right, n_left), dtype=complex), right.power_metric],
    ])
    # A local ``a†Pa`` incoming/outgoing metric is valid for ordinary,
    # lossless guided ports.  In a lossy, evanescent, radiation, or PML basis,
    # forward/backward interference contributes to boundary flux and clipping
    # this reduced metric changes even the exact complex Fresnel solution.
    # Preserve the field equations for those generalized channels and expose
    # any apparent gain through the diagnostic instead.
    ordinary_real_guided = all(
        m.mode_type == "guided"
        and abs(m.k_eff) <= 100.0 * np.finfo(float).eps
        and m.overlap_weights is None
        and tuple(m.pml_cells_xy) == (0, 0)
        for m in (*left_modes, *right_modes)
    )
    apply_passivity_correction = bool(
        enforce_passivity and ordinary_real_guided
    )
    scattering, raw_pv, pv = _positive_metric_passivity_projection(
        raw_scattering,
        metric,
        rcond=rcond,
        enforce=apply_passivity_correction,
    )
    correction_applied = bool(
        apply_passivity_correction
        and not np.allclose(scattering, raw_scattering, rtol=1e-13, atol=1e-15)
    )
    residual = np.linalg.norm(
        k_out @ scattering - k_in, axis=0
    ) / np.maximum(np.linalg.norm(k_in, axis=0), 1e-300)
    guided_inputs = [
        i for i, mode in enumerate(left_modes) if mode.mode_type == "guided"
    ] + [
        n_left + i
        for i, mode in enumerate(right_modes)
        if mode.mode_type == "guided"
    ]
    max_guided_input_residual = float(
        np.max(residual[guided_inputs], initial=0.0)
    )
    s11 = scattering[:n_left, :n_left]
    s12 = scattering[:n_left, n_left:]
    s21 = scattering[n_left:, :n_left]
    s22 = scattering[n_left:, n_left:]
    smatrix: SMatrix = (s11, s12, s21, s22)
    reaction_metric = np.block([
        [left.reaction_gram, np.zeros((n_left, n_right), dtype=complex)],
        [np.zeros((n_right, n_left), dtype=complex), right.reaction_gram],
    ])
    cs = reaction_metric @ scattering
    reciprocity = float(
        np.linalg.norm(cs - cs.T) / max(np.linalg.norm(cs), 1e-300)
    )
    diagnostics = InterfaceDiagnostics(
        n_left=n_left,
        n_right=n_right,
        rank=rank,
        equations=k_out.shape[0],
        condition_number=condition,
        raw_residual_by_input=raw_residual,
        residual_by_input=residual,
        raw_passivity_violation=raw_pv,
        passivity_violation=pv,
        passivity_correction_applied=correction_applied,
        reciprocity_error=reciprocity,
        left_gram_condition=left.gram_condition,
        right_gram_condition=right.gram_condition,
        left_gram_asymmetry=left.gram_asymmetry,
        right_gram_asymmetry=right.gram_asymmetry,
        left_reaction_orthogonality_error=(
            left.reaction_orthogonality_error
        ),
        right_reaction_orthogonality_error=(
            right.reaction_orthogonality_error
        ),
        max_guided_input_residual=max_guided_input_residual,
    )
    return (smatrix, diagnostics) if return_diagnostics else smatrix


def propagation_smatrix(modes: Sequence[VectorMode], length_um: float) -> SMatrix:
    """Diagonal propagation S-matrix for a uniform section of ``length_um``.

    Forward and backward both pick up ``Φ = diag(exp(-i k0 n_eff L - k0 k_eff L))``
    with no inter-mode coupling; reflection blocks are zero. Equivalently this
    is ``exp(-i*beta_eme*L)`` for
    ``beta_eme = k0*mode.n_eff_complex.conjugate()``.

    The generalized interface transforms only its *dual test basis*, so these
    raw primal eigenmodes retain their diagonal propagation constants.  For an
    evanescent mode ``n_eff≈0, k_eff>0`` the same expression is the causal decay.
    """
    n = len(modes)
    if n < 1:
        raise ValueError("propagation needs at least one mode")
    if not np.isfinite(length_um) or length_um < 0.0:
        raise ValueError("section length_um must be finite and >= 0")
    lam_um = modes[0].wavelength_um
    if any(
        not np.isclose(m.wavelength_um, lam_um, rtol=1e-12, atol=1e-15)
        for m in modes[1:]
    ):
        raise ValueError("all propagation modes must share one wavelength")
    k0 = 2.0 * np.pi / lam_um  # 1/µm
    n_eff = np.array([m.n_eff for m in modes], dtype=float)
    k_eff = np.array([m.k_eff for m in modes], dtype=float)
    phi = np.exp(-1j * k0 * n_eff * length_um - k0 * k_eff * length_um)
    phase = np.diag(phi).astype(complex)
    zero = np.zeros((n, n), dtype=complex)
    return zero, phase.copy(), phase.copy(), zero.copy()


def star_product(sa: SMatrix, sb: SMatrix) -> SMatrix:
    """Redheffer star product ``Sa ⋆ Sb`` (``Sa`` on the left, ``Sb`` on the
    right), connecting ``Sa``'s right port to ``Sb``'s left port.

    ``Sa`` may connect ``n0 -> n1`` and ``Sb`` ``n1 -> n2``; only their internal
    port counts must match.  Linear solves are used rather than explicit matrix
    inverses.
    """
    a11, a12, a21, a22 = sa
    b11, b12, b21, b22 = sb
    n_internal = a22.shape[0]
    if (
        a22.shape != (n_internal, n_internal)
        or b11.shape != (n_internal, n_internal)
        or a12.shape[1] != n_internal
        or a21.shape[0] != n_internal
        or b12.shape[0] != n_internal
        or b21.shape[1] != n_internal
    ):
        raise ValueError("Redheffer segments have incompatible internal port sizes")
    eye = np.eye(n_internal, dtype=complex)

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Redheffer feedback matrix is singular; the cascade is at an "
                "exact internal pole or contains an invalid segment"
            ) from exc

    left_feedback = eye - b11 @ a22
    right_feedback = eye - a22 @ b11
    c11 = a11 + a12 @ solve(left_feedback, b11 @ a21)
    c12 = a12 @ solve(left_feedback, b12)
    c21 = b21 @ solve(right_feedback, a21)
    c22 = b22 + b21 @ solve(right_feedback, a22 @ b12)
    return c11, c12, c21, c22


def cascade(segments: Sequence[SMatrix]) -> SMatrix:
    """Fold a left-to-right sequence of S-matrices with the star product."""
    if not segments:
        raise ValueError("cascade needs at least one S-matrix segment")
    total = segments[0]
    for seg in segments[1:]:
        total = star_product(total, seg)
    return total


def run_eme(
    sections: Sequence[Section],
    n_modes: Optional[int] = None,
    *,
    interface_rcond: float = 1e-10,
    enforce_passivity: bool = True,
) -> EMEResult:
    """Cascade a sequence of :class:`Section`s into one device S-matrix.

    Parameters
    ----------
    sections:
        Cross-sections in propagation order. The first and last define the
        input/output ports (typically length-``0`` leads). All must share the
        same transverse grid ``(ny, nx)`` and spacing.
    n_modes:
        Optional cap applied independently to every section.  Adjacent sections
        may retain different counts; their interface blocks are rectangular.
        A cap may not split a degenerate validated Yee beta shell.

    Returns
    -------
    EMEResult
        The device scattering matrix and convenience power ratios.
    """
    if len(sections) < 2:
        raise ValueError("run_eme needs at least two sections (in and out ports)")
    for k, section in enumerate(sections):
        if not np.isfinite(section.length_um) or section.length_um < 0.0:
            raise ValueError(
                f"section {k} length_um must be finite and >= 0"
            )
    counts = [len(s.modes) for s in sections]
    if min(counts) < 1:
        raise ValueError("every section must carry at least one mode")
    if n_modes is not None:
        try:
            cap_is_finite = bool(np.isfinite(n_modes))
        except (TypeError, ValueError):
            cap_is_finite = False
        if (
            isinstance(n_modes, (bool, np.bool_))
            or not cap_is_finite
            or int(n_modes) != n_modes
            or int(n_modes) < 1
        ):
            raise ValueError("n_modes must be an integer >= 1")
        cap = int(n_modes)
        for k, section in enumerate(sections):
            if cap >= len(section.modes):
                continue
            last = section.modes[cap - 1]
            following = section.modes[cap]
            if (
                last.yee_eme_compatible
                and following.yee_eme_compatible
                and abs(last.n_eff_complex - following.n_eff_complex)
                <= 1e-6
                * max(
                    abs(last.n_eff_complex),
                    abs(following.n_eff_complex),
                    1.0,
                )
            ):
                raise ValueError(
                    f"n_modes={cap} splits a degenerate validated Yee beta "
                    f"shell in section {k}; include the complete multiplet "
                    "or use a smaller cap"
                )
    bases: List[Tuple[VectorMode, ...]] = [
        tuple(sec.modes if n_modes is None else sec.modes[:int(n_modes)])
        for sec in sections
    ]
    used_counts = tuple(len(b) for b in bases)

    ref = bases[0][0]
    shape0 = ref.shape
    dl_x = ref.dl_x_um
    dl_y = ref.dl_y_um
    for k, modes in enumerate(bases):
        for m in modes:
            if m.shape != shape0:
                raise ValueError(
                    f"section {k} mode grid {m.shape} != port grid {shape0}; "
                    "all sections must share one transverse grid"
                )
            if not (
                np.isclose(m.dl_x_um, dl_x) and np.isclose(m.dl_y_um, dl_y)
            ):
                raise ValueError(
                    f"section {k} grid spacing differs from the port spacing"
                )

    segments: List[SMatrix] = []
    diagnostics: List[InterfaceDiagnostics] = []
    for k, sec in enumerate(sections):
        modes = bases[k]
        if k > 0:
            matched, diag = interface_smatrix(
                bases[k - 1],
                modes,
                dl_x,
                dl_y,
                rcond=interface_rcond,
                enforce_passivity=enforce_passivity,
                return_diagnostics=True,
            )
            segments.append(matched)
            diagnostics.append(diag)
        if sec.length_um > 0.0:
            segments.append(propagation_smatrix(modes, sec.length_um))

    s11, s12, s21, s22 = cascade(segments)
    left_trace = _basis_trace(
        bases[0], dl_x, dl_y, rcond=interface_rcond
    )
    right_trace = _basis_trace(
        bases[-1], dl_x, dl_y, rcond=interface_rcond
    )
    return EMEResult(
        s11=s11,
        s12=s12,
        s21=s21,
        s22=s22,
        n_modes=min(used_counts),
        n_left_modes=used_counts[0],
        n_right_modes=used_counts[-1],
        section_mode_counts=used_counts,
        left_modes=bases[0],
        right_modes=bases[-1],
        left_power_metric=left_trace.power_metric,
        right_power_metric=right_trace.power_metric,
        interface_diagnostics=tuple(diagnostics),
    )


def waveguide_section(
    *,
    wavelength_um: float,
    dl_um: float,
    core_w_um: float,
    core_h_um: float,
    n_core: float,
    n_clad: float,
    window_w_um: float,
    window_h_um: float,
    num_modes: int,
    num_radiation_modes: int = 0,
    num_evanescent_modes: int = 0,
    pml_cells_xy: Optional[Tuple[int, int]] = None,
    pml_imaginary_thickness_um: Tuple[float, float] = (0.10, 0.10),
    length_um: float = 0.0,
    neff_margin: float = 0.0,
    subpixel: bool = True,
    subpixel_method: str = "tensor",
) -> Section:
    """Solve a centered rectangular-core cross-section and wrap it as a
    :class:`Section`.

    Pass the **same** ``wavelength_um``, ``dl_um``, ``window_w_um`` and
    ``window_h_um`` for every section of a device so they share one transverse
    grid (required by :func:`run_eme`). Only ``core_w_um`` / ``core_h_um`` should
    vary between sections.

    ``neff_margin`` drops modes whose ``n_eff`` is within this margin of the
    cladding index — i.e. **near-cutoff** modes. Those are poorly resolved (large
    evanescent tails into the window walls) and break the within-section
    orthonormality the interface relies on, so a small margin (e.g. ``0.05``)
    keeps the modal basis clean. Default ``0.0`` keeps every guided mode the
    solver returns.
    """
    solver = VectorModeSolver.from_rectangular_core(
        wavelength_um=wavelength_um,
        dl_um=dl_um,
        core_w_um=core_w_um,
        core_h_um=core_h_um,
        n_core=n_core,
        n_clad=n_clad,
        window_w_um=window_w_um,
        window_h_um=window_h_um,
        subpixel=subpixel,
        subpixel_method=subpixel_method,  # type: ignore[arg-type]
    )
    if num_radiation_modes or num_evanescent_modes:
        modes = solver.solve_eme_basis(
            num_guided=num_modes,
            num_radiation=num_radiation_modes,
            num_evanescent=num_evanescent_modes,
            pml_cells_xy=pml_cells_xy,
            pml_imaginary_thickness_um=pml_imaginary_thickness_um,
        )
    else:
        modes = solver.solve(num_modes=num_modes)
    if neff_margin > 0.0:
        modes = tuple(
            m for m in modes
            if m.mode_type != "guided" or m.n_eff > n_clad + neff_margin
        )
    if not modes:
        raise ValueError(
            f"no EME modes for core_w={core_w_um} um at lambda={wavelength_um} "
            f"um (neff_margin={neff_margin} may be too strict)"
        )
    return Section(modes=modes, length_um=length_um)


# ---------------------------------------------------------------------------
# Band / length sweep — parallel F×Z eigensolves
# ---------------------------------------------------------------------------
#
# A frequency (or length) sweep solves the local modes of every cross-section at
# every wavelength. Profiling the canonical taper (dl=0.04, 65×41 cross-section,
# 2N≈5300) shows where the time actually goes:
#
#     rasterize cross-section (from_rectangular_core)  ~0.12 ms
#     FDE eigensolve (solver.solve, 2 modes)          ~52    ms   ← ~99.8% of cost
#     whole S-matrix cascade (run_eme)                ~microseconds
#
# So (a) the eigensolve dominates by ~400×, and (b) the F×Z eigensolves are
# mutually **independent** — the textbook batch-parallel axis. ``run_eme_band``
# rasters each cross-section ONCE (the permittivity is wavelength-independent for
# this non-dispersive prototype; it is reused across the band via
# :meth:`VectorModeSolver.at_wavelength`, which avoids the — already cheap —
# re-raster) and fans the independent solves across a worker pool, then cascades
# each wavelength on the parent. The per-wavelength result is **bit-identical** to
# calling :func:`waveguide_section` + :func:`run_eme` at that wavelength.
#
# This is the CPU "expose the batch" prerequisite for any GPU port: once the F×Z
# eigensolves are collected as one independent set, the same set is what a batched
# GPU eigensolver (CuPy-for-ROCm / rocSOLVER on the MI300X) would consume. See
# ``docs/eme-gpu-acceleration-design.md``.
#
# Multiprocessing note: on macOS/Windows the default start method is *spawn*,
# which re-imports the caller's ``__main__`` — so a script that calls
# ``run_eme_band(..., backend="process")`` MUST guard its entry with
# ``if __name__ == "__main__":``. The *thread* backend has no such constraint and
# skips result pickling; which backend wins depends on how much of the eigensolve
# releases the GIL and on the process-startup cost relative to the workload (see
# ``benchmarks/eme/eme_band_speedup.py`` for measured numbers).

#: Per-section base solvers, stashed in a module global so a pool task ships only
#: ``(section, wavelength)`` indices rather than re-pickling a solver per task.
#: Set directly in the parent (thread/serial backend) or via
#: :func:`_band_pool_init` in each worker (process backend).
_BAND_SOLVERS: Optional[Sequence[VectorModeSolver]] = None


def _band_pool_init(base_solvers: Sequence[VectorModeSolver]) -> None:
    """Process-pool initializer — stash the base solvers once per worker."""
    global _BAND_SOLVERS
    _BAND_SOLVERS = base_solvers


def _solve_section_task(
    task: Tuple[
        int,
        int,
        float,
        int,
        float,
        Optional[float],
        Optional[dict],
        Optional[dict],
    ]
) -> Tuple[int, int, Tuple[VectorMode, ...]]:
    """Solve one ``(section si, wavelength wi)`` cell and return its EME modes.

    Reads the base solvers from the module global :data:`_BAND_SOLVERS` (set by the
    parent for thread/serial, or by :func:`_band_pool_init` in each process)."""
    (
        si,
        wi,
        wl_um,
        num_modes,
        neff_margin,
        n_clad,
        solve_kwargs,
        basis_kwargs,
    ) = task
    assert _BAND_SOLVERS is not None  # set before any task runs
    solver = _BAND_SOLVERS[si].at_wavelength(wl_um)
    if basis_kwargs is None:
        modes = solver.solve(num_modes=num_modes, **(solve_kwargs or {}))
    else:
        modes = solver.solve_eme_basis(
            num_guided=num_modes,
            **basis_kwargs,
        )
    if neff_margin > 0.0 and n_clad is not None:
        modes = tuple(
            m for m in modes
            if m.mode_type != "guided" or m.n_eff > n_clad + neff_margin
        )
    return si, wi, modes


def rectangular_base_section(
    *,
    core_w_um: float,
    length_um: float = 0.0,
    dl_um: float,
    core_h_um: float,
    n_core: float,
    n_clad: float,
    window_w_um: float,
    window_h_um: float,
    ref_wavelength_um: float = 1.31,
    subpixel: bool = True,
    subpixel_method: str = "tensor",
) -> Tuple[VectorModeSolver, float]:
    """Raster a centered rectangular-core cross-section **once** and pair it with
    its length, ready for a band/length sweep via :func:`run_eme_band`.

    Mirrors the geometry arguments of :func:`waveguide_section`, but returns the
    *unsolved* base solver instead of solved modes: the permittivity raster is
    wavelength-independent (non-dispersive prototype), so ``ref_wavelength_um`` is
    only a placeholder — :func:`run_eme_band` rebases every solve to the swept
    wavelength with :meth:`VectorModeSolver.at_wavelength`. Pass the **same**
    ``dl_um`` / ``window_*`` for every section of a device so they share one
    transverse grid (required by :func:`run_eme`)."""
    solver = VectorModeSolver.from_rectangular_core(
        wavelength_um=ref_wavelength_um, dl_um=dl_um, core_w_um=core_w_um,
        core_h_um=core_h_um, n_core=n_core, n_clad=n_clad,
        window_w_um=window_w_um, window_h_um=window_h_um,
        subpixel=subpixel, subpixel_method=subpixel_method,  # type: ignore[arg-type]
    )
    return solver, length_um


def run_eme_band(
    base_sections: Sequence[Tuple[VectorModeSolver, float]],
    wavelengths_um: Sequence[float],
    *,
    num_modes: int,
    neff_margin: float = 0.0,
    n_clad: Optional[float] = None,
    n_modes: Optional[int] = None,
    solve_kwargs: Optional[dict] = None,
    basis_kwargs: Optional[dict] = None,
    workers: Optional[int] = None,
    backend: str = "auto",
) -> Dict[float, EMEResult]:
    """Cascade a staircased device across a band of wavelengths, solving the
    independent per-(wavelength × section) FDE eigenmodes **in parallel**.

    Parameters
    ----------
    base_sections:
        Cross-sections in propagation order as ``(base_solver, length_um)`` pairs
        — e.g. from :func:`rectangular_base_section`. The solver is rasterized
        once and reused across the whole band; ``length_um == 0`` marks a port
        lead (as in :class:`Section`). All sections must share one transverse grid.
    wavelengths_um:
        The free-space wavelengths (microns) to sweep.
    num_modes:
        Guided modes solved per cross-section. Passed to
        :meth:`VectorModeSolver.solve` on the ordinary path, or used as
        ``num_guided`` when ``basis_kwargs`` selects a complete EME basis.
    neff_margin:
        Drop modes whose ``n_eff`` is within this margin of ``n_clad`` (near-cutoff
        — see :func:`waveguide_section`). Requires ``n_clad``.
    n_clad:
        Cladding index, needed only when ``neff_margin > 0``.
    n_modes:
        Cap on the cascade basis size, forwarded to :func:`run_eme`.
    solve_kwargs:
        Extra keyword arguments forwarded to :meth:`VectorModeSolver.solve`
        (e.g. ``n_guess``); must be picklable for the process backend.
    basis_kwargs:
        When provided, solve a complete basis with
        :meth:`VectorModeSolver.solve_eme_basis` instead. ``num_modes`` supplies
        ``num_guided`` and this dictionary supplies options such as
        ``num_radiation``, ``num_evanescent`` and ``pml_cells_xy``. It is
        mutually exclusive with ``solve_kwargs`` and must be picklable for the
        process backend.
    workers:
        Worker count for the eigensolves. ``None`` → ``min(#solves, os.cpu_count)``.
        ``1`` (or a single solve) runs serially with no pool.
    backend:
        ``"auto"`` (default) — ``"process"`` on fork platforms (Linux/CI/cloud,
        ~3.5x), ``"thread"`` on spawn platforms (macOS/Windows, ~1.7x without the
        process-startup tax or ``__main__``-guard requirement). Force ``"process"``
        for a large spawn-platform sweep, or ``"serial"`` to disable parallelism.

    Returns
    -------
    dict[float, EMEResult]
        ``{wavelength_um: EMEResult}`` — one cascaded device S-matrix per
        wavelength, identical to a per-wavelength :func:`run_eme` cascade.
    """
    global _BAND_SOLVERS
    if len(base_sections) < 2:
        raise ValueError("run_eme_band needs at least two sections (in/out ports)")
    wls = [float(w) for w in wavelengths_um]
    if not wls:
        raise ValueError("wavelengths_um must be non-empty")
    if num_modes < (0 if basis_kwargs is not None else 1):
        requirement = ">= 0 with basis_kwargs" if basis_kwargs is not None else ">= 1"
        raise ValueError(f"num_modes must be {requirement}")
    if solve_kwargs is not None and basis_kwargs is not None:
        raise ValueError("solve_kwargs and basis_kwargs are mutually exclusive")
    if basis_kwargs is not None and "num_guided" in basis_kwargs:
        raise ValueError(
            "basis_kwargs must not contain num_guided; use num_modes instead"
        )
    if neff_margin > 0.0 and n_clad is None:
        raise ValueError("neff_margin > 0 requires n_clad")
    if backend not in ("auto", "process", "thread", "serial"):
        raise ValueError(
            f"backend must be auto/process/thread/serial, got {backend!r}")
    if backend == "auto":
        # Process pools shine where the platform *forks* (Linux/CI/the MI300X
        # cloud box): ~0 startup, ~3.5x measured. Under *spawn* (macOS/Windows)
        # the per-worker re-import tax sinks small sweeps and needs a __main__
        # guard, so fall back to threads — a safe, steady ~1.7x with no pickling.
        # Override with backend="process" for big spawn-platform sweeps.
        import multiprocessing as _mp

        backend = (
            "process"
            if _mp.get_start_method(allow_none=False) == "fork"
            else "thread"
        )

    base_solvers = [s for s, _ in base_sections]
    lengths = [float(L) for _, L in base_sections]
    n_sec = len(base_solvers)

    tasks = [
        (
            si,
            wi,
            wl,
            num_modes,
            neff_margin,
            n_clad,
            solve_kwargs,
            basis_kwargs,
        )
        for wi, wl in enumerate(wls)
        for si in range(n_sec)
    ]
    n_tasks = len(tasks)
    if workers is None:
        workers = min(n_tasks, os.cpu_count() or 1)
    workers = max(1, int(workers))

    # Make the base solvers visible to the worker (parent process for
    # thread/serial; each child gets its own copy via the pool initializer).
    _BAND_SOLVERS = base_solvers

    use_serial = backend == "serial" or workers <= 1 or n_tasks == 1
    if use_serial:
        solved = [_solve_section_task(t) for t in tasks]
    elif backend == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            solved = list(ex.map(_solve_section_task, tasks))
    else:  # process
        chunksize = max(1, n_tasks // (workers * 4))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_band_pool_init,
            initargs=(base_solvers,),
        ) as ex:
            solved = list(ex.map(_solve_section_task, tasks, chunksize=chunksize))

    # Regroup solved modes into a [wavelength][section] grid.
    grid: List[List[Optional[Tuple[VectorMode, ...]]]] = [
        [None] * n_sec for _ in wls
    ]
    for si, wi, modes in solved:
        grid[wi][si] = modes

    results: Dict[float, EMEResult] = {}
    for wi, wl in enumerate(wls):
        sections: List[Section] = []
        for si in range(n_sec):
            modes = grid[wi][si]
            if not modes:
                raise ValueError(
                    f"no EME modes for section {si} at lambda={wl} um "
                    f"(neff_margin={neff_margin} may be too strict)"
                )
            sections.append(Section(modes=modes, length_um=lengths[si]))
        results[wl] = run_eme(sections, n_modes=n_modes)
    return results
