"""Multiport scattering-matrix (S-matrix) assembler — pure Python, no engine.

This is the PhotonHub analogue of Tidy3D's ``ComponentModeler`` / Lumerical's
S-parameter sweep, built entirely on top of the mode-resolved **complex** modal
amplitude that :func:`photonhub.plugins.mode_overlap.mode_amplitude` extracts
from a recorded DFT plane. Where :mod:`mode_devices` gives you single-mode power
``T(f) = |c|^2`` per monitor, this module keeps the *phase* of ``c`` and arranges
the per-port amplitudes into a proper scattering matrix.

S-parameter definition
=======================
For an N-port device, mode the field on each port plane as a sum of an inward
(toward the device) and an outward (away from the device) guided wave. The
S-matrix relates the outgoing modal amplitudes ``b`` to the incoming ones ``a``:

    b = S a ,    S_ij(f) = b_i(f) / a_j(f)   with port j the only one driven.

``a_j`` is the amplitude **incident** on the device at the driven port; ``b_i``
is the amplitude **scattered out** of the device at port i. Each is the
**power-normalized wave amplitude** ``c * sqrt(|P_mode|)``, built from the
normalized modal coefficient ``c = a_pm / P_mode`` of
:func:`~photonhub.plugins.mode_overlap.mode_amplitude` and the port mode's own
plane power ``P_mode`` — so

* ``|a|^2`` / ``|b|^2`` is the modal **power** the wave carries through the
  port plane (the same number
  :func:`~photonhub.plugins.mode_devices.ModeMonitor.mode_power` reports), hence
  ``|S_ij|^2 = |b_i|^2 / |a_j|^2`` is a true power transmission / reflection
  **even when ports i and j carry different modes** (unequal widths /
  polarizations, whose per-mode ``P_mode`` differ) and a lossless reciprocal
  device has ``S† S = I``. (The bare ``c`` ratio would be biased by
  ``P_mode,j / P_mode,i ≈ n_eff,j / n_eff,i`` across unequal ports and would
  break reciprocity there; for same-mode ports the two coincide.)
* the phase is ``c``'s phase (``sqrt(|P_mode|)`` is real positive), so ``S``
  carries the physical propagation phase a mode accumulates along the guide.

Incident vs scattered: the directional projection
==================================================
Each :class:`SPort` carries the **outgoing direction** — the sign along the
monitor's propagation ``axis`` that points *out of* the device through that port
(``"+"`` or ``"-"``). The two directional projections of the recorded plane are
then:

* outgoing / scattered ``b_i`` = project onto the mode travelling in the port's
  outgoing direction;
* incoming / incident ``a_i`` = project onto the mode travelling the *opposite*
  way (into the device).

At the **driven** port the recorded plane (placed on the total-field side, just
inside the source) carries incident + reflected: its incoming projection is the
incident ``a_j`` and its outgoing projection is the reflected ``b_j`` (so
``S_jj`` is the port reflection). At every **other** port only scattered light
leaves, so the outgoing projection is the transmitted ``b_i`` and ``S_ij`` is the
port-i transmission. This directional split is exactly what the forward/backward
overlap of :func:`mode_amplitude` provides.

Normalization
=============
``a`` and ``b`` are built from the **source-spectrum-normalized** complex modal
amplitudes already living in the run data: the recorded phasors are divided by
``A0*S(f)`` (NUMERICS.md section 12), so that normalization cancels in the ratio
``S_ij = b_i / a_j`` and leaves a dimensionless scattering parameter. Each port's
amplitude is the power-normalized ``c * sqrt(|P_mode|)`` (see above), so
``|S_ij|^2`` is a power ratio for same-mode AND unequal-mode port pairs alike
(a clean straight through-guide reads ``|S21| ≈ 1``). No de-embedding of the
source-to-monitor or monitor-to-port reference plane is applied — ``S`` is
referenced to the monitor planes as placed (see "What's not handled").

Output
======
:func:`smatrix` returns **one column** of S (the column for the driven port) as
``{(port_out, port_in): {freq_hz: S}}`` — i.e. ``S_ij`` for every port ``i`` with
``j`` fixed to the driven port. Run it once per driven port and feed the columns
to :func:`assemble_smatrix` to get the full matrix as an :class:`xarray.DataArray`
indexed ``(port_out, port_in, f)``. :func:`reciprocity_error` /
:func:`passivity_violation` (and their ``assert_*`` / ``is_*`` wrappers) check the
assembled matrix.

What's not handled (deferred)
=============================
* **Multimode ports.** One mode per port (the monitor's ``mode``). A multimode
  S-matrix would index ports by (port, mode_index); the per-mode amplitudes are
  available by re-projecting onto each mode, but the bookkeeping here is
  single-mode.
* **De-embedding.** S is referenced to the monitor planes; no phase de-embedding
  back to a common reference plane is performed. Do it externally if needed: in
  the engine's ``e^{-i omega t}`` convention an outgoing wave accumulates
  ``e^{+i beta L}`` over the extra lead length ``L``, so multiply by
  ``e^{-i beta L}`` per port to remove it.
* **Source-overlap correction at the driven port.** The incident ``a_j`` is read
  from the recorded total field's incoming projection, which assumes the launched
  mode matches the monitor's ``mode``; a mode-mismatched launch would bias
  ``a_j``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._constants import _TANGENTIAL
from .mode_devices import ModeMonitor
from .mode_overlap import _overlap_terms

__all__ = [
    "SPort",
    "smatrix",
    "assemble_smatrix",
    "reciprocity_error",
    "is_reciprocal",
    "assert_reciprocal",
    "passivity_violation",
    "is_passive",
    "assert_passive",
]

_OPPOSITE = {"+": "-", "-": "+"}


@dataclass(frozen=True)
class SPort:
    """One port of an S-matrix: a :class:`ModeMonitor` + its **outgoing
    direction**.

    Parameters
    ----------
    name:
        Port label (the S-matrix index). Must be unique among the ports.
    monitor:
        The :class:`~photonhub.plugins.mode_devices.ModeMonitor` recording the
        port plane (its ``mode``, ``axis``, ``center_um``, ``thickness_axis`` and
        ``modes_by_freq`` drive the overlap).
    out_direction:
        The sign along the monitor's propagation ``axis`` ('+' or '-') that points
        *out of the device* through this port — the direction a transmitted /
        scattered wave leaves. The incident (incoming) direction is the opposite.
        Defaults to the monitor's own ``direction`` (treated as the outgoing one).
    """

    name: str
    monitor: ModeMonitor
    out_direction: Optional[str] = None

    def __post_init__(self) -> None:
        d = self.out_direction or self.monitor.direction
        if d not in ("+", "-"):
            raise ValueError(
                f"port {self.name!r}: out_direction must be '+' or '-', got {d!r}")
        object.__setattr__(self, "out_direction", d)

    @property
    def in_direction(self) -> str:
        """The incoming (toward-device) direction — opposite the outgoing one."""
        return _OPPOSITE[self.out_direction]

    def _planes(self, data) -> Mapping[str, object]:
        da = data[self.monitor.name]
        ax = self.monitor.axis
        return {c: da.sel(component=c) for c in _TANGENTIAL[ax]}

    def _amplitude(self, data, direction: str, *,
                   colocate: bool = True) -> Dict[float, complex]:
        """Power-normalized complex wave amplitude of the recorded plane onto
        this port's mode, travelling in ``direction`` (the directional
        projection): ``c * sqrt(|P_mode|)`` with ``c = a_pm / P_mode``, so its
        squared magnitude is the modal POWER through the plane (the module
        docstring's normalization — what makes ``|S_ij|^2`` a power ratio for
        unequal-mode port pairs). ``colocate`` Yee-co-locates the staggered E/H
        sim components before the overlap (the default; see
        :func:`mode_overlap.mode_amplitude`).

        The longitudinal Yee de-stagger is applied with the monitor's grid
        ``dl_um`` whenever ``colocate`` is on (mirroring
        ``mode_devices._DESTAGGER_AUTO`` / ``ModeMonitor.mode_power``): the
        driven-port reflection ``S_jj`` is the most stagger-sensitive readout —
        without it the incident wave leaks ``O(beta*dl/2)`` into the outgoing
        projection. Monitors built without a grid (``dl_um is None``) skip it."""
        mon = self.monitor
        terms = _overlap_terms(
            self._planes(data),
            mon.mode,
            axis=mon.axis,
            direction=direction,
            center_um=mon.center_um,
            thickness_axis=mon.thickness_axis,
            # Resolve through the monitor (stored map OR its automatic
            # per-frequency bank) so S-matrix amplitudes project onto the SAME
            # reference modes ModeMonitor.mode_power uses — bypassing the
            # resolver here would silently split the two readouts' conventions.
            modes_by_freq=mon._resolved_modes_by_freq(),
            colocate=colocate,
            destagger_dl=(mon.dl_um if colocate else None),
            # §20 folded-domain quadrature: half-weight a node-registered sample
            # row sitting ON a symmetry plane, exactly as ModeMonitor.mode_power /
            # mode_decomposition do — else the S-matrix readout reverts to the
            # pre-fix quadrature and inflates fold-antinode mode power (a
            # parity-asymmetric bias) on any simulation with a symmetry plane.
            fold_low=mon._fold_low(),
        )
        # c = a_pm/P_mode keeps the phase convention (a clean self-overlap reads
        # c == 1 in either direction, the SIGNED P_mode handling the backward
        # flip); the sqrt(|P_mode|) factor converts it to a wave amplitude whose
        # |.|^2 is the modal power, so per-port P_mode no longer biases S.
        return {
            f: complex(a_pm / p_mode) * float(np.sqrt(abs(p_mode)))
            for f, (a_pm, p_mode) in terms.items()
        }

    def outgoing(self, data, *, colocate: bool = True) -> Dict[float, complex]:
        """``{freq: b}`` — the complex amplitude of the wave scattered *out* of
        the device through this port."""
        return self._amplitude(data, self.out_direction, colocate=colocate)

    def incoming(self, data, *, colocate: bool = True) -> Dict[float, complex]:
        """``{freq: a}`` — the complex amplitude of the wave incident *into* the
        device at this port (the source side at the driven port)."""
        return self._amplitude(data, self.in_direction, colocate=colocate)


def _port_list(ports: Sequence[SPort]) -> List[SPort]:
    names = [p.name for p in ports]
    if len(set(names)) != len(names):
        dup = [n for n in names if names.count(n) > 1]
        raise ValueError(f"duplicate port name(s): {sorted(set(dup))}")
    return list(ports)


def _find_port(ports: Sequence[SPort], driven) -> SPort:
    if isinstance(driven, SPort):
        driven = driven.name
    for p in ports:
        if p.name == driven:
            return p
    raise ValueError(
        f"driven port {driven!r} not among ports {[p.name for p in ports]}")


def smatrix(
    ports: Sequence[SPort],
    driven,
    data,
    *,
    colocate: bool = True,
) -> Dict[Tuple[str, str], Dict[float, complex]]:
    """Assemble **one column** of the S-matrix from a single driven-port run.

    Drives port ``driven`` (only) and returns ``S_ij(f) = b_i / a_j`` for every
    port ``i`` in ``ports``, with ``j = driven`` fixed — i.e. the column of the
    full S-matrix belonging to the driven port. ``a_j`` is the amplitude incident
    on the device at the driven port (its incoming projection); ``b_i`` is the
    amplitude scattered out of the device at port ``i`` (its outgoing projection).

    Parameters
    ----------
    ports:
        The N :class:`SPort`s. Names must be unique.
    driven:
        The driven port — an :class:`SPort` or its ``name``. Must be in ``ports``.
    data:
        The run's ``SimulationData`` (or any mapping ``name -> DataArray``) for the
        run that drove ``driven``. Every port monitor's data must be present.

    Returns
    -------
    dict[(port_out, port_in), dict[float, complex]]
        ``{(i, driven): {freq_hz: S_ij}}`` for each port ``i``. ``S`` is the
        complex scattering parameter (``|S|^2`` is a power ratio).
    """
    ports = _port_list(ports)
    dport = _find_port(ports, driven)

    a_in = dport.incoming(data, colocate=colocate)  # {freq: a_j} incident at driven port
    freqs = list(a_in)
    if not freqs:
        raise ValueError("driven port carries no frequencies")
    for f in freqs:
        if a_in[f] == 0:
            raise ValueError(
                f"incident amplitude a_j at driven port {dport.name!r} is zero "
                f"at f={f} — cannot normalize S_ij (is the port driven?)")

    column: Dict[Tuple[str, str], Dict[float, complex]] = {}
    for p in ports:
        b_out = p.outgoing(data, colocate=colocate)  # {freq: b_i}
        sij: Dict[float, complex] = {}
        for f in freqs:
            if f not in b_out:
                raise ValueError(
                    f"port {p.name!r} has no recorded frequency {f} present at "
                    f"the driven port {dport.name!r}")
            sij[f] = complex(b_out[f] / a_in[f])
        column[(p.name, dport.name)] = sij
    return column


def assemble_smatrix(
    columns: Sequence[Mapping[Tuple[str, str], Mapping[float, complex]]],
    *,
    port_order: Optional[Sequence[str]] = None,
):
    """Combine per-driven-port :func:`smatrix` columns into a full S-matrix as an
    :class:`xarray.DataArray` indexed ``(port_out, port_in, f)``.

    Parameters
    ----------
    columns:
        An iterable of the dicts returned by :func:`smatrix` (one per driven
        port). Together they should cover every (i, j) entry; missing entries are
        filled with ``nan`` and a clear coord layout is still returned.
    port_order:
        Optional explicit ordering of the port labels for both axes. Defaults to
        first-seen order across the columns (driven ports first, then any extra
        out-ports).

    Returns
    -------
    xarray.DataArray
        Complex ``S`` with dims ``("port_out", "port_in", "f")`` and string
        port coords. ``S.sel(port_out=i, port_in=j)`` is ``S_ij(f)``.
    """
    import xarray as xr

    entries: Dict[Tuple[str, str], Dict[float, complex]] = {}
    seen_out: List[str] = []
    seen_in: List[str] = []
    freqs: List[float] = []
    for col in columns:
        for (i, j), per_f in col.items():
            entries[(i, j)] = dict(per_f)
            if i not in seen_out:
                seen_out.append(i)
            if j not in seen_in:
                seen_in.append(j)
            for f in per_f:
                if f not in freqs:
                    freqs.append(f)
    freqs = sorted(freqs)

    if port_order is not None:
        labels = list(port_order)
    else:
        # driven (in) ports first to keep a square matrix's diagonal aligned,
        # then any out-only ports.
        labels = list(seen_in)
        for i in seen_out:
            if i not in labels:
                labels.append(i)

    n = len(labels)
    arr = np.full((n, n, len(freqs)), np.nan, dtype=np.complex128)
    idx = {lbl: k for k, lbl in enumerate(labels)}
    fidx = {f: k for k, f in enumerate(freqs)}
    for (i, j), per_f in entries.items():
        if i not in idx or j not in idx:
            continue
        for f, s in per_f.items():
            arr[idx[i], idx[j], fidx[f]] = s

    return xr.DataArray(
        arr,
        dims=("port_out", "port_in", "f"),
        coords={"port_out": labels, "port_in": labels,
                "f": np.asarray(freqs, dtype=np.float64)},
        name="S",
        attrs={
            "description": "scattering matrix S_ij(f) = b_i / a_j",
            "normalization": (
                "source-spectrum-normalized (NUMERICS.md section 12), "
                "power-normalized wave amplitudes c*sqrt(|P_mode|); |S_ij|^2 is "
                "a power ratio (also across unequal-mode ports), S† S <= I "
                "for a passive device"),
        },
    )


# ----------------------------------------------------------------------------
# Reciprocity & passivity
# ----------------------------------------------------------------------------

def _as_matrices(S) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(stack, freqs)`` where ``stack`` is ``(n_f, n, n)`` complex with
    axis order ``[f, port_out, port_in]``. Accepts the xarray from
    :func:`assemble_smatrix` (dims port_out/port_in/f) or a bare ndarray already
    shaped ``(n_f, n, n)`` / ``(n, n)``."""
    if hasattr(S, "dims"):  # xarray.DataArray
        Sd = S.transpose("f", "port_out", "port_in")
        return np.asarray(Sd.values), np.asarray(Sd.coords["f"].values)
    a = np.asarray(S)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(
            f"expected an (n_f, n, n) or (n, n) S array, got shape {a.shape}")
    return a, np.arange(a.shape[0], dtype=np.float64)


def reciprocity_error(S) -> float:
    """Max ``|S_ij - S_ji|`` over all off-diagonal entries and frequencies.

    A reciprocal (non-magnetic, non-gyrotropic) device has a symmetric S-matrix,
    so this should be ~0. ``nan`` entries (uncomputed) are ignored."""
    stack, _ = _as_matrices(S)
    diff = np.abs(stack - np.transpose(stack, (0, 2, 1)))
    diff = diff[np.isfinite(diff)]
    return float(np.max(diff)) if diff.size else 0.0


def is_reciprocal(S, *, atol: float = 1e-6) -> bool:
    """``True`` when :func:`reciprocity_error` is within ``atol``."""
    return reciprocity_error(S) <= atol


def assert_reciprocal(S, *, atol: float = 1e-6) -> None:
    """Raise ``AssertionError`` if S is not symmetric to ``atol``."""
    err = reciprocity_error(S)
    if err > atol:
        raise AssertionError(
            f"S-matrix not reciprocal: max |S_ij - S_ji| = {err:.3e} > {atol:.1e}")


def passivity_violation(S) -> float:
    """Worst passivity excess ``max(eig(S† S)) - 1`` over all frequencies.

    For a passive device the scattering operator does not create energy:
    ``S† S <= I``, i.e. every eigenvalue of the Hermitian Gram matrix ``S† S`` is
    ``<= 1``. This returns the largest eigenvalue minus one (``<= 0`` means
    passive; a small positive value is numerical slack). Frequencies with any
    ``nan`` S entry are skipped."""
    stack, _ = _as_matrices(S)
    worst = -np.inf
    for Sf in stack:
        if not np.all(np.isfinite(Sf)):
            continue
        gram = Sf.conj().T @ Sf
        # Hermitian by construction; eigvalsh is real and stable.
        lam = np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))
        worst = max(worst, float(np.max(lam)) - 1.0)
    return worst if np.isfinite(worst) else 0.0


def is_passive(S, *, atol: float = 1e-6) -> bool:
    """``True`` when :func:`passivity_violation` is within ``atol`` (so the
    largest singular value of S is ``<= 1 + atol``)."""
    return passivity_violation(S) <= atol


def assert_passive(S, *, atol: float = 1e-6) -> None:
    """Raise ``AssertionError`` if S amplifies energy beyond ``atol`` (i.e.
    ``max eig(S† S) > 1 + atol``)."""
    v = passivity_violation(S)
    if v > atol:
        raise AssertionError(
            f"S-matrix not passive: max eig(S† S) - 1 = {v:.3e} > {atol:.1e}")
