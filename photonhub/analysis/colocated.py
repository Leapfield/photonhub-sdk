"""A recorded field plane with E and H colocated on their true coordinates.

The solver records each field component at its own Yee point (``E_c`` half
a cell along ``c``, ``H_c`` half a cell along the two other axes), while a
``field_dft`` array labels every component with its cell's base node. Products
of components read at those labels, such as a Poynting flux ``E x H*``, are
then wrong at first order in the cell. :func:`colocated_fields` returns the
plane with every component on the same points, the plane's grid nodes, at
their true positions.

It places the samples with the far-field projector's Yee readout (each
component from the run's grid lines, a symmetry-reduced plane mirrored at
each component's own points, E and H brought onto one surface along the
normal as the projector reads a box face) and then moves each component
that sits half a cell off the node onto it, by linear interpolation between
its two samples either side.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .._compat import caller_stacklevel
from ..components import frame as _frame
from ._constants import C0
from .mode_overlap import _TRANSVERSE
from .near_field import (
    _FACE_SUFFIX,
    _YEE_OFFSET,
    _between,
    _face_fields,
    _frame_offset,
    _Grid,
    _grid_block,
    _infer_normal_axis,
    _mesh,
    _normal_samples,
    _recipes_of,
    _run_symmetry,
    _unfold_face,
    _where,
)

__all__ = ["colocated_fields"]

#: The six components, in the order the dataset lists them.
_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: The in-plane axes of a plane normal to each axis, in xyz order: the
#: dataset's dims after ``f``.
_NATURAL: Dict[str, Tuple[str, str]] = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}

#: Relative difference under which a node counts as midway between the two
#: samples it is interpolated from (weights exactly 1/2, as on a uniform
#: axis).
_MIDWAY_RTOL = 1e-9

_WHO = "colocated_fields"

#: What a plane with E and H half a cell apart along its normal costs a
#: product of the two, and how to record one that has them on one surface.
_PRODUCTS = (
    "Each point of a product of E and H, such as the Poynting flux density, is then "
    "wrong at first order in the cell (about 4 % of its peak at 20 cells per wavelength "
    "on a plane 0.6 um from a dipole); summed over the plane, that error leaves a term "
    "along the plane's edges. To bring E and H onto one surface, record the plane one "
    "cell thick: from the quarter point of a cell to that of the next along its normal "
    "(size dl centred at (k + 3/4) dl on a uniform mesh), as a FarFieldMonitor.plane "
    "records it")

#: The ``colocated`` attr's account of the normal for a plane one cell thick.
_SURFACE_TEXT = {"e": "one surface, the E plane of the upper sample",
                 "h": "one surface, the H plane of the lower sample"}


@contextlib.contextmanager
def _named():
    """Raise the projector's refusals under this function's name: the
    shared readout says ``far_field: ...``."""
    try:
        yield
    except ValueError as exc:
        text = str(exc)
        if text.startswith("far_field: "):
            raise ValueError(f"{_WHO}: {text[len('far_field: '):]}") from exc
        raise


def _warn(message: str) -> None:
    warnings.warn(f"{_WHO}: {message}", UserWarning, stacklevel=caller_stacklevel())


def _holders(data, simulation) -> List[object]:
    """Where a far-field recipe can come from: ``simulation=``, then the
    run's own simulation."""
    out = [simulation] if simulation is not None else []
    try:
        out.append(getattr(data, "simulation", None))
    except (AttributeError, OSError, ValueError):
        pass
    return [h for h in out if h is not None]


def _in(data, name: str) -> bool:
    try:
        return name in data
    except TypeError:
        return False


def _face_of_recipe(data, name: str, simulation):
    """``(recipe, axis, sign)`` when ``name`` is a face a
    ``FarFieldMonitor`` compiled, else None."""
    for holder in _holders(data, simulation):
        for rec in _recipes_of(holder).values():
            for face, ax, sg in rec.get("faces", ()):
                if str(face) == name:
                    return rec, str(ax), float(sg)
    return None


def _surface_faces(data, name: str, simulation) -> Optional[Tuple[List[str], bool]]:
    """``(faces, by_name)``: the face monitors of the ``FarFieldMonitor``
    named ``name``, from its recipe (``by_name`` False) or, without one, by
    their names ``<name>.x_minus`` ... as ``far_field`` finds them; or
    None."""
    for holder in _holders(data, simulation):
        rec = _recipes_of(holder).get(name)
        if rec is not None:
            return [str(f[0]) for f in rec["faces"]], False
    try:
        keys = list(data.keys())
    except (AttributeError, TypeError):
        return None
    prefix = f"{name}."
    found = sorted(k for k in keys if isinstance(k, str) and k.startswith(prefix)
                   and k[len(prefix):] in _FACE_SUFFIX)
    return (found, True) if found else None


def _suffix_axis(name: str) -> Optional[Tuple[str, float]]:
    """The axis and outward sign a ``FarFieldMonitor`` face's name gives."""
    tail = name.rsplit(".", 1)[-1] if "." in name else ""
    return _FACE_SUFFIX.get(tail)


def _quasi_2d(data, grid: _Grid, axis: str) -> bool:
    """Whether the run's domain is one cell along ``axis`` (a quasi-2-D
    axis, along which the fields do not vary)."""
    if grid.one_cell_width(axis) is not None:
        return True
    shape = _grid_block(data).get("shape")
    try:
        return int(shape["xyz".index(axis)]) == 1
    except (TypeError, ValueError, IndexError, KeyError):
        return False


def _cell_count(data, sim, axis: str) -> Optional[int]:
    """The run's cell count along ``axis``: its manifest's grid shape, else
    the one ``sim`` resolves (its grid lines, or its size over its cell),
    else None."""
    i = "xyz".index(axis)
    try:
        return int(_grid_block(data)["shape"][i])
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    from ..components.grid import resolved_cell_counts
    try:
        floor = sim._axis_min_cells() if hasattr(sim, "_axis_min_cells") else (1, 1, 1)
        return int(resolved_cell_counts(sim.size_um, sim.grid, floor)[i])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def _period(data, simulation, grid: _Grid, axis: str, base: np.ndarray,
            names: Sequence[str] = ()):
    """``((L_um, phase), None)`` when the plane spans the whole of a
    periodic or Bloch axis of its run, recorded at every cell: the sample of
    a half-offset component before node 0 is then the last one, ``L``
    back, times ``phase = e^{-i k L}`` (the solver's Bloch condition
    ``F(x + L) = F(x) e^{i k L}``; 1 on a plain periodic axis). The
    boundaries come from ``simulation=`` or the run's simulation, the cell
    count from the run's manifest or the simulation. Otherwise ``(None,
    why)``: ``why`` says why a plane on such an axis cannot wrap (recorded
    with a stride across the period, or its cell count not known) when a
    recorded component sits half a cell off the node along it (else no
    row is dropped and ``why`` is None); a plane short of the period is a
    cut plane and gives no reason either."""
    i = "xyz".index(axis)
    sim = next((h for h in _holders(data, simulation)
                if getattr(h, "boundaries", None) is not None), None)
    kind = getattr(getattr(sim, "boundaries", None), axis, None) if sim is not None else None
    if kind not in ("periodic", "bloch"):
        return None, None
    dropping = any(_YEE_OFFSET[nm][i] for nm in names)
    n = _cell_count(data, sim, axis)
    if n is None:
        return None, (f"along the {kind} axis {axis} the run's cell count is not known (no "
                      "manifest, and no simulation= with a grid)") if dropping else None
    b = np.asarray(base, dtype=np.float64)
    try:
        idx = np.asarray(grid.index(axis, b), dtype=np.int64)
    except ValueError:
        return None, None
    step = int(np.max(np.diff(idx))) if idx.size >= 2 else 1
    if n < 2 or idx.size == 0 or idx[0] != 0 or idx[-1] + step < n:
        return None, None                           # short of the period: a cut plane
    if step != 1 or idx.size != n:
        return None, (f"it spans the {kind} axis {axis} but was recorded every {step} cells "
                      "along it") if dropping else None
    q = grid.lines.get(axis)
    length = float(q[-1] - q[0]) if q is not None else n * float(grid.dl)
    k = 0.0
    if kind == "bloch":
        bk = getattr(sim, "bloch_k_per_um", None) or (0.0, 0.0, 0.0)
        k = float(bk[i])
    return (length, complex(np.exp(-1j * k * length))), None


def _nodes(base: np.ndarray, mirrored: bool) -> np.ndarray:
    """An in-plane axis's grid nodes: the recorded base nodes, with their
    mirror images when the plane was unfolded through a symmetry plane
    there (the node on the plane once), as the readout mirrors a component
    that sits on the node."""
    b = np.asarray(base, dtype=np.float64)
    return np.concatenate([-b[1:][::-1], b]) if mirrored else b


def _onto(values: np.ndarray, p: np.ndarray, x: np.ndarray, dim: int) -> np.ndarray:
    """``values`` (sampled at the ascending positions ``p`` along array
    axis ``dim``) interpolated linearly onto the points ``x``, each from
    the two samples either side of it, with the projector's rule
    (:func:`~photonhub.analysis.near_field._between`): weights from the
    point's distances to the two, exactly 1/2 and 1/2 where it lies midway
    (to :data:`_MIDWAY_RTOL`), as on a uniform axis."""
    j = np.searchsorted(p, x)
    d0 = x - p[j - 1]
    d1 = p[j] - x
    d1 = np.where(np.abs(d0 - d1) <= _MIDWAY_RTOL * (d0 + d1), d0, d1)
    shape = [1, 1]
    shape[dim] = x.size
    return _between(np.take(values, j - 1, axis=dim), np.take(values, j, axis=dim),
                    d0.reshape(shape), d1.reshape(shape))


def _colocate(face: Dict[str, object], names: Sequence[str], nodes: Dict[str, np.ndarray],
              quasi: Dict[str, bool], where: str, period: Optional[Dict] = None):
    """The components of ``face`` on the grid nodes: ``({name: values[i_t2,
    i_t1]}, kept nodes along t1, kept nodes along t2)``. Along each in-plane
    axis a component half a cell off the node is interpolated onto it
    (:func:`_onto`); the nodes kept are those with samples of it on both
    sides. On an axis the plane spans whole and the run wraps
    (``period[t] = (L, phase)``, :func:`_period`) the last sample, wrapped,
    stands before the first, so every node is kept. An axis with one sample
    (a quasi-2-D axis, ``quasi``) keeps it."""
    t1, t2 = _TRANSVERSE[face["axis"]]
    fields = {nm: face["fields"][nm] for nm in names}
    vals = {nm: np.asarray(v[0]) for nm, v in fields.items()}
    kept = {}
    for t, dim, pos_i in ((t1, 1, 1), (t2, 0, 2)):
        k = "xyz".index(t)
        x = nodes[t]
        off = [nm for nm in names if _YEE_OFFSET[nm][k]]
        if not off:
            kept[t] = x
            continue
        if x.size == 1:
            if not quasi[t]:
                raise ValueError(
                    f"{_WHO}: {where} has one sample along {t}, so the components "
                    f"half a cell off the node along {t} ({', '.join(off)}) cannot be "
                    f"brought onto it; record the plane over at least two cells "
                    f"along {t}")
            kept[t] = x
            continue
        wrap = (period or {}).get(t)
        if wrap is not None:
            length, phase = wrap
            for nm in off:
                pn = np.asarray(fields[nm][pos_i], dtype=np.float64)
                last = phase * np.take(vals[nm], [-1], axis=dim)
                vals[nm] = np.concatenate([last, vals[nm]], axis=dim)
                pos = (pn[-1:] - length, pn)
                fields[nm] = ((None, np.concatenate(pos), fields[nm][2]) if pos_i == 1
                              else (None, fields[nm][1], np.concatenate(pos)))
        p = np.asarray(fields[off[0]][pos_i], dtype=np.float64)
        mask = (x > p[0]) & (x < p[-1])
        if not np.any(mask):
            raise ValueError(f"{_WHO}: {where} has no grid node between two samples "
                             f"along {t}")
        xk = x[mask]
        for nm in names:
            if _YEE_OFFSET[nm][k]:
                pn = np.asarray(fields[nm][pos_i], dtype=np.float64)
                vals[nm] = _onto(vals[nm], pn, xk, dim)
            else:
                vals[nm] = np.compress(mask, vals[nm], axis=dim)
        kept[t] = xk
    return vals, kept[t1], kept[t2]


def _pick_freqs(da: xr.DataArray, freqs_hz) -> np.ndarray:
    """The recorded frequencies read: every one, or the one nearest each
    asked for."""
    rec = np.atleast_1d(np.asarray(da.coords["f"].values, dtype=np.float64))
    if freqs_hz is None:
        return rec
    want = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64)).ravel()
    if want.size == 0:
        raise ValueError(f"{_WHO}: freqs_hz names no frequency; give at least one, or leave "
                         "it out to read every recorded one")
    return rec[[int(np.argmin(np.abs(rec - f))) for f in want]]


def colocated_fields(
    data,
    monitor_name: str,
    *,
    axis: Optional[str] = None,
    freqs_hz: Union[None, float, Sequence[float], np.ndarray] = None,
    simulation=None,
    dl_um: Optional[float] = None,
) -> xr.Dataset:
    """A recorded field plane with E and H on the same points, at their
    true coordinates.

    The solver records each component at its own Yee point (``E_c`` half a
    cell along ``c``, ``H_c`` half a cell along the two other axes), while
    ``data[name]`` labels every component with its cell's base node, so a
    product of components read at those labels (a Poynting flux ``E x
    H*``, an intensity, an overlap) is wrong at first order in the cell.
    This returns the plane with every recorded component on the plane's
    grid nodes:

    * **In the plane**, on the recorded base nodes (``data[name]``'s own
      coordinates, which are the grid lines, in either order), less the
      first node along an axis where some component sits half a cell past
      the node and has no sample before it. Such a component is
      interpolated linearly onto each node from its two samples either
      side: weights 1/2 and 1/2 on a uniform mesh, from the two cells'
      widths on a graded one, from the distances to two samples a stride
      apart on a plane recorded with ``interval_space``. On an axis the
      plane spans whole, recorded at every cell, where the run wraps
      (periodic or Bloch boundaries, from the run's simulation or
      ``simulation=``; the cell count from the run's manifest or that
      simulation), the sample before the first node is the last one, a
      period back, times the Bloch phase ``e^{-i k L}``, and every node is
      kept. A plane across the period recorded with a stride, or whose cell
      count is not known, cannot wrap and warns.
    * **Along the normal**, a plane recorded one cell thick (two samples on
      neighbouring grid lines: from the quarter point of a cell ``k`` to
      that of cell ``k + 1``, which is size ``dl`` centred at ``(k + 3/4)
      dl`` on a uniform mesh; every face of a
      :class:`~photonhub.FarFieldMonitor`, on any mesh) is read on one
      surface, the E plane of its upper sample: tangential E and normal H
      as recorded there, tangential H and normal E interpolated onto it
      from the two cells' H planes. That is where a ``PowerMonitor`` at
      that node reads H, and how :func:`far_field` brings H onto E on a box
      face. A plane recorded with one sample cannot be moved along its
      normal: tangential E and normal H sit on its plane, tangential H and
      normal E half a cell past it (on a graded mesh, the middle of the
      cell), where the solver recorded them, with a warning when the plane
      holds both. Each variable's ``attrs`` give its position along the
      normal; a scalar coordinate named by the normal axis is present when
      all share one surface.

    The samples are placed with :func:`far_field`'s readout: from the run's
    grid lines (a ``RunResult``'s manifest lists them, so graded meshes
    work), and a symmetry-reduced plane that starts on a symmetry plane is
    first unfolded into the full plane, each component mirrored at its own
    Yee point with the sign the plane gives it (a plane wholly on the kept
    side comes back as recorded).

    **Accuracy.** The interpolation is second order: a plane wave of
    wavenumber ``k_a`` along an interpolated axis ``a`` is read
    ``cos(k_a d / 2)`` of its value (``d`` the cell), ``1 - (k_a d)^2 /
    8``; on a plane recorded every ``s`` cells the node lies half a cell
    from one sample and ``s - 1/2`` cells from the other, and the error is
    about ``2 s - 1`` times larger. On a dipole's exact field 0.3 to 0.6 um
    from the plane, every component reads within 1.9 % of its peak (one
    normal sample) and 2.2 % (two) at 20 cells per wavelength, 0.47 % and
    0.59 % at 40. It grows quickly as the plane nears a source: for a
    plane 0.09 um from a dipole, 24 to 34 % (one sample) and 32 to 62 %
    (two) at 20 cells, 8.0 to 9.7 % and 9.5 to 13 % at 40, over the
    dipole's orientation and the plane's axis and side.

    **Against far_field.** On a uniform mesh, a plane recorded at every
    cell with one sample along its normal is read on :func:`far_field`'s
    own surfaces for a single plane (E on its plane, H half a cell past
    it). There the colocated tangential fields, taken as the currents ``J =
    n x H`` and ``M = -n x E`` and integrated with its quadrature, give its
    radiation integrals times ``cos(k r_a d / 2)`` exactly, ``r_a`` the
    direction's cosine along the axis ``a`` each current's field was
    interpolated along, when the field has died out at the plane's edges:
    the same total current and first moment, and its amplitude along the
    plane's normal to round-off. On a plane that wraps across a period it
    holds at the plane's diffraction orders (``k r_a = k_B + 2 pi m / L``
    on each wrapped axis, ``k_B`` its Bloch wavevector), the directions a
    periodic structure radiates into, but not between them, and not for
    the first moment; on a Bloch plane the normal is not an order (the
    total current then differs, by 2.8 % at 20 cells per wavelength on an
    oblique plane wave). On a
    plane recorded every ``s`` cells the total current and first moment are
    kept but the factor is ``(s - 1/2)/s e^{i k r_a d/2} + e^{-i k r_a
    (s - 1/2) d}/(2 s)``; on a graded mesh the relation holds to second
    order (along the normal 2.7e-4, 6.6e-5 and 1.7e-5 of the amplitude,
    every cell cut in two, then four). A plane recorded one cell thick is
    read on its upper E plane, while ``far_field`` reads a single plane
    from its first sample, so along the normal the two differ by the
    normal interpolation's factor, ``(1 - cos(k dl / 2)) / 2``, about ``(k
    dl)^2 / 16`` of the amplitude: 6.15e-3 at 20 cells per wavelength,
    1.54e-3 at 40. On a ``FarFieldMonitor`` box face the fields are, bit for
    bit, ``far_field``'s H-onto-E pass there brought onto the nodes
    (``far_field`` averages that pass with its E-onto-H pass).
    :func:`far_field` itself reads each component where it was recorded
    and does not interpolate in the plane.

    **Power.** A product of two colocated components carries the factor
    twice. On an FDTD dipole in vacuum (uniform mesh, PML 0.5 um, a plane
    0.6 um above it spanning the domain, its edges in the PML), the flux
    ``1/2 Re (E x H*) . n`` summed over the nodes of a plane recorded one
    cell thick reads 0.46 %, 0.21 % and 0.12 % below a ``PowerMonitor`` on
    the same plane at 20, 30 and 40 cells per wavelength (the monitor pairs
    each component with the one that shares its Yee point, and also
    averages H onto the E plane). With one sample, H half a cell above E,
    each point of ``E x H*`` is wrong at first order (4.2 %, 2.8 % and
    2.1 % of the peak flux density); summed over a plane that error leaves
    a term along the plane's edges, which vanishes where the field has
    died out there or across a periodic cell and remains where the plane
    is cut or runs into the PML, as here: 2.0 %, 1.3 % and 0.91 % low
    (1.6 %, 0.96 % and 0.68 % with the PML 1.5 um from the dipole). Record
    a plane one cell thick for any product of E and H. Those figures hold
    for a plane whose edges reach the PML or where the field has died out.
    A plane cut in the open loses about one cell's strip of flux at its
    edges, first order, since no node precedes a half-offset component's
    first sample: cut 1 um from the dipole axis, 1.8 %, 1.1 % and 0.78 %
    low; cut 0.5 um from it, 5.6 %, 3.7 % and 2.8 %. Record such a plane a
    few cells wider than the region it is integrated over. A plane that
    spans a periodic cell, recorded at every cell, wraps and keeps every
    node: one cell thick across a 1.6 um cell, 0.62 %, 0.28 % and 0.16 %
    low, second order
    (7.4 %, 4.7 % and 3.5 % when its first row and column were dropped).

    Parameters
    ----------
    data:
        A :class:`~photonhub.data.RunResult`, or a mapping of recorded
        ``field_dft`` arrays (in the wire frame, as ``far_field`` takes
        them).
    monitor_name:
        A ``ProfileMonitor`` plane, a face of a
        :class:`~photonhub.FarFieldMonitor` (``"<name>.z_plus"`` ...), or a
        ``FarFieldMonitor`` plane by its own name. A face's dataset carries
        ``n_medium``, the index its recipe recorded, which ``far_field``
        projects into, when the recipe is known (the run keeps it, a
        reloaded result reads it back from ``client.json``, or
        ``simulation=`` gives it). Without it, a surface read by its own
        name (its face found by name, as ``far_field`` finds it) or a face
        of a box (two or more faces recorded under its prefix) warns that
        the index is not known; a lone face read by its own name is an
        ordinary plane. An open box's cut and a plane's edge window are not
        applied: they weight the projection, not the fields.
    axis:
        The plane's normal, ``"x"``, ``"y"`` or ``"z"``. By default the
        axis with one sample, or a ``FarFieldMonitor`` face's own; pass it
        for a plane recorded one cell thick.
    freqs_hz:
        Frequencies to read (Hz): the recorded frequency nearest each. By
        default every recorded one.
    simulation:
        The :class:`~photonhub.Simulation` the data came from: its symmetry
        planes (a ``RunResult`` records them; both must agree), its
        boundaries and grid (a periodic axis wraps; a plain mapping takes
        the cell count from it) and its far-field recipes. Needed for a
        plain mapping from a symmetry-reduced or periodic run.
    dl_um:
        The uniform cell size, for a plain mapping whose cell size nothing
        else tells (a ``RunResult``'s manifest, or a plane's two normal
        samples). Checked as ``far_field`` checks it.

    Returns
    -------
    xarray.Dataset
        Dims ``("f", a, b)``, with ``a``, ``b`` the plane's two in-plane
        axes in xyz order (``("f", "x", "y")`` for a plane normal to z),
        their coordinates the nodes in um in the frame of ``data[name]``;
        ``wlen_um`` (vacuum) beside ``f``, and ``n_medium`` for a
        far-field face whose recipe is known. Data variables ``Ex`` ...
        ``Hz``, those the plane recorded (a ``FarFieldMonitor`` face records
        the four tangential ones), in V/m and A/m for the sources as
        declared (``attrs["normalization"]`` says when the reader could not
        restore them).

    Raises
    ------
    ValueError
        For what ``far_field`` refuses on a plane, with its wording: more
        than two samples along the normal, a ``dl_um`` that disagrees with
        the run or the plane, samples off the run's grid lines, symmetry
        planes that disagree with the run's; and where the readout cannot
        place the samples (a plain array with no cell size, or from a
        graded mesh without the run's grid lines), where ``far_field``
        falls back to its earlier readout with a warning. Also a plane
        with one sample along an in-plane axis (unless the domain is one
        cell there), a ``FarFieldMonitor`` box named instead of one of its
        faces, and an empty ``freqs_hz``.

    Warns
    -----
    UserWarning
        A plane recorded with one sample along its normal that holds both
        E and H (they are half a cell apart along it), or with two samples
        that are not neighbours (read from the first); a plane across a
        periodic or Bloch axis that cannot wrap (recorded with a stride, or
        its cell count not known), once per plane; a far-field surface or
        box face whose recipe is not known (no ``n_medium``).

    Examples
    --------
    The power through a plane recorded one cell thick, from the colocated
    fields, and a map of ``|E|^2``::

        ds = colocated_fields(data, "lid", axis="z").isel(f=0)
        sz = 0.5 * np.real(ds["Ex"] * np.conj(ds["Hy"]) - ds["Ey"] * np.conj(ds["Hx"]))
        power_w = float(sz.integrate(["x", "y"])) * 1e-12      # um^2 -> m^2
        e2 = abs(ds["Ex"]) ** 2 + abs(ds["Ey"]) ** 2 + abs(ds["Ez"]) ** 2
    """
    return _colocated(data, monitor_name, axis=axis, freqs_hz=freqs_hz,
                      simulation=simulation, dl_um=dl_um, surface="e")


def _colocated(data, monitor_name, *, axis, freqs_hz, simulation, dl_um,
               surface: str = "e") -> xr.Dataset:
    """:func:`colocated_fields`, with the surface a plane recorded one cell
    thick is read on (not public): ``"e"`` (the default, H onto the upper
    E plane) or ``"h"`` (E onto the H plane of the lower sample, the
    projector's other pass)."""
    if surface not in ("e", "h"):
        raise ValueError(f"surface must be 'e' or 'h', got {surface!r}")
    if not isinstance(monitor_name, str) or not monitor_name:
        raise ValueError(f"{_WHO}: monitor_name must be a monitor's name, got {monitor_name!r}")
    if axis is not None and axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    name = monitor_name
    surface_by_name = None
    if not _in(data, name):
        listed = _surface_faces(data, name, simulation)
        if listed is not None:
            faces, by_name = listed
            if len(faces) != 1:
                raise ValueError(
                    f"{_WHO}: {name!r} is a far-field surface of {len(faces)} faces; name "
                    f"one of them: {faces}")
            surface_by_name = name if by_name else None
            name = faces[0]
    da = _frame.wire_array(data, name)
    found = _face_of_recipe(data, name, simulation)
    recipe = found[0] if found else None
    sign = found[2] if found else 1.0
    if found:
        if axis is not None and axis != found[1]:
            raise ValueError(
                f"{_WHO}: {name!r} is a far-field face normal to {found[1]}, not {axis}")
        axis = found[1]
    else:
        # a far-field surface with no recipe: named itself, its face found by
        # name (as far_field finds them, and warns), or a face of a box,
        # which records two or more faces under one prefix (a lone
        # "lid.z_plus" is an ordinary plane): its medium is not known
        surface_name = surface_by_name
        if surface_name is None and _suffix_axis(name) is not None:
            prefix = name.rsplit(".", 1)[0]
            listed = _surface_faces(data, prefix, None)
            if listed is not None and listed[1] and len(listed[0]) >= 2:
                surface_name = prefix
        if surface_name is not None:
            _warn(f"the far-field surface {surface_name!r} has no recipe here (the result's "
                  "client.json is missing, or this is not the result of its run), so its "
                  f"faces were found by their names and the medium its face {name!r} sits "
                  "in is not known: the dataset carries no n_medium. Pass simulation= (the "
                  "Simulation the run came from) to read it")
    if "f" not in getattr(da, "dims", ()) or "component" not in getattr(da, "dims", ()):
        raise ValueError(
            f"{_WHO}: {name!r} is not a recorded frequency-domain field plane (a "
            f"ProfileMonitor's field_dft, dims f, component, z, y, x); it has the dims "
            f"{tuple(getattr(da, 'dims', ()))}")
    if axis is None:
        try:
            axis = _infer_normal_axis(da)
        except ValueError:
            guess = _suffix_axis(name)
            if guess is None:
                raise
            axis, sign = guess
    recorded = [str(c) for c in np.asarray(da.coords["component"].values).ravel()]
    names = [c for c in _FIELDS if c in recorded]
    if not names:
        raise ValueError(f"{_WHO}: {name!r} records none of {_FIELDS} (has {recorded})")
    where = _where(name, axis, sign)
    n_normal = _normal_samples(da, axis)
    if n_normal > 2:
        raise ValueError(
            f"{_WHO}: {where} has {n_normal} samples along its normal; a plane needs "
            f"one, a box face one or two (recorded one cell thick). Select them with "
            f".isel({axis}=...)")

    t1, t2 = _TRANSVERSE[axis]
    descending = [t for t in (t1, t2) if t in da.dims and da.sizes[t] > 1
                  and np.any(np.diff(np.asarray(da.coords[t].values, dtype=np.float64)) < 0)]
    if descending:
        da = da.sortby(descending)
    ia = "xyz".index(axis)
    # E and H half a cell apart along the normal unless the plane is one cell
    # thick: tangential E and normal H on the node, tangential H and normal E
    # past it
    split = (len({_YEE_OFFSET[nm][ia] == 0.0 for nm in names}) == 2
             and len({nm[0] for nm in names}) == 2)

    with _named():
        grid, why = _mesh(data, dl_um, [(da, axis, sign)], [name])
        if why:
            raise ValueError(
                f"{_WHO}: the Yee-lattice readout cannot place the samples of {why} "
                "(pass the RunResult the planes came from: its manifest lists the grid "
                "lines)")
        if grid is None:
            raise ValueError(
                f"{_WHO}: the Yee-lattice readout cannot place the samples of planes "
                "whose cell size it cannot tell (pass dl_um=, or the RunResult the "
                "planes came from)")
        symmetry = _run_symmetry(data, simulation,
                                 recipe.get("symmetry") if recipe else None)
        cn = np.asarray(da.coords[axis].values, dtype=np.float64).ravel()
        one_surface = n_normal == 2 and grid.adjacent(axis, cn)
        if n_normal == 2 and not one_surface:
            _warn(f"{where} has two samples along its normal that are not on neighbouring "
                  "grid lines, so E and H cannot be brought onto one surface; it is read "
                  "from its first sample, tangential E and normal H on that plane, "
                  "tangential H and normal E half a cell past it"
                  + (f". {_PRODUCTS}" if split else ""))
        elif n_normal == 1 and split:
            _warn(f"{where} was recorded with one sample along its normal, so tangential H "
                  "and normal E sit half a cell past tangential E and normal H along it "
                  f"(each variable's attrs say where). {_PRODUCTS}")

        freqs = _pick_freqs(da, freqs_hz)
        planes = [0, 0, 0]
        per_f = []
        nodes = quasi = None
        for f in freqs:
            face = _face_fields(da, axis, sign, float(f), grid=grid, positions=True,
                                surface=one_surface, plane=surface, colocate=True,
                                label=name, box=False, components=names)
            if nodes is None:
                for t in (t1, t2):
                    i = "xyz".index(t)
                    base = np.asarray(face["base"][t], dtype=np.float64)
                    step = abs(float(base[1] - base[0])) if base.size >= 2 else 0.0
                    if symmetry and symmetry[i] and base.size and \
                            abs(float(base[0])) <= max(1e-6 * step, 1e-12):
                        planes[i] = symmetry[i]
                nodes = {t: _nodes(face["base"][t], bool(planes["xyz".index(t)]))
                         for t in (t1, t2)}
                read = {t: _period(data, simulation, grid, t, face["base"][t], names)
                        for t in (t1, t2) if not planes["xyz".index(t)]}
                period = {t: r[0] for t, r in read.items()}
                whys = [r[1] for r in read.values() if r[1]]
                if whys:
                    _warn(f"{where} cannot wrap across the period of its run: "
                          f"{'; '.join(whys)}. It loses its first node row along each such "
                          "axis (about one cell's strip of a sum over the plane). Record the "
                          "plane at every cell across the period, and pass the RunResult or "
                          "simulation= so the period is known")
                quasi = {t: _quasi_2d(data, grid, t) for t in (t1, t2)}
            if any(planes):
                face = _unfold_face(face, tuple(planes))
            per_f.append((face, _colocate(face, names, nodes, quasi, where, period)))

    face0, (_, k1, k2) = per_f[0]
    offset = _frame_offset(data)
    a, b = _NATURAL[axis]
    # values are [i_t2, i_t1]; the dataset's in-plane dims run in xyz order
    flip = (t2, t1) != (a, b)
    stack = {nm: np.stack([v[nm].T if flip else v[nm] for _, (v, _, _) in per_f])
             for nm in names}
    coord = {t1: k1 + offset["xyz".index(t1)], t2: k2 + offset["xyz".index(t2)]}
    normal_um = {nm: float(face0["n"][nm]) + float(offset[ia]) for nm in names}

    with np.errstate(divide="ignore"):
        wlen = C0 / freqs * 1e6
    coords: Dict[str, object] = {
        "f": ("f", freqs, {"units": "Hz", "long_name": "frequency"}),
        "wlen_um": ("f", wlen, {"units": "um", "long_name": "vacuum wavelength"}),
        a: (a, coord[a], {"units": "um", "long_name": f"grid node along {a}"}),
        b: (b, coord[b], {"units": "um", "long_name": f"grid node along {b}"}),
    }
    shared = set(normal_um.values())
    if len(shared) == 1:
        coords[axis] = ((), shared.pop(), {"units": "um",
                                           "long_name": "the surface E and H share"})
    if recipe is not None:
        rf = np.asarray(recipe["freqs_hz"], dtype=np.float64)
        rn = np.asarray(recipe["n_medium"], dtype=np.float64)
        n_f = np.array([rn[int(np.argmin(np.abs(rf - f)))] for f in freqs])
        coords["n_medium"] = ("f", n_f, {"long_name": "index of the medium the far-field "
                                                      "surface radiates into (its recipe)"})
    data_vars = {}
    for nm in names:
        kind = "electric" if nm[0] == "E" else "magnetic"
        data_vars[nm] = (("f", a, b), stack[nm], {
            "units": "V/m" if nm[0] == "E" else "A/m",
            "long_name": f"{kind} field {nm[1]} component, on the grid nodes",
            f"{axis}_um": normal_um[nm]})
    src = getattr(da, "attrs", {}) or {}
    attrs: Dict[str, object] = {
        "monitor": name,
        "normal_axis": axis,
        "colocated": ("grid nodes: in the plane each component half a cell off the node "
                      "interpolated linearly onto it; along the normal "
                      + (_SURFACE_TEXT[surface] if one_surface
                         else "as recorded (one sample): tangential E and normal H on "
                              "the plane, tangential H and normal E half a cell past it")),
        "time_convention": "exp(-i omega t)",
    }
    if any(planes):
        attrs["unfolded"] = [t for t in "xyz" if planes["xyz".index(t)]]
    for key in ("normalization", "norm_amplitude"):
        if src.get(key) is not None:
            attrs[key] = src[key]
    if src.get("aborted"):
        attrs["aborted"] = True
        attrs["abort_reason"] = src.get("abort_reason", "")
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)
