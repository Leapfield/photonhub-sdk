"""Near-to-far-field (NTFF) projection, a recorded near-field surface -> the
far-zone radiation pattern, via the surface-equivalence theorem.

This is a pure-host post-processor (no engine / C++ change). Given the
frequency-domain tangential ``E`` and ``H`` recorded by a ``ProfileMonitor``
on a plane (or the six faces of a box), it computes the far field
``E_theta(theta, phi, f)``, ``E_phi(theta, phi, f)``, the radiation intensity
``U = r^2 |E|^2 / (2 eta)`` and the (optionally) normalized directivity, in
arbitrary directions ``(theta, phi)``. This is the near-to-far-field (NTFF)
transformation. The surface radiates into a homogeneous, lossless,
nonmagnetic half-space of index ``n`` (``n_medium``, default 1: vacuum); see
"Medium" below.

Physics / method
================
**Surface equivalence theorem.** On a closed surface with outward unit normal
``n_hat`` carrying the simulated near fields ``(E, H)``, the equivalent surface
currents that reproduce the *exterior* field are

    J = n_hat x H            (electric current)
    M = -n_hat x E           (magnetic current)

For an OPEN surface (a single plane just above a device) the same currents are
used; the projection is then a Kirchhoff/equivalence approximation of the
upper-half-space radiation (exact when the plane captures essentially all of
the outgoing field, i.e. the standard "monitor box lid" use). Both faces of a
closed box are summed with their respective outward normals.

**Radiation vectors.** With observation unit vector

    r_hat = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)) ,

the electric and magnetic radiation vectors are the surface Fourier transforms
of the currents,

    N(theta, phi) = integral_S J(r') exp(-i k r_hat . r') dS'
    L(theta, phi) = integral_S M(r') exp(-i k r_hat . r') dS'

with ``k = n k0 = n 2 pi f / c0`` the wavenumber in the medium the surface
radiates into (``n = 1``, vacuum, unless ``n_medium`` says otherwise). The
``-i k`` kernel sign goes with PhotonHub's
``e^{-i omega t}`` time convention (matching ``mode_overlap.py``), for which an
outgoing wave is ``e^{+i k r}/r``: the far-field expansion ``|r - r'| ~ r -
r_hat . r'`` makes ``e^{i k |r-r'|} ~ e^{i k r} e^{-i k r_hat . r'}``. (Balanis
derives these with ``e^{+j omega t}`` / ``e^{-j k r}`` and so writes ``e^{+j k
r_hat.r'}``; flipping the time convention flips the kernel sign.) A ``+x``-tilted
aperture field then steers the beam toward ``+x``, as it must.

**Far-zone fields.** Projecting the radiation vectors onto the spherical
``(theta_hat, phi_hat)`` basis,

    N_theta =  N_x cos(theta) cos(phi) + N_y cos(theta) sin(phi) - N_z sin(theta)
    N_phi   = -N_x sin(phi)            + N_y cos(phi)
    (idem for L)

the far-zone transverse E components are (e.g. Balanis, *Antenna Theory*,
eq. 12-10, adapted to ``e^{-i omega t}``). Balanis writes them with ``e^{+j omega
t}`` / ``e^{-j k r}``; the conversion is ``j -> -i``, which flips BOTH the
exponential and the leading ``j k``, giving

    E_theta = +(i k e^{i k r}) / (4 pi r) * ( L_phi   + eta * N_theta )
    E_phi   = -(i k e^{i k r}) / (4 pi r) * ( L_theta - eta * N_phi   )

with ``eta = eta0 / n`` the wave impedance of the medium (``eta0`` in vacuum).
The ``e^{i k r}/r`` spherical-spreading
factor is *omitted* from the returned ``E_theta``/``E_phi``: we return the
direction-dependent **far-field amplitude** ``F = r e^{-i k r} E`` (units
V, i.e. V/m times metres), from which the radiation intensity is

    U(theta, phi) = r^2 |E|^2 / (2 eta) = (|F_theta|^2 + |F_phi|^2) / (2 eta) .

The radiated power ``P_rad = integral U dOmega`` and the directivity
``D = 4 pi U / P_rad`` then follow; the pattern SHAPE and ``D`` are independent
of the dropped prefactor.

**Medium.** The projection is the free-space Green's function of ONE
homogeneous medium, so it holds when the half-space the surface radiates into
(the outside of a closed box) is filled by a single lossless, isotropic,
nonmagnetic medium of real index ``n >= 1``: a plane inside a substrate,
projected into that substrate, takes ``n_medium = n_substrate``; a closed box
needs every face in that one medium. Only ``k`` and ``eta`` change; the
currents ``J = n_hat x H``, ``M = -n_hat x E`` are the recorded fields as
before. The default ``n_medium=None`` is index 1, bit for bit the vacuum
projection, and changes no number; with ``simulation=`` the medium at each
face is sampled (half a cell beyond the recorded plane, on the side it
radiates into, over the face's extent, with the sampling of
:func:`~photonhub.analysis.diffraction_orders`) and a medium of another
index gives a ``UserWarning``, never a silent change; so does a closed box
across an interface, which an explicit or ``"auto"`` index refuses.
``n_medium="auto"`` uses the sampled index and refuses a lossy medium (loss
tangent above 1e-3). A layered half-space (a plane in the cladding above a
substrate, a box that straddles an interface) is outside this model: its far
field needs the layered Green's function, which this module does not have.
For a fixed current source the power radiated into index ``n`` is ``n``
times the vacuum value (``P = eta k^2 |I l|^2 / (12 pi)``), and the far-field
amplitude ``F`` of a dipole is index-independent (``k eta = k0 eta0``). The
discretization errors (quadrature, Yee staggering) scale with ``n k0 dl``, the
cell in medium wavelengths: a mesh fine enough in vacuum is ``n`` times
coarser in the medium.

**Area element.** ``dS`` is taken from each face's *real* transverse coordinate
spacings (centered-difference cell widths, reusing
:func:`~photonhub.analysis.mode_overlap._cell_widths`), so graded / non-uniform
meshes integrate correctly, no uniform-spacing assumption.

**Yee staggering.** The engine's DFT monitor records each component at its
own Yee point (``E_c`` half a cell along ``c``, ``H_c`` half a cell along
the two other axes), while the recorded DataArray labels every component
with its base node. The default readout
(``far_field(..., yee="exact")``) reads each component where it was
sampled, with no averaging, which needs the grid: on a uniform mesh the
cell size ``dl`` (``dl_um=``, a ``RunResult``'s manifest, or a face's two
normal samples), on a graded mesh the run's grid lines (a ``RunResult``'s
manifest lists them). A component half a cell along an axis sits at the
midpoint of its cell, ``(q_i + q_(i+1)) / 2`` for the grid lines ``q``
(the solver's convention on a graded mesh), which is ``q_i + dl / 2`` on a
uniform axis:

* **A single plane** is read from one sample along its normal, every
  current at its own point (J on the H plane half a cell above E). For an
  open plane J and M each carry half of the forward field, so they need
  not share a surface: on an exact Gaussian beam this reads the far-field
  intensity to 1e-4 of the peak at 20 cells per wavelength, and the
  in-plane stride (``interval_space``) does not matter.
* **The faces of a box** (several faces, at least one of them bounded by
  others in its plane: a closed box, or one with a face left out) must be
  recorded one cell thick, two samples along the normal (a face of size
  ``dl`` along its normal, centred at ``(k + 3/4) dl``, records the base
  nodes ``k`` and ``k + 1``). Toward the back of each face J and M must
  cancel, and they do not when they sit half a cell apart: read from one
  sample, a vacuum dipole's pattern comes back with a front/back asymmetry
  first order in ``dl`` (0.28 of the peak at 25 cells per wavelength). The
  readout brings E and H onto one surface. Averaging E onto the H plane, or
  H onto the E plane, leaves a smaller imbalance of order ``(k dl)^2`` with
  opposite signs in the two cases; the readout averages the two far
  fields, which cancels that imbalance and halves the
  part of the error that depends on where the source sits. The linear
  interpolation along the normal still scales the averaged field by about
  ``1 - (k_n dl)^2 / 8`` (``k_n`` its wavenumber along the normal), a
  second-order error this readout does not remove: at 25 cells per
  wavelength in vacuum up to about 0.80 % of the peak in front/back
  asymmetry with the source near a box corner and up to about 1.06 % near a box
  edge, within 2 % wherever the build does not warn. Box faces are
  integrated with
  fourth-order weights at the box edges, and box faces recorded with
  ``interval_space`` strides are refused. ``colocate=False`` declares
  synthetic fields that already share their sample points: every component
  is then read at the coordinate given, on one surface.

**Graded meshes.** Each sample is placed from the grid lines as above.
Along a box face's normal, E is averaged onto the H plane of the first
sample (the midpoint of its cell, so the weights stay 1/2 and 1/2), and H is
interpolated linearly onto the E plane of the second sample with weights
from the two cells' widths ``d0`` and ``d1``: ``(d1 H_0 + d0 H_1) / (d0 +
d1)``. In plane, the fourth-order box-edge rule is the integral of the
piecewise-cubic interpolant through the samples (four per cell, at their
true positions), cut at the box edges; the same rule holds on any spacing,
it is exact for cubics, and on a uniform mesh its interior weights are the
cell size (a cell whose stencil spans an abrupt jump in the spacing, more
than 3x, takes its linear interpolant instead). A single plane whose
samples are unevenly spaced takes the same interpolant across its samples
and half a cell past each end; evenly spaced ones keep the centred cell
widths. On a uniform mesh every number is the uniform readout's, bit for
bit. The interpolation along each face's normal still scales the field by
about ``1 - (k_n d)^2 / 8`` for that face's cell ``d``; opposite faces in
cells of different sizes no longer share that factor, so it shows as
front/back asymmetry, up to about ``(k d)^2 / 8`` of the coarser cell (on
exact dipole fields 0.54 % of the peak at 25 cells per wavelength, one z
face in a region twice as fine, against 0.23 % on the uniform mesh). On
solver data that term partly cancels the solver's own error there rather
than adding to it.

**Symmetry planes.** A run with ``Simulation.symmetry`` records the part of
each face on the kept side of its mirror planes, which sit at coordinate 0
of the run's (wire) frame. The readout mirrors every component at its own
Yee point, with the sign that plane gives it (:func:`_mirror_parity`): a
component on the mirror plane keeps its one sample there, one half a cell
off it gains the mirrored samples, and none is averaged (unlike
:func:`unfold_symmetry`, which brings them onto the node for display). A
face that crosses a mirror plane is unfolded into the full face; a box face
normal to a mirror plane gains its mirror image, the box face on the far
side, read as the exact mirror of the recorded one. The symmetry planes
come from the run's manifest, or from ``simulation=``; the two must agree.
The earlier readout cannot add that image, so a symmetry-reduced box that
would fall back to it is refused. A ``FarFieldMonitor`` declared across the
planes records exactly that kept part, and its recipe keeps the planes,
which must agree with the run's too.

**Open boxes and windows.** The readout reads each recorded box face on a
surface half a cell or a cell past its grid line along its normal (the two
passes above). A closed box does not notice where its faces sit, but an
open box does: its pattern is the closed box's less the face it leaves out,
and that face's contribution depends on where it would sit. So a
``FarFieldMonitor`` box with faces left out (``exclude``) ends the faces it
keeps at the declared plane of each face left out, on both passes, wherever
that plane falls between their samples (the fourth-order edge rule cuts
there): the open edge does not move with the mesh, and the projection
converges to the open box as declared. A plane's edge window multiplies
each current at its own sample point, after the unfolding, over the plane's
declared extent (mirrored across a symmetry plane), so a run with symmetry
planes and the full run window the same surface. Both come from the
surface's recipe, so ``far_field(data, name)`` applies them; a list of the
same faces is read as recorded (a box with a face left out is then not cut
along that axis at all, see ``faces``), and the earlier readout refuses
them. An open surface is a truncated one (a Kirchhoff approximation): see
the docs for what it costs.

**Target grids and the exact projection.** The far-field amplitudes are
read on one of three grids: directions ``(theta, phi)``; direction cosines
``(ux, uy)`` about an axis (a k-space grid, the plane-wave directions a
periodic plane's grating orders leave in); or points on a plane, each
read in the direction it lies in from an origin. With ``exact=True`` the
field at a finite distance is evaluated instead, from the same currents on
the same samples with the same quadrature, through the full dyadic
Green's function of the medium (``g = e^{ikR} / (4 pi R)``, ``x = k R``)::

    E = sum [ i k eta g (a J + b (R.J) R) - i k g c (R x M) ] dS
    H = sum [ (i k / eta) g (a M + b (R.M) R) + i k g c (R x J) ] dS
    a = 1 + i/x - 1/x^2,  b = -1 - 3i/x + 3/x^2,  c = 1 + i/x

(``R`` the unit vector from the sample to the point; for large ``x`` this
is the far-field kernel), a direct sum over every (point, sample) pair in
bounded blocks. For points on a plane parallel to one evenly sampled
recorded plane the same field follows faster from the plane's angular
spectrum: each tangential E component Fourier transformed from its own
sample points, each plane wave advanced to the target plane, the normal E
from ``div E = 0`` and H from ``k x E / (k eta)``; the zero padding is
doubled until it no longer moves the result (the periodic copies of the
plane that a finite padding adds reach the points only through waves near
grazing). The two agree to 1e-5 on an exact beam that has decayed at its
plane's edges; on a plane that cuts the field they differ by about the
cut's own error. The direct sum loses accuracy near the currents (its
kernel grows as ``1/R^3``; see ``exact``); the angular spectrum does not.

The earlier readout (``yee="legacy"``) averages each component to the
in-plane node and leaves the half-cell E/H offset along the normal; it is
kept for one release. It is also what the default falls back to, with a
warning, where the Yee-lattice readout cannot place the samples: planes
whose cell size or grid lines cannot be told (a plain array with no run
attached), and box faces recorded with one normal sample.

Coordinates / units. Monitor coordinates are in **microns** (the PhotonHub
convention); they are converted to metres internally so ``k r_hat . r'`` is
dimensionless and the SI ``eta`` is consistent. Frequencies are in Hz.

Dependency-light: numpy + the xarray DataArrays the rest of PhotonHub already
produces. No matplotlib, no engine calls.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .._compat import called_from_package, caller_stacklevel, legacy_keywords
from ..components import frame as _frame
from ._constants import _TANGENTIAL, C0, EPS0, ETA0  # noqa: F401  (C0/ETA0 re-exported)
from .mode_overlap import _TRANSVERSE, _cell_widths, _colocate_to_node

__all__ = [
    "C0",
    "ETA0",
    "FarField",
    "far_field",
    "equivalent_currents",
    "unfold_symmetry",
]

Axis = Literal["x", "y", "z"]

#: Direction-chunk budget for the radiation integral: the dense kernel matrix is
#: [chunk_dirs, n_surface_points]; chunk_dirs is chosen so chunk_dirs*n_pts stays
#: near this many complex entries (~16 MB at 1e6), bounding peak memory for a
#: large monitor projected onto many far-field directions.
_DIR_CHUNK: int = 1_000_000


#: The members of :class:`FarField` that hold or read the flat layout
#: (``[frequency, point]`` over a flat list of points), kept for one release:
#: reading one from outside this package gives a ``FutureWarning`` that names
#: its labelled replacement.
_FLAT_MEMBERS = frozenset({
    "theta", "phi", "e_theta", "e_phi", "e_r", "h_r", "h_theta", "h_phi", "intensity",
    "propagating", "r_um", "ux", "uy", "x_um", "y_um", "radiated_power", "directivity",
    "reshape", "cartesian_fields"})

_FLAT_METHODS = frozenset({"radiated_power", "directivity", "reshape", "cartesian_fields"})


@dataclass(frozen=True)
class FarField:
    """The projected field on a target grid, per frequency.

    **Labelled results.** :attr:`dataset` is the projection as an
    :class:`xarray.Dataset`, labelled by frequency and by the grid's own
    axes:

    * an angle grid (``theta_rad``, ``phi_rad``): dims ``("f", "theta",
      "phi")``, the polar angle from ``+z`` and the azimuth from ``+x`` in
      radians;
    * a k-space grid (``ux``, ``uy``): dims ``("f", "ux", "uy")``, the
      direction cosines along the two in-plane axes of ``plane_axis`` in
      xyz order (each coordinate's ``attrs["axis"]`` names its axis: x and
      z on a plane normal to y), with ``theta``, ``phi`` and
      ``propagating`` as coordinates over ``(ux, uy)`` (``nan`` angles, and
      ``False``, where a point has no direction; a direction an exact
      projection of a plane leaves without field stays ``True``, with
      ``nan`` amplitudes and intensity 0, so ``dataset["e_theta"].notnull()``
      marks the points that carry a field, on every grid).
      :func:`~photonhub.analysis.diffraction_orders` runs its ``(k1,
      k2)`` in its own cyclic order instead (z and x on a plane normal to
      y; the same on a plane normal to z), and its order powers carry the
      period's area in um^2, so they are 1e12 times the watts of ``U
      (lambda / n)^2 / (A cos theta)`` from this intensity;
    * a Cartesian grid (``plane_um``, ``x_um``, ``y_um``): dims ``("f", a,
      b)`` with ``a`` and ``b`` the plane's two in-plane axes in xyz order
      (``("f", "x", "y")`` on a plane normal to z, ``("f", "x", "z")`` on
      one normal to y), in um, the plane's
      position a scalar coordinate named by its normal axis, and ``theta``,
      ``phi`` and ``r_um`` (each point's direction and distance from
      ``origin_um``) as coordinates over the plane;
    * points paired elementwise (``mesh=False``): dims ``("f", "point")``,
      with the grid's coordinates along ``point``.

    Along ``f`` (Hz) every array carries the vacuum wavelength ``wlen_um``
    and the index ``n_medium`` projected into as coordinates. The data
    variables are the far-field amplitudes ``e_theta`` and ``e_phi`` (V,
    see below), for an exact projection also ``e_r`` (V) and ``h_r``,
    ``h_theta``, ``h_phi`` (A), the radiation intensity ``intensity``
    (W/sr), and on a meshed angle grid ``radiated_power`` (W, over ``f``)
    and ``directivity``. An exact projection, or a Cartesian grid, carries
    ``r_um``: one distance (a scalar coordinate) on a sphere, one per point
    on a plane. The labelled values are the arrays of the flat layout below
    reshaped onto the grid's axes, bit for bit, as read-only views.

    The methods :meth:`gain`, :meth:`cross_section`, :meth:`polarization`,
    :meth:`power_density` and :meth:`cartesian` return labelled results
    too, :meth:`renormalize` returns a copy read at a distance, and
    :func:`photonhub.viz.plot_far_field_cut` and
    :func:`photonhub.viz.plot_far_field_map` plot them.

    **The flat layout (deprecated).** The attributes ``theta``, ``phi``,
    ``e_theta``, ``e_phi``, ``e_r``, ``h_r``, ``h_theta``, ``h_phi``,
    ``r_um``, ``ux``, ``uy``, ``x_um``, ``y_um``, the properties
    :attr:`intensity` and :attr:`propagating`, and the methods
    :meth:`radiated_power`, :meth:`directivity`, :meth:`reshape` and
    :meth:`cartesian_fields` hold or read the projection as earlier releases
    returned it: arrays indexed ``[i_freq, i_point]`` over a flat list of
    points (the grid's two axes meshed, the first slowest). They are kept
    for one release; reading one gives a ``FutureWarning`` naming its
    labelled replacement. Their values are unchanged.

    Attributes
    ----------
    theta, phi:
        The polar and azimuthal angles (radians) of each point, a flat
        ``(n_point,)`` list (deprecated: ``dataset["theta"]``,
        ``dataset["phi"]``). On a k-space grid they are the directions of
        its points (``nan`` where a point has none, see ``ux``); on a
        Cartesian grid, the direction of each point seen from ``origin_um``.
    freqs_hz:
        1-D array of frequencies (Hz), the ``f`` coordinate of
        :attr:`dataset`.
    e_theta, e_phi:
        Complex far-field AMPLITUDES ``F = r e^{-i k r} E`` (the spherical
        spreading factor ``e^{i k r}/r`` removed), shape ``(n_freq,
        n_point)`` (deprecated: ``dataset["e_theta"]``, ``dataset["e_phi"]``).
        For an exact projection (``exact``) and on a Cartesian grid they are
        the field at each point, scaled the same way with ``r`` its distance
        from ``origin_um`` (``r_um``): ``E_theta = e_theta e^{i k r} / r``.
    n_medium:
        Index of the medium the field was projected into: a float (1.0, the
        default, is vacuum), or a ``(n_freq,)`` array when it varies with
        frequency. It sets the wave impedance ``eta = eta0 / n_medium`` that
        the intensity and every quantity derived from it use.
    grid:
        ``"angle"`` (``theta_rad``, ``phi_rad``), ``"kspace"`` (``ux``,
        ``uy``) or ``"cartesian"`` (points on a plane, ``x_um``, ``y_um``).
    shape:
        ``(n1, n2)`` of a meshed grid (``mesh=True``), else ``None``: the
        sizes of the grid's two dims in :attr:`dataset`.
    exact:
        Whether the field is the exact one at a finite distance (the full
        Green's function, ``far_field(..., exact=True)``) rather than the
        far-field approximation.
    r_um, origin_um:
        The distance of each point from ``origin_um`` (um, a flat
        ``(n_point,)`` array; deprecated: ``dataset["r_um"]``) and that
        origin (um, the frame of ``data[name]``): set for an exact
        projection, on a Cartesian grid and after :meth:`renormalize`, else
        ``None``.
    e_r, h_r, h_theta, h_phi:
        For an exact projection, the radial E and the three H components at
        each point, scaled like ``e_theta``; else ``None`` (in the far zone
        ``E_r = 0`` and ``H = r_hat x E / eta``). Deprecated:
        ``dataset["e_r"]`` and so on.
    ux, uy, direction:
        On a k-space grid, the direction cosines of each point along the
        two in-plane axes of ``plane_axis`` in xyz order (x and y for
        ``plane_axis="z"``, x and z for ``"y"``, y and z for ``"x"``), and
        the half space, ``"+"`` or ``"-"`` along ``plane_axis``. A point with
        ``ux^2 + uy^2 > 1`` has no direction (only an evanescent wave has
        that transverse wavevector, and it never reaches the far zone): its
        fields are ``nan`` and its intensity 0, as ``diffraction_orders``
        reports evanescent orders. ``ux`` and ``uy`` are flat lists
        (deprecated: ``dataset["ux"]``, ``dataset["uy"]``).
    x_um, y_um, plane_um:
        On a Cartesian grid, each point's coordinates along the two in-plane
        axes of ``plane_axis`` in xyz order (flat lists; deprecated: the
        :attr:`dataset` coordinates named by those axes), and the plane's
        coordinate along it (um, the frame of ``data[name]``).
    plane_axis:
        The axis normal to the ``(ux, uy)`` or ``(x, y)`` plane.
    incident_e_v_per_m:
        The amplitude (V/m) of the plane wave a run driven by one
        :class:`~photonhub.TFSFBox` launches, recorded by :func:`far_field`
        from the run's simulation; ``None`` otherwise. :meth:`cross_section`
        divides by its intensity.
    phase_origin_um:
        The point (um, the frame of ``data[name]``) the amplitudes' phase
        refers to, recorded by :func:`far_field`: ``origin_um`` when it was
        given or implied (an exact projection, a Cartesian grid), else the
        corner of the run's frame (``Simulation.origin_um``; 0 for a plain
        mapping of arrays). :meth:`renormalize` measures its distance from
        it. ``None`` for a ``FarField`` built by hand.
    """

    theta: np.ndarray
    phi: np.ndarray
    freqs_hz: np.ndarray
    e_theta: np.ndarray
    e_phi: np.ndarray
    n_medium: Union[float, np.ndarray] = 1.0
    grid: str = "angle"
    shape: Optional[Tuple[int, int]] = None
    exact: bool = False
    r_um: Optional[np.ndarray] = None
    origin_um: Optional[Tuple[float, float, float]] = None
    e_r: Optional[np.ndarray] = None
    h_r: Optional[np.ndarray] = None
    h_theta: Optional[np.ndarray] = None
    h_phi: Optional[np.ndarray] = None
    ux: Optional[np.ndarray] = None
    uy: Optional[np.ndarray] = None
    direction: Optional[str] = None
    x_um: Optional[np.ndarray] = None
    y_um: Optional[np.ndarray] = None
    plane_um: Optional[float] = None
    plane_axis: Optional[str] = None
    incident_e_v_per_m: Optional[float] = None
    phase_origin_um: Optional[Tuple[float, float, float]] = None

    def __getattribute__(self, name: str):
        # The flat layout is deprecated for the user, not for this package,
        # which builds the labelled form from it.
        if name in _FLAT_MEMBERS and not called_from_package() and not _read_by_dataclasses():
            _warn_flat(self, name)
        return object.__getattribute__(self, name)

    # --- the labelled form ------------------------------------------------------

    @functools.cached_property
    def _labelled(self) -> xr.Dataset:
        from ._far_field_results import labelled
        return labelled(self)

    @property
    def dataset(self) -> xr.Dataset:
        """The projection as an :class:`xarray.Dataset` labelled by
        frequency and the grid's axes (see the class description): the
        amplitudes ``e_theta``, ``e_phi`` (and an exact projection's ``e_r``,
        ``h_r``, ``h_theta``, ``h_phi``), ``intensity`` (W/sr), and on a
        meshed angle grid ``radiated_power`` (W, integrated over the grid
        with a ``sin(theta)`` solid-angle weight) and ``directivity`` (``4
        pi U / P``). Select with ``.sel(f=..., method="nearest")`` or
        ``.sel(phi=0.0)``; its arrays are read-only views of the
        projection's own. ``f`` keeps the order the frequencies were
        projected in, which for a ``FarFieldMonitor`` is monotonic;
        frequencies given unsorted or repeated (``freqs_hz=``) need
        ``.sortby("f")`` or ``.isel(f=...)`` instead of ``method="nearest"``.
        ``radiated_power`` is meaningful when the grid tiles a contiguous
        angular region (a full sphere or a hemisphere),
        and a grid that covers only part of the light makes it, and the
        directivity, relative to that part. Paired directions
        (``mesh=False``) and k-space and Cartesian grids have neither."""
        return self._labelled.copy(deep=False)

    def gain(self, input_power_w=None) -> xr.DataArray:
        """Gain ``G = 4 pi U / P_in`` (dimensionless), labelled like
        :attr:`dataset`'s intensity: the intensity against the power the
        source delivers, so ``G = e D`` with ``e`` the radiation efficiency
        (the share of that power that leaves the surface) and ``D`` the
        directivity. Without losses (and a surface that sees all the light)
        ``G = D``.

        ``input_power_w`` is the power the source delivers (W): one number,
        one per projected frequency, or an :class:`xarray.DataArray` over
        ``f`` read at the projected frequencies, such as the summed outward
        flux of :class:`~photonhub.PowerMonitor` planes that close a small
        box centred on the source (a :class:`~photonhub.RunResult` returns
        each plane's in watts). Off centre, such a box's total is off by an
        amount first order in the cell size times the offset (0.9 % for a
        dipole 0.13 um off the centre of a box of half width 0.3 um at 30
        cells per wavelength): centre it, or check it against a larger box.
        The projection cannot tell it, so it is required:
        ``ValueError`` without it. A point dipole's Hertzian power ``eta k^2
        |p|^2 / (12 pi)`` (``p`` its current moment, ``amplitude`` times its
        cell's volume on a uniform mesh) is the power it delivers only
        alone in a homogeneous lossless medium: nearby structures change it
        (the local density of states), so measure it there."""
        from ._far_field_results import per_frequency
        if input_power_w is None:
            raise ValueError(
                "FarField.gain needs the power the source delivers, input_power_w= (W, one "
                "number, one per frequency, or a DataArray over f): the summed outward flux "
                "of PowerMonitor planes closing a small box centred on the source. The projection "
                "cannot tell it; dataset['directivity'] is the gain of a lossless source")
        p = per_frequency(input_power_w, self.freqs_hz, "input_power_w", "W")
        g = 4.0 * np.pi * self._labelled["intensity"] / xr.DataArray(p, dims="f")
        g.name = "gain"
        g.attrs = {"units": "1", "long_name": "gain 4 pi U / P_in"}
        return g

    def cross_section(self, incident_intensity_w_per_m2=None) -> xr.DataArray:
        """Differential scattering cross section ``dsigma/dOmega = U /
        I_inc`` in um^2/sr, labelled like :attr:`dataset`'s intensity: the
        scattered intensity ``U`` (W/sr) against the intensity ``I_inc``
        (W/m^2) of the incident plane wave. Its integral over the sphere
        (``dataset["radiated_power"] / I_inc`` on a full angle grid) is the
        scattering cross section.

        ``incident_intensity_w_per_m2`` is ``I_inc``: one number, one per
        projected frequency, or an :class:`xarray.DataArray` over ``f`` read
        at the projected frequencies (a :class:`~photonhub.PowerMonitor`
        window inside a :class:`~photonhub.TFSFBox`, divided by its area,
        for instance). By default it is the declared plane wave's,
        ``|E0|^2 / (2 eta)`` with ``E0`` the amplitude of the run's one
        ``TFSFBox`` (:attr:`incident_e_v_per_m`) and ``eta`` the medium's
        impedance, so on the far-field approximation ``dsigma/dOmega =
        (|e_theta|^2 + |e_phi|^2) / |E0|^2``. The plane wave the grid
        carries has ``cos(k dl / 2)`` of that intensity in a
        ``PowerMonitor`` (0.8 % less at 25 cells per wavelength), so a cross
        section divided by an incident flux measured without the scatterer
        reads that much higher. A projection of anything else
        (several sources, another kind, a plain mapping of arrays, a result
        whose simulation is unknown) needs it given: ``ValueError``. The
        surface must hold scattered field only, as a ``TFSFBox``'s outside
        does."""
        from ._far_field_results import impedance, per_frequency
        if incident_intensity_w_per_m2 is None:
            e0 = self.incident_e_v_per_m
            if e0 is None:
                raise ValueError(
                    "FarField.cross_section needs the incident intensity, "
                    "incident_intensity_w_per_m2= (W/m^2): this projection does not come "
                    "from a run driven by one TFSFBox, whose plane wave would give it")
            i_inc = float(e0) ** 2 / (2.0 * impedance(self))
        else:
            i_inc = xr.DataArray(per_frequency(incident_intensity_w_per_m2, self.freqs_hz,
                                               "incident_intensity_w_per_m2", "W/m^2"),
                                 dims="f")
        s = self._labelled["intensity"] / i_inc * 1e12
        s.name = "cross_section"
        s.attrs = {"units": "um^2/sr", "long_name": "differential scattering cross section"}
        return s

    def polarization(self, basis: str = "ludwig3", *, reference: Optional[str] = None,
                     boresight: Optional[str] = None) -> xr.Dataset:
        """The transverse field in a polarization basis, labelled like
        :attr:`dataset` and scaled like ``e_theta`` (V; the intensity in
        each is ``|amplitude|^2 / (2 eta)``).

        * ``"ludwig3"``: ``co`` and ``cross``, Ludwig's third definition
          (the polarization an antenna range measures). About the boresight
          ``b`` with the co-polar reference ``x'`` there (``y' = b x x'``)::

              i_co    = x' - (r . x') (r + b) / (1 + r . b)
              i_cross = y' - (r . y') (r + b) / (1 + r . b)

          For ``b = +z`` and ``x' = x`` this is ``co = cos(phi) E_theta -
          sin(phi) E_phi`` and ``cross = sin(phi) E_theta + cos(phi)
          E_phi``. A Huygens source polarized along ``x'`` has no
          cross-polar field anywhere, and a dipole along ``x'`` none in the
          two principal planes. ``boresight`` is ``"+x"`` ... ``"-z"``: by
          default ``"+z"`` on an angle grid, the grid's own axis and half
          space on a k-space grid, and the side of the plane seen from
          ``origin_um`` on a Cartesian one. ``reference`` is an axis normal
          to it, by default the first of its in-plane axes in xyz order (x
          for z, x for y, y for x). The definition has no value at ``r =
          -b``. For numerical stability, ``co`` and ``cross`` are ``nan``
          throughout the antipodal neighborhood ``1 + r . b <= 1e-12``.
          A ``UserWarning`` identifies the boresight when requested
          directions enter this neighborhood. Choose a boresight suited
          to the hemisphere, or partition a full-sphere calculation.
        * ``"circular"``: ``right`` and ``left``, ``(E_theta - i E_phi) /
          sqrt(2)`` and ``(E_theta + i E_phi) / sqrt(2)``. With the
          ``e^{-i omega t}`` convention and the field leaving along ``r``,
          ``right`` is the field that turns clockwise seen from behind (the
          IEEE right hand: thumb along ``r``; optics texts such as Born and
          Wolf call this left-handed), ``left`` the other way. A
          dipole ``x + i y`` radiates right-hand circular light toward
          ``+z`` and left-hand toward ``-z``: an Ex dipole with an Ey
          dipole whose ``GaussianPulse`` has ``phase=-pi/2`` (its current
          ``sin(omega t)`` against the Ex dipole's ``cos(omega t)``);
          ``phase=+pi/2`` turns both hands over.

        On an exact projection the radial component ``e_r`` is left out (the
        bases are transverse to ``r``)."""
        from ._far_field_results import (NATURAL, grid_dims, ludwig3_vectors,
                                         parse_boresight, spherical_basis, unit)
        ds = self._labelled
        et, ep = ds["e_theta"], ds["e_phi"]
        attrs_out = {"basis": basis, "units": "V", "time_convention": "exp(-i omega t)"}
        if basis == "circular":
            if reference is not None or boresight is not None:
                raise ValueError("the circular basis takes no reference or boresight: it is "
                                 "the handedness about each direction")
            s2 = np.sqrt(2.0)
            right, left = (et - 1j * ep) / s2, (et + 1j * ep) / s2
            right.attrs = {"units": "V", "long_name": "right-hand circular amplitude"}
            left.attrs = {"units": "V", "long_name": "left-hand circular amplitude"}
            return xr.Dataset({"right": right, "left": left}, attrs=attrs_out)
        if basis != "ludwig3":
            raise ValueError(f"basis must be 'ludwig3' or 'circular', got {basis!r}")
        if boresight is None:
            if self.grid == "kspace":
                boresight = f"{self.direction}{self.plane_axis}"
            elif self.grid == "cartesian":
                a = "xyz".index(self.plane_axis)
                o = 0.0 if self.origin_um is None else float(self.origin_um[a])
                boresight = f"{'+' if float(self.plane_um) >= o else '-'}{self.plane_axis}"
            else:
                boresight = "+z"
        b_axis, b_sign = parse_boresight(boresight)
        if reference is None:
            reference = NATURAL[b_axis][0]
        if reference not in ("x", "y", "z") or reference == b_axis:
            raise ValueError(f"reference must be an axis normal to the boresight {boresight!r} "
                             f"({' or '.join(NATURAL[b_axis])}), got {reference!r}")
        dims = grid_dims(self)
        gshape = (np.size(self.theta),) if self.shape is None else tuple(self.shape)
        theta = np.asarray(self.theta, dtype=np.float64).reshape(gshape)
        phi = np.asarray(self.phi, dtype=np.float64).reshape(gshape)
        r_hat, t_hat, p_hat = spherical_basis(theta, phi)
        co_v, cross_v = ludwig3_vectors(r_hat, unit(b_axis, b_sign), unit(reference))
        if np.any(np.all(np.isfinite(r_hat), axis=-1)
                  & ~np.all(np.isfinite(co_v), axis=-1)):
            warnings.warn(
                "FarField.polarization: Ludwig-3 masks the antipodal numerical "
                f"neighborhood of boresight={boresight!r} (1 + r . b <= 1e-12); "
                "co and cross contain NaN there. Choose a boresight suited to the "
                "requested hemisphere, such as boresight='-z' for downward emission, "
                "or partition a full-sphere calculation and choose a boresight for each part.",
                UserWarning, stacklevel=caller_stacklevel(),
            )
        out = {}
        for name, v in (("co", co_v), ("cross", cross_v)):
            a_t = xr.DataArray(np.sum(t_hat * v, -1), dims=dims)
            a_p = xr.DataArray(np.sum(p_hat * v, -1), dims=dims)
            out[name] = et * a_t + ep * a_p
            out[name].attrs = {"units": "V", "long_name": f"Ludwig-3 {name}-polar amplitude"}
        attrs_out.update(boresight=boresight, reference=reference)
        return xr.Dataset(out, attrs=attrs_out)

    def power_density(self) -> xr.DataArray:
        """The radial power density ``S = U / r^2`` (W/m^2) at each point's
        distance, labelled like :attr:`dataset`'s intensity: the
        time-averaged Poynting vector along ``r`` there, the intensity of
        the light a detector at that point receives. It needs distances: an
        exact projection (the exact radial flux at ``r``), a Cartesian grid,
        or far-field amplitudes given one with :meth:`renormalize`."""
        if self.r_um is None:
            raise ValueError(
                "FarField.power_density needs each point's distance: these are far-field "
                "amplitudes, which do not depend on it. Use renormalize(distance_um) first, "
                "project with exact=True and r_um=, or onto a Cartesian grid")
        ds = self._labelled
        s = ds["intensity"] / (ds["r_um"] * 1e-6) ** 2
        s.name = "power_density"
        s.attrs = {"units": "W/m^2", "long_name": "radial power density"}
        return s

    def cartesian(self) -> xr.Dataset:
        """The field at each point in Cartesian components, ``Ex``, ``Ey``,
        ``Ez`` (V/m) and ``Hx``, ``Hy``, ``Hz`` (A/m), labelled like
        :attr:`dataset`, in the units of the recorded fields: the amplitudes
        times ``e^{i k r} / r`` with ``k`` the medium's wavenumber and ``r``
        the point's distance. An exact projection gives all six components
        as computed; in the far-field approximation E is transverse and ``H
        = r_hat x E / eta``. Needs distances, as :meth:`power_density`."""
        if self.r_um is None:
            raise ValueError(
                "FarField.cartesian needs each point's distance: these are far-field "
                "amplitudes, which do not depend on it. Use renormalize(distance_um) first, "
                "project with exact=True and r_um=, or onto a Cartesian grid")
        from ._far_field_results import _frozen, grid_dims
        ds = self._labelled
        dims = ("f",) + grid_dims(self)
        flat = self.cartesian_fields()
        units = {"E": "V/m", "H": "A/m"}
        out = {name: (dims, _frozen(flat[name], ds["e_theta"].shape), {"units": units[name[0]]})
               for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        return xr.Dataset(out, coords=ds.coords, attrs=dict(ds.attrs))

    def renormalize(self, distance_um: float) -> "FarField":
        """The same far-field amplitudes read at ``distance_um`` (um): a
        copy whose points lie that far from the amplitudes' phase reference
        (:attr:`phase_origin_um`: ``origin_um`` when :func:`far_field` was
        given one, else the corner of the run's frame; the magnitudes do not
        depend on it), which the copy records as its ``origin_um``, so that
        :meth:`power_density` and :meth:`cartesian` give the field there.
        The amplitudes themselves do not change: the far field falls as
        ``1/r``. At a finite distance ``R`` the copy's phase is off by about
        ``k d^2 / (2 R)``, with ``d`` the distance from the phase reference
        to the farthest part of the recorded surface (0.025 rad at 1 mm for
        a surface 2.8 um from the run's corner at lambda = 1 um): give
        :func:`far_field` an ``origin_um`` at the emitter, so ``d`` is the
        surface's own size, and keep ``R`` above ``8 d^2 / lambda``, where
        that error is below ``pi / 8``, or project with ``exact=True``.
        Far-field amplitudes on an angle or k-space grid only: an exact
        projection does not fall as ``1/r`` (project again with
        ``r_um=``), and each point of a Cartesian grid has its own
        distance."""
        if self.exact:
            raise ValueError(
                "FarField.renormalize rescales far-field amplitudes; an exact projection's "
                "field does not fall as 1/r: project again with exact=True and r_um=")
        if self.grid == "cartesian":
            raise ValueError(
                "FarField.renormalize: each point of a Cartesian grid has its own distance")
        d = np.asarray(distance_um, dtype=np.float64)
        if d.ndim != 0 or not np.isfinite(d) or d <= 0.0:
            raise ValueError(f"distance_um must be one positive distance (um), got {distance_um!r}")
        r = np.where(self.propagating, float(d), np.nan)
        origin = self.origin_um if self.origin_um is not None else self.phase_origin_um
        return _evolve(self, r_um=r, origin_um=origin)

    # --- the flat layout (deprecated, kept for one release) --------------------

    def _impedance(self) -> Union[float, np.ndarray]:
        """``eta = eta0 / n_medium``: a scalar, or a ``(n_freq, 1)`` column
        that broadcasts over directions. At ``n_medium = 1`` it is ``eta0``
        exactly (IEEE division by 1), so vacuum results are unchanged."""
        eta = ETA0 / np.asarray(self.n_medium, dtype=np.float64)
        return eta[:, None] if np.ndim(eta) else eta

    @property
    def propagating(self) -> np.ndarray:
        """Which points have a direction, shape ``(n_point,)``: every point of
        an angle or Cartesian grid, the points of a k-space grid with
        ``ux^2 + uy^2 <= 1``. A point with a direction can still carry no
        field: in an exact projection of a plane, a direction that ends at
        or behind it reads ``nan`` with intensity 0 while it stays
        ``True`` here. Which points carry a field is
        ``~np.isnan(e_theta)`` (labelled: ``dataset["e_theta"].notnull()``).
        Deprecated (the flat layout): ``dataset["propagating"]`` on a
        k-space grid."""
        if self.ux is None or self.uy is None:
            return np.ones(np.size(self.theta), dtype=bool)
        return np.asarray(self.ux) ** 2 + np.asarray(self.uy) ** 2 <= 1.0

    @property
    def intensity(self) -> np.ndarray:
        """Radiation intensity ``U = (|F_theta|^2 + |F_phi|^2)/(2 eta)``
        (W/sr), with ``eta = eta0 / n_medium`` the medium's wave impedance,
        shape ``(n_freq, n_point)``: the power per unit solid angle in the far
        zone. For an exact projection it is ``r^2`` times the radial
        component of the time-averaged Poynting vector at each point,
        ``Re(E_theta H_phi* - E_phi H_theta*) r^2 / 2``, the power per unit
        solid angle crossing a sphere about ``origin_um`` there, which tends
        to the far-zone value as ``r`` grows. 0 at a k-space point with no
        direction, and at a point of an exact projection with no field (a
        direction that ends at or behind a single plane). Deprecated (the
        flat layout): ``dataset["intensity"]``."""
        if self.h_theta is not None and self.h_phi is not None:
            u = 0.5 * np.real(self.e_theta * np.conj(self.h_phi)
                              - self.e_phi * np.conj(self.h_theta))
            u = np.where(np.isnan(u), 0.0, u)
        else:
            u = ((np.abs(self.e_theta) ** 2 + np.abs(self.e_phi) ** 2)
                 / (2.0 * self._impedance()))
        if self.ux is None:
            return u
        return np.where(self.propagating[None, :], u, 0.0)

    def _angle_grid(self, what: str) -> None:
        if self.grid != "angle":
            raise ValueError(
                f"FarField.{what} integrates over an angle grid (theta_rad, phi_rad); "
                f"this projection is on a {self.grid} grid. Project onto theta_rad and "
                "phi_rad for it")

    def radiated_power(self) -> np.ndarray:
        """Total radiated power ``P_rad = integral U dOmega`` per frequency
        (shape ``(n_freq,)``), integrated over the sampled directions with a
        ``sin(theta)`` solid-angle weight via :func:`_solid_angle_weights`.
        Deprecated: ``dataset["radiated_power"]`` (a meshed angle grid).

        This is only meaningful when the directions tile a contiguous angular
        region (e.g. a full sphere or a hemisphere mesh); for a sparse cut it is
        a crude quadrature. Used as the directivity normalizer. An angle grid
        only (``ValueError`` on a k-space or Cartesian grid); for an exact
        projection it is the power crossing the sphere of radius ``r_um``."""
        self._angle_grid("radiated_power()")
        w = _solid_angle_weights(self.theta, self.phi)
        return np.sum(self.intensity * w[None, :], axis=1)

    def directivity(self) -> np.ndarray:
        """Directivity ``D = 4 pi U / P_rad`` (dimensionless), shape
        ``(n_freq, n_point)``, the radiation pattern normalized so its
        solid-angle average is 1 (for an isotropic radiator D==1 everywhere).
        An angle grid only, as :meth:`radiated_power`. Deprecated:
        ``dataset["directivity"]`` (a meshed angle grid)."""
        p = self.radiated_power()
        p = np.where(p > 0, p, np.nan)
        return 4.0 * np.pi * self.intensity / p[:, None]

    def reshape(self, n_theta: Optional[int] = None,
                n_phi: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reshape the flat per-direction arrays back to a meshed
        ``(n_freq, n_theta, n_phi)`` grid (the layout produced when ``theta``
        and ``phi`` were passed as 1-D axes and meshed by :func:`far_field`).
        Returns a dict with ``e_theta``, ``e_phi``, ``intensity``,
        ``directivity``, ``theta`` (``n_theta,``) and ``phi`` (``n_phi,``).
        The sizes default to :attr:`shape`. On a k-space or Cartesian grid
        the axes are ``ux``, ``uy`` or ``x_um``, ``y_um`` instead, ``theta``
        and ``phi`` are ``(n1, n2)`` arrays, and there is no
        ``directivity``. An exact projection adds ``e_r``, ``h_r``,
        ``h_theta``, ``h_phi`` and ``r_um``. Deprecated: :attr:`dataset`
        holds the same arrays on the grid's axes."""
        if n_theta is None or n_phi is None:
            if self.shape is None:
                raise ValueError(
                    "FarField.reshape needs the grid's two sizes: this projection was "
                    "not meshed (mesh=False), so pass n_theta and n_phi")
            n_theta, n_phi = self.shape
        nf = self.freqs_hz.size
        out = {
            "e_theta": self.e_theta.reshape(nf, n_theta, n_phi),
            "e_phi": self.e_phi.reshape(nf, n_theta, n_phi),
            "intensity": self.intensity.reshape(nf, n_theta, n_phi),
        }
        if self.grid == "angle":
            out["directivity"] = self.directivity().reshape(nf, n_theta, n_phi)
            out["theta"] = self.theta.reshape(n_theta, n_phi)[:, 0]
            out["phi"] = self.phi.reshape(n_theta, n_phi)[0, :]
        else:
            one, two = (("ux", "uy") if self.grid == "kspace" else ("x_um", "y_um"))
            out[one] = np.asarray(getattr(self, one)).reshape(n_theta, n_phi)[:, 0]
            out[two] = np.asarray(getattr(self, two)).reshape(n_theta, n_phi)[0, :]
            out["theta"] = self.theta.reshape(n_theta, n_phi)
            out["phi"] = self.phi.reshape(n_theta, n_phi)
        for name in ("e_r", "h_r", "h_theta", "h_phi"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value.reshape(nf, n_theta, n_phi)
        if self.r_um is not None:
            out["r_um"] = np.asarray(self.r_um).reshape(n_theta, n_phi)
        return out

    def cartesian_fields(self) -> Dict[str, np.ndarray]:
        """The field at each point in Cartesian components, ``Ex``, ``Ey``,
        ``Ez`` (V/m) and ``Hx``, ``Hy``, ``Hz`` (A/m), each ``(n_freq,
        n_point)``, in the units of the recorded fields: the amplitudes times
        ``e^{i k r} / r`` with ``k`` the medium's wavenumber and ``r`` the
        point's distance ``r_um``. An exact projection gives all six
        components; on a Cartesian grid in the far-field approximation E is
        transverse and ``H = r_hat x E / eta``. Needs a distance:
        ``ValueError`` for far-field amplitudes on an angle or k-space grid
        (project with ``exact=True`` and ``r_um=``, or use
        :meth:`renormalize`). Deprecated: :meth:`cartesian` returns them
        labelled."""
        if self.r_um is None:
            raise ValueError(
                "FarField.cartesian_fields needs each point's distance: these are "
                "far-field amplitudes, which do not depend on it. Project with "
                "exact=True and r_um=, or onto a Cartesian grid, or use "
                "renormalize(distance_um)")
        n = np.atleast_1d(np.asarray(self.n_medium, dtype=np.float64))
        k = (2.0 * np.pi * self.freqs_hz / C0 * (n if n.size > 1 else n[0]))[:, None]
        r = np.asarray(self.r_um, dtype=np.float64)[None, :] * 1e-6
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = np.exp(1j * k * r) / r      # nan where a point has no field
        st, ct = np.sin(self.theta), np.cos(self.theta)
        sp, cp = np.sin(self.phi), np.cos(self.phi)
        r_hat = (st * cp, st * sp, ct)
        t_hat = (ct * cp, ct * sp, -st)
        p_hat = (-sp, cp, np.zeros_like(st))
        e_r = self.e_r if self.e_r is not None else np.zeros_like(self.e_theta)
        if self.h_theta is not None:
            h_r, h_t, h_p = self.h_r, self.h_theta, self.h_phi
        else:
            eta = self._impedance()
            h_r, h_t, h_p = np.zeros_like(self.e_theta), -self.e_phi / eta, self.e_theta / eta
        out = {}
        for i, c in enumerate("xyz"):
            out[f"E{c}"] = spread * (e_r * r_hat[i] + self.e_theta * t_hat[i]
                                     + self.e_phi * p_hat[i])
            out[f"H{c}"] = spread * (h_r * r_hat[i] + h_t * t_hat[i] + h_p * p_hat[i])
        return out


def _evolve(ff: FarField, **changes) -> FarField:
    """A copy of ``ff`` with ``changes``, every other field the same object."""
    kw = {f.name: getattr(ff, f.name) for f in dataclasses.fields(ff)}
    kw.update(changes)
    return FarField(**kw)


def _read_by_dataclasses() -> bool:
    """Whether the standard library's dataclass helpers (``replace``,
    ``asdict``, ``astuple``) read the member, not the user: they read every
    field, the flat ones included, without the user having asked for them."""
    frame = sys._getframe(2)
    return frame is not None and frame.f_globals.get("__name__") == "dataclasses"


#: What replaces a flat member that this projection does not have (it is
#: ``None``): the test that tells whether the labelled form has it.
_ABSENT = {
    "e_r": "an exact projection's (ff.exact) FarField.dataset['e_r']",
    "h_r": "an exact projection's (ff.exact) FarField.dataset['h_r']",
    "h_theta": "an exact projection's (ff.exact) FarField.dataset['h_theta']",
    "h_phi": "an exact projection's (ff.exact) FarField.dataset['h_phi']",
    "r_um": ("the coordinate FarField.dataset['r_um'] where the projection has distances "
             "('r_um' in ff.dataset.coords)"),
    "ux": "a k-space grid's (ff.grid == 'kspace') coordinate FarField.dataset['ux']",
    "uy": "a k-space grid's (ff.grid == 'kspace') coordinate FarField.dataset['uy']",
    "x_um": ("a Cartesian grid's (ff.grid == 'cartesian') coordinate named by its first "
             "in-plane axis"),
    "y_um": ("a Cartesian grid's (ff.grid == 'cartesian') coordinate named by its second "
             "in-plane axis"),
}


def _warn_flat(ff: FarField, name: str) -> None:
    """The ``FutureWarning`` of a flat-layout member read by the user."""
    from ._far_field_results import NATURAL, grid_dims
    dims = ("f",) + grid_dims(ff)
    call = "()" if name in _FLAT_METHODS else ""
    meshed_angle = ff.grid == "angle" and ff.shape is not None
    if name in _ABSENT and getattr(ff, name) is None:
        instead = f"{_ABSENT[name]}; this projection has none"
    elif name == "propagating" and ff.grid != "kspace":
        instead = ("nothing: every point of this grid has a direction (a k-space grid's "
                   "FarField.dataset['propagating'] marks those that do)")
    elif name in ("radiated_power", "directivity") and not meshed_angle:
        instead = (f"FarField.dataset[{name!r}] on a meshed angle grid (theta_rad, phi_rad, "
                   "mesh=True); this projection has none")
    elif name == "reshape":
        instead = "FarField.dataset, which holds the same arrays on the grid's axes"
    elif name == "cartesian_fields":
        instead = "FarField.cartesian(), the same fields"
    elif name in ("x_um", "y_um") and ff.plane_axis is not None:
        axis = NATURAL[ff.plane_axis][0 if name == "x_um" else 1]
        instead = f"the coordinate FarField.dataset[{axis!r}]"
    else:
        instead = f"FarField.dataset[{name!r}]"
    warnings.warn(
        f"FarField.{name}{call} reads the flat layout ([frequency, point] over a flat list "
        f"of points), which is kept for one release; use {instead}, labelled by frequency "
        f"and the grid's axes (dims {dims})", FutureWarning,
        stacklevel=caller_stacklevel())


def _solid_angle_weights(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Per-direction solid-angle quadrature weights ``dOmega = sin(theta)
    dtheta dphi`` for a flat list of directions that came from meshing two 1-D
    axes (``theta`` varying slowest). Recovers the 1-D axes from the flat list
    via :func:`numpy.unique` and builds centered-difference cell widths in each
    (reusing :func:`~photonhub.analysis.mode_overlap._cell_widths`).

    Each direction finds its axis cell through the inverse index of the same
    rounded array the axis came from. Looking it up again by Python's
    ``round(x, 12)`` raised ``KeyError`` wherever that rounding and
    :func:`numpy.round` disagree, at an angle whose ``x * 1e12`` lands on a
    tie (``np.linspace(0, np.pi, 113)`` holds one)."""
    th_ax, th_i = np.unique(np.round(theta, 12), return_inverse=True)
    ph_ax, ph_i = np.unique(np.round(phi, 12), return_inverse=True)
    dth = _cell_widths(th_ax)
    dph = _cell_widths(ph_ax) if ph_ax.size > 1 else np.array([2.0 * np.pi])
    return (np.sin(np.asarray(theta, dtype=np.float64))
            * dth[th_i.reshape(-1)] * dph[ph_i.reshape(-1)])


def _directions(
    theta: Union[float, Sequence[float], np.ndarray],
    phi: Union[float, Sequence[float], np.ndarray],
    mesh: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """Normalize the (theta, phi) request into flat 1-D direction arrays.

    If ``mesh`` (default), ``theta`` and ``phi`` are treated as 1-D axes and
    meshed (``theta`` slowest) into ``n_theta*n_phi`` directions; the returned
    shape tuple ``(n_theta, n_phi)`` enables :meth:`FarField.reshape`. If not
    ``mesh``, ``theta`` and ``phi`` must be the same length and are paired
    elementwise (an arbitrary list of directions)."""
    th = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    ph = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    if mesh:
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        return TH.reshape(-1), PH.reshape(-1), (th.size, ph.size)
    if th.size != ph.size:
        raise ValueError(
            f"with mesh=False, theta and phi must be the same length; got "
            f"{th.size} and {ph.size}")
    return th, ph, None


def _plane_component_3d(
    da: xr.DataArray,
    name: str,
    freq_hz: float,
    t1: str,
    t2: str,
    normal: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pull one tangential component off a DFT plane DataArray and return it as a
    2-D complex ``[i_t2, i_t1]`` array, its ``(t1, t2)`` coordinate axes (µm) and
    the plane's normal coordinate (µm).

    Mirrors ``mode_overlap._plane_component`` but also recovers the normal
    position so the radiation kernel ``exp(i k r_hat . r')`` carries the real
    3-D location of the face."""
    if "component" in getattr(da, "dims", ()):
        comp_coords = list(da.coords.get("component", []))
        da = da.sel(component=name) if name in comp_coords \
            else da.squeeze("component", drop=True)
    if "f" in getattr(da, "dims", ()):
        da = da.sel(f=freq_hz, method="nearest")
    # Recover the normal coordinate before squeezing it away.
    normal_um = 0.0
    if normal in getattr(da, "coords", {}):
        nc = np.asarray(da.coords[normal].values, dtype=np.float64).ravel()
        if nc.size >= 1:
            normal_um = float(nc[0])
    # Keep a length-1 TRANSVERSE axis: on a quasi-2-D run (a 1-cell
    # plain-periodic axis) the recorded plane is genuinely one cell deep, and
    # squeezing that axis away left dims like ('y',) and rejected the plane --
    # locking near-to-far out of 2-D scenes. Third site of this pattern after
    # diffraction_orders and the modal readout.
    keep = {t1, t2}
    drop = [d for d in da.dims if d not in keep and da.sizes[d] == 1]
    da = da.squeeze(drop, drop=True) if drop else da
    for ax in (t1, t2):
        if ax not in da.dims:
            da = da.expand_dims(ax)
    if set(da.dims) != keep:
        raise ValueError(
            f"component {name!r}: after reduction dims are {tuple(da.dims)}, "
            f"expected the two transverse axes {{{t1!r}, {t2!r}}}")
    da = da.transpose(t2, t1)
    vals = np.asarray(da.values, dtype=np.complex128)
    c1 = np.asarray(da.coords[t1].values, dtype=np.float64)
    c2 = np.asarray(da.coords[t2].values, dtype=np.float64)
    return vals, c1, c2, normal_um


def equivalent_currents(
    plane_fields: Mapping[str, xr.DataArray],
    *,
    axis: Axis,
    freq_hz: float,
    sign: float = 1.0,
    colocate: bool = True,
) -> Dict[str, np.ndarray]:
    """Surface equivalent currents ``J = n_hat x H``, ``M = -n_hat x E`` on a
    single plane normal to ``axis``, at one frequency.

    ``sign`` is the orientation of the outward normal relative to ``+axis``
    (``+1`` for an outward normal along ``+axis``, ``-1`` along ``-axis``).
    ``colocate`` (default on) averages each Yee-staggered component to the
    in-plane cell node before forming ``J``/``M``, correct for the engine's
    staggered DFT output; pass ``False`` for synthetic already-co-located
    fields (see the module docstring's "Yee staggering").
    Returns a dict with the three Cartesian components of ``J`` and ``M`` (keys
    ``Jx,Jy,Jz,Mx,My,Mz``, each a ``[i_t2, i_t1]`` complex array) plus the
    plane geometry ``t1``,``t2`` (axis names), ``c1``,``c2`` (their coords, µm),
    ``normal_um`` (the normal position, µm) and ``normal_axis`` (``axis``)."""
    if axis not in _TRANSVERSE:
        raise ValueError(f"axis must be one of x/y/z, got {axis!r}")
    t1, t2 = _TRANSVERSE[axis]
    e1n, e2n, h1n, h2n = _TANGENTIAL[axis]
    for key in (e1n, e2n, h1n, h2n):
        if key not in plane_fields:
            raise ValueError(
                f"plane_fields missing {key!r}; for axis={axis!r} need the "
                f"tangential components {e1n!r},{e2n!r},{h1n!r},{h2n!r}")

    E1, c1, c2, n_um = _plane_component_3d(plane_fields[e1n], e1n, freq_hz, t1, t2, axis)
    E2, _, _, _ = _plane_component_3d(plane_fields[e2n], e2n, freq_hz, t1, t2, axis)
    H1, _, _, _ = _plane_component_3d(plane_fields[h1n], h1n, freq_hz, t1, t2, axis)
    H2, _, _, _ = _plane_component_3d(plane_fields[h2n], h2n, freq_hz, t1, t2, axis)

    if colocate:
        # Same offsets as mode_overlap._overlap_terms (the engine's Yee offsets):
        # arrays are [i_t2, i_t1]; E_t1 is +1/2 in t1 (last axis), E_t2 +1/2 in
        # t2 (axis 0), H_t1 +1/2 in t2, H_t2 +1/2 in t1.
        E1 = _colocate_to_node(E1, -1)
        E2 = _colocate_to_node(E2, 0)
        H1 = _colocate_to_node(H1, 0)
        H2 = _colocate_to_node(H2, -1)

    # n_hat = sign * axis_hat. For a transverse pair (a1, a2) along (t1, t2):
    #   n_hat x (a1 t1_hat + a2 t2_hat) = sign * (a1 (axis x t1) + a2 (axis x t2))
    #   and since t1 x t2 = +axis (right-handed), axis x t1 = t2, axis x t2 = -t1.
    # => n_hat x A_t = sign * (-a2 t1_hat + a1 t2_hat).
    # J = n_hat x H ; M = -(n_hat x E).
    J = {t1: sign * (-H2), t2: sign * (H1)}
    M = {t1: -sign * (-E2), t2: -sign * (E1)}

    out: Dict[str, np.ndarray] = {}
    for comp in ("x", "y", "z"):
        out[f"J{comp}"] = J.get(comp, np.zeros_like(E1))
        out[f"M{comp}"] = M.get(comp, np.zeros_like(E1))
    out["t1"] = t1  # type: ignore[assignment]
    out["t2"] = t2  # type: ignore[assignment]
    out["c1"] = c1
    out["c2"] = c2
    out["normal_um"] = n_um  # type: ignore[assignment]
    out["normal_axis"] = axis  # type: ignore[assignment]
    return out


def _cell_widths_clipped(coords: np.ndarray,
                         clip: Optional[Tuple[float, float]]) -> np.ndarray:
    """Per-sample quadrature widths, optionally CLIPPED to a span.

    The plain :func:`_cell_widths` midpoint rule extends a half-cell past each
    end sample, right for an open aperture, wrong for one face of a CLOSED
    box: each face then over-covers its span by a full cell per axis and
    double-counts the edge strips its neighbouring faces already integrate
    (measured +5-17% radiated power on the analytic-dipole box, F11). Clamping
    the cell edges to the box extent tiles the surface exactly once
    (trapezoid-consistent: edge samples carry half-cells, outside samples 0).
    """
    c = np.asarray(coords, dtype=np.float64)
    if clip is None or c.size == 1:
        return _cell_widths(c)
    n = c.size
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    lo, hi = (float(clip[0]), float(clip[1]))
    if hi < lo:
        lo, hi = hi, lo
    np.clip(edges, lo, hi, out=edges)
    return np.abs(np.diff(edges))


def _project_one_face(
    currents: Mapping[str, np.ndarray],
    *,
    k: float,
    rx: np.ndarray,
    ry: np.ndarray,
    rz: np.ndarray,
    clip1: Optional[Tuple[float, float]] = None,
    clip2: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radiation-vector contributions ``(N_x,N_y,N_z, L_x,L_y,L_z)`` of one face
    for a set of direction cosines ``(rx,ry,rz)`` (each a flat ``(n_dir,)``
    array). ``k`` is the wavenumber in the medium radiated into,
    ``n_medium * 2 pi f / c0`` (1/m).

    Computes ``N = integral J exp(-i k r_hat.r') dS`` (and ``L`` from ``M``) ,
    the MINUS-sign kernel of the ``e^{-i omega t}`` convention, matching the
    code below and the module docstring, by forming the kernel on the face's
    grid (coords in µm -> m) and summing with the centered-difference
    cell-area quadrature."""
    t1 = currents["t1"]
    t2 = currents["t2"]
    normal_axis = currents["normal_axis"]
    c1 = np.asarray(currents["c1"], dtype=np.float64) * 1e-6  # µm -> m
    c2 = np.asarray(currents["c2"], dtype=np.float64) * 1e-6
    n_m = float(currents["normal_um"]) * 1e-6

    # A length-1 transverse axis is one cell deep, not a line monitor: give it
    # the grid spacing (taken from the sampled axis -- these monitors ride
    # uniform grids) instead of _cell_widths' single-sample width of 1.0.
    def _w(coords, clip, other):
        w = _cell_widths_clipped(coords, clip)
        if coords.size == 1 and other.size >= 2:
            return np.array([abs(float(other[1] - other[0]))])
        return w

    dA = np.outer(_w(c2, clip2, c1), _w(c1, clip1, c2))  # [i_t2, i_t1] m^2

    # Cartesian position of every surface point: t1, t2 in-plane, normal fixed.
    pos = {t1: c1[None, :] * np.ones((c2.size, 1)),       # [i_t2, i_t1]
           t2: c2[:, None] * np.ones((1, c1.size)),
           normal_axis: np.full((c2.size, c1.size), n_m)}
    X = pos["x"]
    Y = pos["y"]
    Z = pos["z"]

    # Direction cosines per direction; kernel exp(-i k r_hat . r').
    # In the e^{-i omega t} convention the outgoing wave is e^{+i k r}, so the
    # far-field expansion |r - r'| ~ r - r_hat . r' gives
    #   e^{i k |r-r'|} ~ e^{i k r} * e^{-i k r_hat . r'},
    # i.e. the surface-FT kernel carries a MINUS sign here (a +kx0 aperture tilt
    # then steers the beam toward +x, as a +x-travelling wave must). Build
    # via the flat spatial axis. The kernel matrix is [n_dir, n_pts]; for a large
    # monitor x many directions that can be huge, so chunk over directions to keep
    # the working set bounded (see _DIR_CHUNK).
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)
    wdA = dA.reshape(-1)  # [n_pts]
    JM = [currents[c].reshape(-1) * wdA for c in
          ("Jx", "Jy", "Jz", "Mx", "My", "Mz")]

    n_dir = rx.size
    n_pts = Xf.size
    out = [np.empty(n_dir, dtype=np.complex128) for _ in range(6)]
    chunk = max(1, int(_DIR_CHUNK // max(n_pts, 1)))
    for lo in range(0, n_dir, chunk):
        hi = min(lo + chunk, n_dir)
        phase = k * (np.outer(rx[lo:hi], Xf)
                     + np.outer(ry[lo:hi], Yf)
                     + np.outer(rz[lo:hi], Zf))
        kernel = np.exp(-1j * phase)  # [chunk, n_pts], kernel = exp(-i k r_hat.r')
        for i, src in enumerate(JM):
            out[i][lo:hi] = kernel @ src
    return tuple(out)  # type: ignore[return-value]


# --- The Yee-lattice readout ---------------------------------------------------
#
# The engine records each component at its own Yee point (E_c half a cell
# along c, H_c half a cell along the two other axes), while a field_dft
# DataArray reports base-node coordinates for all of them (data.py). The
# default readout of :func:`far_field` places every component where it was
# sampled instead of averaging it to the node and, on the faces of a box,
# brings E and H onto ONE surface from a face recorded one cell thick (two
# normal samples). Uniform and graded meshes (the samples placed from the
# run's grid lines), with the faces of a symmetry-reduced run mirrored at
# their own Yee points first.

#: Offset of each component from its base node, in cells, per axis (x, y, z):
#: the engine's Yee grid, checked against a recorded DFT volume of a run.
_YEE_OFFSET: Dict[str, Tuple[float, float, float]] = {
    "Ex": (0.5, 0.0, 0.0), "Ey": (0.0, 0.5, 0.0), "Ez": (0.0, 0.0, 0.5),
    "Hx": (0.0, 0.5, 0.5), "Hy": (0.5, 0.0, 0.5), "Hz": (0.5, 0.5, 0.0),
}

#: Public ``yee=`` values of :func:`far_field`: the default readout and the
#: earlier one, kept for one release.
_YEE_PUBLIC = ("exact", "legacy")
#: ``yee=`` values of :func:`_far_field_impl`: also the two parts of
#: ``"exact"`` alone (diagnostics, kept so the parts can be compared).
_YEE_MODES = ("exact", "positions", "surface", "legacy")
#: Surfaces a box face can be read on (``_far_field_impl(yee_plane=...)``);
#: :func:`far_field` uses ``"avg"``.
_YEE_PLANES = ("h", "e", "mid", "avg")
#: In-plane quadrature of a box's faces on the Yee-lattice readout: "cubic"
#: (fourth-order weights at the box edges, :func:`_cubic_weights_clipped`)
#: or "clipped" (the earlier readout's clipped midpoint rule, kept so the
#: two can be compared).
_YEE_EDGE_QUADRATURE = "cubic"

#: Direction-chunk budget of the separable kernel: each ``[chunk_dirs, n]``
#: factor matrix stays near this many complex entries (64 MB at 4e6).
_SEPARABLE_CHUNK: int = 4_000_000

#: Relative tolerance under which two cell sizes or sample spacings agree.
_SPACING_RTOL = 1e-6

#: Relative spread of an open plane's sample spacings under which
#: :func:`_plane_weights` takes them as even: a mesher's uniform axis listed
#: as grid lines rounded to 1e-7 um spreads by a few 1e-6, a graded one by
#: 0.1 to 0.4 per cell.
_EVEN_RTOL = 1e-3

#: Largest ratio between the sample spacings of one four-point stencil the
#: box-edge rule (:func:`_cubic_weights_clipped`) integrates with the cubic.
#: Across a jump of 3 the cubic's weights stay positive (0.38 to 1.33 times
#: the local width); across 5 one turns negative (-1.2), across 10 -9.7.
_CUBIC_MAX_JUMP = 3.0


def _where(face: Optional[str], axis: str, sign: float) -> str:
    """How a refusal names a face: ``face 'zhi' (normal to +z)``."""
    side = f"{'+' if sign > 0 else '-'}{axis}"
    return f"face {face!r} (normal to {side})" if face else f"the face normal to {side}"


def _coords(da, axis: str) -> np.ndarray:
    if axis not in getattr(da, "coords", {}):
        return np.empty(0)
    return np.asarray(da.coords[axis].values, dtype=np.float64).ravel()


def _grid_block(data) -> Mapping:
    manifest = getattr(data, "manifest", None)
    return manifest.get("grid", {}) if isinstance(manifest, Mapping) else {}


def _given_symmetry(simulation) -> Optional[Tuple[int, int, int]]:
    """``simulation=``'s symmetry planes as given, for the earlier readout,
    which mirrors only what ``simulation=`` declares (None when it has
    none)."""
    sym = getattr(simulation, "symmetry", None) if simulation is not None else None
    if sym is None or not any(int(v) for v in sym):
        return None
    return tuple(int(v) for v in sym)  # type: ignore[return-value]


def _run_symmetry(data, simulation, recipe=None) -> Optional[Tuple[int, int, int]]:
    """The symmetry planes of the run, ``(sx, sy, sz)`` with each of -1, 0
    or +1, or ``None`` for a run with none: read from the run's manifest
    (``grid.symmetry``), else from the simulation a ``RunResult`` carries,
    and from ``simulation=``. When both the run and ``simulation=`` say,
    they must agree (``ValueError``): a simulation of another run would
    mirror the faces with the wrong signs, or not at all. A plain mapping
    with no ``simulation=`` is read as a full run. ``recipe`` is the planes
    a ``FarFieldMonitor`` was compiled with (its recipe), which must agree
    with the run's too, and stand when nothing else says."""
    def read(sym, where: str) -> Optional[Tuple[int, int, int]]:
        if sym is None:
            return None
        try:
            out = tuple(int(v) for v in sym)
        except (TypeError, ValueError):
            out = ()
        if len(out) != 3 or any(v not in (-1, 0, 1) for v in out):
            raise ValueError(
                f"far_field: {where} gives the symmetry planes {sym!r}; expected "
                "three of -1, 0 and +1")
        return out  # type: ignore[return-value]

    given = (read(getattr(simulation, "symmetry", None), "simulation=")
             if simulation is not None else None)
    recorded = read(_grid_block(data).get("symmetry"), "the run's manifest")
    if recorded is None:
        try:
            sim = getattr(data, "simulation", None)
        except (AttributeError, OSError, ValueError):
            sim = None
        if sim is not None:
            recorded = read(getattr(sim, "symmetry", None), "the run's simulation")
    if given is not None and recorded is not None and given != recorded:
        raise ValueError(
            f"far_field: simulation= has the symmetry planes {given}, but the run "
            f"recorded {recorded}; pass the simulation the run came from")
    sym = recorded if recorded is not None else given
    compiled = read(recipe, "the far-field surface's recipe") if recipe is not None else None
    if compiled is not None:
        if sym is not None and sym != compiled:
            raise ValueError(
                f"far_field: the far-field surface was compiled with the symmetry planes "
                f"{compiled}, but the run says {sym}; project it with the run it was "
                "recorded in")
        sym = compiled
    return sym if sym is not None and any(sym) else None


class _Grid:
    """The grid lines a readout places the samples with, in microns, in the
    run's wire frame: per axis either the uniform cell ``dl`` or, on a
    graded axis, the primary nodes ``q[0 .. n-1]`` with the closing node
    ``q[n]`` appended (the solver closes a graded axis by repeating its last
    spacing).
    A component half a cell off the node along an axis sits at the midpoint
    of its cell, ``q[i] + (q[i+1] - q[i]) / 2``; on a uniform axis that is
    ``q[i] + dl / 2``, the uniform readout's arithmetic bit for bit."""

    def __init__(self, dl: Optional[float] = None,
                 lines: Optional[Mapping[str, np.ndarray]] = None) -> None:
        self.dl = None if dl is None else float(dl)
        self.lines: Dict[str, np.ndarray] = dict(lines or {})

    def index(self, axis: str, base, where: str = "") -> np.ndarray:
        """The grid-line index of each base coordinate along ``axis``;
        ``ValueError`` for a coordinate that is not on a grid line (planes
        from another run, or a manifest that does not belong to them)."""
        c = np.atleast_1d(np.asarray(base, dtype=np.float64))
        q = self.lines.get(axis)
        if q is None:
            if not self.dl:
                raise ValueError(f"far_field: the grid along {axis} is not known")
            i = np.rint(c / self.dl).astype(np.int64)
            off, tol = np.abs(c - i * self.dl), 1e-6 * self.dl
            bad = off > tol
        else:
            j = np.clip(np.searchsorted(q, c), 1, q.size - 1)
            i = np.where(np.abs(c - q[j - 1]) <= np.abs(q[j] - c), j - 1, j)
            last = q.size - 2                  # the last base node (q[-1] closes)
            width = np.diff(q)[np.minimum(i, last)]
            bad = (np.abs(c - q[i]) > 1e-6 * width) | (i > last)
        if np.any(bad):
            c0 = float(c[np.argmax(bad)])
            raise ValueError(
                f"far_field: {where or 'a face'} has a sample at {axis} = {c0:.9g} um, "
                f"which is not on the run's grid lines along {axis}; the planes and "
                "the run's manifest do not belong together")
        return i

    def width(self, axis: str, base):
        """The width of the cell each base node starts (um): the float
        ``dl`` on a uniform axis, ``q[i+1] - q[i]`` on a graded one."""
        q = self.lines.get(axis)
        if q is None:
            return self.dl
        i = self.index(axis, base)
        w = q[i + 1] - q[i]
        return float(w[0]) if np.ndim(base) == 0 else w

    def place(self, axis: str, base, offset: float):
        """Where a component offset by ``offset`` cells (0 or 1/2) from the
        base node ``base`` sits along ``axis``."""
        if offset == 0.0:
            return base
        return base + offset * self.width(axis, base)

    def adjacent(self, axis: str, c: np.ndarray) -> bool:
        """Whether two normal samples sit on neighbouring grid lines."""
        if c.size != 2:
            return False
        if axis not in self.lines:
            return bool(np.isclose(abs(c[1] - c[0]), self.dl, rtol=1e-6))
        i = self.index(axis, c)
        return abs(int(i[1]) - int(i[0])) == 1

    def stride(self, axis: str, c: np.ndarray) -> float:
        """The largest step between neighbouring samples, in cells (1 for a
        face recorded at every cell)."""
        if c.size < 2:
            return 1.0
        if axis not in self.lines:
            return abs(float(c[1] - c[0])) / self.dl
        return float(np.max(np.abs(np.diff(self.index(axis, c)))))

    def one_cell_width(self, axis: str) -> Optional[float]:
        """The cell width of a graded axis one cell long (a quasi-2-D axis),
        else None."""
        q = self.lines.get(axis)
        return float(q[1] - q[0]) if q is not None and q.size == 2 else None


def _run_grid(data) -> Optional[_Grid]:
    """The grid lines of a graded run, from its manifest (``grid.coords_um``,
    one primary-node list per axis, and ``grid.dl_um`` for an axis not
    listed); ``None`` for a uniform run or a plain mapping."""
    grid = _grid_block(data)
    coords = grid.get("coords_um")
    if not coords:
        return None
    base = float(grid["dl_um"]) if grid.get("dl_um") is not None else None
    lines = {}
    for ax in "xyz":
        if ax not in coords:
            continue
        q = np.asarray(coords[ax], dtype=np.float64).ravel()
        if q.size >= 2:
            closing = q[-1] + (q[-1] - q[-2])
        elif q.size == 1 and base:
            closing = q[-1] + base
        else:
            raise ValueError(f"far_field: the run's manifest lists no grid line along {ax}")
        lines[ax] = np.append(q, closing)
    return _Grid(dl=base, lines=lines)


def _mesh(data, dl_um: Optional[float], face_specs, face_names
          ) -> Tuple[Optional[_Grid], Optional[str]]:
    """``(grid, None)`` when the grid is known, ``(None, None)`` when nothing
    tells the cell size, ``(None, why)`` for planes of a graded mesh whose
    grid lines are not known (``why`` names them).

    A graded run's manifest gives its grid lines, and every face sample must
    sit on them; ``dl_um`` with such a run raises ``ValueError`` (it is a
    uniform cell size). Otherwise the uniform cell size is ``dl_um`` when
    given, else a ``RunResult``'s manifest, else the spacing of a face's two
    normal samples. Faces whose samples are not evenly spaced in their
    plane, or are spaced by other than a whole number of cells, are planes
    of a graded mesh; without the run their grid lines are not known. A
    face on an evenly graded patch of a graded mesh, given as a plain
    array with no manifest, cannot be told from a uniform one. ``dl_um``
    that disagrees with the manifest, with a face's two normal samples or
    with the in-plane sample spacing raises ``ValueError``."""
    given = None
    if dl_um is not None:
        given = float(dl_um)
        if not (np.isfinite(given) and given > 0.0):
            raise ValueError(f"dl_um must be a positive cell size, got {dl_um!r}")
    graded = _run_grid(data)
    if graded is not None:
        if given is not None:
            raise ValueError(
                f"far_field: dl_um={given:g} is a uniform cell size, but this run's "
                "mesh is graded: its manifest lists the grid lines, which place the "
                "samples. Leave dl_um out")
        for (da, ax, sg), name in zip(face_specs, face_names):
            for a in "xyz":
                c = _coords(da, a)
                if c.size and (a in graded.lines or graded.dl):
                    graded.index(a, c, _where(name, ax, sg))
        return graded, None
    uneven = "planes of a graded mesh whose grid lines it is not given"
    for (da, ax, sg), name in zip(face_specs, face_names):
        for t in _TRANSVERSE[ax]:
            c = _coords(da, t)
            if c.size >= 3:
                d = np.abs(np.diff(c))
                if np.max(np.abs(d - d[0])) > _SPACING_RTOL * d[0]:
                    if given is not None:
                        raise ValueError(
                            f"far_field: dl_um={given:g} is a uniform cell size, but "
                            f"{_where(name, ax, sg)} is sampled unevenly along {t} (a "
                            "graded mesh). Pass the RunResult the planes came from: its "
                            "manifest lists the grid lines")
                    return None, uneven
    grid = _grid_block(data)
    normal = [(abs(float(c[1] - c[0])), name, ax, sg)
              for (da, ax, sg), name in zip(face_specs, face_names)
              for c in [_coords(da, ax)] if c.size == 2]
    run_dl = float(grid["dl_um"]) if "dl_um" in grid else None

    def close(a, b):
        return abs(a - b) <= _SPACING_RTOL * max(abs(a), abs(b))

    if given is not None:
        if run_dl is not None and not close(given, run_dl):
            raise ValueError(
                f"far_field: dl_um={given:g} disagrees with the run's cell size "
                f"{run_dl:g} um (its manifest); leave dl_um out")
        for s, name, ax, sg in normal:
            if not close(given, s):
                raise ValueError(
                    f"far_field: dl_um={given:g} disagrees with {_where(name, ax, sg)}, "
                    f"whose two samples along its normal are {s:g} um apart")
        dl = given
    elif run_dl is not None:
        dl = run_dl
    elif normal:
        dl = normal[0][0]
    else:
        return None, None
    for (da, ax, sg), name in zip(face_specs, face_names):
        for t in _TRANSVERSE[ax]:
            c = _coords(da, t)
            if c.size < 2:
                continue
            ratio = abs(float(c[1] - c[0])) / dl
            if round(ratio) < 1 or abs(ratio - round(ratio)) > _SPACING_RTOL * ratio:
                if given is not None:
                    raise ValueError(
                        f"far_field: dl_um={given:g} does not fit {_where(name, ax, sg)}, "
                        f"sampled every {abs(float(c[1] - c[0])):g} um along {t}: the "
                        "spacing must be a whole number of cells")
                return None, uneven
    return _Grid(dl=dl), None


def _yee_dl_um(data, dl_um: Optional[float], face_specs, face_names) -> float:
    """The (uniform) cell size the Yee offsets are measured in, for the
    diagnostic ``yee="positions"`` and ``"surface"`` readouts, which refuse
    what they cannot read (:func:`_mesh`)."""
    if _run_grid(data) is not None:
        raise NotImplementedError(
            "far_field(yee=...) covers uniform meshes only; this run has a "
            "graded grid")
    grid, nonuniform = _mesh(data, dl_um, face_specs, face_names)
    if nonuniform:
        raise NotImplementedError(
            "far_field(yee=...) covers uniform meshes only; these planes come "
            "from a graded grid")
    if grid is None:
        raise ValueError(
            "far_field(yee=...) needs the cell size: pass dl_um=, a RunResult, or "
            "faces recorded one cell thick")
    return float(grid.dl)


def _normal_samples(da, axis: str) -> int:
    """How many samples a face carries along its normal."""
    return int(da.sizes[axis]) if axis in getattr(da, "dims", ()) else 1


def _box_faces(face_specs, mirrored: frozenset = frozenset(),
               left_out: Sequence[str] = ()) -> List[bool]:
    """Which faces belong to a box: a face at least one of whose in-plane
    axes is bounded by two faces normal to it (a closed box, or a box with
    a face left out). ``left_out`` holds the normal axis of each face a
    ``FarFieldMonitor`` box left out (its recipe), which bounds the box all
    the same: the box is cut at that face's declared plane. A face
    normal to a symmetry plane (an axis in
    ``mirrored``) counts with its mirror image when it is a box face itself,
    the faces the readout mirrors (:func:`_mirror_face`): starting from every
    such face counted twice, an image whose face turns out not to be a box
    face is dropped and the count redone, until nothing changes. So the
    kept eighth of a box (one face per axis, three planes) is a box, while
    a plane normal to a symmetry plane with no face bounding it stays an
    open plane and bounds nothing. A single plane, or parallel planes
    alone, are open planes."""
    images = [ax in mirrored for _, ax, _ in face_specs]
    while True:
        count: Dict[str, int] = {}
        for (_, ax, _), image in zip(face_specs, images):
            count[ax] = count.get(ax, 0) + (2 if image else 1)
        for ax in left_out:
            count[ax] = count.get(ax, 0) + 1
        bounded = {ax for ax, c in count.items() if c >= 2}
        box = [bool(set(_TRANSVERSE[ax]) & bounded) for _, ax, _ in face_specs]
        kept = [image and b for image, b in zip(images, box)]
        if kept == images:
            return box
        images = kept


def _refuse_unmirrored_box(face_specs, face_names, box, symmetry, cause: str,
                           fix: str) -> None:
    """A symmetry-reduced box read the earlier way (a fallback) would stay
    open: that readout mirrors faces across a symmetry plane but does not
    add the mirror image of a box face normal to one. Measured on the
    kept-side box of a dipole (lambda/20, faces with one normal sample)
    against the full run: the pattern off by 0.91 of its peak, the
    radiated power 0.73 of the full box's. Refuse it, naming the face."""
    if not symmetry:
        return
    for (_, ax, sg), name, b in zip(face_specs, face_names, box):
        if b and symmetry["xyz".index(ax)]:
            raise ValueError(
                f"far_field: {_where(name, ax, sg)} is a box face normal to the "
                f"symmetry plane at {ax} = 0 of the run, and this projection would "
                f"fall back to {_LEGACY_NOTE}, because {cause}. That readout does "
                "not add the mirror image of such a face, so the box would stay "
                f"open on the far side. {fix}")


@dataclass(frozen=True)
class _Readout:
    """How :func:`far_field` reads the faces: ``legacy`` (the earlier
    readout), or per face ``(positions, surface, colocate)``, whether it is
    a box face (its samples must cover the box) and whether a strided face
    is refused, with the ``grid`` the samples are placed on. ``symmetry``
    holds the run's symmetry planes the faces are mirrored through (both
    readouts), or None. ``left_out`` holds an open box's faces left out,
    ``(axis, outward sign, declared plane um)`` each, where the kept faces
    are cut, and ``window`` a plane's edge window,
    ``((axis, fraction, lo_um, hi_um), ...)`` (both from a
    ``FarFieldMonitor``'s recipe)."""

    legacy: bool
    grid: Optional[_Grid] = None
    modes: Tuple[Tuple[bool, bool, bool], ...] = ()
    refuse_stride: Tuple[bool, ...] = ()
    box: Tuple[bool, ...] = ()
    symmetry: Optional[Tuple[int, int, int]] = None
    left_out: Tuple = ()
    window: Optional[Tuple] = None


_LEGACY_NOTE = ("the earlier readout (yee='legacy'), which averages each "
                "component to the cell node and leaves E and H half a cell "
                "apart along the face normal; on a closed box that leaves a "
                "front/back asymmetry of the pattern first order in the cell "
                "size")


def _warn_future(message: str) -> None:
    warnings.warn(f"far_field: {message}", FutureWarning,
                  stacklevel=caller_stacklevel())


def _refuse_recipe_on_legacy(left_out, window, why: str) -> None:
    """An open box's cut and a plane's edge window are applied by the
    Yee-lattice readout only; the earlier one would silently drop them."""
    if not left_out and not window:
        return
    what = "an open box (faces left out)" if left_out else "a plane with an edge window"
    raise ValueError(
        f"far_field: this far-field surface is {what}, which only the default readout "
        f"(yee='exact') applies, and {why}. Pass the RunResult the faces came from "
        "and leave yee at its default")


def _resolve_readout(data, face_specs, face_names, *, yee: str, colocate: bool,
                     simulation, dl_um: Optional[float], recipe_symmetry=None,
                     left_out=(), window=None) -> _Readout:
    """Decide how the faces are read (see :func:`far_field`, ``yee``).
    ``recipe_symmetry``, ``left_out`` and ``window`` come from a
    ``FarFieldMonitor``'s recipe."""
    left_out = tuple(left_out or ())
    if yee == "legacy":
        _refuse_recipe_on_legacy(left_out, window, "yee='legacy' was asked for")
        _warn_future(
            f"yee='legacy' selects {_LEGACY_NOTE}. It is kept for one release; "
            "the default, yee='exact', reads each component on the Yee lattice")
        return _Readout(legacy=True, symmetry=_given_symmetry(simulation))
    symmetry = _run_symmetry(data, simulation, recipe_symmetry)
    if yee in ("positions", "surface"):
        if symmetry:
            raise NotImplementedError(
                f"far_field(yee={yee!r}) does not handle symmetry-reduced runs yet")
        if left_out or window:
            raise NotImplementedError(
                f"far_field(yee={yee!r}) does not handle open boxes or windows")
        box = _box_faces(face_specs)
        grid = _Grid(dl=_yee_dl_um(data, dl_um, face_specs, face_names))
        modes = tuple((yee == "positions", yee == "surface", bool(colocate))
                      for _ in face_specs)
        return _Readout(legacy=False, grid=grid, modes=modes, refuse_stride=tuple(box),
                        box=tuple(box))

    # yee == "exact", the default.
    mirrored = frozenset(a for a, s in zip("xyz", symmetry or (0, 0, 0)) if s)
    box = _box_faces(face_specs, mirrored, [ax for ax, _, _ in left_out])

    def fall_back(why: str, advice: str, fix: str) -> _Readout:
        _refuse_recipe_on_legacy(left_out, window,
                                 f"the Yee-lattice readout cannot place the samples of {why}")
        _refuse_unmirrored_box(face_specs, face_names, box, symmetry,
                               f"the Yee-lattice readout cannot place the samples of {why}",
                               fix)
        _warn(f"the Yee-lattice readout cannot place the samples of {why}, so this "
              f"projection uses {_LEGACY_NOTE}{advice}")
        return _Readout(legacy=True, symmetry=symmetry)

    if not colocate:
        # Synthetic fields that already share their sample points: each
        # component is read at the coordinate given, on one surface.
        return _Readout(legacy=False, grid=_Grid(dl=dl_um) if dl_um else None,
                        modes=tuple((False, False, False) for _ in face_specs),
                        refuse_stride=tuple(False for _ in face_specs), box=tuple(box),
                        symmetry=symmetry, left_out=left_out, window=window)
    grid, why = _mesh(data, dl_um, face_specs, face_names)
    if why:
        return fall_back(why, " (pass the RunResult the planes came from: its "
                              "manifest lists the grid lines)",
                         "Pass the RunResult the planes came from: its manifest lists "
                         "the grid lines, and the default readout then mirrors the box")
    if grid is not None:
        # A strided box face is refused before the one-sample fallback: a
        # face recorded one cell thick with interval_space=(2, 2, 2) keeps
        # one sample along its normal as well.
        for (da, ax, sg), name, b in zip(face_specs, face_names, box):
            if b:
                _refuse_strided_box(da, ax, grid, name, sg)
    thin = [name for (da, ax, _), name, b in zip(face_specs, face_names, box)
            if b and _normal_samples(da, ax) == 1]
    if thin:
        _refuse_recipe_on_legacy(left_out, window,
                                 "its box faces were recorded with one sample along their normal")
        _refuse_unmirrored_box(
            face_specs, face_names, box, symmetry,
            f"the box faces {', '.join(repr(n) for n in thin)} were recorded with one "
            "sample along their normal",
            "Record each box face one cell thick (size dl along its normal, centred "
            "at (k + 3/4) dl) and at every cell: the default readout then mirrors "
            "the box")
        _warn_future(
            f"the box faces {', '.join(repr(n) for n in thin)} were recorded with "
            "one sample along their normal, so E and H cannot be brought onto "
            f"one surface; this projection uses {_LEGACY_NOTE} (about 0.28 of "
            "the peak at 25 cells per wavelength). For the Yee-lattice readout, "
            "record each face one cell thick (size dl along its normal, centred "
            "at (k + 3/4) dl) and at every cell, without interval_space (a "
            "stride along the normal leaves one sample too; pass dl_um= or the "
            "RunResult so a stride shows); this fallback will be removed in a "
            "future release")
        return _Readout(legacy=True, symmetry=symmetry)
    if grid is None:
        return fall_back(
            "planes whose cell size it cannot tell",
            " (pass dl_um=, or the RunResult the planes came from)",
            "Pass dl_um=, or the RunResult the planes came from: the default "
            "readout then mirrors the box")
    modes = tuple((True, b, True) for b in box)
    return _Readout(legacy=False, grid=grid, modes=modes, refuse_stride=tuple(box),
                    box=tuple(box), symmetry=symmetry, left_out=left_out, window=window)


def _face_component(da: xr.DataArray, name: str, freq_hz: float,
                    t1: str, t2: str, normal: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One component of a face as ``[i_normal, i_t2, i_t1]`` complex values,
    with its base-node coordinates ``(cn, c2, c1)`` in microns. Unlike
    :func:`_plane_component_3d` the normal axis is kept (a face recorded one
    cell thick has two samples along it)."""
    if "component" in getattr(da, "dims", ()):
        comp_coords = [str(c) for c in np.asarray(da.coords["component"].values).ravel()]
        if name not in comp_coords:
            raise ValueError(f"the face has no {name!r} component (has {comp_coords})")
        da = da.sel(component=name)
    if "f" in getattr(da, "dims", ()):
        da = da.sel(f=freq_hz, method="nearest")
    for ax in (normal, t1, t2):
        if ax not in da.dims:
            if ax not in da.coords:
                raise ValueError(f"component {name!r} carries no {ax!r} coordinate")
            da = da.expand_dims(ax)
    extra = [d for d in da.dims if d not in (normal, t1, t2)]
    if any(da.sizes[d] != 1 for d in extra):
        raise ValueError(
            f"component {name!r}: unexpected dims {tuple(da.dims)} for a face "
            f"normal to {normal!r}")
    da = da.squeeze(extra, drop=True) if extra else da
    da = da.transpose(normal, t2, t1)
    vals = np.asarray(da.values, dtype=np.complex128)
    return (vals,
            np.asarray(da.coords[normal].values, dtype=np.float64).ravel(),
            np.asarray(da.coords[t2].values, dtype=np.float64).ravel(),
            np.asarray(da.coords[t1].values, dtype=np.float64).ravel())


def _between(a: np.ndarray, b: np.ndarray, d0: float, d1) -> np.ndarray:
    """Linear interpolation from the H planes of two neighbouring cells of
    widths ``d0`` and ``d1`` (their midpoints) onto the grid line between
    them: ``(d1 a + d0 b) / (d0 + d1)``, the uniform readout's ``(a + b) / 2``
    bit for bit when the widths are equal."""
    if np.all(d0 == d1):
        return 0.5 * (a + b)
    return (d1 * a + d0 * b) / (d0 + d1)


def _face_fields(da: xr.DataArray, axis: str, sign: float, freq_hz: float,
                 *, grid: Optional[_Grid], positions: bool, surface: bool,
                 plane: str, colocate: bool, label: Optional[str] = None,
                 box: bool = False,
                 components: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The tangential fields of one face for the Yee-lattice readout, each
    with the points it sits at.

    Returns a dict: ``fields`` maps each of the face's four tangential
    components (or the ``components`` named, with ``positions``: any of
    ``Ex`` ... ``Hz`` the face recorded, a normal component handled like
    the tangential field of the other kind, E_n with H_t and H_n with E_t,
    since it sits where they do along the normal;
    :func:`~photonhub.analysis.colocated_fields` reads all six) to
    ``(values[i_t2, i_t1], c1_um, c2_um)``, ``n`` maps it to
    its position along the normal, ``s`` is the surface position (where a
    box's quadrature is cut), and ``axis``, ``sign``, ``label``, ``box``,
    ``positions`` and ``base`` (the recorded base-node coordinates, by
    axis) describe the face.

    * ``surface``: the face must carry two normal samples on neighbouring
      grid lines (``n0``, ``n1``). With ``plane="h"`` E is averaged onto the
      H plane of ``n0`` (the midpoint of its cell) and H taken there; with
      ``plane="e"`` H is interpolated onto the E plane ``n1`` from the H
      planes of the two cells (:func:`_between`); with ``plane="mid"`` both
      are interpolated onto ``n0 + 3/4`` of a uniform cell. J and M then
      share one surface. (``"avg"`` is handled by
      :func:`_yee_radiation_vectors`.)
    * otherwise the first normal sample is used: E on its plane ``n0``, H
      on the H plane of that cell; ``positions`` puts J there, without it
      both sit on ``n0`` (the earlier readout's assumption, and the right
      one for co-located synthetic fields).
    * ``positions``: each component on its own in-plane Yee point, placed
      from the ``grid``'s lines, no averaging. Without it the in-plane
      handling follows ``colocate``: average to the node, or (``False``)
      read each component at the coordinate given.
    """
    t1, t2 = _TRANSVERSE[axis]
    e1n, e2n, h1n, h2n = _TANGENTIAL[axis]
    names = tuple(components) if components is not None else (e1n, e2n, h1n, h2n)
    i1, i2, ia = "xyz".index(t1), "xyz".index(t2), "xyz".index(axis)
    comps = {}
    for nm in names:
        comps[nm] = _face_component(da, nm, freq_hz, t1, t2, axis)
    ref = names[0]
    cn = comps[ref][1]
    if any(not np.allclose(comps[nm][1], cn) for nm in names[1:]):
        raise ValueError("the face's components disagree on the normal coordinate")
    vals = {nm: comps[nm][0] for nm in comps}
    # Along the normal a component sits on the node (tangential E, normal H)
    # or half a cell past it (tangential H, normal E).
    on_node = {nm: _YEE_OFFSET[nm][ia] == 0.0 for nm in names}

    values: Dict[str, np.ndarray] = {}
    if surface:
        if cn.size != 2 or not grid.adjacent(axis, cn):
            cell = (f"{grid.dl:g} um apart" if axis not in grid.lines
                    else "on neighbouring grid lines")
            raise ValueError(
                f"far_field: {_where(label, axis, sign)} must be recorded one cell "
                f"thick (two samples along its normal, {cell}) to bring E and H "
                f"onto one surface; it has {cn.size} sample(s) at "
                f"{np.round(cn, 6).tolist()} um")
        lo = 0 if cn[1] > cn[0] else 1
        hi = 1 - lo
        d0, d1 = grid.width(axis, float(cn[lo])), grid.width(axis, float(cn[hi]))
        if plane == "h":
            s = float(cn[lo]) + 0.5 * d0
            for nm in names:
                v = vals[nm]
                values[nm] = 0.5 * (v[lo] + v[hi]) if on_node[nm] else v[lo]
        elif plane == "e":
            s = float(cn[hi])
            for nm in names:
                v = vals[nm]
                values[nm] = v[hi] if on_node[nm] else _between(v[lo], v[hi], d0, d1)
        elif plane == "mid":
            # Both onto n0 + 3/4: E sits 3/4 of the way between its samples
            # (n0, n0 + 1), H 1/4 of the way between its own (n0 + 1/2,
            # n0 + 3/2). Linear interpolation at fraction t scales a wave of
            # normal wavenumber k_n by 1 - t (1 - t) (k_n dl)^2 / 2, the same
            # factor for t = 3/4 and t = 1/4: J and M stay balanced to second
            # order for every component of the field. A uniform cell only.
            if d0 != d1:
                raise NotImplementedError(
                    "yee_plane='mid' needs equal cells along the face normal")
            s = float(cn[lo]) + 0.75 * d0
            for nm in names:
                v = vals[nm]
                values[nm] = (0.25 * v[lo] + 0.75 * v[hi] if on_node[nm]
                              else 0.75 * v[lo] + 0.25 * v[hi])
        else:
            raise ValueError(f"yee_plane must be 'h', 'e' or 'mid', got {plane!r}")
        n = {nm: s for nm in names}
    else:
        n_e = float(cn[0])
        n_h = grid.place(axis, n_e, 0.5) if positions else n_e
        s = n_e
        for nm in names:
            values[nm] = vals[nm][0]
        n = {nm: n_e if on_node[nm] else n_h for nm in names}

    def where(nm):
        c2, c1 = comps[nm][2], comps[nm][3]
        if positions:
            off = _YEE_OFFSET[nm]
            return grid.place(t1, c1, off[i1]), grid.place(t2, c2, off[i2])
        return c1, c2

    if not positions and colocate:
        # The earlier readout's in-plane averaging ([i_t2, i_t1] arrays) of
        # the four tangential components.
        node_axis = {e1n: -1, e2n: 0, h1n: 0, h2n: -1}
        if any(nm not in node_axis for nm in names):
            raise ValueError("the earlier readout averages the four tangential "
                             f"components only, not {names}")
        values = {nm: _colocate_to_node(values[nm], node_axis[nm]) for nm in names}

    return {"axis": axis, "sign": float(sign), "s": s, "label": label, "box": box,
            "positions": positions,
            "base": {axis: cn, t1: comps[ref][3], t2: comps[ref][2]},
            "n": n,
            "fields": {nm: (values[nm], *where(nm)) for nm in names}}


def _unfold_face(face: Dict[str, object], symmetry: Tuple[int, int, int]
                 ) -> Dict[str, object]:
    """The face a full run would record, from the part a symmetry-reduced
    run recorded: each component mirrored through every symmetry plane that
    cuts the face (its in-plane axes), at its own points, with the sign
    that plane gives it (:func:`_mirror_parity`). A component on the mirror
    plane (no Yee offset along that axis, or any component of co-located
    fields) keeps its one sample there; one half a cell off it gains the
    mirrored samples. Nothing is averaged. The face must reach the plane,
    which sits at coordinate 0 of the run's frame."""
    axis = face["axis"]
    t1, t2 = _TRANSVERSE[axis]
    fields = dict(face["fields"])
    for a in (t1, t2):
        s = int(symmetry["xyz".index(a)])
        base = np.asarray(face["base"][a], dtype=np.float64)
        if s == 0 or base.size == 0:
            continue
        step = abs(float(base[1] - base[0])) if base.size >= 2 else 0.0
        if base.size >= 2 and np.any(np.diff(base) <= 0.0):
            raise ValueError(
                f"far_field: {_where(face['label'], axis, face['sign'])} is not sampled "
                f"in increasing {a}, so it cannot be mirrored through the symmetry "
                f"plane at {a} = 0")
        if abs(float(base[0])) > max(1e-6 * step, 1e-12):
            raise ValueError(
                f"far_field: {_where(face['label'], axis, face['sign'])} starts at "
                f"{a} = {float(base[0]):g} um, away from the symmetry plane at {a} = 0 "
                "of the run; a face that does not reach a symmetry plane cannot be "
                "mirrored into the full face")
        k = "xyz".index(a)
        dim = 1 if a == t1 else 0                       # values are [i_t2, i_t1]
        for nm, (vals, c1, c2) in fields.items():
            p = _mirror_parity(nm, a, s)
            pos = np.asarray(c1 if a == t1 else c2, dtype=np.float64)
            keep = 0 if face["positions"] and _YEE_OFFSET[nm][k] else 1
            image = -pos[keep:][::-1]
            mirrored = p * np.flip(np.take(vals, np.arange(keep, pos.size), axis=dim),
                                   axis=dim)
            pos = np.concatenate([image, pos])
            vals = np.concatenate([mirrored, vals], axis=dim)
            fields[nm] = (vals, pos, c2) if a == t1 else (vals, c1, pos)
    return dict(face, fields=fields)


def _mirror_face(face: Dict[str, object], symmetry: Tuple[int, int, int]
                 ) -> Dict[str, object]:
    """The mirror image of a box face normal to a symmetry plane: the face
    on the far side of the box, read as the exact mirror of the recorded
    one (each surface and sample reflected through the plane at 0, the
    outward normal reversed, each component times its parity). The face
    must point away from the plane and not sit on it."""
    axis = face["axis"]
    s = int(symmetry["xyz".index(axis)])
    base = np.asarray(face["base"][axis], dtype=np.float64)
    where = _where(face["label"], axis, face["sign"])
    cell = abs(float(base[1] - base[0])) if base.size >= 2 else 0.0
    if face["sign"] < 0 or float(np.min(base)) <= max(1e-6 * cell, 1e-12):
        raise ValueError(
            f"far_field: {where} is a box face normal to the symmetry plane at "
            f"{axis} = 0 of the run, so its mirror image closes the box on the "
            "other side; it must face away from the plane (outward normal "
            f"+{axis}) and lie beyond it. Leave out a face on the plane itself")
    label = f"{face['label']} (mirror image)" if face["label"] else None
    return dict(face, sign=-face["sign"], s=-face["s"], label=label,
                n={nm: -v for nm, v in face["n"].items()},
                fields={nm: (_mirror_parity(nm, axis, s) * vals, c1, c2)
                        for nm, (vals, c1, c2) in face["fields"].items()})


def _edge_window(c_um: np.ndarray, frac: float, lo_um: float, hi_um: float) -> np.ndarray:
    """The edge window along one axis at the sample positions ``c_um``: 1 in
    the middle of the extent ``lo_um .. hi_um``, rising as ``sin^2`` from 0
    at each end over ``frac`` of the extent (0 outside it)."""
    c = np.asarray(c_um, dtype=np.float64)
    width = float(frac) * (float(hi_um) - float(lo_um))
    if width <= 0.0:
        return np.ones(c.shape)
    d = np.minimum(c - float(lo_um), float(hi_um) - c)
    return np.where(d >= width, 1.0, np.sin(0.5 * np.pi * np.clip(d, 0.0, width) / width) ** 2)


def _window_face(face: Dict[str, object], window) -> Dict[str, object]:
    """``face`` with each component multiplied by the edge window at its own
    sample points (so are the currents formed from it): ``window`` gives
    ``(axis, fraction, lo_um, hi_um)`` per in-plane axis."""
    t1, t2 = _TRANSVERSE[face["axis"]]
    parts = {str(a): (float(f), float(lo), float(hi)) for a, f, lo, hi in window}
    fields = {}
    for nm, (vals, c1, c2) in face["fields"].items():
        w1 = _edge_window(c1, *parts[t1]) if t1 in parts else np.ones(np.size(c1))
        w2 = _edge_window(c2, *parts[t2]) if t2 in parts else np.ones(np.size(c2))
        fields[nm] = (vals * (w2[:, None] * w1[None, :]), c1, c2)
    return dict(face, fields=fields)


def _face_terms(face: Dict[str, object]) -> list:
    """The current terms of a face, ``(current, values[i_t2, i_t1], c1_um,
    c2_um, normal_um)``: the Cartesian current component each adds to
    (``"Jx"`` ... ``"Mz"``) and where its samples sit."""
    axis, sign = face["axis"], face["sign"]
    t1, t2 = _TRANSVERSE[axis]
    e1n, e2n, h1n, h2n = _TANGENTIAL[axis]
    f, n = face["fields"], face["n"]
    # n_hat x A_t = sign * (-a2 t1_hat + a1 t2_hat); J = n x H, M = -n x E.
    return [
        (f"J{t1}", -sign * f[h2n][0], f[h2n][1], f[h2n][2], n[h2n]),
        (f"J{t2}", sign * f[h1n][0], f[h1n][1], f[h1n][2], n[h1n]),
        (f"M{t1}", sign * f[e2n][0], f[e2n][1], f[e2n][2], n[e2n]),
        (f"M{t2}", -sign * f[e1n][0], f[e1n][1], f[e1n][2], n[e1n]),
    ]


def _cubic_weights_clipped(coords: np.ndarray,
                           clip: Tuple[float, float]) -> np.ndarray:
    """Quadrature weights for ``integral_a^b f`` over samples that reach
    past both ends: the integral of the piecewise-cubic Lagrange interpolant
    (four-point stencil per cell, centred where the samples allow, shifted
    inward at the array ends), cut exactly at ``a`` and ``b``. Fourth
    order, where the clipped midpoint rule is second order at an end that
    falls mid-cell or on a sample.

    The samples may be unevenly spaced (a graded mesh): the interpolant runs
    through their true positions, so the rule stays exact for cubics and
    fourth order on any smoothly graded spacing. On even spacing the
    interior weights equal the spacing and only the samples around each
    end differ; on a spacing graded by up to 1.4 per cell they stay within
    about 5 % of the local cell width. A cell whose stencil spans spacings
    more than :data:`_CUBIC_MAX_JUMP` apart (an abrupt jump in the mesh,
    where the cubic's weights turn negative) is integrated with the linear
    interpolant of its two samples instead."""
    c = np.asarray(coords, dtype=np.float64)
    n = c.size
    a, b = sorted((float(clip[0]), float(clip[1])))
    w = np.zeros(n)
    # An end sample that meets an edge to rounding (an arange that stops a
    # few ulps short) still reaches it.
    tol = 1e-9 * abs(float(c[1] - c[0])) if n >= 2 else 0.0
    if n < 4 or min(c[0], c[-1]) > a + tol or max(c[0], c[-1]) < b - tol:
        # Too few samples, or samples that stop short of an edge: the
        # clipped midpoint rule, which extends the end samples half a cell.
        return _cell_widths_clipped(c, clip)
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)            # 2-point Gauss: exact for cubics
    for i in range(n - 1):
        lo, hi = sorted((c[i], c[i + 1]))
        lo, hi = max(lo, a), min(hi, b)
        if hi <= lo:
            continue
        s = min(max(i - 1, 0), n - 4)                       # stencil c[s..s+3]
        xs = c[s:s + 4]
        xg = 0.5 * (lo + hi) + 0.5 * (hi - lo) * g
        h = np.abs(np.diff(xs))
        if np.max(h) > _CUBIC_MAX_JUMP * np.min(h):
            # An abrupt jump in the spacing: the cell's own linear interpolant.
            for x in xg:
                t = (x - c[i]) / (c[i + 1] - c[i])
                w[i] += 0.5 * (hi - lo) * (1.0 - t)
                w[i + 1] += 0.5 * (hi - lo) * t
            continue
        for x in xg:
            for j in range(4):
                lj = 1.0
                for m in range(4):
                    if m != j:
                        lj *= (x - xs[m]) / (xs[j] - xs[m])
                w[s + j] += 0.5 * (hi - lo) * lj
    return w


def _plane_weights(coords: np.ndarray) -> np.ndarray:
    """Quadrature weights along one axis of an open plane (no box edge to
    cut at). Evenly spaced samples take the centred cell widths
    (:func:`~photonhub.analysis.mode_overlap._cell_widths`, half a cell past
    each end), on which a field that dies out toward the plane's edges
    integrates to spectral accuracy. Unevenly spaced ones (a graded mesh,
    or a strided plane mirrored across a symmetry plane, whose samples
    either side of the plane sit one cell apart) take the integral of the
    piecewise-cubic interpolant
    across the samples (:func:`_cubic_weights_clipped`) and the same half
    cell past each end: the centred widths are only second order there
    (8e-4 of the peak on the graded-plane beam of the tests, 3.5e-5 with
    this rule)."""
    c = np.asarray(coords, dtype=np.float64)
    if c.size < 4:
        return _cell_widths(c)
    d = np.diff(c)
    if np.max(np.abs(d - d[0])) <= _EVEN_RTOL * abs(d[0]):
        return _cell_widths(c)
    w = _cubic_weights_clipped(c, (float(c[0]), float(c[-1])))
    w[0] += 0.5 * abs(float(c[1] - c[0]))
    w[-1] += 0.5 * abs(float(c[-1] - c[-2]))
    return w


def _term_widths(coords: np.ndarray, clip: Optional[Tuple[float, float]],
                 other: np.ndarray, letter: str, grid: Optional[_Grid]) -> np.ndarray:
    """Quadrature weights (m) along one in-plane axis ``letter`` of a face
    term sampled at ``coords`` (m), ``other`` the samples along the other
    axis: the fourth-order rule cut at the box edges ``clip``
    (:func:`_cubic_weights_clipped`) on a box, :func:`_plane_weights` on a
    plane; a one-sample axis (a quasi-2-D plane) takes the width of its one
    cell on a graded run, else the other axis's spacing. Shared by the
    far-field kernel (:func:`_project_terms`) and the exact one."""
    if coords.size == 1 and other.size >= 2:
        one = grid.one_cell_width(letter) if grid is not None else None
        if one is not None:
            return np.array([one * 1e-6])
        return np.array([abs(float(other[1] - other[0]))])
    if clip is None:
        return _plane_weights(coords)
    if _YEE_EDGE_QUADRATURE == "cubic":
        return _cubic_weights_clipped(coords, clip)
    return _cell_widths_clipped(coords, clip)


def _project_terms(terms, *, axis: str, k: float, rx: np.ndarray,
                   ry: np.ndarray, rz: np.ndarray,
                   clip1: Optional[Tuple[float, float]],
                   clip2: Optional[Tuple[float, float]],
                   grid: Optional[_Grid] = None) -> Dict[str, np.ndarray]:
    """Radiation-vector contributions of a face's current terms, each
    integrated on its own sample grid with the separable kernel
    ``exp(-i k r_hat.r') = exp(-i k r_1 c1) exp(-i k r_2 c2) exp(-i k r_n n)``
    (exactly the dense kernel of :func:`_project_one_face`, factored): two
    small exponential tables and one matrix product per term, instead of an
    exponential per surface point and direction. Terms sampled on the same
    in-plane points share their tables and weights. The directions are
    taken in chunks (:data:`_SEPARABLE_CHUNK`). The weights come from the
    samples' own positions: the fourth-order rule cut at the box edges
    (:func:`_cubic_weights_clipped`) on a box, :func:`_plane_weights` on a
    plane; a one-sample axis (a quasi-2-D plane) takes the width of its one
    cell on a graded run, else the other axis's spacing."""
    t1, t2 = _TRANSVERSE[axis]
    r = {"x": rx, "y": ry, "z": rz}
    n_dir = rx.size
    out: Dict[str, np.ndarray] = {}

    groups: Dict[Tuple[bytes, bytes], list] = {}
    for term in terms:
        c1 = np.asarray(term[2], dtype=np.float64) * 1e-6
        c2 = np.asarray(term[3], dtype=np.float64) * 1e-6
        groups.setdefault((c1.tobytes(), c2.tobytes()), [c1, c2, []])[2].append(term)
    for c1, c2, members in groups.values():
        w = (_term_widths(c2, clip2, c1, t2, grid)[:, None]
             * _term_widths(c1, clip1, c2, t1, grid)[None, :])
        parts = [(name, (vals * w).T, float(n_um) * 1e-6,           # [n1, n2]
                  np.empty(n_dir, dtype=np.complex128))
                 for name, vals, _c1, _c2, n_um in members]
        chunk = max(1, int(_SEPARABLE_CHUNK // max(c1.size, c2.size, 1)))
        for lo in range(0, n_dir, chunk):
            hi = min(lo + chunk, n_dir)
            p1 = np.exp(-1j * k * np.outer(r[t1][lo:hi], c1))     # [nd, n1]
            p2 = np.exp(-1j * k * np.outer(r[t2][lo:hi], c2))     # [nd, n2]
            for _name, gT, n_m, val in parts:
                pn = np.exp(-1j * k * r[axis][lo:hi] * n_m)        # [nd]
                val[lo:hi] = pn * np.einsum("dj,dj->d", p1 @ gT, p2)
        for name, _gT, _n_m, val in parts:
            out[name] = out.get(name, 0.0) + val
    return out


def _yee_radiation_vectors(face_specs, freq_hz: float, *, k: float,
                           rx: np.ndarray, ry: np.ndarray, rz: np.ndarray,
                           readout: _Readout, plane: str,
                           names: Optional[Sequence[str]] = None) -> list:
    """``[N_x, N_y, N_z, L_x, L_y, L_z]`` summed over the faces for the
    Yee-lattice readout. On a symmetry-reduced run each face is first
    mirrored into the face the full run would record (:func:`_unfold_face`)
    and a box face normal to a symmetry plane gains its mirror image
    (:func:`_mirror_face`). A box clips each face's quadrature to the
    surfaces the faces' currents actually sit on (F11's tiling, applied to
    the J/M surface rather than the recorded base plane). ``plane="avg"``
    averages the ``"h"`` and ``"e"`` projections of the faces read on one
    surface, whose J/M imbalances have opposite signs; it leaves the other
    faces as they are. An open box (``readout.left_out``) is cut at the
    declared plane of each face it left out, on every pass; a plane's edge
    window (``readout.window``) multiplies its currents, after the
    unfolding. The faces of each pass come from :func:`_yee_passes`."""
    results = []
    for faces, bounds in _yee_passes(face_specs, freq_hz, readout=readout, plane=plane,
                                     names=names):
        total = {c: np.zeros(rx.size, dtype=np.complex128)
                 for c in ("Jx", "Jy", "Jz", "Mx", "My", "Mz")}
        for fc in faces:
            ax, sg = fc["axis"], fc["sign"]
            t1, t2 = _TRANSVERSE[ax]
            terms = _face_terms(fc)
            if fc["box"]:
                for term in terms:
                    _refuse_open_box(term, t1, t2, bounds, fc["label"], ax, sg)
            contrib = _project_terms(terms, axis=ax, k=k, rx=rx, ry=ry, rz=rz,
                                     clip1=bounds.get(t1), clip2=bounds.get(t2),
                                     grid=readout.grid)
            for name, val in contrib.items():
                total[name] = total[name] + val
        results.append([total[c] for c in ("Jx", "Jy", "Jz", "Mx", "My", "Mz")])
    if len(results) == 1:
        return results[0]
    return [0.5 * (a + b) for a, b in zip(*results)]


def _yee_passes(face_specs, freq_hz: float, *, readout: _Readout, plane: str,
                names: Optional[Sequence[str]] = None) -> list:
    """The surface the Yee-lattice readout integrates, one ``(faces,
    bounds)`` per pass: ``plane="avg"`` reads the faces read on one surface
    twice, E onto the H plane (``"h"``) and H onto the E plane (``"e"``),
    and the two results are averaged; any other ``plane`` is one pass.
    ``faces`` are the :func:`_face_fields` dicts, unfolded through the
    symmetry planes, windowed, each box face normal to a plane followed by
    its mirror image; ``bounds`` maps each axis bounded by two surfaces (a
    box's faces, or the declared plane of a face an open box left out) to
    the ``(lo, hi)`` (m) each face's quadrature is cut at. The far-field kernel
    (:func:`_yee_radiation_vectors`) and the exact one read the same
    surface from here."""
    if plane == "avg" and any(surface for _, surface, _ in readout.modes):
        planes: Tuple[str, ...] = ("h", "e")
    else:
        planes = (plane,)
    labels = (list(names) if names is not None and len(names) == len(face_specs)
              else [None] * len(face_specs))
    for (da, ax, sg), label, refuse in zip(face_specs, labels, readout.refuse_stride):
        if refuse:
            _refuse_strided_box(da, ax, readout.grid, label, sg)
    sym = readout.symmetry
    passes = []
    for pl in planes:
        faces = []
        for (da, ax, sg), (pos, surf, col), label, in_box in zip(
                face_specs, readout.modes, labels, readout.box):
            face = _face_fields(da, ax, sg, freq_hz, grid=readout.grid,
                                positions=pos, surface=surf, plane=pl, colocate=col,
                                label=label, box=in_box)
            if sym:
                face = _unfold_face(face, sym)
            if readout.window:
                face = _window_face(face, readout.window)
            faces.append(face)
            if sym and in_box and sym["xyz".index(ax)]:
                faces.append(_mirror_face(face, sym))
        bounds: Dict[str, Tuple[float, float]] = {}
        if len(faces) + len(readout.left_out) > 1:
            by_axis: Dict[str, list] = {}
            for fc in faces:
                by_axis.setdefault(fc["axis"], []).append(float(fc["s"]) * 1e-6)
            # an open box ends at the declared plane of each face it left
            # out, on every pass, wherever the grid puts the faces it kept
            for ax, _sg, plane_um in readout.left_out:
                by_axis.setdefault(ax, []).append(float(plane_um) * 1e-6)
            bounds = {ax: (min(p), max(p)) for ax, p in by_axis.items() if len(p) >= 2}
        passes.append((faces, bounds))
    return passes


def _refuse_strided_box(da, axis: str, grid: _Grid, face: Optional[str] = None,
                        sign: float = 1.0) -> None:
    """The faces of a box, closed or with a face left out, must be recorded
    at every cell on the Yee-lattice readout (``da``: the face as recorded;
    on a graded mesh a stride is a step of more than one grid line).

    Each face's field does not vanish at the box edges, so the quadrature
    there needs every cell; and a source a few cells from a face gives it
    a near field that varies over those few cells, which no resampling of
    a decimated face recovers. Measured on FDTD (a dipole in a 1.2 um box,
    25 cells per wavelength): a stride of 2, cubic-spline resampled,
    matched stride 1 to about 1e-5 in power with the source 0.36 um or
    more from the faces, and missed it by 17 % with the source near a
    corner of the box (0.10 to 0.12 um from three faces). A single plane,
    or parallel planes alone, keep their stride."""
    for t in _TRANSVERSE[axis]:
        c = _coords(da, t)
        step = grid.stride(t, c) if c.size >= 2 else 1.0
        if step > 1.5:
            raise ValueError(
                "far_field: the faces of a box, closed or with a face left out, "
                f"must be recorded at every cell, but {_where(face, axis, sign)} is "
                f"sampled every {step:.3g} cells "
                "(interval_space). Record the box faces at every cell and one "
                "cell thick, without interval_space; a single plane may keep "
                "its stride")


def _refuse_open_box(term, t1: str, t2: str, bounds, face: Optional[str],
                     axis: str, sign: float) -> None:
    """Each face of a box must cover the box in its plane: the samples of
    every component, with the half sample the quadrature extends past the
    last one, must reach the surfaces the neighbouring faces' currents sit
    on. On the Yee-lattice readout that surface lies up to one cell past a
    face's first sample (H brought onto the E plane of the second sample),
    so a face recorded only over the box's own span leaves
    the box open by a strip: measured on FDTD (a dipole in a box 24 to 48
    cells wide) the default readout's radiated power then reads 0.5 to 1 %
    low, against 0.06 % with the faces one cell past the box. Record each
    box face one cell past the box on every side in its plane."""
    name, _vals, c1_um, c2_um, _n = term
    where = (f"face {face!r} (normal to {'+' if sign > 0 else '-'}{axis})"
             if face else f"the face normal to {'+' if sign > 0 else '-'}{axis}")
    for letter, c in ((t1, np.asarray(c1_um, dtype=np.float64)),
                      (t2, np.asarray(c2_um, dtype=np.float64))):
        if letter not in bounds or c.size < 2:
            continue
        a, b = (v * 1e6 for v in bounds[letter])          # m -> um
        c = np.sort(c)
        half_lo, half_hi = 0.5 * float(c[1] - c[0]), 0.5 * float(c[-1] - c[-2])
        lo, hi = float(c[0]) - half_lo, float(c[-1]) + half_hi
        if lo > a + 1e-6 * half_lo or hi < b - 1e-6 * half_hi:
            raise ValueError(
                f"far_field: {where} does not cover the box: the samples of its "
                f"{name[0]} current reach {letter} = {lo:.6g} .. {hi:.6g} um "
                f"(half a sample past the last one), but the box's surface spans "
                f"{letter} = {a:.6g} .. {b:.6g} um. Record each box face one cell "
                "past the box on every side in its plane")


# --- Symmetry planes: unfolding a reduced domain's plane -------------------

def _mirror_parity(component: str, axis: str, symmetry: int) -> int:
    """Sign a field component picks up on reflection through a symmetry plane
    normal to ``axis``.

    A PEC plane (``symmetry = -1``) pins the tangential E and passes the normal
    component; a PMC plane (``+1``) does the opposite. H is a pseudovector, so
    each of its components takes the opposite sign of the E component with the
    same index."""
    normal = component[1] == axis
    if component[0] == "E":
        return -symmetry if normal else symmetry
    return symmetry if normal else -symmetry


def _half_offset(component: str, axis: str) -> bool:
    """Whether ``component`` sits half a cell past the primary node along
    ``axis`` (the Yee stagger): ``E_c`` is offset along ``c``, ``H_c``
    along the two axes that are not ``c``."""
    if component[0] == "E":
        return component[1] == axis
    return component[1] != axis


def unfold_symmetry(field: xr.DataArray, simulation) -> xr.DataArray:
    """Mirror a plane recorded on a symmetry-reduced domain into the full plane.

    A run with ``Simulation.symmetry`` solves a fraction of the domain and
    records only that fraction. This returns the plane the full domain would
    have recorded: each component is reflected through every symmetry plane
    that cuts the monitor, with the sign that plane gives it
    (:func:`_mirror_parity`), and the mirrored coordinates run negative.

    The symmetry planes sit on the domain's minimum faces, at coordinate 0, so
    the recorded region has to start there; a monitor that stops short of a
    symmetry plane cannot be unfolded and raises ``ValueError``.

    Components that are half a cell off the node along a mirrored axis are
    averaged onto the node first (they have no sample on the plane itself), so
    **the result is node-centred**: pass ``colocate=False`` if you hand it to
    :func:`far_field` yourself. :func:`far_field` mirrors a symmetry-reduced
    run's faces itself, each component at its own Yee point and with no
    averaging (this function's parities, :func:`_mirror_parity`); this one
    serves the plotted, user-frame view of a result and the earlier readout.
    A run with no symmetry is returned unchanged.
    """
    sym = tuple(getattr(simulation, "symmetry", (0, 0, 0)) or (0, 0, 0))
    if not any(sym):
        return field
    if "component" not in getattr(field, "dims", ()):
        raise ValueError(
            "unfold_symmetry needs the monitor's 'component' dim to know each "
            f"component's parity; got dims {tuple(getattr(field, 'dims', ()))}")
    comps = [str(c) for c in np.asarray(field.coords["component"].values).ravel()]
    out = field
    for a in ("x", "y", "z"):
        s = int(sym["xyz".index(a)])
        if s == 0 or a not in out.dims or out.sizes[a] < 2:
            continue
        coord = np.asarray(out.coords[a].values, dtype=np.float64)
        step = float(coord[1] - coord[0])
        if abs(float(coord[0])) > 0.5 * abs(step):
            raise ValueError(
                f"the monitor starts at {a} = {float(coord[0]):g} um, away from "
                f"the symmetry plane at {a} = 0; a plane that does not reach the "
                "symmetry plane cannot be unfolded")
        rest = [d for d in out.dims if d not in ("component", a)]
        t = out.transpose("component", a, *rest)
        v = np.asarray(t.values)
        n = coord.size
        full = np.empty((v.shape[0], 2 * n - 1) + v.shape[2:], dtype=v.dtype)
        for k, name in enumerate(comps):
            p = _mirror_parity(name, a, s)
            col = v[k]
            if _half_offset(name, a):
                node = np.empty_like(col)
                node[1:] = 0.5 * (col[1:] + col[:-1])
                node[0] = 0.5 * (col[0] + p * col[0])  # the ghost below is its image
                col = node
            full[k] = np.concatenate([p * col[1:][::-1], col], axis=0)
        coords = {d: t.coords[d] for d in t.dims if d != a and d in t.coords}
        coords[a] = np.concatenate([-coord[1:][::-1], coord])
        out = xr.DataArray(full, dims=t.dims, coords=coords, attrs=dict(out.attrs),
                           name=out.name)
    return out.transpose(*field.dims)   # the caller's dim order, unchanged


@legacy_keywords(theta="theta_rad", phi="phi_rad")
def far_field(
    data,
    monitor_name: str,
    *,
    theta_rad: Union[None, float, Sequence[float], np.ndarray] = None,
    phi_rad: Union[float, Sequence[float], np.ndarray] = 0.0,
    axis: Optional[Axis] = None,
    sign: float = 1.0,
    freqs_hz: Optional[Sequence[float]] = None,
    mesh: bool = True,
    faces: Optional[Sequence[Tuple[str, str, float]]] = None,
    colocate: bool = True,
    simulation=None,
    n_medium: Union[None, str, float, Sequence[float], np.ndarray] = None,
    yee: str = "exact",
    dl_um: Optional[float] = None,
    ux: Union[None, float, Sequence[float], np.ndarray] = None,
    uy: Union[None, float, Sequence[float], np.ndarray] = None,
    plane_axis: Optional[Axis] = None,
    direction: Optional[str] = None,
    plane_um: Optional[float] = None,
    x_um: Union[None, float, Sequence[float], np.ndarray] = None,
    y_um: Union[None, float, Sequence[float], np.ndarray] = None,
    exact: bool = False,
    r_um: Optional[float] = None,
    origin_um: Optional[Sequence[float]] = None,
) -> FarField:
    """Project a recorded near-field surface onto the far field.

    Single-plane (default) or box projection of the tangential ``E``/``H``
    recorded by a ``ProfileMonitor`` (``data[monitor_name]``), via the
    surface-source currents ``J = n_hat x H``, ``M = -n_hat x E`` and the
    radiation integral (see the module docstring). Returns a :class:`FarField`
    whose :attr:`~FarField.dataset` holds the far-field amplitudes
    ``E_theta``/``E_phi`` and the intensity, radiated power and directivity
    that follow, labelled by frequency and the grid's axes. The surface
    radiates into a homogeneous, lossless medium of index ``n_medium``
    (vacuum by default).

    Each recorded component is read at its own Yee point (``yee="exact"``,
    the default), on a uniform or a graded mesh, and a symmetry-reduced
    run is mirrored into the full surface first. A single plane needs one
    sample along its normal and may keep an ``interval_space`` stride. The
    faces of a box must be recorded one cell thick and at every cell: a
    face of size ``dl`` along its normal, centred at ``(k + 3/4) dl`` (on a
    graded mesh, from a quarter of cell ``k`` to a quarter of cell ``k +
    1``), records the two normal samples the readout brings E and H
    together from.

    Parameters
    ----------
    data:
        A :class:`~photonhub.data.RunResult` (or any mapping) such that
        ``data[monitor_name]`` is the recorded DFT field DataArray, dims
        ``('f','component','z','y','x')`` (a single-plane slice has a singleton
        normal axis). For a box pass per-face monitors via ``faces``.
        A plain mapping of arrays is read in the wire frame (the solver's
        corner-anchored coordinates); a ``RunResult`` hands its planes over
        through ``frame.wire_array``, so a fitted run's user-frame
        coordinates are handled, and its manifest gives the cell size (the
        grid lines of a graded mesh) and the symmetry planes. The
        medium sampling for ``simulation=`` reads the planes in that frame.
    monitor_name:
        Key of the near-field monitor in ``data`` (single-plane mode), or
        the name of a :class:`~photonhub.FarFieldMonitor`: its faces are
        projected as recorded (``faces`` is then not needed), into the
        medium the simulation sampled at them unless ``n_medium`` is given,
        mirrored through the symmetry planes it was compiled with, cut where
        an open box leaves faces out, and with a plane's edge window. The
        run keeps that recipe (a reloaded result reads it back from the
        ``client.json`` beside the run); without it the faces are found by
        their names, ``<name>.x_minus`` ... ``<name>.z_plus``, with a
        warning, the projection is into vacuum unless ``n_medium`` says
        otherwise, and the faces are read as recorded (no open edge, no
        window).
    theta_rad, phi_rad:
        Far-field directions (radians). With ``mesh=True`` (default) these are
        1-D axes meshed into a ``(n_theta, n_phi)`` grid (``theta_rad``
        slowest); with ``mesh=False`` they are paired elementwise.
        ``theta_rad`` is the polar angle from ``+z``, ``phi_rad`` the azimuth
        from ``+x`` in the x-y plane. ``theta`` and ``phi`` are accepted as
        the earlier spellings. One target grid is given: ``theta_rad``
        (directions), ``ux`` and ``uy`` (direction cosines), or
        ``plane_um``, ``x_um`` and ``y_um`` (points on a plane).
    axis:
        Plane normal ``"x"``/``"y"``/``"z"`` (single-plane mode). If ``None``,
        inferred from the monitor's singleton spatial dimension (pass it for
        a plane recorded one cell thick).
    sign:
        Orientation of the outward normal: ``+1`` along ``+axis`` (default,
        i.e. radiating toward larger ``axis``), ``-1`` along ``-axis``.
    freqs_hz:
        Frequencies to project (Hz). Defaults to every frequency on the monitor.
    mesh:
        Whether ``theta_rad``/``phi_rad`` are meshed (default) or paired.
    faces:
        Optional multi-face projection: a sequence of
        ``(monitor_name, axis, sign)`` triples, one per face, summed with
        each face's own outward normal. When given, ``monitor_name``/``axis``/
        ``sign`` are ignored. Each named monitor must be in ``data``. Faces
        whose in-plane extent is bounded by other faces (a closed box, or a
        box with a face left out) are read as a box; parallel planes alone
        are read as open planes. A list of a box with a face left out is read
        without its open edge: along an axis that carries one face the faces
        kept are not cut at all, so they run past that face and past the open
        side alike. For a dipole at the centre of such a box 0.62 um in half
        width, in index 1.45, that reads the pattern 9.4 and 6.2 % of the
        peak (within 30 degrees, on exact dipole fields at 20 and 30 cells
        per wavelength) away
        from the same box declared with ``FarFieldMonitor.box(...,
        exclude=...)``, which ends them at the declared planes. Declare an
        open box that way.
    colocate:
        ``True`` (default): the planes are the engine's Yee-staggered
        output, as a ``ProfileMonitor`` records them. ``False``: synthetic
        fields whose components already share their sample points; each
        component is then read at the coordinate given, on one surface, and
        no cell size is needed. With ``yee="legacy"`` it keeps its earlier
        meaning (average each component to the in-plane node).
    simulation:
        The :class:`~photonhub.Simulation` the data came from. Its
        ``symmetry`` planes say how a symmetry-reduced run is mirrored into
        the full surface (see ``yee``); a ``RunResult`` records them itself,
        and when both say, they must agree (``ValueError``: pass the run's
        own simulation). A plain mapping of arrays from a symmetry-reduced
        run needs it. Required by
        ``n_medium="auto"``. When given, the medium at each face is sampled
        half a cell beyond the recorded plane on the side it radiates into,
        over the face's recorded extent (the sampling
        :func:`~photonhub.analysis.diffraction_orders` uses). A plane that
        is not in one homogeneous medium there, sits in a lossy medium, or
        sits in a medium of another index than the one projected into,
        gives a ``UserWarning``. The faces of a closed box that do not all
        sit in one homogeneous medium raise ``ValueError`` when ``n_medium``
        is given or ``"auto"``; with the default they give a
        ``UserWarning`` and the projection stays the vacuum one.
    n_medium:
        Index of the homogeneous, lossless, nonmagnetic half-space the
        surface radiates into (for a closed box, the medium around it). The
        wavenumber becomes ``k = n_medium * 2 pi f / c0`` and the wave
        impedance ``eta = eta0 / n_medium``, in the radiation integral, the
        far-zone fields, the intensity (``FarField.dataset["intensity"]``), the
        radiated power and the directivity. One of:

        * ``None`` (default): index 1, vacuum, bit for bit the projection
          ``n_medium=1.0`` gives. With ``simulation=`` a face sitting in a
          medium of another index (a plane inside a substrate) gives a
          ``UserWarning`` that names the index; the numbers do not change.
        * a real number ``>= 1``, or one per projected frequency (the
          ``freqs_hz`` used): an index, not a permittivity (a silica
          substrate of permittivity 2.1025 is ``n_medium=1.45``). Equal
          values are reported as one float.
        * ``"auto"``: the index sampled from ``simulation`` (required). In
          a dispersive (pole) medium that is ``sqrt(Re eps(f))`` at each
          projected frequency, and :attr:`FarField.n_medium` is then a
          per-frequency array. A face that crosses structures, a medium
          that is anisotropic, custom or PEC, or a lossy one (a loss tangent
          ``Im eps / Re eps`` above 1e-3 at a projected frequency, from
          damped poles or ``conductivity_s_per_m``) raises ``ValueError``.

        A half-space that is not homogeneous (a plane in the cladding over
        a substrate, a box that straddles an interface) is outside the
        model: its far field needs the layered Green's function. The
        readout's errors scale with the cell in medium wavelengths,
        ``n_medium * k0 * dl``.
    yee:
        How the recorded components are read.

        * ``"exact"`` (default): each component at its own Yee point
          (``E_c`` half a cell along ``c``, ``H_c`` half a cell along the
          two other axes), with no averaging. A single plane (or parallel
          planes alone) is read from its first normal sample, J on the H
          plane half a cell above E. The faces of a box are read from their
          two normal samples: the far fields of E brought onto the H plane
          and of H brought onto the E plane are averaged, which cancels the
          imbalance each leaves between the two currents and halves the part
          of the error that depends on where the source sits. A
          second-order error of the interpolation along the normal remains:
          about ``(k dl)^2 / 8`` of the peak in front/back asymmetry; at 25
          cells per wavelength in vacuum on a uniform mesh up to about
          0.80 % with
          the source near a box corner and up to about 1.06 % near a box
          edge, within 2 % wherever the build does not warn. Box faces take
          fourth-order weights at the
          box edges. Refused, naming the face: a box face recorded with
          ``interval_space`` strides (when the cell size is known), a face
          with more than two samples along its normal, and a box whose
          faces do not reach one cell past it in their plane.

          On a graded mesh each sample is placed from the run's grid lines
          (a ``RunResult``'s manifest lists them), a half-offset component
          at the midpoint of its cell; H is brought onto the E plane with
          the two cells' widths as weights, and the box-edge rule runs on
          the samples' true positions (see the module docstring). Every
          number on a uniform mesh is unchanged. The asymmetry can be
          larger than on a uniform mesh: with a region meshed twice as fine
          around a dipole, ending just outside a box face, it reached 1.79 %
          of the peak at 25 cells per wavelength in the background (0.92 %
          at 35). The solver's time step follows the finest cell, so the
          background runs at a lower Courant number and its dispersion
          error grows, as in a dielectric; and the change of cell size just
          outside the face is an inhomogeneity the projection, which
          assumes one homogeneous medium outside the box, does not model.
          Count the cells per wavelength in the background and use at
          least 30, keep every box face at least 0.35 wavelengths from the
          nearest grid line where the cell size changes, along its normal,
          on either side, and keep every source at least 0.1 wavelengths
          from one (a point source along each axis, a launch along its
          normal): end a refinement inside the box with
          that room, or carry it that far past every face, and keep
          sources that far from where it ends. In that range the
          front/back asymmetry of a vacuum dipole box stays within 2 % of
          the peak at 30 background cells per wavelength for refinements 2
          and 3 times as fine as the background (measured up to 0.90 % and
          0.93 %, a dipole near a box edge; higher ratios are not covered),
          falling at about second order for a source fixed in space.
          Outside it no bound is given: a source on or just past the edge
          of a refined region radiates an asymmetric field by itself, up
          to 1.085 % (three times as fine) and 0.92 % (twice) at 30
          background cells. A :class:`~photonhub.FarFieldMonitor` warns
          about both; see the known limitations.

          On a symmetry-reduced run (its manifest, the simulation a
          ``RunResult`` carries, or ``simulation=`` records symmetry planes)
          each face is mirrored at its own Yee points with the sign each
          plane gives the component: a face across a plane must reach it,
          and a box face normal to a plane gains its mirror image, the face
          on the far side, read as the exact mirror of the recorded one.
          Against the full run's box (its far-side faces the mirror of the
          near ones), the pass that brings E onto the H plane agrees to the
          solver's float32 precision. The pass that brings H onto the E
          plane reads the image face at its outer sample, where the full
          run reads its far-side face at the inner one, so it differs by
          the readout's own second-order error, and so does their mean. On
          FDTD dipole boxes at 16 cells per wavelength (three cases, two
          symmetries, uniform and graded meshes) the mean differs by about
          0.1 to 0.4 % of the peak intensity and falls at about second
          order (0.04 to 0.06 % at 32); the H-onto-E pass by 0.26 to
          0.77 %. On
          exact dipole fields the two differ by up to 8.3e-3 and 4.1e-3 at
          20 cells, 9.1e-4 and 4.5e-4 at 40. Neither the reduced nor the
          full run is systematically closer to the exact pattern. A
          mirror-symmetric field gives a pattern mirror-symmetric to
          round-off.

          Where this readout cannot place the samples it falls back to
          ``"legacy"`` with a warning: planes whose cell size cannot be
          told (pass ``dl_um=``), planes of a graded mesh given as plain
          arrays without the run's grid lines (pass the ``RunResult``), and
          box faces recorded with one sample along their normal (a
          ``FutureWarning``: record them one cell thick). Such a fallback
          mirrors a face across a symmetry plane the earlier way (each
          component averaged onto the node, :func:`unfold_symmetry`) but
          cannot add the mirror image of a box face normal to a plane, so a
          symmetry-reduced box that would fall back is refused, naming the
          face and saying what to record. A plain array is
          from a graded mesh when its samples are not evenly spaced, or are
          spaced by other than a whole number of cells; one cut from an
          evenly graded patch cannot be told from a uniform one.
        * ``"legacy"``: the earlier readout, which averages each component
          to the in-plane node (``colocate``) and leaves E and H half a cell
          apart along the normal; it reads the first sample of a face
          recorded one cell thick. On a closed box its front/back asymmetry
          is first order in the cell size (0.28 of the peak at 25 cells per
          wavelength for a vacuum dipole). It mirrors a symmetry-reduced
          run only when ``simulation=`` declares the planes, as before, and
          never adds the mirror image of a box face normal to a plane (such
          a box stays open on the far side). Kept for one release; it gives
          a ``FutureWarning``.
    dl_um:
        The uniform cell size the Yee offsets are measured in. By default it
        is read from a ``RunResult``'s manifest, or from a face's two normal
        samples. A value that disagrees with the manifest, with a face's
        two normal samples, or with a face's in-plane sample spacing (which
        must be a whole number of cells) raises ``ValueError``, and so does
        any value with a graded run (its manifest's grid lines place the
        samples) or with plain arrays sampled unevenly. On a plain single plane with no run
        attached nothing tells the cell size, so a ``dl_um`` that divides
        the in-plane spacing is read as a stride: ``dl_um=dl / 2`` reads a
        plane recorded every cell as one recorded every second cell of
        half the size.
    ux, uy:
        A k-space grid: direction cosines along the two in-plane axes of
        ``plane_axis`` in xyz order (x and y for ``"z"``, x and z for
        ``"y"``, y and z for ``"x"``), meshed like ``theta_rad`` and
        ``phi_rad`` (``mesh``), in the half space ``direction``: the
        direction with those cosines and ``+-sqrt(1 - ux^2 - uy^2)`` along
        ``plane_axis``. They are fractions of the wavenumber in the medium
        projected into, so ``ux = m lambda / (n L)`` is the m-th grating
        order of a period ``L``. A point with ``ux^2 + uy^2 > 1`` has no
        direction (only an evanescent wave has that transverse wavevector,
        and none reaches the far zone): its fields are ``nan`` and its
        intensity 0, as :func:`~photonhub.analysis.diffraction_orders`
        reports evanescent orders. On a plane recorded over one period of a
        periodic structure the power of order ``m`` is ``U (lambda / n)^2 /
        (A cos theta)`` (W), with ``U`` the intensity at its point, ``A`` the
        period's area (m^2) and ``theta`` the angle from the plane's normal.
        To compare with :func:`~photonhub.analysis.diffraction_orders`: its
        ``(k1, k2)`` run along its ``in_plane_axes``, which follow the
        cyclic order (y and z for a plane normal to x, z and x for y, x and y
        for z), so on a plane normal to y its ``k1`` is ``uy`` and its
        ``k2`` is ``ux`` (times ``k``); and its power carries the period's
        area in um^2, 1e12 times the W of the formula. So compared, on an
        FDTD grating the two agree to about 1e-4 (see the docs).
    plane_axis:
        The axis normal to a k-space grid's ``(ux, uy)`` plane or to a
        Cartesian grid's plane. A single plane's own normal by default; a
        surface of several faces needs it.
    direction:
        ``"+"`` or ``"-"``, the half space of a k-space grid along
        ``plane_axis``: by default a single plane's outward side when
        ``plane_axis`` is its normal, else ``"+"``.
    plane_um, x_um, y_um:
        A Cartesian grid: points on the plane at ``plane_um`` along
        ``plane_axis``, at ``x_um`` and ``y_um`` along its two in-plane axes
        in xyz order (meshed like ``theta_rad``), in the frame of
        ``data[name]`` (a fitted run's user frame). In the far-field
        approximation each point reads the far-field amplitude of its
        direction from ``origin_um``, with ``r_um`` its distance; points
        nearer than ``2 D^2 / lambda`` (``D`` twice the surface's farthest
        reach from ``origin_um``, its diagonal from its centre; ``lambda``
        in the medium), where the approximation's phase error passes ``pi
        / 8``, give a warning: pass ``exact=True`` there.
    exact:
        ``False`` (default): the far-field approximation. ``True``: the
        field the surface's equivalent currents radiate at a finite
        distance, with the full Green's function of the homogeneous medium
        (near, intermediate and far terms), at the points of a Cartesian
        grid, or on the sphere of radius ``r_um`` about ``origin_um`` for an
        angle or k-space grid. The :class:`FarField` holds ``r e^{-ikr}``
        times each spherical component of E and H (``e_r``, ``e_theta``,
        ``e_phi``, ``h_r``, ``h_theta``, ``h_phi``), which tend to the
        far-field amplitudes as ``r`` grows, the difference falling as
        ``1/r``; :meth:`FarField.cartesian` gives the field itself, and its
        ``dataset["intensity"]`` is the power per unit solid angle crossing
        a sphere about ``origin_um`` there. It is computed one
        of two ways, picked automatically:

        * the angular spectrum of the plane's tangential E, for a meshed
          Cartesian grid on a plane parallel to one recorded plane sampled
          evenly along both in-plane axes: zero-padded to at least twice
          the span of the plane and the points, the padding doubled until
          doubling it again moves no value by more than 1e-4 of the
          largest, with at most 2^22 padded points per frequency (about
          0.45 GB at that size), and given up early when the moves, shrinking
          at their last rate, cannot reach that within the cap;
        * otherwise (a box, an angle or k-space grid, paired points, an
          unevenly sampled plane, or a plane whose spectrum does not settle
          within that padding, as a field still strong at its edges or read
          far away), a direct sum over the recorded samples with the
          far-field projection's quadrature, in blocks of about 2^18
          (point, sample) pairs. Its cost grows as the number of points
          times the number of samples (about a second of CPU per ten
          million pairs). Near the surface's currents it loses accuracy,
          because its kernel grows as ``1/R^3`` between samples. On a
          dipole's exact field in a closed box, one sample spacing out it
          is off by 3 to 6 % of the peak at 20 cells per wavelength, and
          over the box's edges the band grows in spacings on finer meshes
          (one spacing out up to 11 % at 40 cells and 21 % at 80; the
          readout's own error is reached from two spacings at 20 cells,
          three at 40 and four at 80, from two over a face's interior). A
          point within two sample spacings or a tenth of a wavelength of
          the currents, whichever is farther, gives a warning.

        On exact fields, clear of that band, both read the field to the
        readout's own error: a dipole in a closed box (uniform mesh, 20
        and 40 cells per wavelength) to 7.0e-3 and 1.6e-3 of the peak at
        1.2 to 20 um from the box centre, the far-field projection's error
        at every distance; a beam that has decayed at its plane's edges,
        the two ways within 1e-5 of each other and of the beam. On a plane
        that cuts the field the two ways are different equivalent sources
        (the tangential E alone against J and M) and differ by about the
        cut's own error (6.9e-5 of the peak with 1.4e-3 of it at the
        plane's edge, where the cut alone is 9.3e-5). A target point
        within a box's span, or a Cartesian point not beyond a plane on
        the side it radiates into, is refused; on an angle or k-space grid
        of a plane the directions that end at or behind it (its grazing
        rim) read ``nan``, with intensity 0 and a warning, and a grid with
        no direction in front of it is refused. Needs the default
        Yee-lattice readout.
    r_um:
        The distance (um) of an exact projection on an angle or k-space
        grid, from ``origin_um``.
    origin_um:
        The point (um, the frame of ``data[name]``) distances are measured
        from: the centre of an exact projection's sphere and of a Cartesian
        grid's far-field approximation, by default the centre of the
        surface's recorded extent (on a mirrored axis the symmetry plane:
        the full surface's centre). Runs that record the same surface on
        different grids, or with and without symmetry planes, can default
        to origins a fraction of a cell apart, which moves the phase
        ``e^{-ikr}`` of the exact amplitudes: pass ``origin_um`` to compare
        runs. Given with far-field amplitudes on an angle or k-space grid,
        it is also their phase reference (``F e^{i k r_hat . o}``); without
        it they keep the phase they always had, referred to the run's
        corner frame.

    Returns
    -------
    FarField
        Its :attr:`~FarField.dataset` labels the projection by frequency and
        the grid's own axes (``(f, theta, phi)``, ``(f, ux, uy)``, the
        plane's two axes, or ``(f, point)``), with the ``n_medium``
        projected into; its methods give the gain, cross section,
        polarization bases and the field at a distance. See
        :class:`FarField`.

    Examples
    --------
    A plane recorded inside a silica substrate (index 1.45) below the
    device, projected into the substrate within 20 degrees of ``-z``::

        theta = np.radians(np.linspace(160.0, 180.0, 41))
        phi = np.radians(np.arange(0.0, 360.0, 10.0))
        ff = far_field(data, "substrate_plane", theta_rad=theta, phi_rad=phi,
                       axis="z", sign=-1.0, n_medium=1.45)
        u_sub = ff.dataset["intensity"].isel(f=0)    # (theta, phi), W/sr

    The same plane on a k-space grid, and a ``FarFieldMonitor`` box's
    exact field on a plane 20 um above its centre::

        u = np.linspace(-1.0, 1.0, 201)
        ks = far_field(data, "substrate_plane", ux=u, uy=u, axis="z", sign=-1.0,
                       n_medium=1.45)              # ks.dataset["intensity"]
        xs = np.linspace(-10.0, 10.0, 81)
        spot = far_field(data, "ff", plane_axis="z", plane_um=20.0, x_um=xs,
                         y_um=xs, exact=True)
        ex = spot.cartesian()["Ex"].isel(f=0)        # (x, y), V/m
    """
    if yee not in _YEE_PUBLIC:
        raise ValueError(f"yee must be one of {_YEE_PUBLIC}, got {yee!r}")
    result = _far_field_impl(
        data, monitor_name, theta_rad=theta_rad, phi_rad=phi_rad, axis=axis,
        sign=sign, freqs_hz=freqs_hz, mesh=mesh, faces=faces, colocate=colocate,
        simulation=simulation, n_medium=n_medium, yee=yee, yee_plane="avg",
        dl_um=dl_um, ux=ux, uy=uy, plane_axis=plane_axis, direction=direction,
        plane_um=plane_um, x_um=x_um, y_um=y_um, exact=exact, r_um=r_um,
        origin_um=origin_um)
    # A run driven by one TF/SF box: the plane wave cross_section() divides
    # by; and the point the amplitudes' phase refers to (renormalize()).
    from ._far_field_results import incident_amplitude
    phase_origin = (result.origin_um if result.origin_um is not None
                    else tuple(float(v) for v in _frame_offset(data)))
    return _evolve(result, incident_e_v_per_m=incident_amplitude(data),
                   phase_origin_um=phase_origin)


@legacy_keywords(theta="theta_rad", phi="phi_rad")
def _far_field_impl(
    data,
    monitor_name: str,
    *,
    theta_rad: Union[None, float, Sequence[float], np.ndarray] = None,
    phi_rad: Union[float, Sequence[float], np.ndarray] = 0.0,
    axis: Optional[Axis] = None,
    sign: float = 1.0,
    freqs_hz: Optional[Sequence[float]] = None,
    mesh: bool = True,
    faces: Optional[Sequence[Tuple[str, str, float]]] = None,
    colocate: bool = True,
    simulation=None,
    n_medium: Union[None, str, float, Sequence[float], np.ndarray] = None,
    yee: str = "exact",
    yee_plane: str = "avg",
    dl_um: Optional[float] = None,
    ux=None,
    uy=None,
    plane_axis: Optional[str] = None,
    direction: Optional[str] = None,
    plane_um: Optional[float] = None,
    x_um=None,
    y_um=None,
    exact: bool = False,
    r_um: Optional[float] = None,
    origin_um: Optional[Sequence[float]] = None,
    exact_path: str = "auto",
) -> FarField:
    """:func:`far_field` with the diagnostic readouts as well (not public,
    may change): ``yee="positions"`` reads every face from one sample at the
    true points; ``yee="surface"`` brings E and H onto one surface with the
    in-plane handling of ``colocate``; both refuse graded meshes and
    symmetry-reduced runs. ``yee_plane`` picks the surface a box face is
    read on: ``"h"`` (E averaged onto the H plane half a cell above the
    first sample), ``"e"`` (H onto the E plane of the second sample),
    ``"mid"`` (both onto the plane three quarters of a cell above the first
    sample) or ``"avg"`` (the mean of the ``"h"`` and ``"e"`` far fields,
    :func:`far_field`'s readout). ``exact_path`` picks how ``exact=True``
    is computed: ``"auto"`` (:func:`far_field`'s rule), ``"direct"`` (the
    direct sum always) or ``"spectral"`` (the angular spectrum, refused
    where it does not apply)."""
    grid_args = dict(theta_rad=theta_rad, phi_rad=phi_rad, ux=ux, uy=uy,
                     plane_axis=plane_axis, direction=direction, plane_um=plane_um,
                     x_um=x_um, y_um=y_um, exact=exact, r_um=r_um, origin_um=origin_um,
                     mesh=mesh)
    _check_grid_args(**grid_args)
    if exact_path not in ("auto", "direct", "spectral"):
        raise ValueError(f"exact_path must be 'auto', 'direct' or 'spectral', got {exact_path!r}")
    # A FarFieldMonitor's name: its faces, the medium they sit in, and what
    # its recipe says of symmetry planes, faces left out and a window.
    surface_n = None
    surface = None
    if faces is None:
        surface = _far_field_surface(data, monitor_name, simulation)
        if surface is not None:
            faces, surface_n = surface.faces, surface.index
    # Assemble the list of (DataArray, axis, sign) faces.
    if faces is not None:
        face_specs = [(_frame.wire_array(data, name), ax, float(sg)) for name, ax, sg in faces]
        face_names = [str(name) for name, _, _ in faces]
    else:
        da = _frame.wire_array(data, monitor_name)
        if axis is None:
            axis = _infer_normal_axis(da)
        face_specs = [(da, axis, float(sign))]
        face_names = [str(monitor_name)]
    for _, ax, _ in face_specs:
        if ax not in _TRANSVERSE:
            raise ValueError(f"axis must be one of x/y/z, got {ax!r}")
    if yee not in _YEE_MODES:
        raise ValueError(f"yee must be one of {_YEE_MODES}, got {yee!r}")
    if yee_plane not in _YEE_PLANES:
        raise ValueError(f"yee_plane must be one of {_YEE_PLANES}, got {yee_plane!r}")
    # A plane has one sample along its normal, a box face one or two (one
    # cell thick); a volume handed over as a plane is refused, as it always
    # was, not read from its first slice.
    for (da, ax, sg), name in zip(face_specs, face_names):
        n_normal = _normal_samples(da, ax)
        if n_normal > 2:
            raise ValueError(
                f"far_field: {_where(name, ax, sg)} has {n_normal} samples along "
                "its normal; a plane needs one, a box face one or two (recorded "
                f"one cell thick). Select them with .isel({ax}=...)")

    # A closed-surface projection needs a surface that is actually CLOSED in
    # 3-D. On a quasi-2-D scene (a 1-cell plain-periodic axis) the invariant
    # axis is periodic, not terminated, so a "box" of four lateral faces is an
    # open tube: the 3-D radiation integral below is then the wrong transform
    # and returns a silently wrong pattern -- measured 13.4% anisotropy and a
    # 0.134 front/back asymmetry for an ISOTROPIC 2-D point dipole, where the
    # 2-D Green's-function transform gives 2.1% and exactly 0. Refuse it
    # rather than answer
    # wrongly; a genuine 2-D near-to-far needs the 2-D kernel, which this
    # plugin does not implement.
    if len(face_specs) > 1:
        axes_present = {ax for _, ax, _ in face_specs}
        for da, ax, _ in face_specs:
            t1, t2 = _TRANSVERSE[ax]
            for letter in (t1, t2):
                if (letter in getattr(da, "sizes", {})
                        and da.sizes[letter] == 1
                        and letter not in axes_present):
                    raise ValueError(
                        "closed-surface far_field needs a surface closed in "
                        f"3-D, but axis {letter!r} is one cell deep and no "
                        "face is normal to it (a quasi-2-D scene). The 3-D "
                        "radiation integral is the wrong transform there; a "
                        "2-D near-to-far needs the 2-D Green's function, "
                        "which this plugin does not implement. Use a full 3-D "
                        "scene, or project a single plane instead of a box.")

    readout = _resolve_readout(data, face_specs, face_names, yee=yee,
                               colocate=colocate, simulation=simulation,
                               dl_um=dl_um,
                               recipe_symmetry=surface.symmetry if surface else None,
                               left_out=surface.left_out if surface else (),
                               window=surface.window if surface else None)
    if readout.legacy:
        # The earlier readout takes one sample along each face's normal: the
        # first, on a face recorded one cell thick.
        face_specs = [(da.isel({ax: [0]}) if _normal_samples(da, ax) > 1 else da, ax, sg)
                      for da, ax, sg in face_specs]

    # The recorded faces as the engine wrote them (wire frame, the modelled
    # part only): where the medium is sampled.
    recorded_specs = list(face_specs)

    # A symmetry-reduced run recorded only its fraction of the plane. On the
    # earlier readout, mirror it back before integrating; that step
    # co-locates, so it takes over from `colocate` (applying both would
    # average the same half cell twice). yee="legacy" mirrors what
    # simulation= declares, as it always did; a fallback from the default
    # mirrors the run's own symmetry planes. The Yee-lattice readout mirrors
    # each component at its own points (_unfold_face).
    if readout.legacy and readout.symmetry:
        planes = SimpleNamespace(symmetry=readout.symmetry)
        face_specs = [(unfold_symmetry(fd, planes), ax, sg)
                      for fd, ax, sg in face_specs]
        colocate = False

    # Frequencies: default to the monitor's own.
    ref_da = face_specs[0][0]
    if freqs_hz is None:
        if "f" in getattr(ref_da, "dims", ()):
            freqs = [float(f) for f in np.asarray(ref_da.coords["f"].values)]
        else:
            freqs = [0.0]
    else:
        freqs = [float(f) for f in np.atleast_1d(freqs_hz)]

    # A FarFieldMonitor's surface radiates into the medium its build sampled
    # at its faces, unless n_medium says otherwise; index 1 stays the
    # default (n_medium=None), bit for bit.
    if n_medium is None and surface_n is not None:
        n_medium = _surface_index(surface_n, freqs)

    # The medium radiated into: index per projected frequency, and what the
    # FarField reports (index 1 by default, the vacuum projection).
    n_f, n_report = _projection_index(n_medium, simulation, recorded_specs,
                                      np.asarray(freqs, dtype=np.float64))

    # The target grid: directions, or points at a finite distance.
    targets = _resolve_targets(data, recorded_specs, readout, **grid_args)
    if exact:
        if readout.legacy:
            raise ValueError(
                "far_field(exact=True) reads the faces with the default Yee-lattice "
                "readout, but this projection uses the earlier one (yee='legacy', or a "
                "fallback the warning above names); record the faces as it says, or "
                "project without exact")
        return _exact_projection(face_specs, face_names, freqs, n_f, n_report, readout,
                                 yee_plane, targets, exact_path)
    if targets.grid == "cartesian":
        _warn_near_zone(targets, freqs, n_f)
    if np.all(targets.live):
        th, ph = targets.theta, targets.phi
    else:
        th, ph = targets.theta[targets.live], targets.phi[targets.live]
    rx = np.sin(th) * np.cos(ph)
    ry = np.sin(th) * np.sin(ph)
    rz = np.cos(th)

    n_freq = len(freqs)
    n_dir = th.size
    e_theta = np.zeros((n_freq, n_dir), dtype=np.complex128)
    e_phi = np.zeros((n_freq, n_dir), dtype=np.complex128)

    cos_t, sin_t = np.cos(th), np.sin(th)
    cos_p, sin_p = np.cos(ph), np.sin(ph)

    # Closed-box tiling (F11), earlier readout: when both faces of an axis
    # are present, their normal planes bound the box on that axis; each
    # face's in-plane quadrature is CLIPPED to those bounds (in meters) so
    # the six faces tile the surface exactly once — no per-face half-cell
    # overhang, no edge double-count. An axis with fewer than two faces (an
    # open surface) stays unclipped. The Yee-lattice readout clips to the
    # surfaces its currents sit on (_yee_radiation_vectors).
    bounds_m: Dict[str, Tuple[float, float]] = {}
    if len(face_specs) > 1:
        by_axis: Dict[str, list] = {}
        for da, ax, sg in face_specs:
            nc = np.asarray(da.coords[ax].values, dtype=np.float64).ravel()
            if nc.size >= 1:
                by_axis.setdefault(ax, []).append(float(nc[0]) * 1e-6)
        for ax, positions in by_axis.items():
            if len(positions) >= 2:
                bounds_m[ax] = (min(positions), max(positions))

    for fi, f in enumerate(freqs):
        # Wavenumber and wave impedance of the medium radiated into. At index
        # 1 both are the vacuum values exactly (1.0 * x and x / 1.0 are exact
        # in IEEE arithmetic), so the default projection is the n_medium=1.0
        # one bit for bit.
        n_idx = n_f[fi]
        k = n_idx * (2.0 * np.pi * f / C0)  # wavenumber in the medium (1/m)
        eta = ETA0 / n_idx                   # wave impedance of the medium
        Ntot = [np.zeros(n_dir, dtype=np.complex128) for _ in range(6)]
        if not readout.legacy:
            Ntot = _yee_radiation_vectors(
                face_specs, f, k=k, rx=rx, ry=ry, rz=rz, readout=readout,
                plane=yee_plane, names=face_names)
        else:
            for da, ax, sg in face_specs:
                currents = equivalent_currents(
                    _as_component_mapping(da, ax), axis=ax, freq_hz=f,
                    sign=sg, colocate=colocate)
                t1, t2 = _TRANSVERSE[ax]
                contribs = _project_one_face(
                    currents, k=k, rx=rx, ry=ry, rz=rz,
                    clip1=bounds_m.get(t1), clip2=bounds_m.get(t2))
                for i in range(6):
                    Ntot[i] = Ntot[i] + contribs[i]
        Nx, Ny, Nz, Lx, Ly, Lz = Ntot

        # Spherical projection of the radiation vectors.
        N_theta = Nx * cos_t * cos_p + Ny * cos_t * sin_p - Nz * sin_t
        N_phi = -Nx * sin_p + Ny * cos_p
        L_theta = Lx * cos_t * cos_p + Ly * cos_t * sin_p - Lz * sin_t
        L_phi = -Lx * sin_p + Ly * cos_p

        # Far-zone E (spreading factor e^{ikr}/r dropped -> far-field amplitude).
        # Prefactor i k / (4 pi); e^{-i omega t} convention. Balanis derives these
        # with e^{+j omega t} / e^{-j k r}; converting to e^{-i omega t} is j -> -i,
        # which flips BOTH the exponential (e^{-jkr} -> e^{+ikr}) AND the leading
        # jk -> -ik. Keeping Balanis' outer signs alongside e^{+ikr} left a global
        # -1 (a constant pi absolute-phase error). On-axis check for a Huygens
        # patch (Ex=g, Hy=g/eta0): these give F_theta = -(ik/2pi) Int g dS, matching
        # the Rayleigh-Sommerfeld reference F = -ik/(2pi) Int E dS in this same
        # convention. In a medium, k and eta are the medium's (Balanis 12-10
        # holds in any homogeneous medium with its own k and eta).
        pref = 1j * k / (4.0 * np.pi)
        e_theta[fi] = pref * (L_phi + eta * N_theta)
        e_phi[fi] = -pref * (L_theta - eta * N_phi)

    return _far_result(targets, e_theta, e_phi, freqs, n_f, n_report)


# --- FarFieldMonitor surfaces ---------------------------------------------------

#: The face monitors a ``FarFieldMonitor`` records: ``<name>.<axis>_<side>``.
_FACE_SUFFIX = {"x_minus": ("x", -1.0), "x_plus": ("x", 1.0), "y_minus": ("y", -1.0),
                "y_plus": ("y", 1.0), "z_minus": ("z", -1.0), "z_plus": ("z", 1.0)}


@dataclass(frozen=True)
class _Surface:
    """A ``FarFieldMonitor``'s surface as ``far_field`` reads it: its faces
    ``(monitor name, axis, outward sign)``, ``{frequency: index}`` of the
    medium they sit in (None: not known), and from its recipe the symmetry
    planes it was compiled with, the faces an open box left out
    ``(axis, sign, declared plane um)`` and a plane's edge window
    ``((axis, fraction, lo_um, hi_um), ...)``."""

    faces: list
    index: Optional[Dict[float, float]]
    symmetry: Optional[Tuple[int, int, int]] = None
    left_out: Tuple = ()
    window: Optional[Tuple] = None


def _recipes_of(holder) -> Mapping:
    """The far-field recipes a simulation holds (``{}`` for none)."""
    recipes = getattr(holder, "_far_field", None) if holder is not None else None
    return recipes if isinstance(recipes, Mapping) else {}


def _far_field_surface(data, name, simulation) -> Optional[_Surface]:
    """The :class:`_Surface` of the ``FarFieldMonitor`` ``name``, or ``None``
    when ``name`` is a recorded monitor or names no surface.

    The recipe comes from the run (a ``RunResult``'s simulation keeps it, and
    a reloaded result reads it back from ``client.json``) or from
    ``simulation=``. Without one, the face monitors are found by their
    names, ``<name>.x_minus`` ... ``<name>.z_plus``, with a warning: the
    medium the faces sit in is then not known, nor is where an open box was
    cut or a plane's window (the faces are read as recorded)."""
    if not isinstance(name, str) or not name:
        return None
    try:
        if name in data:
            return None
    except TypeError:
        return None
    rec = None
    for holder in (simulation, None):
        if holder is None:
            try:
                holder = getattr(data, "simulation", None)
            except (AttributeError, OSError, ValueError):
                holder = None
        rec = _recipes_of(holder).get(name)
        if rec is not None:
            break
    if rec is not None:
        faces = [(str(f[0]), str(f[1]), float(f[2])) for f in rec["faces"]]
        missing = [f[0] for f in faces if f[0] not in data]
        if missing:
            raise KeyError(
                f"far_field: the far-field surface {name!r} records the faces {missing}, "
                "which are not in this result")
        index = {float(f): float(n) for f, n in zip(rec["freqs_hz"], rec["n_medium"])}
        sym = rec.get("symmetry")
        return _Surface(
            faces=faces, index=index,
            symmetry=None if sym is None else tuple(int(v) for v in sym),
            left_out=tuple((str(ax), float(sg), float(plane_um))
                           for ax, sg, plane_um in (rec.get("open") or ())),
            window=None if rec.get("window") is None else tuple(
                (str(a), float(f), float(lo), float(hi)) for a, f, lo, hi in rec["window"]))
    try:
        keys = list(data.keys())
    except (AttributeError, TypeError):
        return None
    prefix = f"{name}."
    faces = sorted((_FACE_SUFFIX[k[len(prefix):]][0], _FACE_SUFFIX[k[len(prefix):]][1], k)
                   for k in keys if isinstance(k, str) and k.startswith(prefix)
                   and k[len(prefix):] in _FACE_SUFFIX)
    if not faces:
        return None
    found = [(k, ax, sg) for ax, sg, k in faces]
    _warn(
        f"the far-field surface {name!r} has no recipe here (the result's client.json "
        "is missing, or this is not the result of its run), so its faces were found by "
        f"their names, {[k for k, _, _ in found]}, and the medium they sit in is not known: "
        "the projection is into vacuum unless n_medium says otherwise. Faces an open box "
        "left out and a plane's edge window are not known either: the faces are read as "
        "recorded")
    return _Surface(faces=found, index=None)


def _surface_index(by_freq: Mapping, freqs) -> Optional[Union[float, list]]:
    """The index a surface's recipe gives each projected frequency (the
    recorded frequency nearest it): ``None`` for vacuum throughout (the
    default projection, bit for bit), one float when it does not change."""
    rec_f = np.asarray(list(by_freq), dtype=np.float64)
    rec_n = np.asarray(list(by_freq.values()), dtype=np.float64)
    n = [float(rec_n[int(np.argmin(np.abs(rec_f - float(f))))]) for f in freqs]
    if all(v == 1.0 for v in n):
        return None
    return n[0] if all(v == n[0] for v in n) else n


# --- the medium radiated into ------------------------------------------------

#: Largest loss tangent ``Im eps / Re eps`` the projection accepts as lossless.
#: The radiation integral uses the real ``k`` and ``eta``; a loss tangent
#: ``t`` gives ``Im n = n t / 2`` to first order, so the field it ignores
#: decays by ``exp(-pi t)`` per medium wavelength of travel: at 1e-3, 0.3 %
#: in field amplitude and 0.6 % in power. As an estimate, the pattern error
#: then grows as about ``pi t`` times the source-to-surface path in medium
#: wavelengths (that distance over ``cos(theta)`` at a polar angle theta, or
#: the surface's size, whichever is larger): small a few wavelengths out, a
#: few percent for a wide plane or a surface 10 wavelengths away read at 60
#: degrees. A pole medium far from its resonance (linewidth 1e-3 of the
#: resonance) sits near 1e-4; a damped pole near its resonance is far above.
_LOSS_TANGENT_MAX: float = 1e-3

#: Relative index difference under which a sampled medium counts as the one
#: projected into: an index given to four significant digits (1.449 for
#: ``sqrt(2.1) = 1.44914``) must not warn, nor one a few parts in 1e4 off.
_INDEX_RTOL: float = 1e-3


def _explicit_index(n_medium, freqs: np.ndarray
                    ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
    """``(n per frequency, the value to report)`` for an index the caller
    gave: a real scalar, or one value per projected frequency, each finite
    and ``>= 1``. Equal values are reported as one float."""
    if isinstance(n_medium, (bool, np.bool_)) or np.iscomplexobj(n_medium):
        raise ValueError(
            f"n_medium must be a real index >= 1, got {n_medium!r}; the "
            "projection is for a lossless medium (a lossy one attenuates the "
            "field before it reaches the far zone)")
    try:
        arr = np.asarray(n_medium, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"n_medium must be None, 'auto', a real index >= 1 or one index "
            f"per projected frequency, got {n_medium!r}") from None
    if arr.ndim == 0:
        n_f = np.full(freqs.shape, float(arr))
    elif arr.shape == freqs.shape:
        n_f = arr.copy()
    else:
        raise ValueError(
            f"n_medium must be a scalar or one value per projected frequency "
            f"({freqs.size}), got shape {arr.shape}")
    if not (np.all(np.isfinite(n_f)) and np.all(n_f >= 1.0)):
        raise ValueError(
            f"n_medium must be a finite real index >= 1, got {n_medium!r}")
    return n_f, _report(n_f)


def _report(n_f: np.ndarray) -> Union[float, np.ndarray]:
    """A float when the index is one value at every frequency, else the
    per-frequency array (a dispersive medium), as ``diffraction_orders``
    reports it."""
    if n_f.size and np.all(n_f == n_f[0]):
        return float(n_f[0])
    return n_f.copy()


def _fmt(n) -> str:
    """An index for a message: ``1.45``, or ``1.44..1.46`` across frequencies."""
    arr = np.atleast_1d(np.asarray(n, dtype=np.float64))
    if arr.size and np.all(arr == arr[0]):
        return f"{float(arr[0]):.6g}"
    return f"{float(arr.min()):.6g}..{float(arr.max()):.6g} (per frequency)"


def _is_simulation(simulation) -> bool:
    from ..components.simulation import Simulation
    return isinstance(simulation, Simulation)


def _complex_eps(m, freq_hz: float) -> complex:
    """``eps(omega)`` of a structure medium under the ``e^{-i omega t}``
    convention (``Im eps > 0`` is loss): eps_inf, every Lorentz and Drude
    term (the forms of ``Medium.permittivity_at_hz``, imaginary parts kept)
    and the conductivity ``i sigma / (omega eps0)``."""
    w = 2.0 * np.pi * float(freq_hz)
    eps = complex(float(m.permittivity))
    for p in m.all_lorentz_poles():
        w0 = 2.0 * np.pi * p.resonance_frequency_hz
        g = 2.0 * np.pi * p.linewidth_hz
        eps += p.delta_eps * w0 * w0 / (w0 * w0 - w * w - 1j * g * w)
    for d in (m.drude or ()):
        wp = 2.0 * np.pi * d.plasma_frequency_hz
        g = 2.0 * np.pi * d.linewidth_hz
        eps += -(wp * wp) / (w * w + 1j * g * w)
    eps += 1j * float(m.conductivity_s_per_m) / (w * EPS0)
    return eps


def _loss_tangent(m, freqs: np.ndarray) -> float:
    """The largest ``Im eps / Re eps`` of medium ``m`` over the projected
    frequencies (0 for a lossless medium; frequencies <= 0 are skipped)."""
    worst = 0.0
    for f in np.atleast_1d(freqs):
        if not f > 0.0:
            continue
        eps = _complex_eps(m, float(f))
        if eps.real > 0.0:
            worst = max(worst, eps.imag / eps.real)
    return worst


def _face_medium(simulation, da: xr.DataArray, axis: str, sign: float,
                 freqs: np.ndarray):
    """Sample the medium one face radiates into.

    Returns ``(n per frequency or None, problem or None, where, loss or
    None)``: ``problem`` says why no index could be read, ``loss`` that the
    medium read is lossy beyond :data:`_LOSS_TANGENT_MAX`. Each recorded
    sample holds the tangential E at its node coordinate and the tangential
    H half a cell above; with ``a`` the outermost sample on the radiating
    side (a face recorded one cell thick has two), the probe sits half a
    cell beyond that slab: ``a + d`` for an outward normal along ``+axis``,
    ``a - d/2`` along ``-axis`` (``d`` the local cell). There the
    face's own extent is sampled at the centres between its recorded nodes
    with ``diffraction._infer_n_medium``: it must be one medium; a pole
    medium gives ``sqrt(Re eps(f))``. ``da`` is the recorded (wire-frame,
    not yet unfolded) array, so the points lie inside the domain."""
    from ..components.grid import snap_mixed_plane
    from ..viz import _geometry as _geom
    from .diffraction import _infer_n_medium, _plane_media

    if axis not in getattr(da, "coords", {}):
        return (None, (f"the recorded plane carries no {axis!r} coordinate, "
                       "so its medium cannot be located"),
                f"the plane normal to {axis}", None)
    nc = np.asarray(da.coords[axis].values, dtype=np.float64).ravel()
    # the outermost sample on the radiating side (a face recorded one cell
    # thick has two)
    pos = (float(nc.max() if sign > 0 else nc.min())) if nc.size else 0.0
    _, d = snap_mixed_plane(simulation, "xyz".index(axis), pos)
    d = float(d) if d else 0.0
    probe = pos + 0.25 * d + (0.75 * d if sign > 0 else -0.75 * d)
    where = (f"the plane at {axis} = {pos:.6g} um (radiating toward "
             f"{'+' if sign > 0 else '-'}{axis}, sampled at {axis} = "
             f"{probe:.6g} um)")
    # The face's own in-plane samples: the centres between its nodes (its
    # cells), or the single coordinate of a one-cell axis.
    axes_c = []
    for letter in _geom.in_plane_axes(axis):
        if letter not in getattr(da, "coords", {}):
            return (None, (f"the recorded plane carries no {letter!r} "
                           "coordinate, so its medium cannot be located"),
                    where, None)
        c = np.unique(np.asarray(da.coords[letter].values, dtype=np.float64))
        axes_c.append(0.5 * (c[:-1] + c[1:]) if c.size >= 2 else c)
    points = tuple(np.meshgrid(axes_c[0], axes_c[1]))
    try:
        n_s, _ = _infer_n_medium(simulation, axis, probe, freqs,
                                 wording="far_field", points=points)
    except ValueError as exc:
        return None, str(exc), where, None
    media = []
    if any(_unsupported_medium(s.medium) or s.medium.is_dispersive
           or s.medium.conductivity_s_per_m for s in simulation.structures):
        media = [m for m in _plane_media(simulation, axis, probe, points)
                 if m is not None]
    for m in media:
        why = _unsupported_medium(m)
        if why:
            return (None, (f"the medium there is {why}, and a far-field "
                           "projection needs an isotropic dielectric"),
                    where, None)
    if not np.all(n_s >= 1.0):
        return (None, (f"the medium there has index {_fmt(n_s)} < 1, and "
                       "far_field projects into a dielectric of index >= 1"),
                where, None)
    loss = None
    tan_d = max((_loss_tangent(m, freqs) for m in media), default=0.0)
    if tan_d > _LOSS_TANGENT_MAX:
        loss = (f"the medium there is lossy (Im eps / Re eps up to {tan_d:.3g} "
                f"at the projected frequencies, above {_LOSS_TANGENT_MAX:g}), "
                "and the projection ignores the absorption on the way to the "
                "far zone")
    return n_s, None, where, loss


def _unsupported_medium(m) -> Optional[str]:
    """Why a structure medium cannot host the projection, or ``None``. Loss
    is judged separately (:func:`_loss_tangent`)."""
    if getattr(m, "pec", None):
        return "a PEC"
    if getattr(m, "permittivity_xyz", None) is not None:
        return "anisotropic (permittivity_xyz)"
    if getattr(m, "permittivity_data", None) is not None:
        return "a custom (permittivity_data) medium"
    return None


def _warn(message: str) -> None:
    warnings.warn(f"far_field: {message}", UserWarning,
                  stacklevel=caller_stacklevel())


def _projection_index(n_medium, simulation, face_specs, freqs: np.ndarray
                      ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
    """``(n per projected frequency, the value FarField reports)``.

    ``None`` is index 1; a number or per-frequency sequence is taken as
    given; ``"auto"`` is sampled from ``simulation``. With a
    :class:`~photonhub.Simulation` the medium at every face is sampled
    (:func:`_face_medium`). ``"auto"`` raises ``ValueError`` when it cannot
    read one lossless medium. A closed box whose faces are not all in one
    homogeneous medium raises with an explicit index or ``"auto"`` and only
    warns with the default, which keeps the earlier numbers. Otherwise a
    medium that is lossy, not homogeneous, or of another index than the one
    projected into gives a ``UserWarning``."""
    auto = isinstance(n_medium, str)
    if auto and n_medium != "auto":
        raise ValueError(
            f"n_medium must be None, 'auto', a real index >= 1 or one index "
            f"per projected frequency, got {n_medium!r}")
    sampled = None
    if _is_simulation(simulation):
        sampled = [_face_medium(simulation, da, ax, sg, freqs)
                   for da, ax, sg in face_specs]
    elif auto:
        raise ValueError(
            "n_medium='auto' reads the medium from the scene: pass "
            "simulation= (the Simulation the data came from), or give the "
            "index, e.g. n_medium=1.45")

    if n_medium is None:
        n_f, report = np.ones(freqs.shape), 1.0
        given = ("n_medium was not given, so the projection assumes index 1 "
                 "(vacuum)")
    elif not auto:
        n_f, report = _explicit_index(n_medium, freqs)
        given = f"the projection uses n_medium={_fmt(n_f)}"
    if sampled is None:
        return n_f, report

    if len(sampled) > 1:
        # A closed box: the equivalence currents radiate through ONE
        # medium's Green's function, so every face must sit in it.
        bad = [f"{w}: {p}" for _, p, w, _ in sampled if p is not None]
        box_problem = None
        if bad:
            box_problem = ("closed-surface far_field needs every face in one "
                           "homogeneous medium, but " + "; ".join(bad))
        elif any(not np.allclose(n_s, sampled[0][0], rtol=1e-9, atol=0.0)
                 for n_s, _, _, _ in sampled[1:]):
            listing = "; ".join(f"{w}: index {_fmt(n_s)}"
                                for n_s, _, w, _ in sampled)
            box_problem = (
                "closed-surface far_field needs every face in one homogeneous "
                f"medium, but the faces sit in different media ({listing}). A "
                "box that straddles an interface radiates into a layered "
                "space, which needs the layered Green's function; enclose the "
                "source in one medium")
        if box_problem is not None:
            if n_medium is None:
                # The default keeps the earlier numbers: say so, do not refuse.
                _warn(f"{box_problem}. {given}, as before.")
                return n_f, report
            raise ValueError(box_problem)

    # A single plane, or a box in one medium: its first face speaks for all.
    n_s, problem, where, loss = sampled[0]
    if len(sampled) > 1:
        where = f"the closed box ({len(sampled)} faces)"
        loss = next((x[3] for x in sampled if x[3] is not None), None)
    if auto:
        if problem is not None or loss is not None:
            raise ValueError(
                f"n_medium='auto' could not read one lossless medium: {where}: "
                f"{problem if problem is not None else loss}. Move the surface "
                "into one homogeneous, lossless medium, or pass n_medium "
                "explicitly")
        return n_s.copy(), _report(n_s)

    if problem is not None:
        _warn(f"{where}: {problem}; {given}.")
        return n_f, report
    # One warning per call: an index mismatch, with the loss appended when
    # the medium is lossy too, or the loss alone.
    if not np.allclose(n_s, n_f, rtol=_INDEX_RTOL, atol=0.0):
        hint = (f"n_medium={_fmt(n_s)} or n_medium='auto'"
                if n_medium is None else f"n_medium={_fmt(n_s)}")
        message = (f"{where} radiates into a medium of index {_fmt(n_s)}, but "
                   f"{given}. Pass {hint} to project into that medium "
                   "(n_medium is an index, not a permittivity).")
        if loss is not None:
            message += f" Besides, {loss}."
        _warn(message)
    elif loss is not None:
        _warn(f"{where}: {loss}; {given}.")
    return n_f, report


def _infer_normal_axis(da: xr.DataArray) -> str:
    """Infer the plane normal from a DFT field DataArray: the spatial axis whose
    coordinate has length 1 (the singleton, swept-out normal)."""
    singles = [ax for ax in ("x", "y", "z")
               if ax in da.dims and da.sizes.get(ax, 0) == 1]
    if len(singles) == 1:
        return singles[0]
    raise ValueError(
        "could not infer the plane normal axis from the monitor (expected "
        f"exactly one singleton spatial dim; dims={tuple(da.dims)}, "
        f"sizes={{ {', '.join(f'{a}:{da.sizes.get(a)}' for a in ('x','y','z') if a in da.dims)} }}). "
        "Pass axis= explicitly.")


def _as_component_mapping(da: xr.DataArray, axis: str) -> Dict[str, xr.DataArray]:
    """Slice the four tangential components for ``axis`` out of a DFT field
    DataArray with a ``component`` dim, into the ``{name: DataArray}`` mapping
    :func:`equivalent_currents` expects. If ``da`` has no ``component`` dim it is
    assumed already pre-sliced and returned per-name unchanged."""
    names = _TANGENTIAL[axis]
    if "component" in getattr(da, "dims", ()):
        return {n: da.sel(component=n) for n in names}
    return {n: da for n in names}


# --- Target grids and the exact projection --------------------------------
#
# far_field projects onto one of three target grids: directions (theta_rad,
# phi_rad), direction cosines (ux, uy) about an axis, or points on a plane
# (plane_axis, plane_um, x_um, y_um). In the far-field approximation every
# grid reads the same amplitudes F(r_hat); exact=True evaluates the field
# the surface's currents radiate at a finite distance with the full Green's
# function of the homogeneous medium, by a direct sum over the samples (the
# same quadrature as the far-field kernel), or, for points on a plane
# parallel to one recorded plane, by the plane's angular spectrum.

#: The two in-plane axes of a target plane normal to each axis, in xyz order:
#: the axes ``ux``/``uy`` and ``x_um``/``y_um`` run along.
_NATURAL: Dict[str, Tuple[str, str]] = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}

#: Point pairs (target, surface sample) per block of the direct sum: about
#: 20 complex arrays of this size are live at once (some 80 MB).
_EXACT_CHUNK: int = 1 << 18

#: Largest padded spectrum (points per frequency) the angular-spectrum path
#: takes before it hands the plane to the direct sum.
_SPECTRAL_MAX_POINTS: int = 1 << 22

#: The angular-spectrum path doubles its zero padding until doubling it once
#: more moves no target value by more than this fraction of the largest.
_SPECTRAL_RTOL: float = 1e-4


def _check_grid_args(*, theta_rad, phi_rad, ux, uy, plane_axis, direction, plane_um,
                     x_um, y_um, exact, r_um, origin_um, mesh) -> None:
    """The target-grid arguments of :func:`far_field`, checked on their own
    (before any data is read)."""
    angle = theta_rad is not None
    kspace = ux is not None or uy is not None
    cart = plane_um is not None or x_um is not None or y_um is not None
    given = [name for name, on in (("theta_rad", angle), ("ux/uy", kspace),
                                   ("plane_um/x_um/y_um", cart)) if on]
    if not given:
        raise ValueError(
            "far_field needs a target grid: theta_rad= (and phi_rad=) for directions, "
            "ux= and uy= for direction cosines, or plane_um=, x_um= and y_um= for "
            "points on a plane")
    if len(given) > 1:
        raise ValueError(
            f"far_field takes one target grid, but got {' and '.join(given)}")
    if kspace and (ux is None or uy is None):
        raise ValueError("a k-space grid needs both ux= and uy=")
    if cart and (plane_um is None or x_um is None or y_um is None):
        raise ValueError("a Cartesian grid needs plane_um=, x_um= and y_um=")
    if not angle and np.any(np.asarray(phi_rad, dtype=np.float64) != 0.0):
        raise ValueError("phi_rad= belongs to an angle grid (theta_rad=)")
    if angle and plane_axis is not None:
        raise ValueError("plane_axis= belongs to a k-space or Cartesian grid, not to "
                         "theta_rad=")
    if plane_axis is not None and plane_axis not in _NATURAL:
        raise ValueError(f"plane_axis must be 'x', 'y' or 'z', got {plane_axis!r}")
    if direction is not None:
        if not kspace:
            raise ValueError("direction= picks the half space of a k-space grid "
                             "(ux=, uy=)")
        if direction not in ("+", "-"):
            raise ValueError(f"direction must be '+' or '-', got {direction!r}")
    if not isinstance(exact, (bool, np.bool_)):
        raise ValueError(f"exact must be True or False, got {exact!r}")
    if r_um is not None:
        if not exact:
            raise ValueError(
                "r_um= is the distance of an exact projection (exact=True); far-field "
                "amplitudes do not depend on it")
        if cart:
            raise ValueError(
                "r_um= belongs to an angle or k-space grid; each point of a Cartesian "
                "grid has its own distance")
        if not (np.ndim(r_um) == 0 and np.isfinite(float(r_um)) and float(r_um) > 0.0):
            raise ValueError(f"r_um must be one positive distance (um), got {r_um!r}")
    elif exact and not cart:
        raise ValueError(
            "exact=True on an angle or k-space grid needs the distance: r_um= (um from "
            "origin_um, by default the surface's centre)")
    if origin_um is not None:
        o = np.asarray(origin_um, dtype=np.float64)
        if o.shape != (3,) or not np.all(np.isfinite(o)):
            raise ValueError(f"origin_um must be three finite coordinates (um), got "
                             f"{origin_um!r}")
    if cart and not (np.ndim(plane_um) == 0 and np.isfinite(float(plane_um))):
        raise ValueError(f"plane_um must be one finite coordinate (um), got {plane_um!r}")


@dataclass(frozen=True)
class _Targets:
    """A target grid resolved against the surface. Positions are in um in
    the wire frame (``offset`` added gives the frame of ``data[name]``).
    ``live`` marks the points with a direction (all but the evanescent part
    of a k-space grid); ``theta``, ``phi`` and ``rhat`` (``(n, 3)``) are
    ``nan`` elsewhere. ``origin`` is where distances and the far-field phase
    are measured from (``None``: the far-field amplitudes keep the phase of
    the wire frame, as they always have). ``points`` and ``r_um`` are set
    for an exact projection and on a Cartesian grid; ``axes`` holds a
    meshed Cartesian grid's two in-plane axes (wire frame). ``lo`` and
    ``hi`` are the corners of the surface's bounding box (wire frame, the
    full surface's with symmetry planes)."""

    grid: str
    theta: np.ndarray
    phi: np.ndarray
    live: np.ndarray
    rhat: np.ndarray
    shape: Optional[Tuple[int, int]]
    offset: np.ndarray
    origin: Optional[np.ndarray]
    points: Optional[np.ndarray]
    r_um: Optional[np.ndarray]
    axes: Optional[Tuple[np.ndarray, np.ndarray]]
    lo: np.ndarray
    hi: np.ndarray
    meta: Dict[str, object]


def _frame_offset(data) -> np.ndarray:
    """``origin_um`` of the run's simulation: what the frame of
    ``data[name]`` adds to the wire frame (0 for a plain mapping), the same
    simulation :func:`~photonhub.components.frame.wire_array` reads it from."""
    sim = None
    try:
        known = getattr(data, "_known_simulation", None)
        sim = known() if callable(known) else getattr(data, "simulation", None)
    except (AttributeError, OSError, ValueError):
        sim = None
    return np.asarray(_frame.frame_origin(sim), dtype=np.float64)


def _pairs(a, b, mesh: bool, what: str) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """Two 1-D axes meshed (``a`` slowest) or paired elementwise, as flat
    arrays, with the meshed shape."""
    a = np.atleast_1d(np.asarray(a, dtype=np.float64)).ravel()
    b = np.atleast_1d(np.asarray(b, dtype=np.float64)).ravel()
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError(f"{what} must be finite")
    if mesh:
        A, B = np.meshgrid(a, b, indexing="ij")
        return A.reshape(-1), B.reshape(-1), (a.size, b.size)
    if a.size != b.size:
        raise ValueError(f"with mesh=False, {what} must be the same length; got "
                         f"{a.size} and {b.size}")
    return a, b, None


def _resolve_targets(data, specs, readout: _Readout, *, theta_rad, phi_rad, ux, uy,
                     plane_axis, direction, plane_um, x_um, y_um, exact, r_um,
                     origin_um, mesh) -> _Targets:
    """The target grid of :func:`far_field` against the recorded faces
    ``specs`` (wire frame, the modelled part) and the readout's symmetry
    planes: directions, the origin, and the points of an exact projection
    or a Cartesian grid."""
    offset = _frame_offset(data)
    sym = readout.symmetry or (0, 0, 0)
    lo, hi = np.zeros(3), np.zeros(3)
    for i, a in enumerate("xyz"):
        c = [_coords(da, a) for da, _, _ in specs]
        c = np.concatenate([v for v in c if v.size] or [np.zeros(1)])
        if sym[i]:
            lo[i], hi[i] = -float(np.max(np.abs(c))), float(np.max(np.abs(c)))
        else:
            lo[i], hi[i] = float(np.min(c)), float(np.max(c))
    centre = 0.5 * (lo + hi)
    planes = {(ax, 1.0 if sg > 0 else -1.0) for _, ax, sg in specs}
    single = next(iter(planes)) if len(planes) == 1 else None
    origin = None if origin_um is None else np.asarray(origin_um, dtype=np.float64) - offset
    meta: Dict[str, object] = {}

    def need_axis(what: str) -> str:
        if plane_axis is not None:
            return plane_axis
        if single is not None:
            return single[0]
        raise ValueError(
            f"far_field: a {what} grid on a surface of several faces needs plane_axis= "
            "(the axis normal to it); a single plane uses its own normal")

    points = r = axes = None
    if theta_rad is not None:
        grid = "angle"
        th, ph, shape = _directions(theta_rad, phi_rad, mesh)
        rhat = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
        live = np.ones(th.size, dtype=bool)
    elif ux is not None:
        grid = "kspace"
        a = need_axis("k-space")
        if direction is None:
            s = single[1] if single is not None and single[0] == a else 1.0
            direction = "+" if s > 0 else "-"
        s = 1.0 if direction == "+" else -1.0
        u1, u2, shape = _pairs(ux, uy, mesh, "ux and uy")
        live = u1 * u1 + u2 * u2 <= 1.0
        p, q = _NATURAL[a]
        rhat = np.full((u1.size, 3), np.nan)
        rhat[live, "xyz".index(p)] = u1[live]
        rhat[live, "xyz".index(q)] = u2[live]
        rhat[live, "xyz".index(a)] = s * np.sqrt(np.maximum(0.0, 1.0 - u1[live] ** 2
                                                            - u2[live] ** 2))
        th = np.arccos(np.clip(rhat[:, 2], -1.0, 1.0))
        ph = np.arctan2(rhat[:, 1], rhat[:, 0])
        meta.update(ux=u1, uy=u2, direction=direction, plane_axis=a)
    else:
        grid = "cartesian"
        a = need_axis("Cartesian")
        p, q = _NATURAL[a]
        v1, v2, shape = _pairs(x_um, y_um, mesh, "x_um and y_um")
        user = np.empty((v1.size, 3))
        user[:, "xyz".index(p)] = v1
        user[:, "xyz".index(q)] = v2
        user[:, "xyz".index(a)] = float(plane_um)
        points = user - offset
        if origin is None:
            origin = centre
        R = points - origin
        r = np.linalg.norm(R, axis=1)
        if np.any(r == 0.0):
            raise ValueError(
                "far_field: a target point sits on the origin (origin_um), where it has "
                "no direction; move the origin or the point")
        rhat = R / r[:, None]
        th = np.arccos(np.clip(rhat[:, 2], -1.0, 1.0))
        ph = np.arctan2(rhat[:, 1], rhat[:, 0])
        live = np.ones(v1.size, dtype=bool)
        if mesh:
            axes = (np.atleast_1d(np.asarray(x_um, dtype=np.float64)).ravel()
                    - offset["xyz".index(p)],
                    np.atleast_1d(np.asarray(y_um, dtype=np.float64)).ravel()
                    - offset["xyz".index(q)])
        meta.update(x_um=v1, y_um=v2, plane_um=float(plane_um), plane_axis=a)
    if exact and grid != "cartesian":
        if origin is None:
            origin = centre
        r = np.where(live, float(r_um), np.nan)
        points = origin[None, :] + float(r_um) * rhat
    return _Targets(grid=grid, theta=th, phi=ph, live=live, rhat=rhat, shape=shape,
                    offset=offset, origin=origin, points=points, r_um=r, axes=axes,
                    lo=lo, hi=hi, meta=meta)


def _far_result(targets: _Targets, e_theta: np.ndarray, e_phi: np.ndarray, freqs,
                n_f: np.ndarray, n_report) -> FarField:
    """The :class:`FarField` of far-field amplitudes on the live points of
    ``targets`` (``(n_freq, n_live)``): referred to ``targets.origin`` when
    it is set (``F_o(r_hat) = F(r_hat) e^{i k r_hat . o}``), ``nan`` at the
    points with no direction."""
    f = np.asarray(freqs, dtype=np.float64)
    live = targets.live
    if targets.origin is not None:
        k = (np.asarray(n_f, dtype=np.float64) * 2.0 * np.pi * f / C0)[:, None]
        shift = targets.rhat[live] @ (targets.origin * 1e-6)
        phase = np.exp(1j * k * shift[None, :])
        e_theta, e_phi = e_theta * phase, e_phi * phase
    if not np.all(live):
        full_t = np.full((f.size, live.size), np.nan + 1j * np.nan)
        full_p = full_t.copy()
        full_t[:, live], full_p[:, live] = e_theta, e_phi
        e_theta, e_phi = full_t, full_p
    return _result(targets, f, e_theta, e_phi, n_report)


def _result(targets: _Targets, f: np.ndarray, e_theta, e_phi, n_report, *, exact=False,
            **fields) -> FarField:
    meta = targets.meta
    origin = (None if targets.origin is None
              else tuple(float(v) for v in targets.origin + targets.offset))
    return FarField(
        theta=targets.theta, phi=targets.phi, freqs_hz=f, e_theta=e_theta, e_phi=e_phi,
        n_medium=n_report, grid=targets.grid, shape=targets.shape, exact=exact,
        r_um=targets.r_um, origin_um=origin, ux=meta.get("ux"), uy=meta.get("uy"),
        direction=meta.get("direction"), x_um=meta.get("x_um"), y_um=meta.get("y_um"),
        plane_um=meta.get("plane_um"), plane_axis=meta.get("plane_axis"), **fields)


def _far_zone_size(targets: _Targets) -> float:
    """``D`` of the far-zone rule: twice the largest distance (um) from
    ``targets.origin`` to a corner of the surface's bounding box (its
    diagonal when the origin is its centre, the default). The far-field
    approximation's phase error at a point ``r`` from the origin is about
    ``k rho^2 / (2 r)`` for a sample ``rho`` from it, ``pi / 8`` at ``r = 2
    D^2 / lambda``."""
    corners = np.array([[(targets.lo if (j >> i) & 1 == 0 else targets.hi)[i]
                         for i in range(3)] for j in range(8)])
    return 2.0 * float(np.max(np.linalg.norm(corners - targets.origin[None, :], axis=1)))


def _warn_near_zone(targets: _Targets, freqs, n_f) -> None:
    """A Cartesian grid read in the far-field approximation: warn when a
    point lies nearer the origin than ``2 D^2 / lambda`` (``D`` from
    :func:`_far_zone_size`, ``lambda`` the shortest wavelength in the
    medium), where the approximation's phase error at the surface's
    farthest point from the origin passes ``pi / 8``."""
    lam = min(C0 / (float(n) * float(f)) for n, f in zip(n_f, freqs) if f > 0) * 1e6 \
        if any(f > 0 for f in freqs) else None
    size = _far_zone_size(targets)
    if lam is None or size <= 0.0:
        return
    zone = 2.0 * size ** 2 / lam
    near = int(np.count_nonzero(targets.r_um < zone))
    if near:
        _warn(
            f"{near} of the {targets.r_um.size} target points lie nearer the origin than "
            f"2 D^2 / lambda = {zone:.4g} um (D = {size:.4g} um, twice the surface's "
            f"farthest reach from origin_um; lambda = {lam:.4g} um in the medium), where "
            "the far-field approximation's phase error passes pi/8. Pass exact=True for "
            "the field at a finite distance")


def _refuse_open_boxes(passes) -> None:
    """Every box face of every pass covers the box (:func:`_refuse_open_box`),
    as the far-field kernel checks it."""
    for faces, bounds in passes:
        for fc in faces:
            if fc["box"]:
                t1, t2 = _TRANSVERSE[fc["axis"]]
                for term in _face_terms(fc):
                    _refuse_open_box(term, t1, t2, bounds, fc["label"], fc["axis"],
                                     fc["sign"])


def _surface_samples(passes, grid: Optional[_Grid]):
    """The weighted current samples of the surface: ``(P, J, M)``, positions
    (m) and the electric and magnetic current per sample times its
    quadrature weight (A m and V m), each ``(n, 3)``, with the weights of
    the far-field kernel (:func:`_term_widths`, cut at the box edges) and
    each pass weighted by ``1 / len(passes)`` (``"avg"`` is their mean)."""
    P, J, M = [], [], []
    share = 1.0 / len(passes)
    for faces, bounds in passes:
        for fc in faces:
            ax = fc["axis"]
            t1, t2 = _TRANSVERSE[ax]
            i1, i2, ia = "xyz".index(t1), "xyz".index(t2), "xyz".index(ax)
            for name, vals, c1, c2, n_um in _face_terms(fc):
                c1m = np.asarray(c1, dtype=np.float64) * 1e-6
                c2m = np.asarray(c2, dtype=np.float64) * 1e-6
                w = (_term_widths(c2m, bounds.get(t2), c1m, t2, grid)[:, None]
                     * _term_widths(c1m, bounds.get(t1), c2m, t1, grid)[None, :])
                pos = np.empty((c2m.size * c1m.size, 3))
                pos[:, i1] = np.broadcast_to(c1m[None, :], w.shape).ravel()
                pos[:, i2] = np.broadcast_to(c2m[:, None], w.shape).ravel()
                pos[:, ia] = float(n_um) * 1e-6
                cur = np.zeros((pos.shape[0], 3), dtype=np.complex128)
                cur[:, "xyz".index(name[1])] = (share * vals * w).ravel()
                P.append(pos)
                (J if name[0] == "J" else M).append(cur)
                (M if name[0] == "J" else J).append(np.zeros_like(cur))
    return np.concatenate(P), np.concatenate(J), np.concatenate(M)


def _green_fields(P: np.ndarray, J: np.ndarray, M: np.ndarray, T: np.ndarray,
                  k: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(E, H)`` (each ``(n_targets, 3)``) radiated at the points ``T`` (m)
    by the weighted current samples ``(P, J, M)`` (:func:`_surface_samples`)
    in a homogeneous medium of wavenumber ``k`` (1/m) and impedance
    ``eta``, with the full dyadic Green's function (``e^{-i omega t}``,
    ``g = e^{ikR} / (4 pi R)``, ``x = k R``)::

        E = sum  i k eta g [a J + b (R.J) R] - i k g c (R x M)
        H = sum  (i k / eta) g [a M + b (R.M) R] + i k g c (R x J)

    with ``R`` the unit vector from the sample to the point, ``a = 1 + i/x -
    1/x^2``, ``b = -1 - 3i/x + 3/x^2``, ``c = 1 + i/x``. In the far zone
    (``x`` large) it is the far-field kernel's ``i k e^{ikr} / (4 pi r)
    (eta (N - (r.N) r) - r x L)``. Summed in blocks of about
    :data:`_EXACT_CHUNK` pairs."""
    nt, npts = T.shape[0], P.shape[0]
    E = np.zeros((nt, 3), dtype=np.complex128)
    H = np.zeros((nt, 3), dtype=np.complex128)
    jc = [c for c in range(3) if np.any(J[:, c])]
    mc = [c for c in range(3) if np.any(M[:, c])]
    ct = max(1, min(nt, _EXACT_CHUNK // max(npts, 1)))
    cp = min(npts, max(1, _EXACT_CHUNK // ct))
    ik = 1j * k
    for t0 in range(0, nt, ct):
        tsl = slice(t0, min(t0 + ct, nt))
        Tc = T[tsl]
        for p0 in range(0, npts, cp):
            psl = slice(p0, min(p0 + cp, npts))
            Rv = [Tc[:, i, None] - P[None, psl, i] for i in range(3)]
            R = np.sqrt(Rv[0] * Rv[0] + Rv[1] * Rv[1] + Rv[2] * Rv[2])
            inv = 1.0 / R
            n = [v * inv for v in Rv]
            ix = inv / k
            g = np.exp(1j * k * R) * (inv / (4.0 * np.pi))
            gc = g * (1.0 + 1j * ix)
            ga = g * (1.0 + 1j * ix - ix * ix)
            gb = g * (-1.0 - 3j * ix + 3.0 * ix * ix)
            gcn = [gc * n[i] for i in range(3)]
            for src, comps, same, cross, s_same, s_cross in (
                    (J, jc, E, H, ik * eta, ik), (M, mc, H, E, ik / eta, -ik)):
                if not comps:
                    continue
                V = src[psl]
                vr = sum(n[c] * V[:, c] for c in comps)
                gbv = gb * vr
                for o in range(3):
                    acc = (gbv * n[o]).sum(axis=1)
                    if o in comps:
                        acc = acc + ga @ V[:, o]
                    same[tsl, o] += s_same * acc
                # (R x V)_o = R_(o+1) V_(o+2) - R_(o+2) V_(o+1)
                for o in range(3):
                    a1, a2 = (o + 1) % 3, (o + 2) % 3
                    acc = np.zeros(Tc.shape[0], dtype=np.complex128)
                    if a2 in comps:
                        acc = acc + gcn[a1] @ V[:, a2]
                    if a1 in comps:
                        acc = acc - gcn[a2] @ V[:, a1]
                    cross[tsl, o] += s_cross * acc
    return E, H


def _check_outside(passes, targets: _Targets) -> None:
    """Every live target point of an exact projection must lie outside the
    surface: outside a box's span (a closed or open box; its equivalent
    currents radiate nothing inside), or beyond a plane on the side it
    radiates into (behind it they radiate nearly nothing, not the field
    there)."""
    T = targets.points[targets.live]
    faces = passes[0][0]
    if any(fc["box"] for fc in faces):
        # the box's span: where each pass cuts its faces' quadrature (the
        # surfaces the currents sit on), both passes together
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for fs, bounds in passes:
            for i, a in enumerate("xyz"):
                if a in bounds:
                    lo[i] = min(lo[i], bounds[a][0] * 1e6)
                    hi[i] = max(hi[i], bounds[a][1] * 1e6)
                else:
                    for fc in fs:
                        for term in _face_terms(fc):
                            t1, t2 = _TRANSVERSE[fc["axis"]]
                            c = {t1: term[2], t2: term[3], fc["axis"]: [term[4]]}[a]
                            lo[i] = min(lo[i], float(np.min(c)))
                            hi[i] = max(hi[i], float(np.max(c)))
        tol = 1e-9 * max(float(np.max(hi - lo)), 1e-6)
        inside = np.all((T >= lo - tol) & (T <= hi + tol), axis=1)
        if np.any(inside):
            p = T[np.argmax(inside)] + targets.offset
            raise ValueError(
                f"far_field: the target point {tuple(round(float(v), 6) for v in p)} um lies "
                "within the box's span; an exact projection reads the field outside the "
                "closed surface, where its equivalent currents radiate it")
        return
    ok = _beyond_planes(faces, T)
    if not np.all(ok):
        p = T[np.argmin(ok)] + targets.offset
        raise ValueError(
            f"far_field: the target point {tuple(round(float(v), 6) for v in p)} um is not "
            "beyond the plane on the side it radiates into; an exact projection of an "
            "open plane reads the field in the half space it faces (behind it the "
            "plane's equivalent currents radiate nearly nothing, not the field there)")


def _beyond_planes(faces, T: np.ndarray) -> np.ndarray:
    """Which points ``T`` (um, wire frame, ``(n, 3)``) lie strictly beyond at
    least one of the open planes ``faces`` on the side it radiates into,
    past its outermost current samples (J sits half a cell past E)."""
    ok = np.zeros(T.shape[0], dtype=bool)
    for fc in faces:
        a = "xyz".index(fc["axis"])
        outer = [float(t[4]) for t in _face_terms(fc)]
        edge = max(outer) if fc["sign"] > 0 else min(outer)
        ok |= (T[:, a] - edge) * fc["sign"] > 1e-9 * max(abs(edge), 1.0)
    return ok


def _warn_near_surface(passes, targets: _Targets, wavelength_um: float) -> None:
    """The direct sum integrates a kernel that grows as ``1/R^3`` near the
    surface on the recorded samples: warn when a live target point lies
    within two sample spacings (the largest in-plane spacing of the
    surface's samples) or a tenth of ``wavelength_um`` (the longest
    projected wavelength in the medium), whichever is farther, of the
    region its currents fill. Measured on the exact field of a dipole in a
    closed box, against the dipole, the worst over planes beside a face
    and its edges: half a spacing out 0.4 to 0.6 times the peak field at
    20 cells per wavelength (up to 2.5 times at 80); one spacing out 3 to
    6 % at 20 cells, up to 11 % at 40 and 21 % at 80 over the box's edges;
    the readout's own error reached from two spacings at 20 cells, three
    at 40 and four at 80 (a twentieth to a tenth of a wavelength), from two
    on every mesh over a face's interior."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    spacing = 0.0
    for fs, bounds in passes:
        for fc in fs:
            t1, t2 = _TRANSVERSE[fc["axis"]]
            for _name, _vals, c1, c2, n_um in _face_terms(fc):
                for a, c in ((t1, c1), (t2, c2), (fc["axis"], [n_um])):
                    c = np.asarray(c, dtype=np.float64).ravel()
                    i = "xyz".index(a)
                    lo_a, hi_a = float(np.min(c)), float(np.max(c))
                    if a in bounds:
                        lo_a = max(lo_a, bounds[a][0] * 1e6)
                        hi_a = min(hi_a, bounds[a][1] * 1e6)
                    lo[i], hi[i] = min(lo[i], lo_a), max(hi[i], hi_a)
                for c in (c1, c2):
                    c = np.asarray(c, dtype=np.float64).ravel()
                    if c.size >= 2:
                        spacing = max(spacing, float(np.max(np.abs(np.diff(c)))))
    if spacing <= 0.0:
        return
    T = targets.points[targets.live]
    gap = np.maximum(np.maximum(lo[None, :] - T, T - hi[None, :]), 0.0)
    dist = np.sqrt(np.sum(gap * gap, axis=1))
    reach = max(2.0 * spacing, 0.1 * float(wavelength_um))
    near = int(np.count_nonzero(dist < reach * (1.0 - 1e-9)))
    if near:
        _warn(
            f"exact=True: {near} of the {T.shape[0]} target points lie within {reach:.4g} um "
            f"of the surface's currents (two sample spacings, {2.0 * spacing:.4g} um, or a "
            "tenth of a wavelength, whichever is farther; the nearest "
            f"{float(np.min(dist)):.4g} um), where the direct sum over the recorded samples "
            "loses accuracy: on a dipole's exact field 3 to 6 % of the peak one spacing out "
            "at 20 cells per wavelength, up to 21 % at 80 over a box's edges. Read the "
            "field farther from the surface")


def _spectral_plane(passes, targets: _Targets):
    """The one face the angular-spectrum path can propagate, or ``None``: a
    single plane (not a box face) normal to the target plane's axis, its two
    tangential E components each sampled evenly along both in-plane axes,
    and a meshed Cartesian grid."""
    if targets.grid != "cartesian" or targets.axes is None or len(passes) != 1:
        return None
    faces = passes[0][0]
    if len(faces) != 1 or faces[0]["box"] or faces[0]["axis"] != targets.meta["plane_axis"]:
        return None
    fc = faces[0]
    e1n, e2n = _TANGENTIAL[fc["axis"]][:2]
    for nm in (e1n, e2n):
        _vals, c1, c2 = fc["fields"][nm]
        for c in (np.asarray(c1, dtype=np.float64), np.asarray(c2, dtype=np.float64)):
            if c.size < 2:
                return None
            d = np.diff(c)
            if np.max(np.abs(d - d[0])) > _SPACING_RTOL * abs(d[0]) or d[0] <= 0.0:
                return None
    return fc


def _spectral_fields(fc, targets: _Targets, k: float, eta: float
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(E, H)`` (each ``(n_points, 3)``) at the meshed target plane from
    the angular spectrum of the plane ``fc``'s tangential E: each component
    is Fourier transformed from its own sample positions, each plane wave
    advanced to the target plane (evanescent ones decay), the normal E
    completed from ``div E = 0`` and H from ``k x E / (k eta)``, and the sum
    evaluated at the targets. The transform is zero-padded to at least twice
    the span of the plane and the targets, and the padding doubled until
    doubling it again moves no value by more than :data:`_SPECTRAL_RTOL` of
    the largest |E| (``eta`` |H| counted alike); ``None`` when that needs
    more than :data:`_SPECTRAL_MAX_POINTS` points (the direct sum takes
    over). Exact for a field that leaves the plane: the J and M of the
    direct sum carry the same field there."""
    a, s = fc["axis"], float(fc["sign"])
    t1, t2 = _TRANSVERSE[a]
    e1n, e2n = _TANGENTIAL[a][:2]
    n_e = float(fc["n"][e1n]) * 1e-6
    p, q = _NATURAL[a]
    ax_p, ax_q = (v * 1e-6 for v in targets.axes)
    u1, u2 = (ax_p, ax_q) if (t1, t2) == (p, q) else (ax_q, ax_p)
    dz = (float(targets.points[0, "xyz".index(a)]) * 1e-6 - n_e) * s
    comps = []
    for nm in (e1n, e2n):
        vals, c1, c2 = fc["fields"][nm]
        comps.append((np.asarray(vals, dtype=np.complex128),
                      np.asarray(c1, dtype=np.float64) * 1e-6,
                      np.asarray(c2, dtype=np.float64) * 1e-6))
    d1 = float(comps[0][1][1] - comps[0][1][0])
    d2 = float(comps[0][2][1] - comps[0][2][0])

    def span(c_list, u):
        lo_ = min([float(np.min(c)) for c in c_list] + [float(np.min(u))])
        hi_ = max([float(np.max(c)) for c in c_list] + [float(np.max(u))])
        return hi_ - lo_

    s1 = span([c[1] for c in comps], u1) + d1
    s2 = span([c[2] for c in comps], u2) + d2
    N1 = 1 << int(np.ceil(np.log2(max(2.0 * s1 / d1, 2.0))))
    N2 = 1 << int(np.ceil(np.log2(max(2.0 * s2 / d2, 2.0))))

    def at(N1, N2):
        # Each component is evaluated at the points as soon as it is formed,
        # so about seven padded arrays are alive at once (some 0.45 GB at
        # the largest padding).
        k1 = 2.0 * np.pi * np.fft.fftfreq(N1, d=d1)
        k2 = 2.0 * np.pi * np.fft.fftfreq(N2, d=d2)
        K1, K2 = k1[None, :], k2[:, None]                 # spectra are [k2, k1]
        kz = np.sqrt((k * k - K1 * K1 - K2 * K2).astype(np.complex128))
        kz = np.where(kz.real < 0, -kz, kz)
        kz = np.where((kz.real == 0) & (kz.imag < 0), np.conj(kz), kz)
        A1 = np.exp(1j * np.outer(u1, k1)) / N1             # [n_u1, N1]
        A2 = np.exp(1j * np.outer(u2, k2)) / N2             # [n_u2, N2]

        def ev(X):
            return A1 @ X.T @ A2.T                          # [n_u1, n_u2]

        prop = np.exp(1j * kz * dz)
        E = []
        for vals, c1, c2 in comps:
            X = np.zeros((N2, N1), dtype=np.complex128)
            X[:vals.shape[0], :vals.shape[1]] = vals
            X = np.fft.fft2(X)
            X *= np.exp(-1j * k2 * c2[0])[:, None]
            X *= np.exp(-1j * k1 * c1[0])[None, :]
            X *= prop
            E.append(X)
        del prop
        out = {"e1": ev(E[0]), "e2": ev(E[1])}
        En = K1 * E[0]
        En += K2 * E[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            En /= -s * kz
        En[np.abs(kz) <= 1e-12 * k] = 0.0
        out["en"] = ev(En)
        ce = 1.0 / (k * eta)
        tmp = K2 * En
        tmp -= s * kz * E[1]
        out["h1"] = ce * ev(tmp)
        tmp = s * kz * E[0]
        tmp -= K1 * En
        out["h2"] = ce * ev(tmp)
        del En
        tmp = K1 * E[1]
        tmp -= K2 * E[0]
        out["hn"] = ce * ev(tmp)
        return out

    if 4 * N1 * N2 > _SPECTRAL_MAX_POINTS:
        return None
    prev = at(N1, N2)
    last = None
    while 4 * N1 * N2 <= _SPECTRAL_MAX_POINTS:
        N1, N2 = 2 * N1, 2 * N2
        cur = at(N1, N2)
        peak = max(float(np.max(np.abs(cur[c]))) for c in ("e1", "e2", "en"))
        moved = max(float(np.max(np.abs(cur[c] - prev[c]))) * (eta if c[0] == "h" else 1.0)
                    for c in cur)
        goal = _SPECTRAL_RTOL * max(peak, 1e-300)
        if moved <= goal:
            break
        if last is not None:
            # Give up early when the moves, shrinking at their last rate,
            # cannot reach the goal within the doublings the cap leaves.
            left = 0
            while 4 ** (left + 1) * N1 * N2 <= _SPECTRAL_MAX_POINTS:
                left += 1
            rate = moved / last
            if rate >= 1.0 or moved * rate ** left > goal:
                return None
        last = moved
        prev = cur
    else:
        return None
    E = np.empty((targets.points.shape[0], 3), dtype=np.complex128)
    H = np.empty_like(E)
    swap = (t1, t2) != (p, q)
    for key, comp, dest in (("e1", t1, E), ("e2", t2, E), ("en", a, E),
                            ("h1", t1, H), ("h2", t2, H), ("hn", a, H)):
        grid2 = cur[key].T if swap else cur[key]           # [n_p, n_q]
        dest[:, "xyz".index(comp)] = grid2.reshape(-1)
    return E, H


def _exact_projection(face_specs, face_names, freqs, n_f, n_report, readout: _Readout,
                      plane: str, targets: _Targets, path: str) -> FarField:
    """``far_field(..., exact=True)``: the field the surface's currents
    radiate at the target points, per frequency, by the angular spectrum of
    one plane (:func:`_spectral_fields`) where it applies and the direct sum
    (:func:`_green_fields`) otherwise, returned as amplitudes ``r e^{-ikr}``
    times each spherical component about ``targets.origin``."""
    f = np.asarray(freqs, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError("far_field(exact=True) needs positive frequencies")
    wavelength_um = max(C0 / (float(n) * float(fr)) for n, fr in zip(n_f, f)) * 1e6
    first = _yee_passes(face_specs, f[0], readout=readout, plane=plane, names=face_names)
    faces0 = first[0][0]
    if targets.grid != "cartesian" and not any(fc["box"] for fc in faces0):
        # Directions on an open plane's sphere that end at or behind the
        # plane (its grazing rim, or a sliver past it while the sphere
        # starts on the plane) carry no field of the plane: nan, as an
        # evanescent k-space point, rather than a refusal of the grid.
        ok = targets.live.copy()
        ok[ok] = _beyond_planes(faces0, targets.points[ok])
        dropped = int(np.count_nonzero(targets.live & ~ok))
        if dropped and not np.any(ok):
            toward = sorted({("+" if fc["sign"] > 0 else "-") + fc["axis"] for fc in faces0})
            hint = ("theta_rad and phi_rad in that half space" if targets.grid == "angle"
                    else "direction= and plane_axis= for that half space")
            raise ValueError(
                f"far_field: every direction of this exact projection ends, at r_um = "
                f"{float(np.nanmax(targets.r_um)):.4g}, no farther than the plane, where "
                "its equivalent currents radiate nearly nothing, not the field there: the "
                f"plane radiates toward {' and '.join(toward)}. Pick {hint}, or a larger "
                "r_um or an origin_um that puts the sphere in front of it")
        if dropped:
            _warn(
                f"exact=True: {dropped} of the {int(np.count_nonzero(targets.live))} directions "
                f"reach, at r_um = {float(np.nanmax(targets.r_um)):.4g}, no farther than the "
                "plane (its grazing rim, or behind it), where its equivalent currents radiate "
                "nearly nothing, not the field there: their fields are nan and their "
                "intensity 0")
            targets = dataclasses.replace(targets, live=ok,
                                          r_um=np.where(ok, targets.r_um, np.nan))
    live = targets.live
    n_live = int(np.count_nonzero(live))
    out = {c: np.full((f.size, live.size), np.nan + 1j * np.nan, dtype=np.complex128)
           for c in ("e_r", "e_theta", "e_phi", "h_r", "h_theta", "h_phi")}
    R = (targets.points[live] - targets.origin) * 1e-6
    r = np.linalg.norm(R, axis=1)
    th, ph = targets.theta[live], targets.phi[live]
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    basis = {"r": np.stack([st * cp, st * sp, ct], 1),
             "theta": np.stack([ct * cp, ct * sp, -st], 1),
             "phi": np.stack([-sp, cp, np.zeros_like(st)], 1)}
    T = targets.points[live] * 1e-6
    checked = warned = False
    for fi, fr in enumerate(f):
        k = float(n_f[fi]) * (2.0 * np.pi * fr / C0)
        eta = ETA0 / float(n_f[fi])
        passes = first if fi == 0 else _yee_passes(face_specs, fr, readout=readout,
                                                   plane=plane, names=face_names)
        _refuse_open_boxes(passes)
        if not checked:
            _check_outside(passes, targets)
            checked = True
        fields = None
        if path != "direct":
            plane_fc = _spectral_plane(passes, targets) if n_live == live.size else None
            if plane_fc is not None:
                fields = _spectral_fields(plane_fc, targets, k, eta)
            if fields is None and path == "spectral":
                raise ValueError(
                    "far_field: the angular-spectrum path applies to a meshed Cartesian "
                    "grid on a plane parallel to one evenly sampled recorded plane, whose "
                    f"padding settles within {_SPECTRAL_MAX_POINTS} points")
        if fields is None:
            if not warned:
                _warn_near_surface(passes, targets, wavelength_um)
                warned = True
            P, J, M = _surface_samples(passes, readout.grid)
            fields = _green_fields(P, J, M, T, k, eta)
        E, H = fields
        scale = r * np.exp(-1j * k * r)
        for name, vec in (("e", E), ("h", H)):
            for comp, b in basis.items():
                out[f"{name}_{comp}"][fi, live] = scale * np.sum(vec * b, axis=1)
    return _result(targets, f, out["e_theta"], out["e_phi"], n_report, exact=True,
                   e_r=out["e_r"], h_r=out["h_r"], h_theta=out["h_theta"],
                   h_phi=out["h_phi"])
