"""Periodic boundary condition utilities.

Under periodic boundary conditions an atom pair is identified by the two atom
indices plus an integer lattice translation, ``(i, j, n1, n2, n3)``.  The
corresponding distance vector is ``x_i - (x_j + n @ cell)``.  The lattice
translation of every pair is determined once (by a periodic neighbor search on
the unwrapped path) and then kept fixed.  This keeps the internal coordinates
smooth when atoms cross cell boundaries, which would not be the case if the
minimum image were re-evaluated at every step.

Cells follow the ASE convention: the rows of the 3x3 ``cell`` matrix are the
lattice vectors and ``pbc`` holds three booleans for the periodic directions.
"""
import itertools
import logging

import numpy as np
from scipy.spatial import KDTree


logger = logging.getLogger(__name__)


def normalize_cell(cell, pbc=None):
    """Validate a cell and its periodicity flags.

    Args:
        cell:   3x3 matrix with lattice vectors as rows, or 3 box lengths.
                ``None`` means no periodicity.
        pbc:    Periodic flags, either one bool or three.  Defaults to fully
                periodic when a cell is given.

    Returns:
        (cell, pbc) as a float 3x3 array and a bool array of length 3, or
        (None, None) when no direction is periodic.  Zero lattice vectors along
        non-periodic directions (as ASE uses for slabs) are replaced by
        orthogonal unit vectors so the cell can be inverted.
    """
    if cell is None:
        if pbc is not None and np.any(pbc):
            raise ValueError("A cell is required for periodic boundary conditions")
        return None, None
    cell = np.array(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    if cell.shape != (3, 3):
        raise ValueError("Cell must be a 3x3 matrix or 3 box lengths")
    if pbc is None:
        pbc = True
    pbc = np.array(np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)))
    if not pbc.any():
        return None, None
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths[pbc] < 1e-8):
        raise ValueError("Periodic directions must have non-zero lattice vectors")
    for k in np.flatnonzero(lengths < 1e-8):
        others = cell[lengths >= 1e-8]
        _, sv, vt = np.linalg.svd(others)
        cell[k] = vt[np.sum(sv > 1e-8)]
        lengths[k] = 1.0
    if abs(np.linalg.det(cell)) < 1e-8:
        raise ValueError("Cell is singular")
    return cell, pbc


def _shift_grid(nrep):
    """All integer lattice translations with |n_k| <= nrep[k]."""
    ranges = [range(-n, n + 1) for n in nrep]
    return np.array(list(itertools.product(*ranges)), dtype=int).reshape(-1, 3)


def wrap_positions(geom, cell, pbc):
    """Wrap positions into the cell along periodic directions.

    Returns:
        wrapped:    Wrapped positions.
        offsets:    Integer lattice translations with ``geom = wrapped + offsets @ cell``.
    """
    geom = np.asarray(geom, dtype=float)
    frac = geom @ np.linalg.inv(cell)
    offsets = np.zeros(frac.shape, dtype=int)
    offsets[..., pbc] = np.floor(frac[..., pbc]).astype(int)
    return geom - offsets @ cell, offsets


def minimum_image(dvec, cell, pbc):
    """Minimum image convention for an array of displacement vectors."""
    dvec = np.asarray(dvec, dtype=float)
    flat = dvec.reshape(-1, 3)
    frac = flat @ np.linalg.inv(cell)
    frac[:, pbc] -= np.round(frac[:, pbc])
    reduced = frac @ cell
    # Rounding fractional coordinates is only exact for orthogonal cells, so also check the
    # neighboring images to handle skewed cells.
    candidates = reduced[:, None, :] + (_shift_grid(pbc.astype(int)) @ cell)[None]
    best = np.argmin(np.einsum('ijk,ijk->ij', candidates, candidates), axis=1)
    return candidates[np.arange(len(flat)), best].reshape(dvec.shape)


def bonded_pairs(geom, radii, cell, pbc, scale=1.2):
    """Bonded atom pairs: distance below `scale` times the sum of covalent radii.

    Returns:
        Dict mapping `(i, j)` to the lattice translation `n` of the shortest bonded image of j,
        i.e. the bond vector is ``x_i - (x_j + n @ cell)``.
    """
    radii = np.asarray(radii, dtype=float)
    found = find_periodic_pairs(geom, scale * 2 * radii.max(), cell, pbc)
    bonds = {}
    for (i, j, *shift), d in found.items():
        if d < scale * (radii[i] + radii[j]) and ((i, j) not in bonds or d < bonds[i, j][1]):
            bonds[i, j] = (np.array(shift), d)
    return {pair: shift for pair, (shift, _) in bonds.items()}


def unwrap_geom(refgeom, geom, cell, pbc, radii=None):
    """Translate atoms of `geom` by lattice vectors to make it continuous with `refgeom`.

    Without `radii`, every atom is imaged independently with the minimum image convention.  This
    fails when atoms move by about half a lattice vector or more: atoms of one molecule can then
    be imaged in opposite directions, tearing it apart.  With covalent `radii`, bonds present in
    both geometries keep their periodic image, so every bonded fragment is translated as a whole,
    by the lattice vector that minimizes the displacement of the fragment from the reference.
    Atoms without persistent bonds are imaged independently, as with the minimum image convention.

    Returns:
        The unwrapped geometry, which differs from `geom` only by lattice translations of atoms.
    """
    refgeom = np.asarray(refgeom, dtype=float)
    geom = np.asarray(geom, dtype=float)
    inv = np.linalg.inv(cell)
    disp = geom - refgeom
    shifts = np.rint((minimum_image(disp, cell, pbc) - disp) @ inv).astype(int)
    # Per-atom imaging is unambiguous for fragments if no atom moves more than a quarter cell
    if radii is None or np.all(np.abs(((disp + shifts @ cell) @ inv)[:, pbc]) < 0.25):
        return geom + shifts @ cell
    bonds_ref = bonded_pairs(refgeom, radii, cell, pbc)
    bonds = bonded_pairs(geom, radii, cell, pbc)
    neighbors = {i: [] for i in range(len(geom))}
    for pair in set(bonds_ref) & set(bonds):
        i, j = pair
        delta = bonds[pair] - bonds_ref[pair]   # shift[j] = shift[i] + delta keeps the bond intact
        neighbors[i].append((j, delta))
        neighbors[j].append((i, -delta))
    shifts = np.zeros((len(geom), 3), dtype=int)
    visited = np.zeros(len(geom), dtype=bool)
    candidates = _shift_grid(pbc.astype(int))
    conflicts = 0
    for root in range(len(geom)):
        if visited[root]:
            continue
        visited[root] = True
        fragment, queue = [root], [root]
        while queue:
            i = queue.pop()
            for j, delta in neighbors[i]:
                if not visited[j]:
                    visited[j] = True
                    shifts[j] = shifts[i] + delta
                    fragment.append(j)
                    queue.append(j)
                elif not np.array_equal(shifts[j], shifts[i] + delta):
                    conflicts += 1
        # Rigid lattice translation of the fragment closest to the reference
        d = geom[fragment] + shifts[fragment] @ cell - refgeom[fragment]
        center = np.where(pbc, -np.rint(d.mean(axis=0) @ inv), 0).astype(int)
        cost = [np.sum((d + (center + k) @ cell) ** 2) for k in candidates]
        shifts[fragment] += center + candidates[np.argmin(cost)]
    if conflicts:
        logger.warning("%d persistent bonds could not be kept in the same periodic image while unwrapping",
                       conflicts // 2)
    return geom + shifts @ cell


def unwrap_path(path, cell, pbc, radii=None):
    """Make atoms move continuously along a path.

    Each image is translated atom-by-atom by lattice vectors to be continuous with the previous
    image, see `unwrap_geom`.  The first image is left unchanged.
    """
    path = np.array(path, dtype=float)
    for i in range(1, len(path)):
        path[i] = unwrap_geom(path[i - 1], path[i], cell, pbc, radii)
    return path


def canonical_pair(i, j, shift):
    """Unique representation ``(i, j, n1, n2, n3)`` with ``i < j`` of a periodic pair."""
    if i > j:
        i, j, shift = j, i, -np.asarray(shift)
    return (int(i), int(j)) + tuple(int(s) for s in shift)


def find_periodic_pairs(geom, cutoff, cell, pbc, centers=None):
    """Find all pairs of distinct atoms within `cutoff` of each other, including periodic images.

    Pairs of an atom with its own periodic images are excluded since their distance is constant.

    Args:
        geom:       Cartesian geometry with shape (natoms, 3).  Need not be wrapped.
        cutoff:     Distance cut-off.
        cell, pbc:  Normalized cell, see `normalize_cell`.
        centers:    Only search for neighbors of these atoms.  Defaults to all atoms.

    Returns:
        Dict mapping canonical pairs ``(i, j, n1, n2, n3)`` to their distances, where the lattice
        translation refers to the given (unwrapped) positions.
    """
    geom = np.asarray(geom, dtype=float).reshape(-1, 3)
    natoms = len(geom)
    wrapped, offsets = wrap_positions(geom, cell, pbc)
    # Distance between lattice planes determines how many images are needed to cover the cut-off
    spacing = 1 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    nrep = np.where(pbc, np.ceil(cutoff / spacing), 0).astype(int)
    shifts = _shift_grid(nrep)
    images = (wrapped[None] + (shifts @ cell)[:, None]).reshape(-1, 3)
    image_atom = np.tile(np.arange(natoms), len(shifts))
    image_shift = np.repeat(shifts, natoms, axis=0)
    centers = np.arange(natoms) if centers is None else np.asarray(centers, dtype=int)
    tree = KDTree(images)
    found = {}
    for i, neighbors in zip(centers, tree.query_ball_point(wrapped[centers], cutoff)):
        for m in neighbors:
            j = image_atom[m]
            if i == j:
                continue
            pair = canonical_pair(i, j, image_shift[m] + offsets[i] - offsets[j])
            found[pair] = np.linalg.norm(wrapped[i] - images[m])
    return found
