"""Coordinate utilities used by the interpolation program"""
import logging

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.data.vdw_alvarez import vdw_radii
from scipy import sparse
from scipy.spatial import KDTree

from .pbc import canonical_pair, find_periodic_pairs, normalize_cell, unwrap_geom, unwrap_path


logger = logging.getLogger(__name__)


def align_path(path, cell=None, pbc=None, rigid=True, radii=None):
    """Rotate and translate images to minimize RMSD movements along the path.
    Also moves the geometric center of all images to the origin.

    Under periodic boundary conditions (`cell` given), rotations are not meaningful.  The path
    is instead unwrapped with the minimum image convention so that atoms move continuously, and
    is otherwise left in place.  Rigid-body alignment is also skipped when `rigid` is False,
    which is needed when some atoms are fixed in space.  With covalent `radii`, bonded fragments
    are unwrapped as a whole (see `pbc.unwrap_geom`).
    """
    cell, pbc = normalize_cell(cell, pbc)
    if cell is not None or not rigid:
        path = np.array(path, dtype=float)
        if cell is not None:
            path = unwrap_path(path, cell, pbc, radii)
        max_rmsd = max([np.sqrt(np.mean((g - nextg) ** 2)) for g, nextg in zip(path, path[1:])],
                       default=0)
        return max_rmsd, path
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0
    for g, nextg in zip(path, path[1:]):
        rmsd, nextg[:] = align_geom(g, nextg)
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(refgeom, geom, cell=None, pbc=None, rigid=True, radii=None):
    """Find translation/rotation that moves a given geometry to maximally overlap
    with a reference geometry. Implemented with Kabsch algorithm.

    Under periodic boundary conditions, atoms are only translated by lattice vectors
    to their minimum images closest to the reference.  If `rigid` is False, the geometry
    is not rotated or translated.

    Args:
        refgeom:    The reference geometry to be rotated to
        geom:       The geometry to be rotated and shifted
        cell, pbc:  Periodic cell and flags (ASE convention).  None for molecules.
        rigid:      Whether rigid-body rotation and translation are allowed.
        radii:      Covalent radii; under PBC, bonded fragments are then imaged as a whole.

    Returns:
        RMSD:       Root-mean-squared difference between the rotated geometry
                    and the reference
        new_geom:   The rotated geometry that maximumally overal with the reference
    """
    cell, pbc = normalize_cell(cell, pbc)
    if cell is not None or not rigid:
        refgeom = np.asarray(refgeom)
        new_geom = np.array(geom, dtype=float)
        if cell is not None:
            new_geom = unwrap_geom(refgeom, new_geom, cell, pbc, radii)
        return np.sqrt(np.mean((new_geom - refgeom) ** 2)), new_geom
    center = np.mean(refgeom, axis=0)   # Find the geometric center
    ref2 = refgeom - center
    geom2 = geom - np.mean(geom, axis=0)
    cov = np.dot(geom2.T, ref2)
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    new_geom = np.dot(geom2, u) + center
    rmsd = np.sqrt(np.mean((new_geom - refgeom) ** 2))
    return rmsd, new_geom


def fixed_mask(fixed, natoms):
    """Boolean mask of shape (natoms, 3) marking fixed Cartesian coordinates.

    Args:
        fixed:  None, indices of fixed atoms, a boolean mask of fixed atoms, or a boolean
                (natoms, 3) mask of fixed Cartesian components.

    Returns:
        The mask, or None if nothing is fixed.
    """
    if fixed is None:
        return None
    fixed = np.asarray(fixed)
    if fixed.dtype == bool:
        if fixed.shape == (natoms,):
            fixed = np.repeat(fixed[:, None], 3, axis=1)
        elif fixed.shape != (natoms, 3):
            raise ValueError("Boolean mask of fixed atoms must have shape (natoms,) or (natoms, 3)")
        mask = fixed.copy()
    else:
        mask = np.zeros((natoms, 3), dtype=bool)
        mask[fixed.astype(int).ravel()] = True
    return mask if mask.any() else None


def covalent_radius(symbol):
    """Covalent radius from Cordero et al., Dalton Trans. 2008, 2832 (via ASE).
    Unknown symbols get 1.5 Angstrom."""
    number = atomic_numbers.get(symbol.capitalize(), 0)
    return covalent_radii[number] if number else 1.5


def get_radii(atoms):
    """Covalent radii of a list of atom symbols, or None if no symbols are given."""
    return None if atoms is None else np.array([covalent_radius(atom) for atom in atoms])


def get_vdw_radii(atoms):
    """Van der Waals radii from Alvarez, Dalton Trans. 2013, 42, 8617 (via ASE); 2 Angstrom if unknown."""
    radii = []
    for atom in atoms:
        number = atomic_numbers.get(atom.capitalize(), 0)
        radius = vdw_radii[number] if number else np.nan
        radii.append(2.0 if np.isnan(radius) else radius)
    return np.array(radii)


def get_bond_list(geom, atoms=None, threshold=4, min_neighbors=4, snapshots=30, bond_threshold=1.8,
                  enforce=(), cell=None, pbc=None):
    """Get the list of all the important atom pairs.
    Samples a number of snapshots from a list of geometries to generate all
    distances that are below a given threshold in any of them.

    Args:
        atoms:      Symbols for each atoms.
        geom:       One or a list of geometries to check for pairs
        threshold:  Threshold for including a bond in the bond list
        min_neighbors: Minimum number of neighbors to include for each atom.
                    If an atom has smaller than this number of bonds, additional
                    distances will be added to reach this number.
        snapshots:  Number of snapshots to be used in the generation, useful
                    for speeding up the process if the path is long and
                    atoms numerous.
        cell, pbc:  Periodic cell and flags (ASE convention).  When given, the geometries must be
                    unwrapped (see `align_path`) and pairs are returned as `(i, j, n1, n2, n3)`
                    including the lattice translation of atom j.

    Returns:
        List of all the included interatomic distance pairs.
    """
    # Type casting and value checks on input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # If there is only one geometry or it is flattened, promote to 3d
        geom = geom.reshape(1, -1, 3)
    cell, pbc = normalize_cell(cell, pbc)
    if cell is not None:
        rijlist = _get_periodic_bond_list(geom, threshold, min_neighbors, snapshots, bond_threshold,
                                          enforce, cell, pbc)
        return rijlist, _equilibrium_distances(rijlist, atoms)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Determine which images to be used to determine distances
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    # Get neighbor list for included geometry and merge them
    rijset = set(enforce)
    for image in images:
        image_tree = KDTree(geom[image])
        pairs = image_tree.query_pairs(threshold)
        rijset.update(pairs)
        bonded = image_tree.query_pairs(bond_threshold)
        neighbors = {i: {i} for i in range(geom.shape[1])}
        for i, j in bonded:
            neighbors[i].add(j)
            neighbors[j].add(i)
        for i, j in bonded:
            for ni in neighbors[i]:
                for nj in neighbors[j]:
                    if ni != nj:
                        pair = tuple(sorted([ni, nj]))
                        if pair not in rijset:
                            rijset.add(pair)
    rijlist = sorted(rijset)
    # Check neighbor count to make sure `min_neighbors` is satisfied
    count = np.zeros(geom.shape[1], dtype=int)
    for i, j in rijlist:
        count[i] += 1
        count[j] += 1
    final_geom = geom[-1]
    final_tree = KDTree(final_geom)
    for idx, ct in enumerate(count):
        if ct < min_neighbors:
            _, neighbors = final_tree.query(final_geom[idx], k=min_neighbors + 1)
            for i in neighbors:
                if i == idx:
                    continue
                pair = tuple(sorted([i, idx]))
                if pair in rijset:
                    continue
                else:
                    rijset.add(pair)
                    rijlist.append(pair)
                    count[i] += 1
                    count[idx] += 1
    return rijlist, _equilibrium_distances(rijlist, atoms)


def _get_periodic_bond_list(geom, threshold, min_neighbors, snapshots, bond_threshold, enforce,
                            cell, pbc):
    """Periodic version of `get_bond_list`.  Pairs are `(i, j, n1, n2, n3)` tuples."""
    natoms = geom.shape[1]
    if natoms < 2:
        min_neighbors = 0   # Periodic images of the same atom are always at a fixed distance
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    zero = (0, 0, 0)
    rijset = set(enforce)
    for image in images:
        rijset.update(find_periodic_pairs(geom[image], threshold, cell, pbc))
        bonded = find_periodic_pairs(geom[image], bond_threshold, cell, pbc)
        # neighbors[i] holds (k, n) for every atom k whose image k + n is bonded to atom i
        neighbors = {i: {(i, zero)} for i in range(natoms)}
        for i, j, *shift in bonded:
            neighbors[i].add((j, tuple(shift)))
            neighbors[j].add((i, tuple(-s for s in shift)))
        for i, j, *shift in bonded:
            for ni, si in neighbors[i]:
                for nj, sj in neighbors[j]:
                    if ni != nj:
                        rijset.add(canonical_pair(ni, nj, np.add(shift, sj) - si))
    rijlist = sorted(rijset)
    # Check neighbor count to make sure `min_neighbors` is satisfied
    count = np.zeros(natoms, dtype=int)
    for i, j, *_ in rijlist:
        count[i] += 1
        count[j] += 1
    final_geom = geom[-1]
    for idx, ct in enumerate(count):
        if ct < min_neighbors:
            cutoff = max(threshold, 1.0)
            found = find_periodic_pairs(final_geom, cutoff, cell, pbc, centers=[idx])
            while len(found) < min_neighbors:
                cutoff *= 1.5
                found = find_periodic_pairs(final_geom, cutoff, cell, pbc, centers=[idx])
            for pair in sorted(found, key=found.get)[:min_neighbors]:
                if pair in rijset:
                    continue
                rijset.add(pair)
                rijlist.append(pair)
                count[pair[0]] += 1
                count[pair[1]] += 1
    return rijlist


def _equilibrium_distances(rijlist, atoms):
    """Reference distances used by the scaling functions, from sums of covalent radii."""
    if atoms is None:
        re = np.full(len(rijlist), 2.0)
    else:
        radius = np.array([covalent_radius(atom) for atom in atoms])
        re = np.array([radius[i] + radius[j] for i, j, *_ in rijlist])
    logger.debug("Pair list contain %d pairs", len(rijlist))
    return re


def _pair_vectors(geom, rij_list, cell=None):
    """Atom indices and distance vectors `x_i - x_j` (minus the lattice translation for
    periodic pairs) of a list of pairs."""
    pairs = np.asarray(rij_list, dtype=int).reshape(len(rij_list), -1)
    dvec = geom[pairs[:, 0]] - geom[pairs[:, 1]]
    if pairs.shape[1] == 5:
        if cell is None:
            raise ValueError("Periodic atom pairs require a cell")
        dvec -= pairs[:, 2:] @ np.asarray(cell, dtype=float)
    return pairs[:, :2], dvec


def compute_rij(geom, rij_list, cell=None):
    """Calculate a list of distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: list of indexes of all the atom pairs, `(i, j)` or `(i, j, n1, n2, n3)`
            for periodic pairs with lattice translation `n` of atom j
        cell: Lattice vectors as rows.  Required for periodic pairs.

    Returns:
        rij (array): Array of all the distances.
        bmat (3d array): Cartesian gradients of all the distances."""
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    bmat = np.zeros((nrij, len(geom), 3))
    if nrij == 0:
        return np.zeros(0), bmat
    pairs, dvec = _pair_vectors(geom, rij_list, cell)
    rij = np.linalg.norm(dvec, axis=1)
    grad = dvec / rij[:, None]
    bmat[np.arange(nrij), pairs[:, 0]] = grad
    bmat[np.arange(nrij), pairs[:, 1]] = -grad
    return rij, bmat


def compute_wij(geom, rij_list, func, cell=None):
    """Calculate a list of scaled distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.  The distances are scaled with
    a given scaling function.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: 2d numpy array of indexes of all the atom pairs
        func: A scaling function, which returns both the value and derivative.  Must
            qualify as a numpy Ufunc in order to be broadcasted to array elements.
        cell: Lattice vectors as rows.  Required for periodic pairs.

    Returns:
        wij (array): Array of all the scaled distances.
        bmat (2d array): Cartesian gradients of all the scaled distances, with the
            second dimension flattened (need this to be used in scipy.optimize)."""
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    rij, bmat = compute_rij(geom, rij_list, cell)
    wij, dwdr = func(rij)
    for idx, grad in enumerate(dwdr):
        bmat[idx] *= grad
    return wij, bmat.reshape(nrij, -1)


def compute_rij_sparse(geom, rij_list, cell=None):
    """Calculate distances and sparse Cartesian gradients for selected pairs.

    This is equivalent to :func:`compute_rij`, except the derivative matrix is
    returned as CSR with shape ``(nrij, 3 * natoms)``.
    """
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    if nrij == 0:
        return np.zeros(0), sparse.csr_matrix((0, geom.size))

    pairs, dvec = _pair_vectors(geom, rij_list, cell)
    rij = np.linalg.norm(dvec, axis=1)
    grad = dvec / rij[:, None]

    rows = np.repeat(np.arange(nrij), 6)
    atom_cols = pairs[:, :, None] * 3 + np.arange(3)
    cols = atom_cols.reshape(nrij, 6).ravel()
    data = np.concatenate([grad, -grad], axis=1).ravel()
    bmat = sparse.csr_matrix((data, (rows, cols)), shape=(nrij, geom.size))
    return rij, bmat


def compute_wij_sparse(geom, rij_list, func, cell=None):
    """Calculate scaled distances and sparse Cartesian gradients.

    The scaled coordinate values match :func:`compute_wij`. The derivative
    matrix is CSR with shape ``(nrij, 3 * natoms)``.
    """
    rij, bmat = compute_rij_sparse(geom, rij_list, cell)
    wij, dwdr = func(rij)
    dwdr = np.asarray(dwdr)
    return wij, bmat.multiply(dwdr[:, None]).tocsr()


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        dval = -alpha / re * val1 - val2 / x
        return val1 + val2, dval
    return scaler


def elu_scaler(re=2, alpha=2, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        val1 = (1 - x / re) * alpha + 1
        dval = np.full(x.shape, -alpha / re)
        large = x > re
        v1l = np.exp(alpha * (1 - x[large] / re))
        val1[large] = v1l
        dval[large] = -alpha / re * v1l
        val2 = beta * re / x
        return val1 + val2, dval - val2 / x
    return scaler
