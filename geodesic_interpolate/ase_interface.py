"""Interface to the Atomic Simulation Environment (ASE).

Example:
    >>> from ase.io import read
    >>> from geodesic_interpolate import interpolate
    >>> images = interpolate([read('initial.traj'), read('final.traj')], nimages=9)
    >>> from ase.mep import NEB
    >>> neb = NEB(images)

The cell and periodic flags of the input images are honored.  Atoms may be wrapped differently
in each image: the path is unwrapped using the minimum image convention, so the displacement of
every atom between consecutive images must be shorter than half a lattice vector.

`FixAtoms`, `FixCartesian` and `FixScaled` constraints are enforced: coordinates fixed in all
input images are never changed by the interpolation.  Coordinates fixed in only some images (e.g.
atoms pinned while relaxing one end point) move freely along the path.  The end points of the
returned path keep their own constraints, all other images get the common ones.
"""
import itertools
import logging
import os

import numpy as np
from ase.constraints import FixAtoms, FixCartesian, FixScaled
from ase.io import write

from .coord_utils import fixed_mask
from .interpolation import redistribute, smooth_path


logger = logging.getLogger(__name__)


def _scaled_to_cartesian_mask(cell, mask):
    """Cartesian components fixed by fixing the scaled coordinates in `mask`, or None if the
    fixed subspace is not spanned by Cartesian axes (e.g. a single direction in a skewed cell)."""
    # Scaled coordinate k is the projection onto the k-th reciprocal vector
    reciprocal = np.linalg.inv(np.asarray(cell, dtype=float))[:, mask].T
    if not len(reciprocal):
        return np.zeros(3, dtype=bool)
    basis = np.linalg.svd(reciprocal, full_matrices=False)[2]
    cartesian = np.isclose(np.linalg.norm(basis, axis=0), 1.0, atol=1e-8)
    return cartesian if cartesian.sum() == len(reciprocal) else None


def get_fixed_mask(atoms):
    """Boolean (natoms, 3) mask of the Cartesian coordinates fixed by `FixAtoms`, `FixCartesian`
    and `FixScaled` (selective dynamics of POSCAR files) constraints.  `FixScaled` is supported
    when the fixed directions are Cartesian axes, e.g. fixing a and b of a slab with c along z.
    Other constraint types are not enforced."""
    fixed = np.zeros((len(atoms), 3), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed[constraint.get_indices()] = True
        elif isinstance(constraint, FixCartesian):
            fixed[constraint.index] |= constraint.mask
        elif isinstance(constraint, FixScaled):
            mask = _scaled_to_cartesian_mask(atoms.cell, constraint.mask)
            if mask is None:
                logger.warning("FixScaled constraint with mask %s does not fix Cartesian directions of "
                               "this cell and is not enforced.", constraint.mask)
            else:
                fixed[constraint.index] |= mask
        else:
            logger.warning("Constraint %s is not enforced by the geodesic interpolation.",
                           type(constraint).__name__)
    return fixed


def atoms_to_path(images, use_pbc=True):
    """Extract symbols, coordinates, cell and periodic flags from a list of ASE Atoms.

    Args:
        images:     Sequence of `ase.Atoms` with identical atoms in identical order.
        use_pbc:    If False, ignore the cell and treat the system as a molecule.

    Returns:
        symbols, coords (nimages x natoms x 3), cell (or None), pbc (or None).
        Constraints are obtained separately with `get_fixed_mask`.
    """
    images = list(images)
    if len(images) < 2:
        raise ValueError("Need at least two images.")
    symbols = images[0].get_chemical_symbols()
    for atoms in images[1:]:
        if atoms.get_chemical_symbols() != symbols:
            raise ValueError("All images must contain the same atoms in the same order.")
    coords = np.array([atoms.get_positions() for atoms in images])
    pbc = np.array(images[0].pbc, dtype=bool)
    if not use_pbc or not pbc.any():
        return symbols, coords, None, None
    cell = np.array(images[0].cell)
    for atoms in images[1:]:
        if not np.array_equal(atoms.pbc, pbc) or not np.allclose(np.array(atoms.cell), cell, atol=1e-6):
            raise ValueError("All images must have the same cell and periodic boundary conditions.")
    return symbols, coords, cell, pbc


def get_path_fixed_mask(images):
    """Cartesian coordinates fixed along a path: those fixed in all of the images."""
    masks = [get_fixed_mask(atoms) for atoms in images]
    common = np.logical_and.reduce(masks)
    if any(not np.array_equal(mask, common) for mask in masks):
        logger.info("Constraints differ between the input images.  Only the %d coordinates fixed in all "
                    "images are fixed along the path; end points keep their own constraints.", common.sum())
    return common


def mask_to_constraints(mask):
    """ASE constraints fixing the Cartesian coordinates in a (natoms, 3) mask."""
    constraints = []
    full = np.flatnonzero(mask.all(axis=1))
    if len(full):
        constraints.append(FixAtoms(indices=full))
    partial = np.flatnonzero(mask.any(axis=1) & ~mask.all(axis=1))
    for components in {tuple(mask[i]) for i in partial}:
        indices = [i for i in partial if tuple(mask[i]) == components]
        constraints.append(FixCartesian(indices, mask=components))
    return constraints


def path_to_atoms(images, coords, fixed=None, wrap=False):
    """Create a list of ASE Atoms with the given coordinates.

    Args:
        images:     Input images.  The first and last image are the templates of the first and last
                    image of the path, and keep their constraints.  Interior images are copies of
                    the first image.
        coords:     Coordinates of the path.
        fixed:      (natoms, 3) mask of the coordinates fixed along the path, used as constraints of
                    the interior images.  Defaults to `get_path_fixed_mask(images)`.
        wrap:       Wrap atoms into the cell.
    """
    if fixed is None:
        fixed = get_path_fixed_mask(images)
    interior = mask_to_constraints(fixed)
    path = []
    for i, X in enumerate(coords):
        if i == 0 or i == len(coords) - 1:
            atoms = images[0 if i == 0 else -1].copy()
        else:
            atoms = images[0].copy()
            atoms.set_constraint(interior)
        atoms.set_positions(X, apply_constraint=False)
        if wrap:
            atoms.wrap()
        path.append(atoms)
    return path


def _cartesian_to_scaled(atoms):
    """Copy of `atoms` with `FixCartesian` replaced by equivalent `FixScaled` constraints, since
    only the latter can be written as selective dynamics of POSCAR files."""
    atoms = atoms.copy()
    constraints = []
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixCartesian):
            constraints.append(constraint)
            continue
        for scaled in itertools.product([False, True], repeat=3):
            cartesian = _scaled_to_cartesian_mask(atoms.cell, np.array(scaled))
            if cartesian is not None and np.array_equal(cartesian, constraint.mask):
                constraints.extend(FixScaled(i, scaled, atoms.cell) for i in constraint.index)
                break
        else:
            logger.warning("FixCartesian constraint with mask %s cannot be written as selective "
                           "dynamics for this cell and is dropped.", constraint.mask)
    atoms.set_constraint(constraints)
    return atoms


def write_vasp_neb(images, directory='.'):
    """Write images as VASP NEB input: `directory/00/POSCAR` ... `directory/NN/POSCAR`.

    Positions are written in direct coordinates and in the original atom order.  Constraints
    are written as selective dynamics.
    """
    if len(images) > 100:
        raise ValueError("VASP NEB supports at most 100 image directories (00-99)")
    for i, atoms in enumerate(images):
        atoms = _cartesian_to_scaled(atoms)
        folder = os.path.join(directory, '%02d' % i)
        os.makedirs(folder, exist_ok=True)
        write(os.path.join(folder, 'POSCAR'), atoms, format='vasp', direct=True, sort=False)
    logger.info("Wrote VASP NEB images 00-%02d to %s", len(images) - 1, directory)


def interpolate(images, nimages=17, sweep=None, tol=2e-3, maxiter=15, microiter=20, scaling=1.7,
                friction=1e-2, dist_cutoff=3, use_pbc=True, wrap=False, return_raw=False, fixed=None,
                max_rounds=5, seed=None):
    """Geodesic interpolation between ASE Atoms objects.

    Args:
        images:     Two or more `ase.Atoms`.  Images are added by bisection or removed until
                    `nimages` remain, then the path is smoothed into a geodesic.
        nimages:    Number of images in the final path, including end points.
        sweep:      Optimize one image at a time.  Defaults to True for more than 35 atoms.
        tol:        Convergence tolerance.
        maxiter:    Maximum number of minimization iterations.
        microiter:  Maximum number of micro iterations for the sweeping algorithm.
        scaling:    Exponential parameter for the Morse scaled coordinates.
        friction:   Size of friction term used to prevent very large changes of geometry.
        dist_cutoff: Cut-off value for the distance between a pair of atoms to be included in the
                    coordinate system.
        use_pbc:    Honor the cell and periodic flags of the images (minimum image convention).
        wrap:       Wrap the positions of the returned images back into the cell.  By default the
                    path is returned unwrapped so that atoms move continuously.
        return_raw: Also return the path before geodesic smoothing.
        fixed:      Additional fixed atoms (indices or mask) or (natoms, 3) mask of fixed Cartesian
                    components, combined with the constraints common to all images.
        max_rounds: Maximum number of smoothing rounds; after each round, atom pairs that came into
                    contact without being coordinates are added and the path is smoothed again.
        seed:       Seed for NumPy's global random generator (used for the random nudges of the
                    bisection).  Results also depend on the number of BLAS threads.

    Returns:
        List of `ase.Atoms` with new positions, and the raw path as a second list if `return_raw` is
        True.  End points keep the constraints of the first and last input image, interior images
        get the constraints enforced along the path.
    """
    images = list(images)
    if seed is not None:
        np.random.seed(seed)
    symbols, coords, cell, pbc = atoms_to_path(images, use_pbc)
    mask = get_path_fixed_mask(images)
    extra = fixed_mask(fixed, len(symbols))
    if extra is not None:
        mask |= extra
    raw = redistribute(symbols, coords, nimages, tol=tol * 5, cell=cell, pbc=pbc, fixed=mask)
    smoother = smooth_path(symbols, raw, scaling, threshold=dist_cutoff, friction=friction, sweep=sweep, tol=tol,
                           max_iter=maxiter, micro_iter=microiter, max_rounds=max_rounds, cell=cell, pbc=pbc,
                           fixed=mask)
    path = path_to_atoms(images, smoother.path, mask, wrap)
    if return_raw:
        return path, path_to_atoms(images, raw, mask, wrap)
    return path
