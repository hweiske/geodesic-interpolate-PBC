"""Quality checks for interpolated paths.

Geodesic interpolation only sees interatomic distances at discrete images, so a path can be short in
its metric and still be unphysical.  The checks here look for the failure modes seen in practice:

  - bond stretch: a bond present in both end points is torn along the path (raw, unwrapped positions,
    so a periodic-image error shows up as a bond across the box),
  - close contact: atoms that are more than three bonds apart come too close (one part of a molecule
    pushed into another, or into a surface),
  - bond crossing: two bonds of a molecule pass through each other between images; atom-atom distances
    can stay normal while this happens, so points along the bonds are compared on sub-steps between
    images,
  - large steps between consecutive images.

Usage:  geodesic_analyze band.traj [more.traj ...]
"""
import argparse

import numpy as np
from scipy.spatial import KDTree

from .coord_utils import align_path, get_radii, get_vdw_radii
from .pbc import find_periodic_pairs, normalize_cell


BOND_SCALE = 1.2           # bonded: distance below BOND_SCALE * sum of covalent radii
BOND_POINTS = np.linspace(0.1, 0.9, 5)
THRESHOLDS = dict(stretch=1.3, contact=0.5, crossing=0.8)


def _pairs_within(X, cutoff, cell, pbc):
    """Dict {(i, j, n1, n2, n3): distance} of pairs closer than `cutoff` (n is zero without a cell)."""
    if cell is None:
        tree = KDTree(X)
        return {(i, j, 0, 0, 0): np.linalg.norm(X[i] - X[j]) for i, j in tree.query_pairs(cutoff)}
    return find_periodic_pairs(X, cutoff, cell, pbc)


def _bonds(X, radii, cell, pbc):
    return {pair for pair, d in _pairs_within(X, BOND_SCALE * 2 * radii.max(), cell, pbc).items()
            if d < BOND_SCALE * (radii[pair[0]] + radii[pair[1]])}


def _finite_fragment_bonds(bonds, natoms):
    """Bonds of finite fragments (molecules); periodic networks such as a slab are excluded, since
    a bond graph that wraps around the cell has no meaningful notion of bonds crossing."""
    neighbors = [[] for _ in range(natoms)]
    for i, j, *n in bonds:
        neighbors[i].append((j, np.array(n)))
        neighbors[j].append((i, -np.array(n)))
    offset = np.zeros((natoms, 3), dtype=int)
    component = np.full(natoms, -1)
    periodic = set()
    for root in range(natoms):
        if component[root] >= 0:
            continue
        component[root] = root
        queue = [root]
        while queue:
            i = queue.pop()
            for j, n in neighbors[i]:
                if component[j] < 0:
                    component[j], offset[j] = root, offset[i] + n
                    queue.append(j)
                elif not np.array_equal(offset[j], offset[i] + n):
                    periodic.add(root)
    return [bond for bond in bonds if component[bond[0]] not in periodic]


def _near_in_graph(bonds, natoms, depth=3):
    """Pairs of atoms at most `depth` bonds apart."""
    neighbors = [set() for _ in range(natoms)]
    for i, j, *_ in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    near = set()
    for i in range(natoms):
        seen = frontier = {i}
        for _ in range(depth):
            frontier = {k for f in frontier for k in neighbors[f]} - seen
            seen = seen | frontier
        near.update((min(i, k), max(i, k)) for k in seen)
    return near


def analyze_path(path, symbols, cell=None, pbc=None, substeps=10):
    """Quality metrics of a path whose first and last images are the end points.

    Args:
        path:       Unwrapped Cartesian coordinates, nimages x natoms x 3.
        symbols:    Atom symbols.
        cell, pbc:  Periodic cell and flags (ASE convention), None for molecules.
        substeps:   Sub-steps per interval between images for the bond-crossing check.

    Returns:
        Dict with the worst value of each metric and where it occurs:
          stretch         largest ratio of a bond (present in both end points) to its longer end-point length
          contact         smallest ratio of a non-bonded distance (atoms > 3 bonds apart) to its reference,
                          the smaller of the sum of van der Waals radii and the distance in the end points
          contact_dist    that distance in Angstrom
          crossing        smallest distance between points along two bonds sharing no atom (molecules only)
          max_step        largest displacement of an atom between consecutive images
          length          Cartesian length of the path (sum of image-to-image distances)
    """
    path = np.asarray(path, dtype=float)
    natoms = path.shape[1]
    cell, pbc = normalize_cell(cell, pbc)
    radii, vdw = get_radii(symbols), get_vdw_radii(symbols)
    lattice = np.zeros((3, 3)) if cell is None else cell
    bonds_start, bonds_end = _bonds(path[0], radii, cell, pbc), _bonds(path[-1], radii, cell, pbc)
    persistent = sorted(bonds_start & bonds_end)
    near = _near_in_graph(bonds_start | bonds_end, natoms)
    result = dict(images=len(path))

    # Bond stretch on raw positions with the end-point lattice translation of each bond
    if persistent:
        pairs = np.array([bond[:2] for bond in persistent])
        shifts = np.array([bond[2:] for bond in persistent], dtype=float) @ lattice
        lengths = np.linalg.norm(path[:, pairs[:, 0]] - path[:, pairs[:, 1]] - shifts, axis=2)
        ratio = lengths / np.maximum(lengths[0], lengths[-1])
        k, b = np.unravel_index(np.argmax(ratio), ratio.shape)
        result.update(stretch=float(ratio[k, b]), stretch_image=int(k), stretch_bond=tuple(map(int, pairs[b])),
                      stretch_dist=float(lengths[k, b]))
    else:
        result.update(stretch=1.0, stretch_image=0, stretch_bond=None, stretch_dist=0.0)

    # Close contacts between atoms more than three bonds apart
    cutoff = 0.9 * 2 * vdw.max()
    best = (np.inf, None, None, None)
    for k, X in enumerate(path):
        for (i, j, *n), d in _pairs_within(X, cutoff, cell, pbc).items():
            if (min(i, j), max(i, j)) in near:
                continue
            shift = np.dot(n, lattice)
            ends = min(np.linalg.norm(path[0, i] - path[0, j] - shift), np.linalg.norm(path[-1, i] - path[-1, j] - shift))
            ratio = d / min(vdw[i] + vdw[j], ends)
            if ratio < best[0]:
                best = (ratio, d, k, (i, j))
    result.update(contact=float(best[0]), contact_dist=float(best[1]) if best[1] is not None else np.inf,
                  contact_image=best[2], contact_pair=best[3])

    # Bonds of molecules passing through each other, checked on sub-steps between images
    molecular = _finite_fragment_bonds(persistent, natoms)
    best = (np.inf, None, None)
    if len(molecular) > 1:
        pairs = np.array([bond[:2] for bond in molecular])
        shifts = np.array([bond[2:] for bond in molecular], dtype=float) @ lattice
        owner = np.repeat(np.arange(len(molecular)), len(BOND_POINTS))
        u = BOND_POINTS[None, :, None]
        for k in range(len(path) - 1):
            for t in np.linspace(0, 1, substeps + 1)[:-1 if k < len(path) - 2 else None]:
                X = (1 - t) * path[k] + t * path[k + 1]
                points = (X[pairs[:, 0], None] * (1 - u) + (X[pairs[:, 1]] + shifts)[:, None] * u).reshape(-1, 3)
                for (p, q, *_), d in _pairs_within(points, 1.5, cell, pbc).items():
                    if d < best[0] and not set(pairs[owner[p]]) & set(pairs[owner[q]]):
                        best = (d, k + t, (tuple(map(int, pairs[owner[p]])), tuple(map(int, pairs[owner[q]]))))
    result.update(crossing=float(best[0]), crossing_at=best[1], crossing_bonds=best[2])

    steps = np.linalg.norm(np.diff(path, axis=0), axis=2)
    result.update(max_step=float(steps.max()) if len(steps) else 0.0,
                  length=float(np.linalg.norm(np.diff(path, axis=0).reshape(len(path) - 1, -1), axis=1).sum()))
    return result


def violations(result, thresholds=None):
    """Names of the checks a path fails."""
    limits = dict(THRESHOLDS, **(thresholds or {}))
    failed = []
    if result['stretch'] > limits['stretch']:
        failed.append('torn bond')
    if result['contact'] < limits['contact']:
        failed.append('close contact')
    if result['crossing'] < limits['crossing']:
        failed.append('bond crossing')
    return failed


def format_report(rows, symbols, thresholds=None):
    """Table of `analyze_path` results; `rows` is a list of (label, result)."""
    name = lambda i: '%s%d' % (symbols[i], i)
    width = max([14] + [len(label) for label, _ in rows])
    lines = ['%-*s %8s %8s %10s %9s %8s  %s' % (width, 'run', 'stretch', 'contact', 'crossing', 'max step', 'length',
                                                'verdict'),
             '%-*s %8s %8s %10s %9s %8s' % (width, '', '(ratio)', '(ratio)', '(A)', '(A)', '(A)')]
    for label, r in rows:
        failed = violations(r, thresholds)
        detail = []
        if 'torn bond' in failed:
            detail.append('%s-%s %.2f A at image %d' % (name(r['stretch_bond'][0]), name(r['stretch_bond'][1]),
                                                        r['stretch_dist'], r['stretch_image']))
        if 'close contact' in failed:
            detail.append('%s...%s %.2f A at image %d' % (name(r['contact_pair'][0]), name(r['contact_pair'][1]),
                                                          r['contact_dist'], r['contact_image']))
        if 'bond crossing' in failed:
            (a, b), (c, d) = r['crossing_bonds']
            detail.append('%s-%s x %s-%s %.2f A near image %.1f' % (name(a), name(b), name(c), name(d),
                                                                    r['crossing'], r['crossing_at']))
        verdict = 'ok' if not failed else 'FAIL: ' + '; '.join(detail)
        lines.append('%-*s %8.2f %8.2f %10.2f %9.2f %8.1f  %s' % (
            width, label, r['stretch'], r['contact'], r['crossing'], r['max_step'], r['length'], verdict))
    return '\n'.join(lines)


def main():
    """Command line entry point: analyze one or more band files."""
    from ase.io import read
    from .ase_interface import atoms_to_path

    ps = argparse.ArgumentParser(description="Check interpolated paths for torn bonds, close contacts and "
                                 "bonds passing through each other.  The first and last image of each file are "
                                 "taken as end points.")
    ps.add_argument("filename", nargs='+', help="Path files in any format ASE can read")
    ps.add_argument("--stretch", type=float, default=THRESHOLDS['stretch'],
                    help="Maximum ratio of a bond to its end-point length")
    ps.add_argument("--contact", type=float, default=THRESHOLDS['contact'],
                    help="Minimum ratio of a non-bonded distance to the smaller of the sum of van der Waals radii "
                    "and its distance in the end points")
    ps.add_argument("--crossing", type=float, default=THRESHOLDS['crossing'],
                    help="Minimum distance (A) between points along two bonds sharing no atom")
    args = ps.parse_args()
    thresholds = dict(stretch=args.stretch, contact=args.contact, crossing=args.crossing)
    rows = []
    for filename in args.filename:
        images = read(filename, index=':')
        symbols, coords, cell, pbc = atoms_to_path(images)
        _, coords = align_path(coords, cell, pbc, rigid=False, radii=get_radii(symbols))
        rows.append((filename, analyze_path(coords, symbols, cell, pbc)))
    print(format_report(rows, symbols, thresholds))
    if any(violations(r, thresholds) for _, r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
