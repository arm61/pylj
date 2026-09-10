"""The placement of initial configurations."""

import math
from typing import NamedTuple

import numpy as np

from pylj.configuration import Configuration
from pylj.model import Model
from pylj.potentials import Species, check_positive_finite
from pylj.simulation import _check_potentials_at_the_cut_off, _resolve_cut_off

#: Number of trial positions tried for a single atom by Metropolis
#: placement before it gives up and raises ``ValueError``.
PLACEMENT_ATTEMPTS = 1000


def place_square(number_of_atoms: int, species: tuple[Species, ...], box: float) -> Configuration:
    """Place atoms on a square lattice.

    The lattice has ``ceil(sqrt(number_of_atoms))`` sites along each side of
    the box, and the atoms fill those sites in order, taking the species in
    turn. On a lattice with an even number of sites per side, a mixture
    therefore starts out in stripes of one species and then the other.
    Diffusion mixes them over the course of the run. No check is made for
    overlapping atoms. A lattice packed too tightly for the potential
    stores a large potential energy, and for a potential with a hard core that
    energy is infinite.

    Args:
        number_of_atoms: The number of atoms.
        species: The species, assigned to the atoms in turn.
        box: The side length of the box, in metres.

    Returns:
        The configuration.
    """
    m = int(np.ceil(np.sqrt(number_of_atoms)))
    spacing = box / m
    sites = [((i + 0.5) * spacing, (j + 0.5) * spacing) for i in range(m) for j in range(m)]
    position = np.array(sites[:number_of_atoms], dtype=float).reshape(-1, 2)
    species_index = np.arange(number_of_atoms) % len(species)
    return Configuration(position, species, species_index, box)


#: The rows of a triangular lattice sit ``sqrt(3) / 2`` of a column spacing
#: apart.
TRIANGULAR_RATIO = math.sqrt(3) / 2


class Lattice(NamedTuple):
    """A triangular lattice fitted to a square box.

    Attributes:
        columns: The number of columns.
        rows: The number of rows.
        strain: How far the ratio of the row spacing to the column spacing
            sits from :data:`TRIANGULAR_RATIO`, as a fraction of it. The six
            neighbours of an atom split into two distances that differ by
            about three quarters of the strain.
    """

    columns: int
    rows: int
    strain: float


#: The largest strain :func:`place_triangular` accepts. Beyond a third the
#: six neighbours of an atom split by more than a quarter, and the lattice is
#: no longer a triangular one.
MOST_STRAIN = 1 / 3


def _triangular_lattice(number_of_atoms: int) -> Lattice | None:
    """Return the columns, rows and strain of the best triangular lattice.

    The lattice has ``columns * rows`` sites and an even number of rows. Its
    strain is how far the ratio of its row spacing to its column spacing sits
    from :data:`TRIANGULAR_RATIO`, as a fraction of that ratio, and the best
    lattice is the one with the smallest strain. The return is ``None`` when
    no even number of rows divides ``number_of_atoms``.
    """
    best: Lattice | None = None
    for rows in range(2, number_of_atoms + 1, 2):
        if number_of_atoms % rows:
            continue
        columns = number_of_atoms // rows
        strain = abs(columns / rows / TRIANGULAR_RATIO - 1)
        if best is None or strain < best.strain:
            best = Lattice(columns, rows, strain)
    return best


def place_triangular(
    number_of_atoms: int,
    species: tuple[Species, ...],
    box: float,
    max_strain: float = 0.05,
) -> Configuration:
    """Place atoms on a triangular lattice that fills the box.

    A triangular lattice gives every atom six neighbours at one distance,
    which is the arrangement a two-dimensional solid settles into. Fitting
    one to a square box strains it a little, so the six split into two
    distances differing by about three quarters of the strain. Each row is
    offset along x by half a column spacing from the row below it, and there
    is an even number of rows so that the offset keeps alternating across
    the periodic boundary.

    The lattice fills the box, so the number of atoms has to be a number of
    columns times an even number of rows. For the six to stay close to one
    distance, the ratio of columns to rows has to be close to
    ``sqrt(3) / 2``, and ``max_strain`` is the largest fraction of that
    ratio a lattice may sit away from it, at most :data:`MOST_STRAIN`. At
    the default the counts up to 300 that fit are 30, 56, 90, 120, 168, 224,
    270 and 288.

    The atoms fill the sites row by row, taking the species in turn. On a
    lattice with an even number of columns, a mixture therefore starts out
    in stripes of one species and then the other. Diffusion mixes them over
    the course of the run. No check is made for overlapping atoms: a lattice
    packed too tightly for the potential stores a large potential energy,
    and for a potential with a hard core that energy is infinite.

    Args:
        number_of_atoms: The number of atoms.
        species: The species, assigned to the atoms in turn.
        box: The side length of the box, in metres.
        max_strain: How far the fitted lattice may sit from
            ``sqrt(3) / 2``, as a fraction of that ratio.

    Returns:
        The configuration.

    Raises:
        ValueError: If ``max_strain`` is not positive and finite or is
            above :data:`MOST_STRAIN`, or no lattice within ``max_strain``
            has this many sites.
    """
    check_positive_finite("max_strain", max_strain)
    if max_strain > MOST_STRAIN:
        raise ValueError(
            f"max_strain of {max_strain:.3f} is above {MOST_STRAIN:.3f}, beyond which the six "
            "neighbours of an atom split by more than a quarter and the lattice is no longer a "
            "triangular one. max_strain is a fraction, so 0.05 is five per cent."
        )
    best = _triangular_lattice(number_of_atoms)
    if best is None or best.strain > max_strain:
        raise ValueError(_no_lattice_message(number_of_atoms, best, max_strain))
    columns, rows = best.columns, best.rows
    column_spacing = box / columns
    row_spacing = box / rows
    sites = [
        ((i + 0.5 * (j % 2)) * column_spacing, (j + 0.5) * row_spacing)
        for j in range(rows)
        for i in range(columns)
    ]
    position = np.array(sites, dtype=float)
    species_index = np.arange(number_of_atoms) % len(species)
    return Configuration(position, species, species_index, box)


def _odd_row_lattice(number_of_atoms: int) -> Lattice | None:
    """Return the best lattice an odd number of rows would give.

    ``None`` when no odd number of rows divides ``number_of_atoms``, or when
    the best one is no better than the best even-row lattice.
    """
    best: Lattice | None = None
    for rows in range(1, number_of_atoms + 1, 2):
        if number_of_atoms % rows:
            continue
        columns = number_of_atoms // rows
        strain = abs(columns / rows / TRIANGULAR_RATIO - 1)
        if best is None or strain < best.strain:
            best = Lattice(columns, rows, strain)
    even = _triangular_lattice(number_of_atoms)
    if best is None or (even is not None and even.strain <= best.strain):
        return None
    return best


def _no_lattice_message(
    number_of_atoms: int, best: Lattice | None, max_strain: float
) -> str:
    """Say why a triangular lattice was refused and which counts would fit."""
    if best is None:
        reason = (
            f"{number_of_atoms} atoms cannot fill a triangular lattice: it needs a number "
            "of columns times an even number of rows"
        )
    else:
        reason = (
            f"{number_of_atoms} atoms fill a {best.columns} column by {best.rows} row "
            f"triangular lattice, straining it by {best.strain:.3f}, above max_strain "
            f"of {max_strain:.3f}"
        )
    odd = _odd_row_lattice(number_of_atoms)
    if odd is not None and odd.strain <= max_strain:
        reason += (
            f", and the {odd.columns} by {odd.rows} lattice that would fit has an odd number "
            "of rows"
        )
    nearby = []
    for direction in (-1, 1):
        candidate = number_of_atoms
        for _ in range(300):
            candidate += direction
            if candidate < 2:
                break
            fit = _triangular_lattice(candidate)
            if fit is not None and fit.strain <= max_strain:
                nearby.append(candidate)
                break
    advice = []
    if nearby:
        advice.append(f"use {' or '.join(str(one) for one in sorted(nearby))} atoms")
    if best is not None and best.strain <= MOST_STRAIN:
        advice.append("raise max_strain")
    if not advice:
        return f"{reason}."
    return f"{reason}. To go on, {' or '.join(advice)}."


def place_metropolis(
    number_of_atoms: int,
    box: float,
    model: Model,
    cut_off: float,
    placement_temperature: float,
    rng: np.random.Generator,
) -> Configuration:
    """Place atoms one at a time by Metropolis insertion.

    Each atom in turn is given a uniform trial position in the box,
    accepted by :func:`mc.accept` at ``placement_temperature`` on its
    interaction energy with the atoms already placed, and redrawn on
    rejection. Before the atom is added it interacts with nothing, so its
    interaction energy with the atoms already placed is exactly the energy
    change the insertion causes.

    Placing the atoms one after another gives a reasonable starting point,
    not a configuration drawn from equilibrium. The atoms avoid the close
    contacts the potential penalises at the placement temperature, and raising
    that temperature makes closer contacts more likely. The run itself brings
    the configuration to equilibrium.

    Args:
        number_of_atoms: The number of atoms.
        box: The side length of the box, in metres.
        model: The species, assigned to the atoms in turn, and the potential
            between each pair of them.
        cut_off: The cut-off, in metres.
        placement_temperature: The temperature of the acceptance, in kelvin.
        rng: The generator to draw trial positions and acceptances from.

    Returns:
        The configuration.

    Raises:
        ValueError: If :data:`PLACEMENT_ATTEMPTS` trial positions are all
            rejected for a single atom. At the highest densities this many
            attempts can reach, success depends on the positions that happen to
            be drawn, so the same call may succeed with one seed and raise with
            another.
    """
    # mc imports this module for place, so the acceptance criterion is
    # imported here rather than at the top of the module.
    from pylj.mc import accept

    species = model.species
    species_index = np.arange(number_of_atoms) % len(species)
    placed = Configuration(np.zeros((0, 2)), species, species_index[:0], box)
    for i in range(number_of_atoms):
        for _attempt in range(PLACEMENT_ATTEMPTS):
            trial = rng.uniform(0, box, size=2)
            energy = placed.insertion_energy(trial, int(species_index[i]), model, cut_off)
            if accept(energy, placement_temperature, rng=rng):
                placed = placed.replace(
                    position=np.vstack([placed.position, trial]),
                    species_index=species_index[: i + 1],
                )
                break
        else:
            raise ValueError(
                f"Could not place atom {i + 1} of {number_of_atoms} in a "
                f"{box * 1e10:.1f} Angstrom box at a placement temperature of "
                f"{placement_temperature:g} K after {PLACEMENT_ATTEMPTS} attempts; "
                "reduce the number of atoms or use a larger box; for a soft "
                "potential, raising placement_temperature tolerates closer contacts."
            )
    return placed


def place(
    number_of_atoms: int,
    temperature: float,
    box: float,
    *,
    model: Model,
    init_conf: str,
    placement_temperature: float | None,
    max_strain: float,
    cut_off: float | None,
    rng: np.random.Generator,
) -> tuple[Configuration, float]:
    """Build the initial configuration for a simulation factory.

    Takes the box and cut-off in Angstrom, as the factories do, checks the
    potentials at the cut-off before any placement is attempted, and places
    the atoms.

    Args:
        number_of_atoms: The number of atoms.
        temperature: The temperature of the run, in kelvin.
        box: The side length of the box, in Angstrom, from 4 to 600.
        model: The species, assigned to the atoms in turn, and the potential
            between each pair of them.
        init_conf: ``'square'`` for a square lattice, ``'triangular'`` for a
            triangular one, or ``'metropolis'`` for sequential Metropolis
            insertion.
        placement_temperature: The temperature of the Metropolis acceptance
            used by ``'metropolis'``, in kelvin; ``None`` for the run
            temperature.
        max_strain: How far the fitted lattice may sit from
            ``sqrt(3) / 2``, as a fraction of that ratio, when ``'triangular'``
            fits its lattice to the box.
        cut_off: The cut-off, in Angstrom; ``None`` for 15 Angstrom or half
            the box, whichever is smaller.
        rng: The generator for Metropolis placement.

    Returns:
        The configuration and the cut-off, both in metres.

    Raises:
        ValueError: If no atoms are requested, a temperature is
            not positive and finite, the box is outside 4 to 600 Angstrom,
            the cut-off exceeds half the box, a potential has not died away
            at the cut-off, ``init_conf`` is unknown, ``max_strain`` is not
            positive and finite or is above :data:`MOST_STRAIN`, no
            triangular lattice within ``max_strain`` has this many sites, or
            Metropolis placement exhausts its trial budget.
    """
    if number_of_atoms < 1:
        raise ValueError("A simulation needs at least one atom")
    check_positive_finite("temperature", temperature)
    if placement_temperature is None:
        placement_temperature = temperature
    check_positive_finite("placement_temperature", placement_temperature)
    if not 4 <= box <= 600:
        raise ValueError(
            f"box must be between 4 and 600 Angstrom, not {box}: below 4 the cell cannot "
            "hold more than one atom, and above 600 the atoms are too small to be "
            "seen in the viewer."
        )
    box_m = box * 1e-10
    cut_off_m = _resolve_cut_off(box_m, None if cut_off is None else cut_off * 1e-10)
    _check_potentials_at_the_cut_off(model, cut_off_m, temperature, box_m)
    if init_conf == "square":
        configuration = place_square(number_of_atoms, model.species, box_m)
    elif init_conf == "triangular":
        configuration = place_triangular(number_of_atoms, model.species, box_m, max_strain)
    elif init_conf == "metropolis":
        configuration = place_metropolis(
            number_of_atoms,
            box_m,
            model,
            cut_off_m,
            placement_temperature,
            rng,
        )
    else:
        raise ValueError(
            f"init_conf must be 'square', 'triangular' or 'metropolis', not {init_conf!r}"
        )
    return configuration, cut_off_m
