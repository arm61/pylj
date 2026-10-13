"""Pictures and animations of the atoms in the box, for Simulation.view and
Simulation.animate."""

from collections.abc import Iterable, Sequence
from typing import cast

from matplotlib.colors import is_color_like, to_rgba
from matplotlib.typing import ColorType

from pylj.configuration import Configuration
from pylj.model import Model
from pylj.potentials import check_positive_finite
from pylj.sample.panes import _potential_minimum

RGBA = tuple[float, float, float, float]


def _atom_diameters(
    configuration: Configuration, model: Model, diameter: float | Iterable[float] | None
) -> list[float]:
    """Returns the drawn diameter of each atom, in Angstrom.

    Args:
        configuration: The configuration to draw.
        model: The model, which gives each species its default diameter.
        diameter: ``None`` gives each species the separation at the minimum
            of its own pair energy. A single value applies to every atom, and
            a sequence gives one value per atom.

    Returns:
        One diameter per atom.

    Raises:
        ValueError: If a sequence does not have one value per atom, or a
            value is not positive and finite.
    """
    atoms = configuration.number_of_atoms
    if diameter is None:
        defaults = [_potential_minimum(model.potential(one, one)) for one in configuration.species]
        return [defaults[index] for index in configuration.species_index]
    if isinstance(diameter, Iterable):
        values = [float(value) for value in diameter]
        if len(values) != atoms:
            raise ValueError(f"diameter has {len(values)} values, but there are {atoms} atoms")
    else:
        values = [float(diameter)] * atoms
    for value in values:
        check_positive_finite("diameter", value)
    return values


def _atom_colours(
    configuration: Configuration, colour: ColorType | Sequence[ColorType] | None
) -> list[RGBA]:
    """Returns the colour of each atom.

    Args:
        configuration: The configuration to draw.
        colour: ``None`` gives each species its own colour, the first from
            matplotlib's colour cycle for the first species and so on. A
            single matplotlib colour applies to every atom, and a sequence
            gives one colour per atom.

    Returns:
        One colour per atom, as red, green, blue and alpha.

    Raises:
        ValueError: If a colour is not one matplotlib recognises, or a
            sequence does not have one colour per atom.
    """
    atoms = configuration.number_of_atoms
    if colour is None:
        return [to_rgba(f"C{index}") for index in configuration.species_index]
    if isinstance(colour, str) or is_color_like(colour):
        return [to_rgba(cast(ColorType, colour))] * atoms
    values = [to_rgba(one) for one in cast(Sequence[ColorType], colour)]
    if len(values) != atoms:
        raise ValueError(f"colour has {len(values)} values, but there are {atoms} atoms")
    return values
