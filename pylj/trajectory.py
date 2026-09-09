"""The configurations a simulation samples, kept in order for analysis
after the run."""

from collections.abc import Iterable, Iterator
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylj import pairwise
from pylj.configuration import Configuration
from pylj.scattering import bin_centres, bin_counts, debye_sum, max_separation


class Trajectory:
    """The configurations a simulation has sampled, in order.

    A simulation appends its current configuration each time ``sample`` is
    called, so a trajectory holds one frame per sample. Every frame comes
    from the same system: the same box and the same number of atoms. A frame
    is a :class:`~pylj.configuration.Configuration`, so ``trajectory[i]`` has
    positions, a box and, for molecular dynamics, velocities. A slice such as
    ``trajectory[100:]`` is a trajectory of the later frames.

    Args:
        frames: Configurations to start with, in order.
    """

    def __init__(self, frames: Iterable[Configuration] = ()) -> None:
        self._frames: list[Configuration] = []
        for one in frames:
            self.append(one)

    def append(self, configuration: Configuration) -> None:
        """Add a frame to the end.

        Args:
            configuration: The frame to add.

        Raises:
            ValueError: If the frame's box or number of atoms differs from
                the first frame's.
        """
        if self._frames:
            first = self._frames[0]
            if configuration.box != first.box:
                raise ValueError(
                    f"The frame's box is {configuration.box} m but the trajectory's "
                    f"is {first.box} m"
                )
            if configuration.number_of_atoms != first.number_of_atoms:
                raise ValueError(
                    f"The frame has {configuration.number_of_atoms} atoms but the trajectory "
                    f"has {first.number_of_atoms}"
                )
        self._frames.append(configuration)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Configuration]:
        return iter(self._frames)

    @overload
    def __getitem__(self, index: int) -> Configuration: ...

    @overload
    def __getitem__(self, index: slice) -> "Trajectory": ...

    def __getitem__(self, index: int | slice) -> "Configuration | Trajectory":
        if isinstance(index, slice):
            return Trajectory(self._frames[index])
        return self._frames[index]

    @property
    def position(self) -> NDArray[np.float64]:
        """The positions of every frame, shape ``(frames, N, 2)``, in metres."""
        if not self._frames:
            return np.empty((0, 0, 2))
        return np.stack([one.position for one in self._frames])

    def __repr__(self) -> str:
        if not self._frames:
            return "Trajectory(no frames)"
        atoms = self._frames[0].number_of_atoms
        return f"Trajectory({len(self._frames)} frames, {atoms} atoms)"

    def rdf(
        self, bins: int = 100, r_max: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return g(r) averaged over the frames.

        Each frame's :meth:`~pylj.configuration.Configuration.rdf` is taken
        and the mean over frames returned.

        Args:
            bins: The number of bins.
            r_max: The largest distance binned, in metres; by default half
                the box.

        Returns:
            The centre of each bin, in metres, and the mean g(r) at each.

        Raises:
            ValueError: If the trajectory has no frames.
        """
        self._check_frames()
        curves = [one.rdf(bins, r_max) for one in self._frames]
        return curves[0][0], np.mean([gr for _, gr in curves], axis=0)

    def scattering(self, q: ArrayLike, bins: int | None = None) -> NDArray[np.float64]:
        """Return I(q) averaged over the frames.

        Args:
            q: The magnitudes of the scattering vector, in 1/m.
            bins: If given, the pair distances of every frame are binned
                together and the sum is over bins rather than pairs, as in
                :meth:`~pylj.configuration.Configuration.scattering`. One
                sum then serves the whole trajectory, which is much faster
                than summing each frame in turn, at the cost of taking every
                distance in a bin to be at the bin centre. By default every
                pair of every frame is summed exactly.

        Returns:
            The mean I(q) at each value of ``q``.

        Raises:
            ValueError: If the trajectory has no frames.
        """
        self._check_frames()
        if bins is None:
            return np.mean([one.scattering(q) for one in self._frames], axis=0)
        first = self._frames[0]
        r_max = max_separation(first.box)
        total = np.zeros(bins)
        for one in self._frames:
            distance, _ = pairwise.dist(one.position, one.box)
            total += bin_counts(distance, bins, r_max)
        centre = bin_centres(bins, r_max)
        return debye_sum(centre, q, first.number_of_atoms, weight=total / len(self._frames))

    def _check_frames(self) -> None:
        if not self._frames:
            raise ValueError("The trajectory has no frames: call sample() during the run")
