import os
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

__all__ = [
    "FAR3DHarmonic",
    "FAR3DEigenvector",
    "FAR3DParser",
]

# Note that this current module only handles the FAR3D initial value problem output.


@dataclass
class FAR3DHarmonic:
    """
    Represents a harmonic in the Fourier decomposition of an eigenvector.

    Attributes:
        m (int): Poloidal mode number.
        n (int): Toroidal mode number.
        amplitudes (np.ndarray): Array of amplitudes corresponding to radial points.
    """

    m: int
    n: int
    amplitudes: np.ndarray
    phase: float


@dataclass
class FAR3DEigenvector:
    """
    Stores the details of a specific eigenvector from FAR3D,
    including the eigenvalue, radial coordinates, and a detailed breakdown
    of harmonics with their amplitudes.
    """

    eigenvalue: float
    s_coords: np.ndarray
    harmonics: list[FAR3DHarmonic]
    sort_by_amplitude: bool = True

    def __init__(
        self,
        eigenvalue: float,
        s_coords: np.ndarray,
        harmonics: list[FAR3DHarmonic],
        sort_by_amplitude: bool = True,
    ):
        self.eigenvalue = eigenvalue
        self.s_coords = s_coords
        self.harmonics = harmonics
        self.sort_by_amplitude = sort_by_amplitude

        if self.sort_by_amplitude:
            self.amplitude_sort()

    def amplitude_sort(self):
        """
        Sorts 'harmonics' by the peak amplitude in each harmonic.
        """
        peak_amplitudes = np.zeros(len(self.harmonics))

        # obtain array of peak amplitudes
        count = 0
        for h in self.harmonics:
            peak_amplitudes[count] = np.max(h.amplitudes)
            count += 1

        # sort by peak amplitude
        indices_sorted = np.argsort(peak_amplitudes)

        sorted_harmonics = self.harmonics.copy()

        count = 0
        for i in indices_sorted:
            sorted_harmonics[count] = self.harmonics[i]
            count += 1

        self.harmonics = sorted_harmonics

    def export_to_numpy(
        self, filename: str, num_harmonics: int = None, resolution_step: int = 1
    ):
        """
        Exports the harmonics to a NumPy file.

        Args:
            filename (str): The name of the file to export to.
            num_harmonics (int, optional): The number of harmonics to
                export (in order of peak amplitude if 'sort_by_amplitude' is
                set to true). If None, all harmonics are exported.
        """
        num_harmonics = num_harmonics or len(self.harmonics)
        harmonics_data = {
            "eigenvalue": self.eigenvalue,
            "s_coords": self.s_coords[0:-1:resolution_step],
            "harmonics": np.array(
                [
                    (h.m, h.n, h.amplitudes[0:-1:resolution_step], h.phase)
                    for h in self.harmonics[:num_harmonics]
                ],
                dtype=object,
            ),
        }
        np.save(filename, harmonics_data)
        print(f"Harmonics exported to {filename}")

    @classmethod
    def load_from_numpy(cls, filename: str, sort_by_amplitude: bool = True):
        """
        Loads harmonics from a NumPy file into an FAR3DEigenvector instance.

        Args:
            filename (str): The name of the file to load from.

        Returns:
            FAR3DEigenvector: An instance of FAR3DEigenvector with loaded data.
        """
        harmonics_data = np.load(filename, allow_pickle=True).item()
        harmonics = [
            FAR3DHarmonic(m=m, n=n, amplitudes=amplitudes, phase=phase)
            for m, n, amplitudes, phase in harmonics_data["harmonics"]
        ]
        return cls(
            eigenvalue=harmonics_data["eigenvalue"],
            s_coords=harmonics_data["s_coords"],
            harmonics=harmonics,
            sort_by_amplitude=sort_by_amplitude,
        )


@dataclass
class FAR3DParser:
    """
    A class to handle the parsing and storage of eigenmode data from
    FAR3D. It currently parses the 'phi_####' files for eigenvectors
    and the corresponding modes.
    """

    @classmethod
    def from_file(cls, filename, eigenvalue):
        """
        Load harmonics data from a file.

        Parameters:
        -----------
        filename : str
            Path to the file containing harmonic data
        eigenvalue : str
            The growth rate (imaginary component of the eigenvalue) from
            FAR3D 'tmp_grwth_omega'.

        Returns:
        --------
        Harmonics
            A new Harmonics instance with data loaded from the file
        """
        harmonics = []

        with open(filename) as f:
            head = f.readline()

        phiarr = np.loadtxt(filename, skiprows=1)

        columns = head.split("\t")
        for nc, c in enumerate(columns):
            # special case for the first column
            if c == "r":
                # convert from rho to s
                s_coords = np.array(phiarr[:, nc] ** 2)
                # skips the current loop iteration
                continue

            m = int(c.split(" ")[-2][:-1])
            n = int(c.split(" ")[-1][0])

            phase = 0.0
            if c[0] == "R":
                # cosine
                phase = np.pi / 2.0
            elif c[0] == "I":
                # sine
                phase = 0.0
            else:
                raise ValueError("Failed to parse first line")

            temp_harmonic = FAR3DHarmonic(
                m, n, amplitudes=np.array(phiarr[:, nc]), phase=phase
            )

            harmonics.append(temp_harmonic)

        return FAR3DEigenvector(
            eigenvalue=eigenvalue, s_coords=s_coords, harmonics=harmonics
        )
