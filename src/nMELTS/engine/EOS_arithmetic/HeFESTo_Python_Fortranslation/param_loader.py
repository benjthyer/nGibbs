from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .param_state import (
    DEFAULT_PARAMETER_DIR,
    HeFESToParameterRecord,
    PARAMETER_FIELD_NAMES,
    _load_hefesto_parameter_directory_uncached,
)


class ParameterStore:
    """Singleton-backed container for HeFESTo parameter records.

    Provides cached, convenient access to parsed parameter files and a
    numeric `apar` array suitable for inserting into `HeFESToState`.
    """

    def __init__(self, directory: Optional[Path | str] = None) -> None:
        directory = Path(directory) if directory is not None else DEFAULT_PARAMETER_DIR
        self.directory = Path(directory)
        self._records_by_species: Dict[str, HeFESToParameterRecord] = {}
        self._load_records()

    def _load_records(self) -> None:
        raw = _load_hefesto_parameter_directory_uncached(self.directory)
        # Convert to mapping by species_label for stable lookups
        for fname, rec in raw.items():
            key = rec.species_label
            self._records_by_species[key] = rec

    @property
    def species(self) -> List[str]:
        return sorted(self._records_by_species.keys())

    @property
    def npar(self) -> int:
        return len(PARAMETER_FIELD_NAMES)

    @property
    def nspecies(self) -> int:
        return len(self._records_by_species)

    def get_record(self, species_label: str) -> HeFESToParameterRecord:
        return self._records_by_species[species_label]

    def build_apar(self, species_list: Optional[List[str]] = None, nspecp: Optional[int] = None) -> np.ndarray:
        """Return an `apar` array (ns x npar).

        - `species_list`: if provided, order rows according to this list; missing
          species will raise KeyError.
        - `nspecp`: if provided, pad the matrix to `nspecp` rows with zeros.
        """
        if species_list is None:
            species_list = self.species

        nrows = len(species_list)
        ncols = self.npar
        apar = np.zeros((nrows, ncols), dtype=float)
        for i, s in enumerate(species_list):
            rec = self.get_record(s)
            apar[i, : len(rec.values)] = np.asarray(rec.values, dtype=float)

        if nspecp is not None and nspecp > nrows:
            pad = np.zeros((nspecp - nrows, ncols), dtype=float)
            apar = np.vstack([apar, pad])

        return apar


@lru_cache(maxsize=1)
def get_parameter_store(directory: Optional[str] = None) -> ParameterStore:
    """Return a cached `ParameterStore` for the given directory.

    Call with no argument to use the packaged `DEFAULT_PARAMETER_DIR`.
    """
    return ParameterStore(directory)
