"""
Common software interface for all V2X dataset adapters.

The canonical schema defines WHAT every adapter must return.
This base adapter defines HOW the rest of the code interacts with
every dataset.

Every dataset-specific adapter must implement:

    1. __len__()
    2. get_sample(index)

and get_sample(index) must return a CanonicalV2XSample.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from project_paths import resolve_path

from schema import CanonicalV2XSample


class BaseAdapter(ABC):
    """
    Base class for all dataset-specific V2X adapters.

    Examples of subclasses:

        UrbanIngAdapter
        TUMTrafAdapter
        V2XPnPAdapter
        DeepAccidentAdapter

    Regardless of the original dataset format, every subclass must return
    the same CanonicalV2XSample structure.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        dataset_root:
            Root directory of the original dataset.

        split:
            Optional dataset split such as:
                "train"
                "val"
                "test"

            Some datasets may organize their data differently, so this is
            optional rather than mandatory.
        """

        self.dataset_root = resolve_path(dataset_root)
        self.split = split

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of canonical samples available through
        this adapter.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> CanonicalV2XSample:
        """
        Read one sample from the native dataset and convert it into
        the canonical V2X representation.

        A dataset-specific implementation is responsible for:

            1. Finding the correct frame/timestamp.
            2. Loading agent information.
            3. Loading LiDAR data.
            4. Loading camera data.
            5. Loading calibration and poses.
            6. Loading annotations.
            7. Converting dataset-specific geometry and conventions
               into the canonical conventions.
            8. Constructing a CanonicalV2XSample.

        Parameters
        ----------
        index:
            Integer sample index.

        Returns
        -------
        CanonicalV2XSample
            Sample following the common schema defined in schema.py.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common behaviour
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> CanonicalV2XSample:
        """
        Convenience interface allowing:

            sample = adapter[index]

        rather than:

            sample = adapter.get_sample(index)

        This method also validates the returned canonical sample.
        """

        if not isinstance(index, int):
            raise TypeError(
                f"Sample index must be an integer, got {type(index).__name__}."
            )

        num_samples = len(self)

        # Support normal Python negative indexing.
        if index < 0:
            index += num_samples

        if index < 0 or index >= num_samples:
            raise IndexError(
                f"Sample index {index} is out of range for "
                f"dataset with {num_samples} samples."
            )

        sample = self.get_sample(index)

        if not isinstance(sample, CanonicalV2XSample):
            raise TypeError(
                f"{self.__class__.__name__}.get_sample() must return "
                "a CanonicalV2XSample."
            )

        sample.validate()

        return sample

    def __repr__(self) -> str:
        """
        Simple readable description of the adapter.
        """

        return (
            f"{self.__class__.__name__}("
            f"dataset_root='{self.dataset_root}', "
            f"split={self.split!r}, "
            f"num_samples={len(self)})"
        )