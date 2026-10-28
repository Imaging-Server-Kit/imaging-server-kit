from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from imaging_server_kit.types._array import _ArrayLayer

# Largest mask (in pixels) for which the number of labels is shown in `repr`
MAX_SIZE_COUNT_LABELS = 2**24


def invalidate_label_features(features, labels: Sequence[int]) -> Dict:
    """Set NaN values to all the features (except `label`) of the given labels.

    Used for objects whose features are no longer accurate, e.g. objects cut by a tile border.

    Parameters
    ----------
    features : dict or DataFrame
        Features with a `label` column.
    labels : sequence of int
        The labels to invalidate.

    Returns
    -------
    The features, as a dict of 1D Numpy arrays.
    """
    df = pd.DataFrame(features)
    rows = df["label"].isin(labels)
    if rows.any():
        for c in df.columns.difference(["label"]):
            if df[c].dtype.kind in "iub":
                df[c] = df[c].astype(float)  # To hold NaN values
            df.loc[rows, c] = np.nan
    return {c: df[c].values for c in df.columns}


class Mask(_ArrayLayer):
    """Data layer for segmentation masks.

    Masks are label images, where integer values encode either object classes (e.g.
    for pixel classification) or object instances.

    Parameters
    ----------
    data : numpy.ndarray, optional
        The label image. Boolean arrays are accepted for binary masks.
    name : str, default="Mask"
        Name of the layer.
    description : str, default="Segmentation mask"
        Description of the layer, displayed on the algorithm documentation page.
    dimensionality : list of int, optional
        Accepted numbers of dimensions, for example `[2, 3]`. By default, any number
        of dimensions is accepted.
    channel_axis : int, optional
        Index of the channel axis, if any. The channel axis is not counted as a
        spatial dimension, and it is never split into tiles.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties. Use `merger="instances"` for
        instance segmentation masks, so that labels stay unique when the mask is
        assembled from tiles.

    Examples
    --------
    >>> mask = sk.Mask(labels, name="Nuclei", merger="instances")
    """

    kind = "mask"
    _zeros_dtype = np.uint32

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name: str = "Mask",
        description: str = "Segmentation mask",
        dimensionality: Optional[List[int]] = None,
        channel_axis: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            dimensionality=dimensionality,
            channel_axis=channel_axis,
            **kwargs,
        )

    def _summary(self) -> str:
        data = self.data
        if data is None:
            return "empty"
        summary = f"{data.dtype} {data.shape}"
        # Counting labels requires a full pass over the data, so we skip it for large masks
        if data.size <= MAX_SIZE_COUNT_LABELS:
            summary += f", {self.n_objects} labels"
        return summary

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            # pd.unique (hash-based) is several times faster than np.unique (sort-based)
            labels = pd.unique(self.data.ravel())
            n_labels = len(labels) - int(0 in labels)
            return n_labels

    def _erase(self, slices_with_channel) -> None:
        erased_labels = np.unique(self.data[slices_with_channel])
        super()._erase(slices_with_channel)

        # Objects cut by the domain keep part of their pixels: their features are no longer accurate
        features = self.meta.get("features")
        if isinstance(features, Dict) and ("label" in features):
            cut_labels = np.unique(self.data[np.isin(self.data, erased_labels[erased_labels > 0])])
            try:
                self.meta = {**self.meta, "features": invalidate_label_features(features, cut_labels)}
            except ValueError:
                pass  # Features that don't fit in a table (e.g. columns of different lengths) are left as-is
