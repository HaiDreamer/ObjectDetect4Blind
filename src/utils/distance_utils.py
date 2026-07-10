import numpy as np

def _fast_percentile_1d(vals: np.ndarray, q: float) -> float | None:
    """
    Compute the q-th percentile of a 1D NumPy array efficiently.

    Unlike np.percentile(), this implementation uses np.partition(),
    which avoids fully sorting the array. This reduces the average
    time complexity from O(n log n) to O(n).

    Parameters
    ----------
    vals : np.ndarray
        Input one-dimensional array. NaN and infinite values are ignored.

    q : float
        Desired percentile in the range [0, 100].
        Examples:
            0   -> minimum value
            50  -> median
            100 -> maximum value

    Returns
    -------
    float | None
        The estimated q-th percentile.
        Returns None if the input is None or contains no finite values.
    """

    if vals is None:
        return None
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])

def _compute_box_distance(
    depth_map_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    frac: float = 0.5,
    mode: str = "center",
    q: float = 10.0,
    subsample: int = 1,
) -> float | None:
    """
    Estimate the distance to an object from a depth map within a bounding box.

    The function extracts a representative region of the bounding box and
    computes a robust depth estimate using a low percentile of the valid
    depth values. This approach reduces the influence of background pixels,
    noise, and invalid measurements.

    Two region selection modes are supported:

    - ``center``: Uses a square/rectangular patch centered inside the
      bounding box. The patch dimensions are determined by ``frac``.
    - ``bottom``: Uses only the bottom portion of the bounding box and
      restricts the horizontal region to the center band. This mode is
      particularly suitable for road scenes, where the lower part of an
      object's bounding box usually corresponds to the point closest to
      the camera.

    Parameters
    ----------
    depth_map_m : np.ndarray
        Depth map in meters with shape (H, W).

    x1, y1 : int
        Coordinates of the upper-left corner of the bounding box.

    x2, y2 : int
        Coordinates of the lower-right corner of the bounding box.

    frac : float, default=0.5
        Fraction of the bounding box dimensions used to define the sampling
        region.

    mode : {"center", "bottom"}, default="center"
        Strategy for selecting the region from which depth values are
        extracted.

    q : float, default=10.0
        Percentile of valid depth values used as the distance estimate.
        Lower percentiles bias the estimate toward the nearest visible
        surface within the sampled region.

    subsample : int, default=1
        Sampling interval applied to the extracted patch. Values greater
        than one reduce computation by processing every n-th pixel.

    Returns
    -------
    float | None
        Estimated object distance in meters, or ``None`` if the bounding
        box is invalid or no valid depth values are available.

    Notes
    -----
    Before computing the percentile, the function:
    - Clips the bounding box to image boundaries.
    - Removes zero, NaN, and infinite depth values.
    - Optionally subsamples the selected region.
    - Computes the specified percentile using
      ``_fast_percentile_1d()`` for efficiency.
    """
    H, W = depth_map_m.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    if mode == "bottom":
        ch = int(h * frac)
        if ch <= 0:
            return None
        y_start = max(y1, y2 - ch)

        center_band_width = int(w * 0.5)
        if center_band_width <= 0:
            return None

        cx = (x1 + x2) // 2
        x_start = max(x1, cx - center_band_width // 2)
        x_end = min(x2, x_start + center_band_width)
        if x_end <= x_start:
            return None

        patch = depth_map_m[y_start:y2, x_start:x_end]
    else:
        cw = int(w * frac)
        ch = int(h * frac)
        if cw <= 0 or ch <= 0:
            return None

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cx1 = max(0, cx - cw // 2)
        cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw)
        cy2 = min(H, cy1 + ch)
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        patch = depth_map_m[cy1:cy2, cx1:cx2]

    if patch.size == 0:
        return None

    if subsample > 1:
        patch = patch[::subsample, ::subsample]

    valid = patch[(patch > 0) & np.isfinite(patch)].reshape(-1)
    return _fast_percentile_1d(valid, q=q)

def _nearest_sidewalk_distance(
    depth_map_m: np.ndarray,
    sidewalk_mask: np.ndarray,
    max_depth: float = 80.0,
    band_start_frac: float = 0.1,
    q: float = 10.0,
    subsample: int = 1,
):
    """
    Estimate the nearest sidewalk distance from a depth map.

    The function extracts valid sidewalk pixels from a segmentation mask,
    filters out invalid or excessively large depth values, and estimates
    the nearest sidewalk distance using a low percentile of the remaining
    depth values. To improve robustness and computational efficiency, the
    search is optionally restricted to the lower portion of the image,
    where sidewalks are more likely to appear.

    Parameters
    ----------
    depth_map_m : np.ndarray
        Depth map in meters with shape (H, W).

    sidewalk_mask : np.ndarray
        Binary sidewalk segmentation mask with the same shape as
        ``depth_map_m``. Sidewalk pixels should have value 1.

    max_depth : float, default=80.0
        Maximum valid depth value (meters). Pixels beyond this threshold
        are ignored.

    band_start_frac : float, default=0.1
        Fraction of the image height defining the start of the search
        region. Only pixels below this row are considered initially. If
        no valid pixels are found, the entire mask is searched.

    q : float, default=10.0
        Percentile used to estimate the nearest sidewalk distance. Lower
        percentiles emphasize the closest visible sidewalk surface.

    subsample : int, default=1
        Sampling interval applied to the valid sidewalk pixels. Values
        greater than one reduce computation by processing every n-th pixel.

    Returns
    -------
    tuple[float | None, int | None, int | None]
        A tuple containing:

        - Estimated sidewalk distance (meters).
        - x-coordinate of the representative sidewalk pixel.
        - y-coordinate of the representative sidewalk pixel.

        Returns ``(None, None, None)`` if no valid sidewalk pixels are
        available.
    """
    assert depth_map_m.shape == sidewalk_mask.shape, "Depth and mask must have same size"
    H, W = depth_map_m.shape

    base_cond = (
        (sidewalk_mask == 1) &
        (depth_map_m > 0) &
        (depth_map_m < max_depth) &
        np.isfinite(depth_map_m)
    )
    if not np.any(base_cond):
        return None, None, None

    y_band_start = int(H * band_start_frac)
    band_mask = np.zeros_like(sidewalk_mask, dtype=bool)
    band_mask[y_band_start:H, :] = True

    cond = base_cond & band_mask
    if not np.any(cond):
        cond = base_cond

    ys, xs = np.where(cond)

    if subsample > 1 and ys.size > 0:
        take = np.arange(0, ys.size, subsample, dtype=np.int64)
        ys = ys[take]
        xs = xs[take]

    vals = depth_map_m[ys, xs].astype(np.float32)
    finite = np.isfinite(vals)
    vals = vals[finite]
    ys = ys[finite]
    xs = xs[finite]

    d_q = _fast_percentile_1d(vals, q=q)
    if d_q is None:
        return None, None, None

    idx = int(np.argmin(np.abs(vals - d_q)))
    return float(d_q), int(xs[idx]), int(ys[idx])