from pathlib import Path
import numpy as np

from utils.config_loader import SEG_CLASS_NAMES


def _load_seg_regions_from_border_txt(border_txt_path: Path):
    """
    Load segmentation polygon annotations from a border text file.

    Each non-empty line in the input file is expected to follow the format::

        <class_id> <confidence> x1 y1 x2 y2 x3 y3 ...

    where:
        - class_id  : Integer semantic class identifier.
        - confidence: Detection/segmentation confidence score.
        - (xi, yi)  : Polygon vertex coordinates in image pixels.

    The polygon must contain at least three vertices. Coordinates are rounded
    to the nearest integer and converted into the OpenCV polygon format
    ``(N, 1, 2)`` with dtype ``np.int32``.

    Parameters
    ----------
    border_txt_path : Path
        Path to the segmentation border annotation file.

    Returns
    -------
    list[dict]
        A list of segmentation region dictionaries. Each dictionary contains:

        - ``poly`` : np.ndarray
            Polygon vertices in OpenCV contour format.
        - ``class_id`` : int
            Semantic class identifier.
        - ``class_name`` : str
            Human-readable class name retrieved from ``SEG_CLASS_NAMES``.
        - ``confidence`` : float
            Confidence score associated with the segmentation region.

    Notes
    -----
    Invalid annotation lines are skipped silently. A line is considered invalid
    if it:
        - contains fewer than three polygon vertices,
        - has an odd number of coordinate values,
        - is empty.

    If the annotation file does not exist, an empty list is returned.
    """
    # Return an empty result if the annotation file is unavailable.
    if not border_txt_path.exists():
        print(f"[SEG] border file not found: {border_txt_path}")
        return []

    regions = []

    # Parse each annotation line independently.
    with open(border_txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()

            # Skip empty lines.
            if not ln:
                continue

            vals = ln.split()

            # Require:
            #   class_id + confidence + at least 3 (x, y) vertex pairs.
            if len(vals) < 2 + 6:
                continue

            # Read annotation metadata.
            cls_id = int(float(vals[0]))
            conf = float(vals[1])

            # Remaining values represent polygon coordinates.
            coords = list(map(float, vals[2:]))

            # Polygon coordinates must be provided as complete (x, y) pairs.
            if len(coords) % 2 != 0:
                continue

            pts = []

            # Convert coordinate sequence into a list of integer vertices.
            it = iter(coords)
            for x, y in zip(it, it):
                pts.append([int(round(x)), int(round(y))])

            # A valid polygon requires at least three vertices.
            if len(pts) < 3:
                continue

            # Convert vertices into OpenCV contour representation.
            poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)

            # Resolve the semantic class name if available.
            cls_name = SEG_CLASS_NAMES.get(cls_id, str(cls_id))

            # Store the parsed segmentation region.
            regions.append({
                "poly": poly,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf
            })

    return regions