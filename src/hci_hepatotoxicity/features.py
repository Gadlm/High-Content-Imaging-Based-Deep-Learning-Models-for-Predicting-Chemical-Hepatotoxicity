import numpy as np
from skimage import filters, measure


STATISTICS = ("max", "min", "median", "mean", "range", "variance")


def summarize(values, trim_fraction):
    values = np.asarray(values, dtype=float).ravel()
    values = np.sort(values[np.isfinite(values)])
    if not 0 <= trim_fraction < 0.5 or values.size == 0:
        raise ValueError("Invalid values or trim fraction")
    cut = int(np.floor(values.size * trim_fraction))
    if cut and 2 * cut < values.size:
        values = values[cut:-cut]
    return {
        "max": float(values.max()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "range": float(np.ptp(values)),
        "variance": float(values.var()),
    }


def _regions(mask):
    mask = np.asarray(mask)
    labels = measure.label(mask > 0) if np.unique(mask).size <= 2 else mask.astype(int)
    regions = measure.regionprops(labels)
    if not regions:
        raise ValueError("No segmented objects were found")
    return regions


def _region_features(mask, prefix, trim_fraction, roughness_scale):
    rows = []
    for region in _regions(mask):
        perimeter = float(region.perimeter)
        edge = filters.sobel(region.image.astype(float))
        rows.append(
            {
                "area": float(region.area),
                "perimeter": perimeter,
                "major_axis": float(region.axis_major_length),
                "minor_axis": float(region.axis_minor_length),
                "roughness": float(np.abs(edge).sum() * roughness_scale / (2 * perimeter)) if perimeter else 0.0,
                "circularity": float(4 * np.pi * region.area / perimeter**2) if perimeter else 0.0,
            }
        )
    output = {}
    for descriptor in rows[0]:
        for statistic, value in summarize([row[descriptor] for row in rows], trim_fraction).items():
            output[f"{prefix}_{descriptor}_{statistic}"] = value
    return output, len(rows)


def extract_morphometric_features(cell_mask, nucleus_mask, blue_channel, params):
    trim_fraction = float(params["trim_fraction"])
    roughness_scale = float(params["roughness_scale"])
    cell, cell_count = _region_features(cell_mask, "cell", trim_fraction, roughness_scale)
    nucleus, nucleus_count = _region_features(nucleus_mask, "nucleus", trim_fraction, roughness_scale)
    blue_values = np.asarray(blue_channel, dtype=float)[np.asarray(cell_mask) > 0]
    blue = {f"blue_{name}": value for name, value in summarize(blue_values, trim_fraction).items()}
    return {**cell, **nucleus, "cell_count": cell_count, "nucleus_count": nucleus_count, **blue}


def extract_fluorescence_features(channels, params):
    trim_fraction = float(params["trim_fraction"])
    excluded = set(params.get("excluded_features", []))
    output = {}
    for channel, image in channels.items():
        for statistic, value in summarize(image, trim_fraction).items():
            name = f"{channel}_{statistic}"
            if name not in excluded:
                output[name] = value
    return output
