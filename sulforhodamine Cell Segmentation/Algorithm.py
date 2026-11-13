"""Segment GFP-labelled mast cells with Cellpose and/or Otsu thresholding.

Coarse masks from either detector are snapped onto the nearest intensity edge
with a marker-controlled gradient watershed (see refine_boundaries).

Examples (from this folder):
    python Algorithm.py "..\\cell1\\Duplicate1 gfp for OnlyIge.jpg"
    python Algorithm.py ..\\cell2 --pattern "*gfp*"
    python Algorithm.py ..\\cell2 --method threshold --edge-gamma 1.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, watershed
import tifffile


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def read_grayscale(path: Path) -> np.ndarray:
    """Read a 2D grayscale image, converting RGB images to intensity."""
    image = np.asarray(iio.imread(path))
    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            image = image[..., :3].astype(np.float32).mean(axis=-1)
        elif image.shape[0] in (3, 4):
            image = np.moveaxis(image[:3], 0, -1).astype(np.float32).mean(axis=-1)
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            raise ValueError(f"Expected a 2D grayscale or RGB image: {path}")
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"Expected a non-empty 2D image: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"Image contains non-finite pixel values: {path}")
    return image


def watershed_split(
    cellpose_labels: np.ndarray,
    min_distance: int = 30,
    min_area: int = 100,
) -> np.ndarray:
    """Split merged Cellpose masks using distance-map peaks as watershed seeds.

    Existing Cellpose instances seed the watershed first. A single Cellpose
    instance is split only when the distance map contains additional, well-
    separated peaks.
    """
    foreground = cellpose_labels > 0
    components, component_count = ndi.label(
        foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )
    result = np.zeros(cellpose_labels.shape, dtype=np.uint32)
    next_label = 1

    for component_id in range(1, component_count + 1):
        component = components == component_id
        distance = ndi.distance_transform_edt(component)
        original_ids = np.unique(cellpose_labels[component])
        original_ids = original_ids[original_ids > 0]
        markers = np.zeros(cellpose_labels.shape, dtype=np.int32)
        marker_id = 1

        for original_id in original_ids:
            original_region = component & (cellpose_labels == original_id)
            distances = np.where(original_region, distance, -1)
            seed = np.unravel_index(np.argmax(distances), distances.shape)
            markers[seed] = marker_id
            marker_id += 1

        if len(original_ids) == 1:
            peaks = peak_local_max(
                distance,
                labels=component.astype(np.uint8),
                min_distance=min_distance,
                threshold_abs=max(2.0, float(distance.max()) * 0.5),
                exclude_border=False,
            )
            for row, column in peaks:
                if markers[row, column] == 0:
                    markers[row, column] = marker_id
                    marker_id += 1

        split_labels = watershed(
            -distance,
            markers,
            mask=component,
            connectivity=2,
        )
        for split_id in np.unique(split_labels):
            if split_id == 0:
                continue
            split_region = split_labels == split_id
            if int(np.count_nonzero(split_region)) < min_area:
                continue
            result[split_region] = next_label
            next_label += 1

    return result


def threshold_candidates(
    image: np.ndarray,
    threshold_factor: float = 0.5,
    threshold_min_area: int = 1000,
) -> np.ndarray:
    """Label connected Otsu-foreground components as coarse cell candidates."""
    smoothed = filters.gaussian(image, sigma=3, preserve_range=True)
    threshold = filters.threshold_otsu(smoothed) * threshold_factor
    foreground = morphology.closing(
        smoothed > threshold,
        footprint=morphology.disk(5),
    )

    components, _ = ndi.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(components.ravel())
    keep_component = areas >= threshold_min_area
    keep_component[0] = False
    foreground = keep_component[components]

    component_labels, _ = ndi.label(
        foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )
    return component_labels.astype(np.uint32)


def estimate_diameter(candidates: np.ndarray) -> float | None:
    """Median equivalent diameter of candidate cells, for Cellpose rescaling."""
    diameters = [region.equivalent_diameter_area for region in regionprops(candidates)]
    if not diameters:
        return None
    return float(np.median(diameters))


def refine_boundaries(
    image: np.ndarray,
    labels: np.ndarray,
    foreground_fraction: float = 0.3,
    background_fraction: float = 0.1,
    edge_gamma: float = 0.5,
    sigma: float = 2.0,
) -> np.ndarray:
    """Snap each coarse mask onto the strongest nearby intensity edge.

    For every labelled cell, pixels brighter than ``foreground_fraction`` of
    the way from background to the cell's 95th-percentile intensity seed the
    cell, and pixels dimmer than ``background_fraction`` (or well outside the
    coarse mask, or in another cell) seed the background. A marker-controlled
    watershed on the gradient magnitude then places the boundary on the edge
    between the two seeds. ``edge_gamma`` compresses intensity before taking
    the gradient: 1.0 favours the sharpest bright edge (rejects glow around
    saturated objects), lower values pick up dim protrusions.
    """
    smoothed = filters.gaussian(image.astype(np.float32), sigma=sigma, preserve_range=True)
    background_level = float(np.median(smoothed))
    compressed = np.clip(smoothed - background_level, 0, None) ** edge_gamma
    gradient = filters.sobel(compressed)
    result = np.zeros(labels.shape, dtype=np.uint32)

    for region in regionprops(labels):
        margin = max(25, round(0.15 * region.equivalent_diameter_area))
        row0, column0, row1, column1 = region.bbox
        window = (
            slice(max(row0 - 2 * margin, 0), min(row1 + 2 * margin, labels.shape[0])),
            slice(max(column0 - 2 * margin, 0), min(column1 + 2 * margin, labels.shape[1])),
        )
        cell = labels[window] == region.label
        other_cells = (labels[window] > 0) & ~cell
        local = smoothed[window]
        span = float(np.percentile(local[cell], 95)) - background_level
        if span <= 0:
            continue

        foreground = cell & (local > background_level + foreground_fraction * span)
        background = (
            (local < background_level + background_fraction * span)
            | ~morphology.dilation(cell, morphology.disk(margin))
            | other_cells
        ) & ~foreground
        if not foreground.any() or not background.any():
            continue

        markers = np.zeros(cell.shape, dtype=np.int32)
        markers[background] = 1
        markers[foreground] = 2
        refined = ndi.binary_fill_holes(watershed(gradient[window], markers) == 2)

        # Drop fragments that do not contain a confident foreground seed.
        pieces, _ = ndi.label(refined)
        seeded = np.unique(pieces[foreground])
        refined = np.isin(pieces, seeded[seeded > 0])

        target = result[window]
        target[refined & (target == 0)] = region.label

    return result


def relabel_min_area(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove masks smaller than ``min_area`` and number the rest 1..N."""
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    lookup = np.zeros(areas.shape, dtype=np.uint32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint32)
    return lookup[labels]


def detect_granules(
    image: np.ndarray,
    threshold_mads: float = 4.0,
    min_area: int = 20,
    max_area: int = 3000,
    sigma: float = 2.0,
) -> np.ndarray:
    """Label bright exocytosed-granule spots in an avidin image.

    The avidin channel is dominated by shot noise around a flat background, so
    spots are pixels whose smoothed intensity exceeds the median by
    ``threshold_mads`` robust standard deviations. Components larger than
    ``max_area`` (saturated debris or beads) are discarded.
    """
    smoothed = filters.gaussian(image.astype(np.float32), sigma=sigma, preserve_range=True)
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median))) * 1.4826
    if mad <= 0:
        return np.zeros(image.shape, dtype=np.uint32)
    spots, _ = ndi.label(
        smoothed > median + threshold_mads * mad,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    areas = np.bincount(spots.ravel())
    keep = (areas >= min_area) & (areas <= max_area)
    keep[0] = False
    lookup = np.zeros(areas.shape, dtype=np.uint32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint32)
    return lookup[spots]


def assign_granules(
    cells: np.ndarray, spots: np.ndarray, max_distance: float
) -> dict[int, tuple[int, float]]:
    """Map each spot to the nearest cell as {spot: (cell, distance_px)}.

    Granules are released outside the cell, so distance is measured from the
    spot's closest pixel to the cell mask; spots farther than ``max_distance``
    are assigned to cell 0 (unassigned).
    """
    if not cells.any():
        return {region.label: (0, float("inf")) for region in regionprops(spots)}
    distance, (rows, columns) = ndi.distance_transform_edt(
        cells == 0, return_indices=True
    )
    nearest_cell = cells[rows, columns]
    assignments = {}
    for region in regionprops(spots):
        coords = region.coords
        pixel_distances = distance[coords[:, 0], coords[:, 1]]
        closest = int(np.argmin(pixel_distances))
        spot_distance = float(pixel_distances[closest])
        cell = int(nearest_cell[coords[closest, 0], coords[closest, 1]])
        assignments[region.label] = (
            (cell, spot_distance) if spot_distance <= max_distance else (0, spot_distance)
        )
    return assignments


def save_degranulation(
    avidin: np.ndarray,
    cells: np.ndarray,
    spots: np.ndarray,
    assignments: dict[int, tuple[int, float]],
    output_dir: Path,
    stem: str,
) -> None:
    """Write an avidin overlay plus per-spot and per-cell granule tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{stem}_degranulation_overlay.png"
    spots_path = output_dir / f"{stem}_granules.csv"
    summary_path = output_dir / f"{stem}_granules_per_cell.csv"

    gray = display_grayscale(avidin)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    overlay[find_boundaries(cells, mode="outer")] = (0, 255, 0)
    overlay[find_boundaries(spots, mode="outer")] = (255, 0, 255)
    iio.imwrite(overlay_path, overlay)

    summary = {int(cell): [0, 0, 0.0] for cell in np.unique(cells) if cell > 0}
    with spots_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "granule",
                "cell",
                "distance_to_cell_px",
                "area_px",
                "mean_intensity",
                "centroid_row_px",
                "centroid_column_px",
            ]
        )
        for region in regionprops(spots, intensity_image=avidin):
            cell, distance = assignments[region.label]
            row, column = region.centroid
            writer.writerow(
                [
                    region.label,
                    cell,
                    round(distance, 2),
                    region.area,
                    round(float(region.intensity_mean), 2),
                    round(row, 2),
                    round(column, 2),
                ]
            )
            if cell in summary:
                summary[cell][0] += 1
                summary[cell][1] += int(region.area)
                summary[cell][2] += float(region.intensity_mean) * region.area

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["cell", "granule_count", "granule_area_px", "integrated_intensity"]
        )
        for cell, (count, area, integrated) in summary.items():
            writer.writerow([cell, count, area, round(integrated, 2)])

    print(f"Saved {overlay_path}")
    print(f"Saved {spots_path}")
    print(f"Saved {summary_path}")


def display_grayscale(image: np.ndarray) -> np.ndarray:
    """Scale an image to 8-bit for the boundary overlay."""
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def save_results(image: np.ndarray, labels: np.ndarray, output_dir: Path, stem: str) -> None:
    """Write a label image, boundary overlay, and per-cell measurements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / f"{stem}_labels.tif"
    overlay_path = output_dir / f"{stem}_overlay.png"
    csv_path = output_dir / f"{stem}_measurements.csv"

    tifffile.imwrite(label_path, labels)
    gray = display_grayscale(image)
    overlay = np.repeat(gray[..., None], 3, axis=-1)
    overlay[find_boundaries(labels, mode="outer")] = (0, 255, 0)
    iio.imwrite(overlay_path, overlay)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["label", "area_px", "centroid_row_px", "centroid_column_px"]
        )
        for region in regionprops(labels):
            row, column = region.centroid
            writer.writerow(
                [region.label, region.area, round(row, 2), round(column, 2)]
            )

    print(f"Saved {label_path}")
    print(f"Saved {overlay_path}")
    print(f"Saved {csv_path}")


def input_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files = sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No supported images matched {pattern!r} in {path}"
        )
    return files


def input_files_or_none(folder: Path, pattern: str) -> list[Path]:
    """Like input_files for a folder, but an empty match is not an error."""
    return sorted(
        candidate
        for candidate in folder.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment GFP-labelled cells with Cellpose, Otsu thresholding, "
            "or both; watershed splits merged masks."
        ),
        epilog=(
            "Inspect every *_overlay.png before using measurements. "
            "Pretrained Cellpose may need a measured --diameter or a model "
            "trained on manually annotated images."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="An image file or a folder containing images.",
    )
    parser.add_argument(
        "--pattern",
        default="*gfp*",
        help='Folder filename pattern (default: "*gfp*").',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output folder (default: segmentation_results beside the input).",
    )
    parser.add_argument(
        "--method",
        choices=("cellpose", "threshold", "both"),
        default="both",
        help="Segmentation method(s) to run (default: both).",
    )
    parser.add_argument(
        "--model",
        default="cpsam_v2",
        help="Cellpose pretrained model name or model path (default: cpsam_v2).",
    )
    parser.add_argument(
        "--diameter",
        type=positive_float,
        help=(
            "Approximate cell diameter in pixels; leave unset for automatic "
            "sizing, or set it if the overlay misses cells."
        ),
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Split touching cells with a distance-map watershed. Off by "
            "default because it cuts large, irregular single cells apart."
        ),
    )
    parser.add_argument(
        "--min-distance",
        type=positive_int,
        default=30,
        help="Minimum spacing in pixels between --split seeds (default: 30).",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip snapping mask boundaries onto intensity edges.",
    )
    parser.add_argument(
        "--foreground-fraction",
        type=positive_float,
        default=0.3,
        help=(
            "Refinement: fraction of each cell's background-to-peak intensity "
            "range above which pixels are certainly cell (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--background-fraction",
        type=positive_float,
        default=0.1,
        help=(
            "Refinement: fraction below which pixels are certainly background "
            "(default: 0.1). Must be below --foreground-fraction."
        ),
    )
    parser.add_argument(
        "--edge-gamma",
        type=positive_float,
        default=0.5,
        help=(
            "Refinement: intensity exponent before edge detection (default: "
            "0.5). Raise toward 1.0 to exclude glow around saturated objects; "
            "lower it to capture dim protrusions."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=positive_int,
        default=100,
        help="Discard final masks smaller than this area in pixels (default: 100).",
    )
    parser.add_argument(
        "--threshold-factor",
        type=positive_float,
        default=0.5,
        help="Otsu threshold multiplier for threshold mode (default: 0.5).",
    )
    parser.add_argument(
        "--threshold-min-area",
        type=positive_int,
        default=1000,
        help=(
            "Remove threshold components smaller than this pixel area "
            "(default: 1000)."
        ),
    )
    parser.add_argument(
        "--avidin-pattern",
        default="*avidin*",
        help=(
            "Avidin images beside each GFP image to analyse for degranulation "
            '(default: "*avidin*"). Pass "" to skip.'
        ),
    )
    parser.add_argument(
        "--granule-threshold",
        type=positive_float,
        default=4.0,
        help=(
            "Granule spots must exceed the avidin background by this many "
            "robust standard deviations (default: 4.0)."
        ),
    )
    parser.add_argument(
        "--granule-max-area",
        type=positive_int,
        default=3000,
        help=(
            "Ignore avidin spots larger than this pixel area, such as "
            "saturated debris (default: 3000)."
        ),
    )
    parser.add_argument(
        "--granule-distance",
        type=positive_float,
        default=150.0,
        help=(
            "Assign a granule to the nearest cell only within this many "
            "pixels of the cell mask (default: 150)."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a CUDA GPU if the installed PyTorch build supports it.",
    )
    args = parser.parse_args()
    if args.background_fraction >= args.foreground_fraction:
        parser.error("--background-fraction must be below --foreground-fraction")
    return args


def main() -> None:
    args = parse_args()
    files = input_files(args.input, args.pattern)
    output_dir = args.output or (
        args.input / "segmentation_results"
        if args.input.is_dir()
        else args.input.parent / "segmentation_results"
    )
    model = None
    if args.method in ("cellpose", "both"):
        from cellpose import models

        model = models.CellposeModel(gpu=args.gpu, pretrained_model=args.model)

    def finish(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if args.split:
            labels = watershed_split(
                labels, min_distance=args.min_distance, min_area=args.min_area
            )
        if not args.no_refine:
            labels = refine_boundaries(
                image,
                labels,
                foreground_fraction=args.foreground_fraction,
                background_fraction=args.background_fraction,
                edge_gamma=args.edge_gamma,
            )
        return relabel_min_area(labels, args.min_area)

    def degranulation(path: Path, labels: np.ndarray, method: str) -> None:
        if not args.avidin_pattern:
            return
        for avidin_path in input_files_or_none(path.parent, args.avidin_pattern):
            avidin = read_grayscale(avidin_path)
            if avidin.shape != labels.shape:
                print(
                    f"Skipping {avidin_path.name}: size {avidin.shape} does not "
                    f"match {path.name} {labels.shape}"
                )
                continue
            spots = detect_granules(
                avidin,
                threshold_mads=args.granule_threshold,
                max_area=args.granule_max_area,
            )
            assignments = assign_granules(labels, spots, args.granule_distance)
            save_degranulation(
                avidin,
                labels,
                spots,
                assignments,
                output_dir,
                f"{avidin_path.stem}_{method}",
            )
            assigned = sum(1 for cell, _ in assignments.values() if cell > 0)
            print(
                f"{avidin_path.name} ({method} cells from {path.name}): "
                f"{assigned} of {len(assignments)} granules assigned to cells"
            )

    for path in files:
        image = read_grayscale(path)
        candidates = threshold_candidates(
            image,
            threshold_factor=args.threshold_factor,
            threshold_min_area=args.threshold_min_area,
        )
        if args.method in ("cellpose", "both"):
            if model is None:
                raise RuntimeError("Cellpose model was not initialized.")
            # Cellpose expects ~30 px cells; without a diameter it misses
            # cells hundreds of pixels across, so estimate one if not given.
            diameter = args.diameter or estimate_diameter(candidates)
            model_input = np.repeat(image[..., None], 3, axis=-1)
            cellpose_labels, *_ = model.eval(
                model_input,
                channel_axis=-1,
                normalize=True,
                diameter=diameter,
            )
            labels = finish(image, np.asarray(cellpose_labels, dtype=np.uint32))
            save_results(image, labels, output_dir, f"{path.stem}_cellpose")
            diameter_text = f"{diameter:.0f} px" if diameter else "auto"
            print(
                f"{path.name} (Cellpose, diameter {diameter_text}): "
                f"segmented {int(labels.max())} cells"
            )
            degranulation(path, labels, "cellpose")

        if args.method in ("threshold", "both"):
            labels = finish(image, candidates)
            save_results(image, labels, output_dir, f"{path.stem}_threshold")
            print(f"{path.name} (threshold): segmented {int(labels.max())} cells")
            degranulation(path, labels, "threshold")


if __name__ == "__main__":
    main()