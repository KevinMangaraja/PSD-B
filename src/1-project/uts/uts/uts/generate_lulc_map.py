from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import json
import re

import folium
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_window, rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from sklearn.metrics import accuracy_score, classification_report

from extract_sentinel_pixels import iter_sentinel_tile_rasters


CLASS_ORDER = (
    "Sawah",
    "Pemukiman",
    "Hutan non mangrove",
    "Hutan mangrove",
    "Air",
)
CLASS_COLORS = {
    "Sawah": "#f1c40f",
    "Pemukiman": "#e74c3c",
    "Hutan non mangrove": "#228b22",
    "Hutan mangrove": "#006d5b",
    "Air": "#168aad",
}
BANDS = ("B02", "B03", "B04", "B08", "B11")
FEATURES = (*BANDS, "NDVI", "NDWI", "NDBI")
CLASS_IDS = {label: index for index, label in enumerate(CLASS_ORDER, 1)}
SCENE_PATTERN = re.compile(
    r"(?P<tile>T\d{2}[A-Z]{3})_(?P<date>\d{8}T\d{6})_"
    r"(?P<band>B\d{2})_\d+m\.jp2$"
)
BLOCK_SIZE = 512
BOUNDARY_SOURCE = (
    "geoBoundaries gbOpen Indonesia ADM1, sourced from OpenStreetMap/Wambacher "
    "(boundary year 2017; ODbL 1.0)"
)
BOUNDARY_SOURCE_URL = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM1/"
BOUNDARY_MASK_VERSION = "2"
COASTLINE_SOURCE = "Natural Earth 1:10m coastline"
COASTLINE_SOURCE_URL = (
    "https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/"
)
COASTLINE_LICENSE = "Public domain"
MANGROVE_COASTAL_BUFFER_METERS = 5_000
MANGROVE_COAST_MASK_VERSION = "1"


def _latest_complete_scenes(raster_root: Path) -> dict[str, dict[str, Path]]:
    scenes: dict[tuple[str, str], dict[str, Path]] = {}
    for path in iter_sentinel_tile_rasters(raster_root):
        match = SCENE_PATTERN.fullmatch(path.name)
        if not match or match.group("band") not in BANDS:
            continue
        key = (match.group("tile"), match.group("date"))
        band = match.group("band")
        if band in scenes.setdefault(key, {}):
            raise ValueError(f"Band raster duplikat ditemukan untuk scene {key}: {band}")
        scenes[key][band] = path

    if not scenes:
        raise FileNotFoundError(
            f"Tidak menemukan raster Sentinel-2 JP2 di {raster_root.resolve()}."
        )

    latest: dict[str, tuple[str, dict[str, Path]]] = {}
    for (tile, acquisition), files in scenes.items():
        missing = set(BANDS) - set(files)
        if missing:
            raise ValueError(
                f"Scene {tile} {acquisition} tidak lengkap; band hilang: "
                f"{sorted(missing)}."
            )
        if tile not in latest or acquisition > latest[tile][0]:
            latest[tile] = (acquisition, files)
    return {tile: value[1] for tile, value in sorted(latest.items())}


def _feature_matrix(arrays: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    values = arrays[:, valid].T
    b02, b03, b04, b08, b11 = values.T
    return pd.DataFrame(
        {
            "B02": b02,
            "B03": b03,
            "B04": b04,
            "B08": b08,
            "B11": b11,
            "NDVI": (b08 - b04) / (b08 + b04),
            "NDWI": (b03 - b08) / (b03 + b08),
            "NDBI": (b11 - b08) / (b11 + b08),
        },
        columns=FEATURES,
    )


def _class_palette() -> list[int]:
    palette = [0] * 768
    for class_id, label in enumerate(CLASS_ORDER, start=1):
        color = CLASS_COLORS[label].lstrip("#")
        palette[class_id * 3 : class_id * 3 + 3] = [
            int(color[offset : offset + 2], 16) for offset in (0, 2, 4)
        ]
    return palette


def _apply_mangrove_coastal_constraint(
    raster_path: Path, coastline, coastline_crs
) -> int:
    changed_pixels = 0
    mangrove_id = CLASS_IDS["Hutan mangrove"]
    replacement_id = CLASS_IDS["Hutan non mangrove"]
    with rasterio.open(raster_path, "r+") as prediction:
        if prediction.crs is None or not prediction.crs.is_projected:
            raise ValueError(
                f"CRS raster harus terproyeksi untuk membatasi mangrove: "
                f"{raster_path}"
            )
        coastal_zone = gpd.GeoSeries(
            [coastline], crs=coastline_crs
        ).to_crs(prediction.crs).iloc[0].buffer(MANGROVE_COASTAL_BUFFER_METERS)
        if coastal_zone.is_empty:
            raise ValueError("Zona pesisir mangrove kosong setelah transformasi CRS.")

        for _, window in prediction.block_windows(1):
            values = prediction.read(1, window=window)
            mangrove_pixels = values == mangrove_id
            if not mangrove_pixels.any():
                continue
            bounds = box(*rasterio.windows.bounds(window, prediction.transform))
            block_zone = coastal_zone.intersection(bounds)
            if block_zone.is_empty:
                outside_coast = mangrove_pixels
            else:
                coastal_pixels = rasterize(
                    [(mapping(block_zone), 1)],
                    out_shape=values.shape,
                    transform=prediction.window_transform(window),
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                outside_coast = mangrove_pixels & ~coastal_pixels
            changed_count = int(outside_coast.sum())
            if changed_count:
                values[outside_coast] = replacement_id
                prediction.write(values, 1, window=window)
                changed_pixels += changed_count
    return changed_pixels


def _write_class_previews(
    class_ids: np.ndarray, output_dir: Path, stem: str
) -> dict[str, Path]:
    paths = {}
    palette = _class_palette()
    for class_id, label in enumerate(CLASS_ORDER, start=1):
        preview_ids = np.where(class_ids == class_id, class_id, 0).astype(np.uint8)
        preview = Image.fromarray(preview_ids, mode="P")
        preview.putpalette(palette)
        preview.info["transparency"] = bytes([0, *([255] * 255)])
        path = output_dir / f"{stem}_streamlit_{class_id}.png"
        preview.save(path, format="PNG", optimize=True)
        preview.close()
        paths[label] = path
    return paths


def _classify_scene(
    tile: str,
    files: dict[str, Path],
    boundary,
    model,
    output_path: Path,
) -> tuple[int, rasterio.crs.CRS]:
    with ExitStack() as stack:
        sources = {
            band: stack.enter_context(rasterio.open(path))
            for band, path in files.items()
        }
        reference = sources["B02"]
        if reference.crs is None:
            raise ValueError(f"CRS raster B02 tidak tersedia untuk tile {tile}.")
        for band in ("B03", "B04", "B08"):
            source = sources[band]
            if (
                source.crs != reference.crs
                or source.transform != reference.transform
                or source.shape != reference.shape
            ):
                raise ValueError(f"Grid band {band} tidak sejajar dengan B02 pada {tile}.")

        b11_source = sources["B11"]
        if b11_source.crs != reference.crs or b11_source.bounds != reference.bounds:
            raise ValueError(f"CRS/cakupan B11 tidak cocok dengan B02 pada {tile}.")
        b11_10m = stack.enter_context(
            WarpedVRT(
                b11_source,
                crs=reference.crs,
                transform=reference.transform,
                width=reference.width,
                height=reference.height,
                src_nodata=0,
                nodata=0,
                resampling=Resampling.bilinear,
            )
        )

        projected_boundary = boundary.to_crs(reference.crs).geometry.iloc[0]
        clipped_boundary = projected_boundary.intersection(box(*reference.bounds))
        if clipped_boundary.is_empty or clipped_boundary.area == 0:
            print(f"Tile {tile}: tidak beririsan dengan batas wilayah.")
            return 0, reference.crs

        window = geometry_window(reference, [mapping(clipped_boundary)])
        row_start = max(0, int(window.row_off))
        col_start = max(0, int(window.col_off))
        row_stop = min(reference.height, row_start + int(window.height))
        col_stop = min(reference.width, col_start + int(window.width))
        profile = reference.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=BLOCK_SIZE,
            blockysize=BLOCK_SIZE,
            BIGTIFF="IF_SAFER",
        )

        pixel_count = 0
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.set_band_description(1, "kelas_lulc")
            destination.update_tags(
                model="Random Forest",
                source_scenes=", ".join(path.name for path in files.values()),
                class_mapping="; ".join(
                    f"{index}: {label}"
                    for index, label in enumerate(CLASS_ORDER, start=1)
                ),
                cloud_masking="Tidak dilakukan; band SCL tidak tersedia",
            )
            for row_off in range(row_start, row_stop, BLOCK_SIZE):
                for col_off in range(col_start, col_stop, BLOCK_SIZE):
                    height = min(BLOCK_SIZE, row_stop - row_off)
                    width = min(BLOCK_SIZE, col_stop - col_off)
                    block_window = rasterio.windows.Window(
                        col_off, row_off, width, height
                    )
                    transform = reference.window_transform(block_window)
                    block_bounds = box(
                        *rasterio.windows.bounds(block_window, reference.transform)
                    )
                    block_geometry = clipped_boundary.intersection(block_bounds)
                    if block_geometry.is_empty:
                        continue
                    inside = rasterize(
                        [(mapping(block_geometry), 1)],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                    ).astype(bool)
                    arrays = np.stack(
                        [
                            sources[band].read(1, window=block_window)
                            for band in ("B02", "B03", "B04", "B08")
                        ]
                        + [b11_10m.read(1, window=block_window)]
                    ).astype(np.float32)
                    valid = inside & np.all(np.isfinite(arrays), axis=0)
                    valid &= np.all(arrays > 0, axis=0)
                    classified = np.zeros((height, width), dtype=np.uint8)
                    if valid.any():
                        features = _feature_matrix(arrays, valid)
                        predictions = model.predict(features)
                        classified[valid] = np.fromiter(
                            (CLASS_IDS[label] for label in predictions),
                            dtype=np.uint8,
                            count=len(predictions),
                        )
                        pixel_count += int(valid.sum())
                    destination.write(classified, 1, window=block_window)

    print(f"{tile}: {pixel_count:,} piksel valid diklasifikasikan.")
    return pixel_count, reference.crs


def _model_fingerprint(model_path: Path) -> str:
    digest = sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_existing_prediction(
    raster_path: Path, boundary_geometry, boundary_crs
) -> int:
    with rasterio.open(raster_path, "r+") as prediction:
        projected = gpd.GeoSeries(
            [boundary_geometry], crs=boundary_crs
        ).to_crs(prediction.crs).iloc[0]
        clipped = projected.intersection(box(*prediction.bounds))
        if clipped.is_empty or clipped.area == 0:
            for _, window in prediction.block_windows(1):
                prediction.write(
                    np.zeros(
                        (int(window.height), int(window.width)), dtype=np.uint8
                    ),
                    1,
                    window=window,
                )
            return 0

        valid_count = 0
        for _, block in prediction.block_windows(1):
            values = prediction.read(1, window=block)
            if not values.any():
                continue
            block_bounds = box(
                *rasterio.windows.bounds(block, prediction.transform)
            )
            block_geometry = clipped.intersection(block_bounds)
            if block_geometry.is_empty:
                values.fill(0)
            else:
                inside = rasterize(
                    [(mapping(block_geometry), 1)],
                    out_shape=values.shape,
                    transform=prediction.window_transform(block),
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                values[~inside] = 0
            valid_count += int(np.count_nonzero(values))
            prediction.write(values, 1, window=block)
    return valid_count


def _write_folium_overlay(
    raster_path: Path, image_path: Path, preview_path: Path
) -> tuple[list[list[float]], int, dict[str, Path]]:
    with rasterio.open(raster_path) as source:
        class_ids = source.read(1)
        scale = min(1, 2048 / source.width, 2048 / source.height)
        preview_ids = source.read(
            1,
            out_shape=(
                max(1, round(source.height * scale)),
                max(1, round(source.width * scale)),
            ),
            resampling=Resampling.nearest,
        )
        bounds = transform_bounds(
            source.crs, "EPSG:4326", *source.bounds, densify_pts=21
        )
    palette = _class_palette()
    image = Image.fromarray(class_ids, mode="P")
    image.putpalette(palette)
    image.info["transparency"] = bytes([0, *([255] * 255)])
    image.save(image_path, format="PNG", compress_level=1)
    preview = Image.fromarray(preview_ids, mode="P")
    preview.putpalette(palette)
    preview.info["transparency"] = bytes([0, *([255] * 255)])
    preview.save(preview_path, format="PNG", optimize=True)
    class_preview_paths = _write_class_previews(
        preview_ids, preview_path.parent, raster_path.stem
    )
    class_count = int(np.count_nonzero(class_ids))
    image.close()
    preview.close()
    del class_ids
    west, south, east, north = bounds
    return [[south, west], [north, east]], class_count, class_preview_paths


def refresh_streamlit_previews(project_dir: Path) -> None:
    project_dir = Path(project_dir).resolve()
    output_dir = project_dir / "data" / "hasil_klasifikasi"
    metadata_path = output_dir / "ringkasan_klasifikasi.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preview_paths = {}
    preview_bounds = {}
    palette = _class_palette()
    class_preview_paths_by_tile = {}

    for relative_path in metadata["prediction_rasters"]:
        raster_path = project_dir / relative_path
        tile = raster_path.stem.removeprefix("klasifikasi_").removesuffix("_10m")
        preview_path = output_dir / f"{raster_path.stem}_streamlit.png"
        with rasterio.open(raster_path) as source:
            scale = min(1, 2048 / source.width, 2048 / source.height)
            preview_ids = source.read(
                1,
                out_shape=(
                    max(1, round(source.height * scale)),
                    max(1, round(source.width * scale)),
                ),
                resampling=Resampling.nearest,
            )
            bounds = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
            )
        preview = Image.fromarray(preview_ids, mode="P")
        preview.putpalette(palette)
        preview.info["transparency"] = bytes([0, *([255] * 255)])
        preview.save(preview_path, format="PNG", optimize=True)
        preview.close()
        class_preview_paths_by_tile[tile] = _write_class_previews(
            preview_ids, output_dir, raster_path.stem
        )
        west, south, east, north = bounds
        preview_paths[tile] = preview_path.relative_to(project_dir).as_posix()
        preview_bounds[tile] = [[south, west], [north, east]]

    boundary = gpd.read_file(
        project_dir / "data" / "geojson" / "batas_administratif_jawa_timur.geojson"
    ).to_crs("EPSG:4326")
    west, south, east, north = boundary.total_bounds
    metadata["streamlit_previews"] = list(preview_paths.values())
    metadata["streamlit_previews_by_tile"] = preview_paths
    metadata["streamlit_class_previews_by_tile"] = {
        tile: {
            label: path.relative_to(project_dir).as_posix()
            for label, path in paths.items()
        }
        for tile, paths in class_preview_paths_by_tile.items()
    }
    metadata["overlay_bounds"] = preview_bounds
    metadata["map_bounds"] = [[float(south), float(west)], [float(north), float(east)]]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_lulc_map(project_dir: Path) -> dict[str, Path]:
    project_dir = Path(project_dir).resolve()
    data_dir = project_dir / "data"
    raster_root = data_dir / "geojson"
    output_dir = data_dir / "hasil_klasifikasi"
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_path = raster_root / "batas_administratif_jawa_timur.geojson"
    if not boundary_path.is_file():
        raise FileNotFoundError(
            f"Batas administratif rinci Jawa Timur tidak ditemukan: {boundary_path}"
        )
    boundary = gpd.read_file(boundary_path)
    samples = gpd.read_file(raster_root / "jawa_timur.geojson")
    if boundary.empty or samples.empty:
        raise ValueError("GeoJSON batas wilayah atau poligon digitalisasi kosong.")
    if "nama_provinsi" not in boundary or not boundary["nama_provinsi"].eq(
        "East Java"
    ).all():
        raise ValueError(
            "GeoJSON batas administratif harus berisi provinsi East Java."
        )
    if "Type" not in samples or "ID_Poligon" not in samples:
        raise ValueError("GeoJSON sampel harus memiliki kolom Type dan ID_Poligon.")
    boundary = boundary.to_crs("EPSG:4326")
    if samples.crs is None:
        raise ValueError("CRS GeoJSON poligon digitalisasi tidak tersedia.")
    samples = samples.to_crs("EPSG:4326")
    if set(samples["Type"].dropna().unique()) != set(CLASS_ORDER):
        raise ValueError("Label poligon digitalisasi tidak sesuai dengan lima kelas model.")

    model_path = data_dir / "model_rf_lulc.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model Random Forest tidak ditemukan: {model_path}")
    model = joblib.load(model_path)
    if tuple(getattr(model, "feature_names_in_", FEATURES)) != FEATURES:
        raise ValueError("Urutan fitur pada model tidak cocok dengan fitur Sentinel-2.")
    if set(model.classes_) != set(CLASS_ORDER):
        raise ValueError("Kelas pada model tidak cocok dengan lima kelas peta.")
    model_fingerprint = _model_fingerprint(model_path)
    boundary_fingerprint = _model_fingerprint(boundary_path)

    coastline_path = raster_root / "garis_pantai_jawa_timur.geojson"
    if not coastline_path.is_file():
        raise FileNotFoundError(f"Data garis pantai tidak ditemukan: {coastline_path}")
    coastline = gpd.read_file(coastline_path)
    if coastline.empty or coastline.crs is None:
        raise ValueError("GeoJSON garis pantai kosong atau tidak memiliki CRS.")
    coastline_geometry = coastline.geometry.union_all()
    if coastline_geometry.is_empty:
        raise ValueError("Geometri garis pantai kosong.")
    coastline_fingerprint = _model_fingerprint(coastline_path)

    previous_metadata_path = output_dir / "ringkasan_klasifikasi.json"
    previous_metadata = (
        json.loads(previous_metadata_path.read_text(encoding="utf-8"))
        if previous_metadata_path.is_file()
        else {}
    )
    existing_rasters = sorted(output_dir.glob("klasifikasi_*_10m.tif"))
    available_scenes = list(iter_sentinel_tile_rasters(raster_root))
    if available_scenes:
        selected_scenes = _latest_complete_scenes(raster_root)
        print("Scene Sentinel-2 yang dipakai (scene terbaru per tile):")
    elif existing_rasters:
        selected_scenes = {}
        print(
            "Citra Sentinel-2 sumber tidak tersedia; memperbarui GeoTIFF "
            "klasifikasi yang ada."
        )
    else:
        selected_scenes = _latest_complete_scenes(raster_root)

    raster_paths = []
    total_pixels = 0
    projected_boundary = boundary.to_crs("EPSG:32749")
    boundary_wgs84 = boundary.to_crs("EPSG:4326")
    source_scene_tags = {
        tile: json.dumps(
            {band: path.name for band, path in files.items()}, sort_keys=True
        )
        for tile, files in selected_scenes.items()
    }
    for tile, files in selected_scenes.items():
        output_path = output_dir / f"klasifikasi_{tile}_10m.tif"
        same_model_and_scenes = False
        same_boundary = False
        same_coastline = False
        if output_path.is_file():
            with rasterio.open(output_path) as existing:
                tags = existing.tags()
                same_model_and_scenes = (
                    tags.get("model") == "Random Forest"
                    and tags.get("model_sha256") == model_fingerprint
                    and tags.get("source_scenes") == source_scene_tags[tile]
                )
                same_boundary = tags.get("boundary_sha256") == boundary_fingerprint
                same_boundary &= (
                    tags.get("boundary_mask_version") == BOUNDARY_MASK_VERSION
                )
                same_coastline = (
                    tags.get("coastline_sha256") == coastline_fingerprint
                    and tags.get("mangrove_coastal_buffer_m")
                    == str(MANGROVE_COASTAL_BUFFER_METERS)
                    and tags.get("mangrove_coast_mask_version")
                    == MANGROVE_COAST_MASK_VERSION
                )
                if same_model_and_scenes and same_boundary:
                    count = int(
                        np.count_nonzero(existing.read(1, masked=True).compressed())
                    )
        if same_model_and_scenes and same_boundary and same_coastline:
            print(f"{tile}: memakai ulang GeoTIFF klasifikasi yang sudah ada.")
        elif same_model_and_scenes:
            if not same_boundary:
                print(f"{tile}: memperbarui batas administratif klasifikasi.")
                count = _clip_existing_prediction(
                    output_path,
                    boundary.geometry.iloc[0],
                    boundary.crs,
                )
            if not same_coastline:
                changed = _apply_mangrove_coastal_constraint(
                    output_path, coastline_geometry, coastline.crs
                )
                print(
                    f"{tile}: {changed:,} piksel mangrove di luar zona pesisir "
                    "dikoreksi."
                )
        else:
            count, _ = _classify_scene(tile, files, boundary, model, output_path)
            changed = _apply_mangrove_coastal_constraint(
                output_path, coastline_geometry, coastline.crs
            )
            print(
                f"{tile}: {changed:,} piksel mangrove di luar zona pesisir "
                "dikoreksi."
            )
        with rasterio.open(output_path, "r+") as classified:
            classified.update_tags(
                model_sha256=model_fingerprint,
                source_scenes=source_scene_tags[tile],
                boundary_sha256=boundary_fingerprint,
                boundary_mask_version=BOUNDARY_MASK_VERSION,
                coastline_sha256=coastline_fingerprint,
                coastline_source=COASTLINE_SOURCE,
                mangrove_coastal_buffer_m=str(MANGROVE_COASTAL_BUFFER_METERS),
                mangrove_coast_mask_version=MANGROVE_COAST_MASK_VERSION,
            )
        total_pixels += count
        raster_paths.append(output_path)
        print(f"  {tile}: {next(iter(files.values())).name}")

    if not selected_scenes:
        for output_path in existing_rasters:
            tile = output_path.stem.removeprefix("klasifikasi_").removesuffix("_10m")
            same_boundary = False
            same_coastline = False
            with rasterio.open(output_path) as existing:
                tags = existing.tags()
                same_boundary = (
                    tags.get("boundary_sha256") == boundary_fingerprint
                    and tags.get("boundary_mask_version") == BOUNDARY_MASK_VERSION
                )
                same_coastline = (
                    tags.get("coastline_sha256") == coastline_fingerprint
                    and tags.get("mangrove_coastal_buffer_m")
                    == str(MANGROVE_COASTAL_BUFFER_METERS)
                    and tags.get("mangrove_coast_mask_version")
                    == MANGROVE_COAST_MASK_VERSION
                )
                count = int(
                    np.count_nonzero(existing.read(1, masked=True).compressed())
                )
            if not same_boundary:
                count = _clip_existing_prediction(
                    output_path,
                    boundary.geometry.iloc[0],
                    boundary.crs,
                )
            if not same_coastline:
                changed = _apply_mangrove_coastal_constraint(
                    output_path, coastline_geometry, coastline.crs
                )
                print(f"{tile}: {changed:,} piksel mangrove di luar zona pesisir dikoreksi.")
            with rasterio.open(output_path, "r+") as classified:
                classified.update_tags(
                    model_sha256=model_fingerprint,
                    boundary_sha256=boundary_fingerprint,
                    boundary_mask_version=BOUNDARY_MASK_VERSION,
                    coastline_sha256=coastline_fingerprint,
                    coastline_source=COASTLINE_SOURCE,
                    mangrove_coastal_buffer_m=str(MANGROVE_COASTAL_BUFFER_METERS),
                    mangrove_coast_mask_version=MANGROVE_COAST_MASK_VERSION,
                )
            total_pixels += count
            raster_paths.append(output_path)

    if total_pixels == 0:
        raise ValueError("Tidak ada piksel Sentinel-2 valid di dalam batas Jawa Timur.")

    raster_tiles = [
        path.stem.removeprefix("klasifikasi_").removesuffix("_10m")
        for path in raster_paths
    ]
    overlay_paths = []
    preview_paths = []
    class_preview_paths_by_tile = {}
    overlay_bounds = {}
    predicted_counts = {label: 0 for label in CLASS_ORDER}
    for path in raster_paths:
        with rasterio.open(path) as source:
            values, counts = np.unique(source.read(1), return_counts=True)
            for value, count in zip(values, counts):
                if value:
                    predicted_counts[CLASS_ORDER[int(value) - 1]] += int(count)
        overlay_path = output_dir / f"{path.stem}_folium.png"
        preview_path = output_dir / f"{path.stem}_streamlit.png"
        image_bounds, _, class_preview_paths = _write_folium_overlay(
            path, overlay_path, preview_path
        )
        overlay_paths.append(overlay_path)
        preview_paths.append(preview_path)
        tile = path.stem.removeprefix("klasifikasi_").removesuffix("_10m")
        class_preview_paths_by_tile[tile] = class_preview_paths
        overlay_bounds[path.stem] = image_bounds

    projected_boundary_geometry = projected_boundary.geometry.union_all()
    raster_footprints = []
    for files in selected_scenes.values():
        with rasterio.open(files["B02"]) as source:
            footprint = box(*source.bounds).intersection(
                projected_boundary_geometry
            )
            if not footprint.is_empty:
                raster_footprints.append(footprint)
    covered_area = (
        gpd.GeoSeries(raster_footprints, crs="EPSG:32749").union_all().area
        if raster_footprints
        else 0.0
    )
    coverage_percent = (
        covered_area * 100.0 / projected_boundary_geometry.area
        if selected_scenes
        else previous_metadata.get("raster_coverage_percent_of_boundary", 0.0)
    )
    test_pixels = pd.read_csv(
        data_dir / "csv" / "split_dataset" / "test_pixels.csv"
    )
    test_predictions = model.predict(test_pixels.loc[:, FEATURES])
    test_accuracy = accuracy_score(test_pixels["Type"], test_predictions)
    test_report = classification_report(
        test_pixels["Type"],
        test_predictions,
        labels=list(CLASS_ORDER),
        zero_division=0,
        output_dict=True,
    )
    macro_f1 = test_report["macro avg"]["f1-score"]

    m = folium.Map(
        location=[
            float(boundary_wgs84.geometry.iloc[0].centroid.y),
            float(boundary_wgs84.geometry.iloc[0].centroid.x),
        ],
        zoom_start=7,
        tiles=None,
        control_scale=False,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Citra satelit",
        overlay=False,
        control=False,
        max_zoom=19,
    ).add_to(m)
    bounds = [
        [float(boundary_wgs84.total_bounds[1]), float(boundary_wgs84.total_bounds[0])],
        [float(boundary_wgs84.total_bounds[3]), float(boundary_wgs84.total_bounds[2])],
    ]
    for class_id, label in enumerate(CLASS_ORDER, start=1):
        layer = folium.FeatureGroup(name=label, show=True).add_to(m)
        for tile, paths in class_preview_paths_by_tile.items():
            folium.raster_layers.ImageOverlay(
                image=f"https://lulc-assets.invalid/{paths[label].name}",
                bounds=overlay_bounds[f"klasifikasi_{tile}_10m"],
                name=label,
                opacity=1.0,
                interactive=False,
                cross_origin=False,
                zindex=class_id,
            ).add_to(layer)

    legend_rows = "".join(
        f'<div><span style="display:inline-block;width:13px;height:13px;'
        f'background:{CLASS_COLORS[label]};margin-right:7px;'
        f'border:1px solid #555"></span>{label}</div>'
        for label in CLASS_ORDER
    )
    legend = folium.Element(
        f"""
        <div style="position:fixed;bottom:25px;left:25px;z-index:9999;
          background:rgba(255,255,255,.94);padding:12px 14px;
          border:1px solid #777;border-radius:5px;
          font:13px/1.45 Arial,sans-serif;box-shadow:0 1px 5px #777">
          {legend_rows}
        </div>
        """
    )
    m.get_root().html.add_child(legend)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds(bounds)

    map_path = output_dir / "peta_klasifikasi_lulc_jawa_timur.html"
    m.save(map_path)
    map_html = map_path.read_text(encoding="utf-8")
    map_html = "\n".join(line.rstrip() for line in map_html.splitlines()) + "\n"
    asset_prefix = "https://lulc-assets.invalid/"
    if map_html.count(asset_prefix) != len(overlay_paths) * len(CLASS_ORDER):
        raise RuntimeError("URL overlay Folium tidak sesuai dengan jumlah tile.")
    map_path.write_text(
        map_html.replace(asset_prefix, "./"),
        encoding="utf-8",
    )
    metadata_path = output_dir / "ringkasan_klasifikasi.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": str(model_path.relative_to(project_dir)),
                "method": "Random Forest",
                "boundary_file": boundary_path.relative_to(project_dir).as_posix(),
                "boundary_source": BOUNDARY_SOURCE,
                "boundary_source_url": BOUNDARY_SOURCE_URL,
                "boundary_license": "Open Data Commons Open Database License (ODbL) 1.0",
                "coastline_file": coastline_path.relative_to(project_dir).as_posix(),
                "coastline_source": COASTLINE_SOURCE,
                "coastline_source_url": COASTLINE_SOURCE_URL,
                "coastline_license": COASTLINE_LICENSE,
                "mangrove_coastal_buffer_m": MANGROVE_COASTAL_BUFFER_METERS,
                "mangrove_outside_coastal_zone": "Reclassified as Hutan non mangrove",
                "classes": list(CLASS_ORDER),
                "features": list(FEATURES),
                "scene_files": {
                    tile: {band: path.name for band, path in files.items()}
                    for tile, files in selected_scenes.items()
                }
                if selected_scenes
                else previous_metadata.get("scene_files", {}),
                "prediction_rasters": [
                    path.relative_to(project_dir).as_posix() for path in raster_paths
                ],
                "folium_tile_overlays": [
                    path.relative_to(project_dir).as_posix() for path in overlay_paths
                ],
                "streamlit_previews": [
                    path.relative_to(project_dir).as_posix() for path in preview_paths
                ],
                "streamlit_previews_by_tile": {
                    tile: path.relative_to(project_dir).as_posix()
                    for tile, path in zip(raster_tiles, preview_paths)
                },
                "streamlit_class_previews_by_tile": {
                    tile: {
                        label: path.relative_to(project_dir).as_posix()
                        for label, path in paths.items()
                    }
                    for tile, paths in class_preview_paths_by_tile.items()
                },
                "overlay_bounds": {
                    tile: overlay_bounds[f"klasifikasi_{tile}_10m"]
                    for tile in raster_tiles
                },
                "map_bounds": [
                    [
                        float(boundary_wgs84.total_bounds[1]),
                        float(boundary_wgs84.total_bounds[0]),
                    ],
                    [
                        float(boundary_wgs84.total_bounds[3]),
                        float(boundary_wgs84.total_bounds[2]),
                    ],
                ],
                "map_serving_note": (
                    "Serve the output folder over HTTP so the Folium map can load "
                    "the adjacent full-resolution PNG overlays."
                ),
                "prediction_valid_pixels": total_pixels,
                "raster_coverage_percent_of_boundary": coverage_percent,
                "predicted_pixel_count_by_class": predicted_counts,
                "test_accuracy": test_accuracy,
                "test_macro_f1": macro_f1,
                "cloud_masking": "Not applied; SCL band is unavailable",
                "uncovered_area": "No prediction; no valid Sentinel-2 raster coverage",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nPeta Folium: {map_path}")
    print(f"GeoTIFF klasifikasi 10 m: {len(raster_paths)} tile di {output_dir}")
    print(f"Overlay Folium 10 m: {len(overlay_paths)} PNG tile")
    print(f"Ringkasan metode dan metrik: {metadata_path}")
    return {
        "map": map_path,
        "overlays": output_dir,
        "metadata": metadata_path,
        "rasters": output_dir,
    }


if __name__ == "__main__":
    generate_lulc_map(Path(__file__).resolve().parent)
