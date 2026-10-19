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
    raster_path: Path, image_path: Path
) -> tuple[list[list[float]], int]:
    with rasterio.open(raster_path) as source:
        class_ids = source.read(1)
        bounds = transform_bounds(
            source.crs, "EPSG:4326", *source.bounds, densify_pts=21
        )
    image = Image.fromarray(class_ids, mode="P")
    palette = [0] * 768
    for class_id, label in enumerate(CLASS_ORDER, start=1):
        color = CLASS_COLORS[label].lstrip("#")
        palette[class_id * 3 : class_id * 3 + 3] = [
            int(color[offset : offset + 2], 16) for offset in (0, 2, 4)
        ]
    image.putpalette(palette)
    image.info["transparency"] = bytes([0, *([255] * 255)])
    image.save(image_path, format="PNG", compress_level=1)
    class_count = int(np.count_nonzero(class_ids))
    image.close()
    del class_ids
    west, south, east, north = bounds
    return [[south, west], [north, east]], class_count


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

    selected_scenes = _latest_complete_scenes(raster_root)
    print("Scene Sentinel-2 yang dipakai (scene terbaru per tile):")
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
        reusable = False
        if output_path.is_file():
            with rasterio.open(output_path) as existing:
                tags = existing.tags()
                reusable = (
                    tags.get("model") == "Random Forest"
                    and tags.get("model_sha256") == model_fingerprint
                    and tags.get("source_scenes") == source_scene_tags[tile]
                )
                same_boundary = tags.get("boundary_sha256") == boundary_fingerprint
                same_boundary &= (
                    tags.get("boundary_mask_version") == BOUNDARY_MASK_VERSION
                )
                if reusable and same_boundary:
                    count = int(
                        np.count_nonzero(existing.read(1, masked=True).compressed())
                    )
        if reusable and same_boundary:
            print(f"{tile}: memakai ulang GeoTIFF klasifikasi yang sudah ada.")
        elif reusable:
            print(
                f"{tile}: memotong prediksi lama ke batas administratif rinci."
            )
            count = _clip_existing_prediction(
                output_path,
                boundary.geometry.iloc[0],
                boundary.crs,
            )
        else:
            count, _ = _classify_scene(tile, files, boundary, model, output_path)
        with rasterio.open(output_path, "r+") as classified:
            classified.update_tags(
                model_sha256=model_fingerprint,
                source_scenes=source_scene_tags[tile],
                boundary_sha256=boundary_fingerprint,
                boundary_mask_version=BOUNDARY_MASK_VERSION,
            )
        total_pixels += count
        raster_paths.append(output_path)
        print(f"  {tile}: {next(iter(files.values())).name}")

    if total_pixels == 0:
        raise ValueError("Tidak ada piksel Sentinel-2 valid di dalam batas Jawa Timur.")

    overlay_paths = []
    overlay_bounds = {}
    overlay_tiles = {}
    predicted_counts = {label: 0 for label in CLASS_ORDER}
    for path in raster_paths:
        with rasterio.open(path) as source:
            values, counts = np.unique(source.read(1), return_counts=True)
            for value, count in zip(values, counts):
                if value:
                    predicted_counts[CLASS_ORDER[int(value) - 1]] += int(count)
        overlay_path = output_dir / f"{path.stem}_folium.png"
        image_bounds, _ = _write_folium_overlay(path, overlay_path)
        overlay_paths.append(overlay_path)
        overlay_bounds[path.stem] = image_bounds
        overlay_tiles[overlay_path.stem] = path.stem

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
    coverage_percent = covered_area * 100.0 / projected_boundary_geometry.area
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
    for tile, image_path in zip(selected_scenes, overlay_paths):
        folium.raster_layers.ImageOverlay(
            image=f"https://lulc-assets.invalid/{image_path.name}",
            bounds=overlay_bounds[overlay_tiles[image_path.stem]],
            name=f"RF {tile}",
            opacity=1.0,
            interactive=True,
            cross_origin=False,
            zindex=1,
            show=True,
        ).add_to(m)

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
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(bounds)

    map_path = output_dir / "peta_klasifikasi_lulc_jawa_timur.html"
    m.save(map_path)
    map_html = map_path.read_text(encoding="utf-8")
    asset_prefix = "https://lulc-assets.invalid/"
    if map_html.count(asset_prefix) != len(overlay_paths):
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
                "boundary_file": str(boundary_path.relative_to(project_dir)),
                "boundary_source": BOUNDARY_SOURCE,
                "boundary_source_url": BOUNDARY_SOURCE_URL,
                "boundary_license": "Open Data Commons Open Database License (ODbL) 1.0",
                "classes": list(CLASS_ORDER),
                "features": list(FEATURES),
                "scene_files": {
                    tile: {band: path.name for band, path in files.items()}
                    for tile, files in selected_scenes.items()
                },
                "prediction_rasters": [
                    str(path.relative_to(project_dir)) for path in raster_paths
                ],
                "folium_tile_overlays": [
                    str(path.relative_to(project_dir)) for path in overlay_paths
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
