from contextlib import ExitStack
from hashlib import sha256
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_window, rasterize
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import box, mapping


BANDS = ("B02", "B03", "B04", "B08", "B11")
FEATURES = [*BANDS, "NDVI", "NDWI", "NDBI"]
COLUMNS = ["ID_Poligon", "Type", "Status_Split", *FEATURES]
FILENAME_PATTERN = re.compile(
    r"(?P<tile>T\d{2}[A-Z]{3})_(?P<date>\d{8}T\d{6})_"
    r"(?P<band>B\d{2})_\d+m\.jp2$"
)


def stratified_polygon_split(gdf, test_percent=30):
    if not 0 < test_percent < 100:
        raise ValueError("Persentase testing harus di antara 0 dan 100.")
    if gdf.empty:
        raise ValueError("Tidak ada poligon untuk dibagi.")

    class_counts = gdf["Type"].value_counts().sort_index()
    test_total = math.ceil(len(gdf) * test_percent / 100)
    quotas = class_counts * test_percent
    test_counts = (quotas // 100).astype(int)
    remainder = test_total - int(test_counts.sum())
    fractional = (quotas % 100).rename("remainder").reset_index()
    fractional.columns = ["Type", "remainder"]
    fractional = fractional.sort_values(
        ["remainder", "Type"], ascending=[False, True], kind="stable"
    )
    for class_name in fractional["Type"].head(remainder):
        test_counts[class_name] += 1

    test_ids = set()
    for class_name, count in test_counts.items():
        class_ids = gdf.loc[gdf["Type"] == class_name, "ID_Poligon"].tolist()
        class_ids.sort(
            key=lambda polygon_id: sha256(polygon_id.encode("utf-8")).hexdigest()
        )
        test_ids.update(class_ids[:count])

    result = gdf.copy()
    result["Status_Split"] = result["ID_Poligon"].map(
        lambda polygon_id: "Testing" if polygon_id in test_ids else "Training"
    )
    return result


def create_pixel_datasets(gdf, train_polygons, test_polygons, raster_root, output_dir):
    raster_root = Path(raster_root)
    output_dir = Path(output_dir)
    train_path = output_dir / "train_pixels.csv"
    test_path = output_dir / "test_pixels.csv"

    required_columns = {"ID_Poligon", "Type", "Status_Split", "geometry"}
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        raise ValueError(f"Kolom GeoJSON tidak lengkap: {sorted(missing_columns)}")
    if gdf["ID_Poligon"].duplicated().any():
        raise ValueError("ID_Poligon pada GeoJSON harus unik.")
    split_ids = set(train_polygons["ID_Poligon"]) | set(test_polygons["ID_Poligon"])
    if split_ids != set(gdf["ID_Poligon"]):
        raise ValueError("Daftar poligon train/test tidak mencakup GeoJSON secara tepat.")
    if not raster_root.is_dir():
        raise FileNotFoundError(f"Folder raster tidak ditemukan: {raster_root.resolve()}")

    scenes = {}
    for path in raster_root.glob("TILE_*/*.jp2"):
        match = FILENAME_PATTERN.fullmatch(path.name)
        if not match or match.group("band") not in BANDS:
            continue
        key = (match.group("tile"), match.group("date"))
        band = match.group("band")
        if band in scenes.setdefault(key, {}):
            raise ValueError(f"Band duplikat untuk scene {key}: {band}")
        scenes[key][band] = path

    if not scenes:
        raise FileNotFoundError(f"Tidak menemukan raster JP2 Sentinel-2 di {raster_root.resolve()}.")
    for key, files in scenes.items():
        missing_bands = set(BANDS) - set(files)
        if missing_bands:
            raise ValueError(f"Scene {key} tidak lengkap; band hilang: {sorted(missing_bands)}")

    latest = {}
    for (tile, acquisition), files in scenes.items():
        if tile not in latest or acquisition > latest[tile][0]:
            latest[tile] = (acquisition, files)

    print("=== SCENE SENTINEL-2 YANG DIPAKAI ===")
    for tile, (acquisition, _) in sorted(latest.items()):
        print(f"{tile}: {acquisition}")

    pixel_counts = {polygon_id: 0 for polygon_id in gdf["ID_Poligon"]}
    sample_limit = 50
    sampled_rows = {polygon_id: [] for polygon_id in gdf["ID_Poligon"]}
    random_generators = {
        polygon_id: np.random.default_rng(
            int.from_bytes(sha256(polygon_id.encode("utf-8")).digest()[:8], "little")
        )
        for polygon_id in gdf["ID_Poligon"]
    }
    block_size = 4096
    ordered_scenes = sorted(latest.items(), key=lambda item: item[1][0], reverse=True)
    for tile, (_, files) in ordered_scenes:
            with ExitStack() as stack:
                sources = {
                    band: stack.enter_context(rasterio.open(path))
                    for band, path in files.items()
                }
                reference = sources["B02"]
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

                tile_bounds = box(*reference.bounds)
                projected = gdf.to_crs(reference.crs)
                polygons = []
                chunks = {}
                for polygon in projected.itertuples():
                    geometry = polygon.geometry.intersection(tile_bounds)
                    if geometry.is_empty or geometry.area == 0:
                        continue
                    polygon_index = len(polygons)
                    polygons.append((polygon, geometry))
                    window = geometry_window(reference, [mapping(geometry)])
                    col_start = max(0, int(window.col_off))
                    row_start = max(0, int(window.row_off))
                    col_end = min(reference.width, int(window.col_off + window.width))
                    row_end = min(reference.height, int(window.row_off + window.height))
                    for block_row in range(
                        row_start // block_size, (row_end - 1) // block_size + 1
                    ):
                        for block_col in range(
                            col_start // block_size, (col_end - 1) // block_size + 1
                        ):
                            chunks.setdefault((block_row, block_col), []).append(
                                polygon_index
                            )

                id_values = np.asarray([item[0].ID_Poligon for item in polygons])
                type_values = np.asarray([item[0].Type for item in polygons])
                status_values = np.asarray([item[0].Status_Split for item in polygons])

                for block_row, block_col in sorted(chunks):
                    row_start = block_row * block_size
                    col_start = block_col * block_size
                    window = Window(
                        col_start,
                        row_start,
                        min(block_size, reference.width - col_start),
                        min(block_size, reference.height - row_start),
                    )
                    shape = (int(window.height), int(window.width))
                    transform = reference.window_transform(window)
                    window_bounds = box(
                        *rasterio.windows.bounds(window, reference.transform)
                    )
                    shapes = []
                    for polygon_index in chunks[(block_row, block_col)]:
                        _, geometry = polygons[polygon_index]
                        geometry = geometry.intersection(window_bounds)
                        if not geometry.is_empty:
                            shapes.append((mapping(geometry), polygon_index + 1))
                    labels = rasterize(
                        shapes,
                        out_shape=shape,
                        transform=transform,
                        fill=0,
                        dtype="int32",
                    )
                    arrays = np.stack(
                        [
                            sources[band].read(1, window=window)
                            for band in ("B02", "B03", "B04", "B08")
                        ]
                        + [b11_10m.read(1, window=window)]
                    ).astype(np.float32)
                    valid = (
                        (labels > 0)
                        & np.all(np.isfinite(arrays), axis=0)
                        & np.all(arrays > 0, axis=0)
                    )
                    row_indices, column_indices = np.where(valid)
                    if not len(row_indices):
                        continue
                    polygon_indices = labels[row_indices, column_indices] - 1
                    values = arrays[:, row_indices, column_indices].T
                    ids = id_values[polygon_indices]
                    types = type_values[polygon_indices]
                    statuses = status_values[polygon_indices]

                    for index in range(len(polygon_indices)):
                        polygon_id = ids[index]
                        pixel_counts[polygon_id] += 1
                        sample = sampled_rows[polygon_id]
                        b02, b03, b04, b08, b11 = values[index]
                        row = {
                            "ID_Poligon": polygon_id,
                            "Type": types[index],
                            "Status_Split": statuses[index],
                            "B02": b02,
                            "B03": b03,
                            "B04": b04,
                            "B08": b08,
                            "B11": b11,
                            "NDVI": (b08 - b04) / (b08 + b04),
                            "NDWI": (b03 - b08) / (b03 + b08),
                            "NDBI": (b11 - b08) / (b11 + b08),
                        }
                        if len(sample) < sample_limit:
                            sample.append(row)
                        else:
                            replacement = random_generators[polygon_id].integers(
                                pixel_counts[polygon_id]
                            )
                            if replacement < sample_limit:
                                sample[replacement] = row

    empty_polygons = [
        polygon_id for polygon_id, count in pixel_counts.items() if count == 0
    ]
    available_ids = set(pixel_counts) - set(empty_polygons)
    eligible_polygons = stratified_polygon_split(
        gdf.loc[gdf["ID_Poligon"].isin(available_ids)].copy()
    )
    train_polygons = eligible_polygons.loc[
        eligible_polygons["Status_Split"] == "Training",
        ["ID_Poligon", "Type", "Status_Split"],
    ].copy()
    test_polygons = eligible_polygons.loc[
        eligible_polygons["Status_Split"] == "Testing",
        ["ID_Poligon", "Type", "Status_Split"],
    ].copy()
    print(
        "\n=== PEMBAGIAN POLIGON FINAL SETELAH FILTER CAKUPAN RASTER ==="
    )
    print(
        pd.crosstab(
            eligible_polygons["Type"],
            eligible_polygons["Status_Split"],
            margins=True,
        )
    )
    final_status = eligible_polygons.set_index("ID_Poligon")["Status_Split"]
    for samples in sampled_rows.values():
        for row in samples:
            if row["ID_Poligon"] in final_status:
                row["Status_Split"] = final_status[row["ID_Poligon"]]
    rows = [row for samples in sampled_rows.values() for row in samples]
    if not rows:
        raise ValueError("Tidak ada sampel piksel Sentinel-2 yang valid.")
    pixels = pd.DataFrame(rows, columns=COLUMNS)
    train = pixels.loc[pixels["Status_Split"] == "Training"].copy()
    test = pixels.loc[pixels["Status_Split"] == "Testing"].copy()
    if train.empty or test.empty:
        raise ValueError("Ekstraksi menghasilkan CSV training atau testing kosong.")

    expected_classes = set(eligible_polygons["Type"])
    for split, data, polygons, status in (
        ("Training", train, train_polygons, "Training"),
        ("Testing", test, test_polygons, "Testing"),
    ):
        expected_ids = set(polygons["ID_Poligon"])
        if set(data["ID_Poligon"]) != expected_ids:
            raise ValueError(f"ID poligon CSV {split} tidak cocok dengan split GeoJSON.")
        if not data["Status_Split"].eq(status).all():
            raise ValueError(f"Status_Split CSV {split} tidak valid.")
        expected_types = polygons.set_index("ID_Poligon")["Type"]
        if not data["Type"].eq(data["ID_Poligon"].map(expected_types)).all():
            raise ValueError(f"Label kelas CSV {split} tidak cocok dengan GeoJSON.")
        if set(data["Type"]) != expected_classes:
            raise ValueError(
                f"Kelas pada split {split} tidak lengkap setelah poligon tanpa "
                f"data dikeluarkan."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    train_polygons.to_csv(output_dir / "train_polygons.csv", index=False)
    test_polygons.to_csv(output_dir / "test_polygons.csv", index=False)
    if empty_polygons:
        print(
            f"[WARNING] {len(empty_polygons)} poligon tidak memiliki piksel valid pada "
            "tile yang tersedia dan tidak dimasukkan ke CSV split:"
        )
        print(
            gdf.loc[gdf["ID_Poligon"].isin(empty_polygons), ["ID_Poligon", "Type", "Status_Split"]]
            .to_string(index=False)
        )
    print(f"Training: {len(train):,} piksel dari {train['ID_Poligon'].nunique()} poligon")
    print(f"Testing : {len(test):,} piksel dari {test['ID_Poligon'].nunique()} poligon")
    print(
        pd.crosstab(
            pixels["Type"], pixels["Status_Split"], margins=True
        )
    )
    print(
        f"Total piksel valid yang diperiksa: {sum(pixel_counts.values()):,}; "
        f"maksimal {sample_limit} sampel per poligon."
    )
    print("Catatan: nilai band bernilai nol dibuang; cloud masking tidak tersedia karena band SCL tidak disertakan.")
    print(f"CSV baru: {train_path} dan {test_path}")
    return train, test, train_polygons, test_polygons
