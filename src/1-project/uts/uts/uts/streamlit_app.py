import base64
import io
import json
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import rasterio
import streamlit as st
from folium import Element
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from generate_lulc_map import (
    CLASS_COLORS,
    CLASS_ORDER,
    _latest_complete_scenes,
    generate_lulc_map,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RASTER_DIR = DATA_DIR / "geojson"
OUTPUT_DIR = DATA_DIR / "hasil_klasifikasi"
METADATA_PATH = OUTPUT_DIR / "ringkasan_klasifikasi.json"
BOUNDARY_PATH = RASTER_DIR / "batas_administratif_jawa_timur.geojson"
PREVIEW_SIZE = (2048, 2048)


@st.cache_data(show_spinner=False, max_entries=16)
def _preview_image_url(path_string: str, modified_ns: int, size: int) -> str:
    del modified_ns, size
    buffer = io.BytesIO()
    with rasterio.open(path_string) as source:
        scale = min(
            1,
            PREVIEW_SIZE[0] / source.width,
            PREVIEW_SIZE[1] / source.height,
        )
        width = max(1, round(source.width * scale))
        height = max(1, round(source.height * scale))
        class_ids = source.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )
    palette = np.zeros((256, 4), dtype=np.uint8)
    for class_id, label in enumerate(CLASS_ORDER, start=1):
        color = CLASS_COLORS[label].lstrip("#")
        palette[class_id] = [
            int(color[offset : offset + 2], 16) for offset in (0, 2, 4)
        ] + [255]
    image = Image.fromarray(palette[class_ids])
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _expected_scenes() -> dict[str, dict[str, str]]:
    return {
        tile: {band: path.name for band, path in files.items()}
        for tile, files in _latest_complete_scenes(RASTER_DIR).items()
    }


def _ensure_current_classification() -> dict:
    expected_scenes = _expected_scenes()
    metadata = (
        json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if METADATA_PATH.is_file()
        else {}
    )
    outputs_exist = all(
        (PROJECT_DIR / relative_path).is_file()
        for relative_path in metadata.get("prediction_rasters", [])
    )
    if metadata.get("scene_files") != expected_scenes or not outputs_exist:
        with st.spinner("Mengklasifikasikan seluruh tile Sentinel-2..."):
            generate_lulc_map(PROJECT_DIR)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return metadata


def _build_map(metadata: dict, opacity: float) -> folium.Map:
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
    bounds = boundary.total_bounds
    center = boundary.geometry.iloc[0].representative_point()
    map_view = folium.Map(
        location=[float(center.y), float(center.x)],
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
    ).add_to(map_view)

    for tile in metadata["scene_files"]:
        raster_path = OUTPUT_DIR / f"klasifikasi_{tile}_10m.tif"
        if not raster_path.is_file():
            raise FileNotFoundError(f"Output klasifikasi tidak ditemukan: {raster_path}")
        stat = raster_path.stat()
        image_url = _preview_image_url(
            str(raster_path), stat.st_mtime_ns, stat.st_size
        )
        with rasterio.open(raster_path) as source:
            image_bounds = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds, densify_pts=21
            )
        west, south, east, north = image_bounds
        folium.raster_layers.ImageOverlay(
            image=image_url,
            bounds=[[south, west], [north, east]],
            name=f"RF {tile}",
            opacity=opacity,
            interactive=False,
            zindex=1,
            show=True,
        ).add_to(map_view)

    legend_rows = "".join(
        f'<div><span style="display:inline-block;width:13px;height:13px;'
        f'background:{CLASS_COLORS[label]};margin-right:7px;'
        f'border:1px solid #555"></span>{label}</div>'
        for label in CLASS_ORDER
    )
    map_view.get_root().html.add_child(
        Element(
            f"""
            <div style="position:fixed;bottom:25px;left:25px;z-index:9999;
              background:rgba(255,255,255,.94);padding:10px 12px;
              border:1px solid #777;border-radius:5px;
              font:13px/1.45 Arial,sans-serif">
              {legend_rows}
            </div>
            """
        )
    )
    folium.LayerControl(collapsed=True).add_to(map_view)
    map_view.fit_bounds(
        [[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]]
    )
    return map_view


def _map_document(map_view: folium.Map) -> str:
    rendered = map_view.get_root().render()
    start = rendered.find("<body")
    if start == -1:
        raise RuntimeError("HTML Folium tidak memiliki elemen body.")
    start = rendered.find(">", start) + 1
    end = rendered.rfind("</body>")
    if start == 0 or end == -1:
        raise RuntimeError("Struktur body HTML Folium tidak lengkap.")
    body = rendered[start:end]
    body += """
    <script>
      window.addEventListener("load", function () {
        window.setTimeout(function () {
          const map = window["MAP_NAME"];
          if (map) {
            map.invalidateSize(true);
          }
        }, 100);
      });
    </script>
    """.replace("MAP_NAME", map_view.get_name())
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css">'
        '<style>html,body{width:100%;height:100%;margin:0;padding:0;overflow:hidden}</style>'
        '</head><body>'
        + body
        + "</body></html>"
    )


def main() -> None:
    st.set_page_config(page_title="Peta LULC Jawa Timur", layout="wide")
    metadata = _ensure_current_classification()
    full_map = st.query_params.get("view") == "folium"

    if not full_map:
        st.title("Klasifikasi LULC Jawa Timur")
        st.link_button("Buka peta Folium", "?view=folium")

    opacity_percent = st.slider(
        "Transparansi klasifikasi",
        min_value=0,
        max_value=100,
        value=100,
        format="%d%%",
    )
    map_view = _build_map(metadata, opacity_percent / 100)
    st.iframe(_map_document(map_view), height=780)


if __name__ == "__main__":
    main()
