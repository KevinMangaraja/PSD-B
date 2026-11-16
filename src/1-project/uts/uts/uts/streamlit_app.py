import base64
import json
from io import BytesIO
from pathlib import Path

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium
from folium import Element


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "hasil_klasifikasi"
METADATA_PATH = OUTPUT_DIR / "ringkasan_klasifikasi.json"
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


def _project_path(relative_path: str) -> Path:
    parts = relative_path.replace("\\", "/").split("/")
    if not relative_path or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Path asset tidak valid: {relative_path!r}")
    return PROJECT_DIR.joinpath(*parts)


@st.cache_data(show_spinner=False)
def _load_metadata(metadata_path: str, modified_ns: int, size: int) -> dict:
    del modified_ns, size
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, max_entries=80)
def _preview_image_url(
    path_string: str, modified_ns: int, size: int, class_id: int
) -> str:
    del modified_ns, size
    with Image.open(path_string) as source:
        image = source.convert("P")
    transparency = bytearray([0, *([255] * 255)])
    for other_class_id in range(1, len(CLASS_ORDER) + 1):
        if other_class_id != class_id:
            transparency[other_class_id] = 0
    image.info["transparency"] = bytes(transparency)
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image.close()
    encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_map(metadata: dict, opacity: float) -> folium.Map:
    south_west, north_east = metadata["map_bounds"]
    map_view = folium.Map(
        location=[
            (south_west[0] + north_east[0]) / 2,
            (south_west[1] + north_east[1]) / 2,
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
    ).add_to(map_view)

    for class_id, label in enumerate(CLASS_ORDER, start=1):
        layer = folium.FeatureGroup(name=label, show=True).add_to(map_view)
        for tile, fallback_path in metadata["streamlit_previews_by_tile"].items():
            relative_path = metadata.get(
                "streamlit_class_previews_by_tile", {}
            ).get(tile, {}).get(label, fallback_path)
            image_path = _project_path(relative_path)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Preview klasifikasi tidak ditemukan: {image_path}"
                )
            stat = image_path.stat()
            image_url = _preview_image_url(
                str(image_path), stat.st_mtime_ns, stat.st_size, class_id
            )
            folium.raster_layers.ImageOverlay(
                image=image_url,
                bounds=metadata["overlay_bounds"][tile],
                name=label,
                opacity=opacity,
                interactive=False,
                zindex=class_id,
            ).add_to(layer)

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
    folium.LayerControl(collapsed=False, position="topright").add_to(map_view)
    map_view.fit_bounds([south_west, north_east])
    return map_view


def main() -> None:
    st.set_page_config(page_title="Peta LULC Jawa Timur", layout="wide")
    if not METADATA_PATH.is_file():
        st.error(
            "Data peta belum tersedia. Jalankan generate_lulc_map.py secara lokal, "
            "lalu commit metadata dan preview PNG ke repository."
        )
        st.stop()
    metadata_stat = METADATA_PATH.stat()
    metadata = _load_metadata(
        str(METADATA_PATH), metadata_stat.st_mtime_ns, metadata_stat.st_size
    )
    required_metadata = {
        "streamlit_previews_by_tile",
        "overlay_bounds",
        "map_bounds",
    }
    if not required_metadata.issubset(metadata):
        st.error(
            "Metadata peta belum memiliki bounds dan preview deployment. "
            "Jalankan ulang generate_lulc_map.py, lalu commit hasilnya."
        )
        st.stop()
    preview_paths = [
        _project_path(relative_path)
        for relative_path in metadata["streamlit_previews_by_tile"].values()
    ]
    missing_previews = [str(path) for path in preview_paths if not path.is_file()]
    if missing_previews:
        st.error(
            "File preview klasifikasi belum ada di repository. Jalankan "
            "generate_lulc_map.py lalu commit file streamlit PNG berikut:\n\n"
            + "\n".join(missing_previews)
        )
        st.stop()
    full_map = st.query_params.get("view") == "folium"

    if not full_map:
        st.title("Klasifikasi LULC Jawa Timur")
        st.link_button("Buka peta Folium", "?view=folium")

    transparency_percent = st.slider(
        "Transparansi klasifikasi (0% = jelas, 100% = transparan)",
        min_value=0,
        max_value=100,
        value=0,
        format="%d%%",
    )
    map_view = _build_map(metadata, 1 - transparency_percent / 100)
    st_folium(map_view, height=780, use_container_width=True, returned_objects=[])


if __name__ == "__main__":
    main()
