import numpy as np
from PIL import Image, ImageFilter, ImageStat


FEATURE_COLUMNS = [
    "mean_red",
    "std_red",
    "mean_green",
    "std_green",
    "mean_blue",
    "std_blue",
    "mean_overall",
    "std_overall",
    "mean_hue",
    "std_hue",
    "mean_sat",
    "std_sat",
    "mean_val",
    "std_val",
    "gray_mean",
    "gray_std",
    "edge_mean",
    "edge_std",
    "quad_mean_tl",
    "quad_mean_tr",
    "quad_mean_bl",
    "quad_mean_br",
    "quad_std_tl",
    "quad_std_tr",
    "quad_std_bl",
    "quad_std_br",
] + [
    f"hog_r{row}_c{column}_b{orientation}"
    for row in range(4)
    for column in range(4)
    for orientation in range(9)
]


def extract_image_features(image: Image.Image) -> dict[str, float | int]:
    image = image.convert("RGB")
    width, height = image.size
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)

    rgb_stats = ImageStat.Stat(resized)
    red_mean, green_mean, blue_mean = rgb_stats.mean
    red_std, green_std, blue_std = rgb_stats.stddev

    hsv_stats = ImageStat.Stat(resized.convert("HSV"))
    hue_mean, saturation_mean, value_mean = hsv_stats.mean
    hue_std, saturation_std, value_std = hsv_stats.stddev

    grayscale = resized.convert("L")
    gray_stats = ImageStat.Stat(grayscale)
    edge_stats = ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES))

    quadrants = [
        grayscale.crop((0, 0, 112, 112)),
        grayscale.crop((112, 0, 224, 112)),
        grayscale.crop((0, 112, 112, 224)),
        grayscale.crop((112, 112, 224, 224)),
    ]
    quadrant_means = [ImageStat.Stat(quadrant).mean[0] for quadrant in quadrants]
    quadrant_stds = [ImageStat.Stat(quadrant).stddev[0] for quadrant in quadrants]

    gray_array = np.asarray(grayscale, dtype=np.float32)
    gradient_y, gradient_x = np.gradient(gray_array)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.degrees(np.arctan2(gradient_y, gradient_x)), 180.0)
    orientation_bin = np.floor(orientation / 20.0).astype(np.int32) % 9

    hog_features = {}
    for row in range(4):
        for column in range(4):
            top, bottom = row * 56, (row + 1) * 56
            left, right = column * 56, (column + 1) * 56
            cell_bins = orientation_bin[top:bottom, left:right].ravel()
            cell_magnitude = magnitude[top:bottom, left:right].ravel()
            histogram = np.bincount(
                cell_bins, weights=cell_magnitude, minlength=9
            ).astype(np.float32)
            histogram /= np.linalg.norm(histogram) + 1e-8
            for bin_index, value in enumerate(histogram):
                hog_features[f"hog_r{row}_c{column}_b{bin_index}"] = round(
                    float(value), 6
                )

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4),
        "mean_red": round(red_mean, 4),
        "std_red": round(red_std, 4),
        "mean_green": round(green_mean, 4),
        "std_green": round(green_std, 4),
        "mean_blue": round(blue_mean, 4),
        "std_blue": round(blue_std, 4),
        "mean_overall": round((red_mean + green_mean + blue_mean) / 3, 4),
        "std_overall": round((red_std + green_std + blue_std) / 3, 4),
        "mean_hue": round(hue_mean, 4),
        "std_hue": round(hue_std, 4),
        "mean_sat": round(saturation_mean, 4),
        "std_sat": round(saturation_std, 4),
        "mean_val": round(value_mean, 4),
        "std_val": round(value_std, 4),
        "gray_mean": round(gray_stats.mean[0], 4),
        "gray_std": round(gray_stats.stddev[0], 4),
        "edge_mean": round(edge_stats.mean[0], 4),
        "edge_std": round(edge_stats.stddev[0], 4),
        "quad_mean_tl": round(quadrant_means[0], 4),
        "quad_mean_tr": round(quadrant_means[1], 4),
        "quad_mean_bl": round(quadrant_means[2], 4),
        "quad_mean_br": round(quadrant_means[3], 4),
        "quad_std_tl": round(quadrant_stds[0], 4),
        "quad_std_tr": round(quadrant_stds[1], 4),
        "quad_std_bl": round(quadrant_stds[2], 4),
        "quad_std_br": round(quadrant_stds[3], 4),
        **hog_features,
    }
