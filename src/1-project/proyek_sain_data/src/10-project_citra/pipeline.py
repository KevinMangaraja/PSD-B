import json
import re
import shutil
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_COLUMNS, extract_image_features


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parents[1] / "data"
PREPARATION_DIR = PROJECT_DIR / "3-data_preparation"
PROCESSED_DIR = PREPARATION_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "4-modeling" / "models"
EVALUATION_DIR = PROJECT_DIR / "5-evaluation" / "results"
DEPLOYMENT_DIR = PROJECT_DIR / "6-deployment"

CLASS_FOLDERS = {"RAPI": 1, "BERANTAKAN": 0}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TEST_GROUP = "kelompok 2"
def _normalize_group(name: str) -> str:
    match = re.search(r"\d+", name)
    if not match:
        raise ValueError(f"Nama kelompok tidak memiliki nomor: {name}")
    return f"kelompok {match.group(0)}"


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Fitur tidak ditemukan pada dataset: {sorted(missing)}")
    return FEATURE_COLUMNS.copy()


def prepare_dataset() -> pd.DataFrame:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {DATASET_DIR}")

    records = []
    label_review_candidates = []
    for folder_name, label in CLASS_FOLDERS.items():
        class_dir = DATASET_DIR / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(image_path) as image:
                features = extract_image_features(image)
            filename_hint = image_path.stem.casefold()
            if (label == 1 and "berantakan" in filename_hint) or (
                label == 0 and "rapi" in filename_hint
            ):
                label_review_candidates.append(
                    {
                        "rel_path": image_path.relative_to(DATASET_DIR).as_posix(),
                        "folder_label": "Rapi" if label == 1 else "Berantakan",
                        "filename_hint": "Berantakan" if label == 1 else "Rapi",
                        "review_reason": "Nama berkas mengindikasikan kelas yang berbeda dari folder.",
                    }
                )
            records.append(
                {
                    **features,
                    "filename": image_path.name,
                    "rel_path": image_path.relative_to(DATASET_DIR).as_posix(),
                    "kelompok": _normalize_group(image_path.parent.name),
                    "class": "Rapi" if label == 1 else "Berantakan",
                    "label": label,
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Dataset tidak memiliki citra yang dapat diproses.")
    if set(frame["label"]) != {0, 1}:
        raise ValueError("Dataset harus memuat citra dari kedua kelas.")
    if frame["rel_path"].duplicated().any():
        raise ValueError("Manifest dataset memiliki path gambar duplikat.")

    groups = set(frame["kelompok"])
    if TEST_GROUP not in groups:
        raise ValueError(
            f"Kelompok uji harus tersedia ({TEST_GROUP}); kelompok ditemukan: {sorted(groups)}"
        )

    frame["split"] = "train"
    frame.loc[frame["kelompok"] == TEST_GROUP, "split"] = "test"

    split_groups = {
        split: set(part["kelompok"])
        for split, part in frame.groupby("split")
    }
    if split_groups["train"] & split_groups["test"]:
        raise ValueError("Ditemukan kelompok yang bocor lintas split.")
    for split, part in frame.groupby("split"):
        if set(part["label"]) != {0, 1}:
            raise ValueError(f"Split {split} tidak memuat kedua kelas.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PREPARATION_DIR / "dataset_manifest.csv"
    frame.to_csv(PROCESSED_DIR / "dataset_manifest.csv", index=False)
    frame.to_csv(manifest_path, index=False)
    review_path = PROCESSED_DIR / "label_review_candidates.csv"
    pd.DataFrame(
        label_review_candidates,
        columns=["rel_path", "folder_label", "filename_hint", "review_reason"],
    ).to_csv(review_path, index=False)
    for split in ("train", "test"):
        frame.loc[frame["split"] == split].to_csv(
            PROCESSED_DIR / f"{split}_features.csv", index=False
        )

    feature_columns = _feature_columns(frame)

    print(f"Dataset: {len(frame)} citra; fitur: {len(feature_columns)}")
    print(frame.groupby(["split", "class"]).size().unstack(fill_value=0))
    print(f"Train: semua kelompok kecuali {TEST_GROUP}; test: {TEST_GROUP}")
    if label_review_candidates:
        print(
            f"Peringatan: {len(label_review_candidates)} kandidat label perlu ditinjau; "
            f"lihat {review_path}"
        )
    print(f"Manifest tersimpan: {manifest_path}")
    return frame


def train_model(frame: pd.DataFrame | None = None) -> dict:
    if frame is None:
        manifest_path = PROCESSED_DIR / "dataset_manifest.csv"
        if not manifest_path.is_file():
            frame = prepare_dataset()
        else:
            frame = pd.read_csv(manifest_path)

    feature_columns = _feature_columns(frame)
    training_frame = frame.loc[frame["split"] != "test"].reset_index(drop=True)
    if training_frame.empty or set(training_frame["label"]) != {0, 1}:
        raise ValueError("Data pengembangan kosong atau tidak memuat kedua kelas.")
    if training_frame["kelompok"].nunique() < 3:
        raise ValueError("Validasi group-aware membutuhkan setidaknya tiga kelompok train.")

    X = training_frame[feature_columns]
    y = training_frame["label"].to_numpy()
    groups = training_frame["kelompok"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(X, y, groups))

    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42
        ),
        "SVC": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
    }

    results = []
    best_name = None
    best_model = None
    best_macro_f1 = -1.0
    for name, candidate in models.items():
        cv_results = cross_validate(
            make_pipeline(StandardScaler(), candidate),
            X,
            y,
            cv=folds,
            scoring={
                "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
                "accuracy": "accuracy",
                "precision_macro": make_scorer(
                    precision_score, average="macro", zero_division=0
                ),
                "recall_macro": make_scorer(
                    recall_score, average="macro", zero_division=0
                ),
            },
            n_jobs=1,
        )
        macro_f1 = float(cv_results["test_macro_f1"].mean())
        result = {
            "model": name,
            "validation_accuracy_mean": float(cv_results["test_accuracy"].mean()),
            "validation_accuracy_std": float(cv_results["test_accuracy"].std()),
            "validation_macro_f1": float(macro_f1),
            "validation_macro_f1_std": float(cv_results["test_macro_f1"].std()),
            "validation_precision_macro_mean": float(cv_results["test_precision_macro"].mean()),
            "validation_recall_macro_mean": float(cv_results["test_recall_macro"].mean()),
        }
        results.append(result)
        if macro_f1 > best_macro_f1:
            best_name = name
            best_model = candidate
            best_macro_f1 = macro_f1

    if best_model is None or best_name is None:
        raise RuntimeError("Tidak ada model yang berhasil dilatih.")

    final_scaler = StandardScaler().fit(training_frame[feature_columns].to_numpy())
    X_training = final_scaler.transform(training_frame[feature_columns].to_numpy())
    best_model.fit(X_training, training_frame["label"].to_numpy())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_DIR / "feature_extraction.py", DEPLOYMENT_DIR / "feature_extraction.py")
    with (MODEL_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)
    with (MODEL_DIR / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(final_scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(best_model, DEPLOYMENT_DIR / "best_model.joblib")
    joblib.dump(final_scaler, DEPLOYMENT_DIR / "scaler.joblib")
    with (DEPLOYMENT_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    print(f"Model terpilih: {best_name} (macro-F1 validasi {best_macro_f1:.3f})")
    print(f"Model akhir dilatih pada {len(training_frame)} citra data pengembangan.")
    return {"model_name": best_name, "model": best_model, "results": results}


def evaluate_model() -> dict:
    test_path = PROCESSED_DIR / "test_features.csv"
    model_path = MODEL_DIR / "best_model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    feature_path = MODEL_DIR / "feature_columns.json"
    if not all(path.is_file() for path in (test_path, model_path, scaler_path, feature_path)):
        raise FileNotFoundError("Artefak test/model belum lengkap; jalankan prepare dan train.")

    test_frame = pd.read_csv(test_path)
    with feature_path.open(encoding="utf-8") as file:
        feature_columns = json.load(file)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if set(test_frame["split"]) != {"test"}:
        raise ValueError("Berkas test_features.csv berisi data selain split test.")
    test_groups = set(test_frame["kelompok"])
    train_val = pd.read_csv(PROCESSED_DIR / "dataset_manifest.csv")
    train_val_groups = set(
        train_val.loc[train_val["split"].isin(["train", "val"]), "kelompok"]
    )
    if test_groups & train_val_groups:
        raise ValueError("Kelompok pada test set juga muncul pada train/validasi.")

    X_test = scaler.transform(test_frame[feature_columns].to_numpy())
    y_test = test_frame["label"].to_numpy()
    predictions = model.predict(X_test)
    probability_by_class = {
        int(label): probability
        for label, probability in zip(model.classes_, model.predict_proba(X_test).T)
    }
    positive_probabilities = probability_by_class[1]

    metrics = {
        "model_name": type(model).__name__,
        "test_groups": sorted(test_groups),
        "test_samples": int(len(test_frame)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "test_roc_auc": float(roc_auc_score(y_test, positive_probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Berantakan", "Rapi"],
            output_dict=True,
            zero_division=0,
        ),
        "target_accuracy_met": bool(accuracy_score(y_test, predictions) >= 0.85),
        "target_macro_f1_met": bool(
            f1_score(y_test, predictions, average="macro", zero_division=0) >= 0.83
        ),
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with (EVALUATION_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame(
        {
            "rel_path": test_frame["rel_path"],
            "kelompok": test_frame["kelompok"],
            "actual_label": y_test,
            "predicted_label": predictions,
            "probability_rapi": positive_probabilities,
        }
    ).to_csv(EVALUATION_DIR / "test_predictions.csv", index=False)

    print(
        f"Evaluasi pada {metrics['test_samples']} citra dari kelompok "
        f"{', '.join(metrics['test_groups'])}: "
        f"accuracy={metrics['test_accuracy']:.3f}, "
        f"macro-F1={metrics['test_f1_macro']:.3f}, "
        f"ROC-AUC={metrics['test_roc_auc']:.3f}"
    )
    return metrics


def run_pipeline() -> dict:
    frame = prepare_dataset()
    training = train_model(frame)
    metrics = evaluate_model()
    return {"training": training, "evaluation": metrics}
