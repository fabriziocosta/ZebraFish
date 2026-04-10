from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import TiffFile


DEFAULT_WORKBOOK = "compounds (MJW V2).xlsx"
DEFAULT_SHEET = "REVISED TABLE 25 NOV 2019"
DEFAULT_IMAGE_ROOT = "images"
DEFAULT_MANIFEST = "/tmp/zebrafish_image_dirs.txt"

UNIT_PATTERNS = [
    (r"(\d+(?:[._]\d+)?)\s*(microM|microm|um|μm)", "uM"),
    (r"(\d+(?:[._]\d+)?)\s*(mm|mM)", "mM"),
    (r"(\d+(?:[._]\d+)?)\s*(%|percent)", "%"),
]


def load_raw_workbook(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    return pd.read_excel(Path(workbook_path), sheet_name=sheet_name, header=None)


def load_compound_classification(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    raw = load_raw_workbook(workbook_path=workbook_path, sheet_name=sheet_name)
    df = (
        raw.iloc[:, [2, 3, 6, 7]]
        .rename(
            columns={
                2: "compound",
                3: "seizure_link_strength",
                6: "compound_class",
                7: "mechanism_of_action",
            }
        )
        .copy()
    )
    df = df[df["seizure_link_strength"].astype("string").str.strip().isin(list("ABCDEF"))].copy()
    df["compound"] = df["compound"].astype("string").str.strip()
    df["compound_class"] = df["compound_class"].astype("string").str.strip().replace({"": pd.NA}).ffill()
    df["mechanism_of_action"] = df["mechanism_of_action"].astype("string").str.strip()
    return df[["compound", "compound_class", "mechanism_of_action"]].reset_index(drop=True)


def load_exposure_map(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> dict[str, str]:
    raw = load_raw_workbook(workbook_path=workbook_path, sheet_name=sheet_name)
    df = raw.iloc[:, [2, 3, 8]].rename(
        columns={2: "compound", 3: "seizure_link_strength", 8: "exposure_conditions"}
    )
    df = df[df["seizure_link_strength"].astype("string").str.strip().isin(list("ABCDEF"))].copy()
    df["compound"] = df["compound"].astype("string").str.strip()
    df["exposure_conditions"] = df["exposure_conditions"].astype("string").str.strip()
    return (
        df[["compound", "exposure_conditions"]]
        .drop_duplicates(subset=["compound"])
        .set_index("compound")["exposure_conditions"]
        .to_dict()
    )


def ensure_image_manifest(
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> Path:
    image_root = Path(image_root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        result = subprocess.run(
            ["find", "-L", str(image_root), "-mindepth", "2", "-maxdepth", "3", "-type", "d"],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest_path.write_text(result.stdout)
    return manifest_path


def load_all_image_dirs(
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> list[Path]:
    ensure_image_manifest(image_root=image_root, manifest_path=manifest_path)
    manifest_path = Path(manifest_path)
    return [Path(line.strip()) for line in manifest_path.read_text().splitlines() if line.strip()]


def normalize_name(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("δ", "delta").replace("μ", "mu")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def folder_status(path: Path) -> str:
    text = " ".join(part.lower() for part in path.parts)
    if any(flag in text for flag in ["do not use", "dont use", "not using", "superceded", "abandoned", "failed"]):
        return "do_not_use"
    return "active"


def child_folder_status(child_name: str, run_folder_status: str) -> str:
    lower = child_name.lower()
    if "dont use" in lower or "do not use" in lower:
        return "do_not_use"
    return run_folder_status


def contains_alias(norm_dir: str, norm_alias: str) -> bool:
    return f" {norm_alias} " in f" {norm_dir} "


def get_compound_alias_map() -> dict[str, list[str]]:
    return {
        "Bicuculline": ["bicuculline"],
        "Pentylenetetrazole": ["ptz", "pentylenetetrazole"],
        "Picrotoxin": ["picrotoxin"],
        "Bemegride": ["bemegride"],
        "Gabazine": ["gabazine"],
        "DMCM*": ["dmcm"],
        "Strychnine": ["strychnine"],
        "Kainic acid": ["kainic acid"],
        "N-methyl-D-aspartate": ["nmda", "n methyl d aspartate"],
        "Domoic acid": ["domoic acid"],
        "Cis-ACPD**": ["cisacpd", "cis acpd"],
        "(RS)-(Tetrazol-5-yl)glycine": ["rs tetrazol glycine", "rst"],
        "4-aminopyridine": ["4ap", "4 aminopyridine"],
        "Tetraethylammonium Cl": ["tetraethylammonium chloride", "tetraethylammonium cl"],
        "XE991***": ["xe991"],
        "Pilocarpine": ["pilocarpine"],
        "Bethanechol": ["bethanechol"],
        "Carbachol": ["carbachol"],
        "Oxotremorine": ["oxotremorine"],
        "Muscarine": ["muscarine"],
        "PF06767832****": ["pf06767832"],
        "BW373U86*******": ["bw373u86"],
        "SNC80*****": ["snc80"],
        "Fentanyl": ["fentanyl"],
        "Meperidine": ["meperidine", "pethidine"],
        "Morphine": ["morphine"],
        "SB205607******": ["sb205607"],
        "Donepezil": ["donepezil"],
        "Physostigmine": ["physostigmine", "physostigmine salicylate"],
        "Tacrine": ["tacrine"],
        "Galantamine": ["galantamine"],
        "Neostigmine": ["neostigmine"],
        "Phenserine": ["phenserine"],
        "Bupropion": ["bupropion"],
        "Nomifensine": ["nomifensine"],
        "Maprotiline": ["maprotiline"],
        "Amoxapine": ["amoxapine", "amoxipine"],
        "Amitriptyline": ["amitriptyline", "amitripityline"],
        "Clomipramine": ["clomipramine"],
        "Desipiramine": ["desipramine", "desipiramine"],
        "Protriptyline": ["protriptyline"],
        "Chlorpromazine": ["chlorpromazine"],
        "Clozapine": ["clozapine"],
        "Olanzapine": ["olanzapine"],
        "Risperidone": ["risperidone"],
        "Aminophylline": ["aminophylline"],
        "Theophylline": ["theophylline"],
        "Caffeine": ["caffeine"],
        "Theobromine": ["theobromine"],
        "Paraxanthine": ["paraxanthine"],
        "Cocaine": ["cocaine"],
        "Amphetamine": ["amphetamine"],
        "Phenyclidine": ["phencyclidine"],
        "Apomorphine": ["apomorphine"],
        "Ethanol": ["ethanol"],
        "Rolipram": ["rolipram"],
        "Yohimbine": ["yohimbine"],
        "4-Aminophenyl sulfone": ["4 aminophenyl sulfone", "aminophenyl sulfone"],
        "Cisplatin": ["cisplatin"],
        "Clonidine": ["clonidine"],
        "Emetine": ["emetine", "emetic drug"],
        "Ketamine": ["ketamine"],
        "Ketoconazole": ["ketoconazole", "ketoconozole"],
        "Mizolastine": ["mizolastine"],
        "Quinine HCl": ["quinine hcl", "quinine"],
    }


def select_candidate_image_dirs(
    all_dirs: Iterable[Path],
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
) -> list[Path]:
    image_root = Path(image_root).resolve()
    candidates: list[Path] = []
    for path in all_dirs:
        rel = path.relative_to(image_root)
        parts = rel.parts
        if len(parts) not in (2, 3):
            continue
        name = path.name.lower()
        if any(flag in name for flag in ["summary", "summaries", "analysis", "data summaries", "older data", "additional groups", "figures"]):
            continue
        if any(flag in name for flag in ["output", "outputs"]):
            continue
        if len(parts) == 3:
            keep_nested = (
                parts[0] == "FIRST BATCH convulsants" and parts[1] == "Brain_Imaging_Rerun_-_New_Alignment"
            ) or (
                parts[0] == "THIRD BATCH convulsants"
                and parts[1] in ["EMETIC COMPOUNDS NOT CORRECTED FOR ANTERIOR COMMISSURE", "EMETICS RERUN"]
            )
            if not keep_nested:
                continue
        candidates.append(path)
    return candidates


def build_compound_image_run_map(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> pd.DataFrame:
    compound_df = load_compound_classification(workbook_path=workbook_path, sheet_name=sheet_name)
    all_dirs = load_all_image_dirs(image_root=image_root, manifest_path=manifest_path)
    candidates = select_candidate_image_dirs(all_dirs, image_root=image_root)
    image_root = Path(image_root).resolve()
    alias_map = get_compound_alias_map()

    rows = []
    for record in compound_df.to_dict("records"):
        aliases = [normalize_name(alias) for alias in alias_map.get(record["compound"], [record["compound"]])]
        for image_dir in candidates:
            normalized_dir_name = normalize_name(image_dir.name)
            if any(contains_alias(normalized_dir_name, alias) for alias in aliases):
                relative_path = image_dir.relative_to(image_root)
                rows.append(
                    {
                        "compound": record["compound"],
                        "compound_class": record["compound_class"],
                        "mechanism_of_action": record["mechanism_of_action"],
                        "image_run_dir": str(image_dir),
                        "image_run_dir_relative": str(relative_path),
                        "source_batch": relative_path.parts[0],
                        "dir_name": image_dir.name,
                        "folder_status": folder_status(image_dir),
                    }
                )

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["compound", "image_run_dir"])
        .sort_values(["compound", "image_run_dir"])
        .reset_index(drop=True)
    )


def build_compound_image_index(run_map_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_map_df.groupby(["compound", "compound_class", "mechanism_of_action"], dropna=False)
        .agg(
            n_image_run_dirs=("image_run_dir", "size"),
            n_active_image_run_dirs=("folder_status", lambda s: int((s == "active").sum())),
            image_run_dirs=("image_run_dir", lambda s: " | ".join(s)),
            active_image_run_dirs=(
                "image_run_dir",
                lambda s: " | ".join(
                    run_map_df.loc[s.index].loc[run_map_df.loc[s.index, "folder_status"] == "active", "image_run_dir"]
                ),
            ),
        )
        .reset_index()
        .sort_values("compound")
        .reset_index(drop=True)
    )


def infer_run_unit(exposure_text: str | None) -> str | None:
    text = str(exposure_text)
    for pattern, unit in UNIT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return unit
    if "% v/v" in text.lower() or "%" in text:
        return "%"
    return None


def normalize_num(text: str) -> float:
    return float(text.replace("_", "."))


def parse_concentration(folder_name: str, fallback_unit: str | None = None) -> dict:
    lower = folder_name.lower()
    if "control" in lower or "water" in lower:
        return {
            "condition_kind": "control",
            "concentration_value": None,
            "concentration_unit": None,
            "concentration_value_uM": None,
            "concentration_label": "control",
        }
    for pattern, canonical_unit in UNIT_PATTERNS:
        match = re.search(pattern, folder_name, flags=re.IGNORECASE)
        if match:
            value = normalize_num(match.group(1))
            value_uM = value if canonical_unit == "uM" else value * 1000 if canonical_unit == "mM" else None
            return {
                "condition_kind": "treatment",
                "concentration_value": value,
                "concentration_unit": canonical_unit,
                "concentration_value_uM": value_uM,
                "concentration_label": f"{value:g} {canonical_unit}",
            }
    match = re.search(r"(\d+(?:[._]\d+)?)", folder_name)
    if match and fallback_unit is not None:
        value = normalize_num(match.group(1))
        value_uM = value if fallback_unit == "uM" else value * 1000 if fallback_unit == "mM" else None
        return {
            "condition_kind": "treatment",
            "concentration_value": value,
            "concentration_unit": fallback_unit,
            "concentration_value_uM": value_uM,
            "concentration_label": f"{value:g} {fallback_unit}",
        }
    return {
        "condition_kind": "unknown",
        "concentration_value": None,
        "concentration_unit": fallback_unit,
        "concentration_value_uM": None,
        "concentration_label": None,
    }


def build_compound_image_condition_map(
    run_map_df: pd.DataFrame,
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    exposure_map = load_exposure_map(workbook_path=workbook_path, sheet_name=sheet_name)

    rows = []
    for record in run_map_df.to_dict("records"):
        run_dir = Path(record["image_run_dir"])
        fallback_unit = infer_run_unit(exposure_map.get(record["compound"]))
        try:
            child_dirs = sorted([path for path in run_dir.iterdir() if path.is_dir()])
        except Exception:
            child_dirs = []

        parsed_children = [(child, parse_concentration(child.name, fallback_unit=fallback_unit)) for child in child_dirs]
        treatment_values = sorted(
            {
                info["concentration_value_uM"] if info["concentration_value_uM"] is not None else info["concentration_value"]
                for _, info in parsed_children
                if info["condition_kind"] == "treatment" and info["concentration_value"] is not None
            }
        )
        n_treatment = len(treatment_values)

        for child, info in parsed_children:
            if info["condition_kind"] == "control":
                band = "control"
                rank = None
            elif info["condition_kind"] == "treatment" and info["concentration_value"] is not None:
                sortable = (
                    info["concentration_value_uM"]
                    if info["concentration_value_uM"] is not None
                    else info["concentration_value"]
                )
                rank = treatment_values.index(sortable) + 1
                if n_treatment <= 1:
                    band = "high"
                elif rank == 1:
                    band = "low"
                elif rank == n_treatment:
                    band = "high"
                else:
                    band = "mid"
            else:
                band = "unknown"
                rank = None

            rows.append(
                {
                    **record,
                    "exposure_conditions": exposure_map.get(record["compound"]),
                    "image_condition_dir": str(child),
                    "image_condition_dir_name": child.name,
                    "condition_folder_status": child_folder_status(child.name, record["folder_status"]),
                    **info,
                    "concentration_rank_in_run": rank,
                    "concentration_band": band,
                    "n_treatment_concentrations_in_run": n_treatment,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["compound", "image_run_dir", "condition_kind", "concentration_rank_in_run", "image_condition_dir_name"]
        )
        .reset_index(drop=True)
    )


def build_compound_image_condition_index(condition_map_df: pd.DataFrame) -> pd.DataFrame:
    return (
        condition_map_df.groupby(
            ["compound", "compound_class", "mechanism_of_action", "concentration_band"],
            dropna=False,
        )
        .agg(
            n_condition_dirs=("image_condition_dir", "size"),
            n_active_condition_dirs=("condition_folder_status", lambda s: int((s == "active").sum())),
            condition_dirs=("image_condition_dir", lambda s: " | ".join(s)),
        )
        .reset_index()
        .sort_values(["compound", "concentration_band"])
        .reset_index(drop=True)
    )


def timepoint_sort_key(path: Path):
    match = re.search(r"TL(\d+)", path.name)
    if match:
        return int(match.group(1))
    return path.name


def list_timepoint_files(condition_dir: str | Path) -> list[Path]:
    condition_dir = Path(condition_dir)
    direct_files = sorted(condition_dir.glob("*.tif*"), key=timepoint_sort_key)
    if direct_files:
        return direct_files
    return sorted(condition_dir.rglob("*.tif*"), key=timepoint_sort_key)


def choose_sample_files(files: list[Path], n: int = 10) -> list[Path]:
    if not files:
        return []
    if len(files) <= n:
        return files
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    return [files[i] for i in indices]


def load_mid_z_slice(path: str | Path) -> np.ndarray:
    path = Path(path)
    with TiffFile(path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = getattr(series, "axes", "")
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 2:
        return arr
    if axes and "Z" in axes:
        z_axis = axes.index("Z")
        while z_axis >= arr.ndim:
            z_axis -= 1
        mid_z = arr.shape[z_axis] // 2
        return np.take(arr, mid_z, axis=z_axis).squeeze()
    if arr.ndim >= 3:
        return arr[arr.shape[0] // 2].squeeze()
    return arr


def select_condition_choices(
    condition_map_df: pd.DataFrame,
    selected_compound: str,
    selector_column: str,
    selected_concentration: str,
    only_active: bool = True,
) -> pd.DataFrame:
    matches = condition_map_df[
        (condition_map_df["compound"] == selected_compound)
        & (condition_map_df[selector_column].astype("string") == str(selected_concentration))
    ].copy()
    if only_active:
        matches = matches[matches["condition_folder_status"] == "active"].copy()
    return (
        matches[
            [
                "compound",
                "compound_class",
                "mechanism_of_action",
                "concentration_band",
                "concentration_label",
                "image_condition_dir",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def plot_midz_time_samples(
    condition_dir: str | Path,
    sample_count: int = 10,
    title: str | None = None,
):
    condition_dir = Path(condition_dir)
    timepoint_files = list_timepoint_files(condition_dir)
    sample_files = choose_sample_files(timepoint_files, n=sample_count)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.ravel()

    for ax, file_path in zip(axes, sample_files):
        image = load_mid_z_slice(file_path)
        ax.imshow(image, cmap="gray")
        ax.set_title(file_path.name.replace("_Angle0.ome.tiff", ""), fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample_files) :]:
        ax.axis("off")

    fig.suptitle(title or condition_dir.name, fontsize=14)
    fig.tight_layout()
    return fig, axes
