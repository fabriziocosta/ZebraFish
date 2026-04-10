from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from zebrafish_tensor_utils import load_image_condition_tensor as _load_image_condition_tensor


DEFAULT_WORKBOOK = "compounds (MJW V2).xlsx"
DEFAULT_SHEET = "REVISED TABLE 25 NOV 2019"
DEFAULT_IMAGE_ROOT = "images"
DEFAULT_MANIFEST = "/tmp/zebrafish_image_dirs.txt"

UNIT_PATTERNS = [
    (r"(\d+(?:[._]\d+)?)\s*(microM|microm|um|μm)", "uM"),
    (r"(\d+(?:[._]\d+)?)\s*(mm|mM)", "mM"),
    (r"(\d+(?:[._]\d+)?)\s*(%|percent)", "%"),
]


def configure_full_dataframe_display() -> None:
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.expand_frame_repr", False)


def load_raw_workbook(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    return pd.read_excel(Path(workbook_path), sheet_name=sheet_name, header=None)


def load_compound_classification_raw(
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


def load_compound_classification(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    raw_df = load_compound_classification_raw(workbook_path=workbook_path, sheet_name=sheet_name)
    mechanism_alias_map = get_mechanism_of_action_alias_map()
    missing_aliases = raw_df.loc[~raw_df["mechanism_of_action"].isin(mechanism_alias_map), "mechanism_of_action"].unique()
    if len(missing_aliases):
        raise KeyError(f"Missing mechanism_of_action mnemonic aliases for: {missing_aliases.tolist()}")

    return (
        raw_df.assign(
            compound=lambda df: df["compound"].map(clean_compound_name),
            mechanism_of_action=lambda df: df["mechanism_of_action"].map(mechanism_alias_map),
        )[["compound", "compound_class", "mechanism_of_action"]]
        .reset_index(drop=True)
    )


def load_exposure_map(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> dict[str, str]:
    raw = load_raw_workbook(workbook_path=workbook_path, sheet_name=sheet_name)
    df = raw.iloc[:, [2, 3, 8]].rename(
        columns={2: "compound", 3: "seizure_link_strength", 8: "exposure_conditions"}
    )
    df = df[df["seizure_link_strength"].astype("string").str.strip().isin(list("ABCDEF"))].copy()
    df["compound"] = df["compound"].astype("string").str.strip().map(clean_compound_name)
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


def clean_compound_name(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    return " ".join(text.split()).strip()


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


def build_compound_standardization_map(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    compound_df = load_compound_classification_raw(workbook_path=workbook_path, sheet_name=sheet_name)
    alias_map = get_compound_alias_map()

    rows = []
    for compound_name in compound_df["compound"].drop_duplicates():
        standardized_name = clean_compound_name(compound_name)
        rows.append(
            {
                "compound_name": compound_name,
                "standardized_compound_name": standardized_name,
                "normalized_compound_name": normalize_name(compound_name),
                "mapping_source": "canonical",
            }
        )
        for alias in alias_map.get(compound_name, []):
            if alias == compound_name:
                continue
            rows.append(
                {
                    "compound_name": alias,
                    "standardized_compound_name": standardized_name,
                    "normalized_compound_name": normalize_name(alias),
                    "mapping_source": "alias",
                }
            )

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["compound_name", "standardized_compound_name"])
        .sort_values(["standardized_compound_name", "mapping_source", "compound_name"])
        .reset_index(drop=True)
    )


def get_mechanism_of_action_alias_map() -> dict[str, str]:
    return {
        "GABAAR antagonism": "GABAAR_Antagonist",
        "Reversible AChE inhibition": "AChE_Inhibitor_Reversible",
        "P1 antagonism and PDE inhibition": "P1_Antagonist_PDE_Inhibitor",
        "NMDAR activation": "NMDAR_Activation",
        "mAChR agonism (non selective)": "mAChR_Agonist_NonSelective",
        "δ-opioidR agonism": "DOR_Agonist",
        "KAR activation": "KAR_Activation",
        "Non-selective Kv blockade": "Kv_Blocker_NonSelective",
        "mAChR and nAChR agonism": "mAChR_nAChR_Agonist",
        "NA/DA reuptake inhibition": "NET_DAT_ReuptakeInhibitor",
        "NMDAR antagonism": "NMDAR_Antagonist",
        "GABAAR negative allosteric modulation": "GABAAR_NegativeAllostericModulator",
        "GlyR antagonism": "GlyR_Antagonist",
        "Kv7 (KCNQ) channel blockade": "Kv7_Blocker",
        "M1 selective mACHR agonist": "M1_mAChR_Agonist_Selective",
        "μ-opioidR agonism": "MOR_Agonist",
        "μ-opioid R agonism": "MOR_Agonist",
        "μ-δ-opioidR activation": "MOR_DOR_Activation",
        "Reversible AChE inhibition (no BBB penetration)": "AChE_Inhibitor_Reversible_NoBBB",
        "NA reuptake inhibitor (weak 5-HT), mAChR, a1/2R/H1R antagonism": "NET_ReuptakeInhibitor_Weak5HT_MultiReceptorAntagonist",
        "NA/5-HT reuptake inhibitor,DAR/5-HTR/a1/2R/H1R antagonism": "NET_SERT_ReuptakeInhibitor_MultiReceptorAntagonist",
        "NA/5-HT reuptake inhibitor,5-HTR/a1R/H1/2R/mAChR antagonism": "NET_SERT_ReuptakeInhibitor_MultiReceptorAntagonist",
        "NA/5-HT reuptake inhibitor,5-HTR/a1R/DAR/H1/2R/mAChR antagonism": "NET_SERT_ReuptakeInhibitor_MultiReceptorAntagonist",
        "NA/5-HT reuptake inhibitor, a1R/H1R/mAChR antagonism": "NET_SERT_ReuptakeInhibitor_MultiReceptorAntagonist",
        "NA/5-HT reuptake inhibitor,5-HT2R/a1R/DAR/H1R/mAChR antagonism": "NET_SERT_ReuptakeInhibitor_MultiReceptorAntagonist",
        "D1/2/3/4R/a1/2R/H1R/mACh1/2R/5-HT1/2R antagonism": "D1_D2_D3_D4R_MultiReceptorAntagonist",
        "D1/2R/a1/2R/H1R/mAChR/5-HT2R antagonism": "D1_D2R_MultiReceptorAntagonist",
        "D2R/a1R/H1R/mAChR/5-HT2R antagonism": "D2R_MultiReceptorAntagonist",
        "D2R/a1/2R/H1R/mAChR/5-HT2R antagonism": "D2R_MultiReceptorAntagonist",
        "DA/NA/5-HT reuptake inhibiton and Na2+ channel blockade": "DAT_NET_SERT_ReuptakeInhibitor_NaChannelBlocker",
        "DA/NA/5-HT reuptake inhibiton": "DAT_NET_SERT_ReuptakeInhibitor",
        "D1/2R/5-HT/aR agonism": "D1_D2R_5HTR_aR_Agonist",
        "GABAAR positIve allosteric modulation": "GABAAR_PositiveAllostericModulator",
        "PDE-4 inhibition": "PDE4_Inhibitor",
        "a2R antagonism": "a2R_Antagonist",
        "Antibiotic (sulfonamide-like)": "SulfonamideLike_Antibiotic",
        "DNA cross linking chemotherapeutic": "DNA_CrossLinking_Chemotherapeutic",
        "a2R agonism (in CNS)": "a2R_Agonist_CNS",
        "Alkaloid antiprotozoal": "Alkaloid_Antiprotozoal",
        "Cytochrome P450 inhibitor": "CYP450_Inhibitor",
        "H1R antagonism": "H1R_Antagonist",
        "Alkaloid antimalarial": "Alkaloid_Antimalarial",
    }


def build_mechanism_of_action_alias_map(
    workbook_path: str | Path = DEFAULT_WORKBOOK,
    sheet_name: str = DEFAULT_SHEET,
) -> pd.DataFrame:
    compound_df = load_compound_classification_raw(workbook_path=workbook_path, sheet_name=sheet_name)
    alias_map = get_mechanism_of_action_alias_map()

    mechanism_df = (
        compound_df[["mechanism_of_action"]]
        .drop_duplicates()
        .assign(mechanism_of_action_mnemonic=lambda df: df["mechanism_of_action"].map(alias_map))
        .sort_values("mechanism_of_action")
        .reset_index(drop=True)
    )

    missing_aliases = mechanism_df["mechanism_of_action_mnemonic"].isna()
    if missing_aliases.any():
        missing_values = mechanism_df.loc[missing_aliases, "mechanism_of_action"].tolist()
        raise KeyError(f"Missing mechanism_of_action mnemonic aliases for: {missing_values}")

    return mechanism_df


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
    compound_df_raw = load_compound_classification_raw(workbook_path=workbook_path, sheet_name=sheet_name)
    compound_records = compound_df.assign(compound_raw=compound_df_raw["compound"])
    all_dirs = load_all_image_dirs(image_root=image_root, manifest_path=manifest_path)
    candidates = select_candidate_image_dirs(all_dirs, image_root=image_root)
    image_root = Path(image_root).resolve()
    alias_map = get_compound_alias_map()

    rows = []
    for record in compound_records.to_dict("records"):
        raw_compound = record["compound_raw"]
        aliases = [normalize_name(alias) for alias in alias_map.get(raw_compound, [raw_compound])]
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


def load_image_condition_tensor(
    condition_dir: str | Path | None = None,
    *,
    condition_df: pd.DataFrame | None = None,
    selected_compound: str | None = None,
    selector_column: str | None = None,
    selected_concentration: str | None = None,
    only_active: bool = True,
    selected_condition_index: int = 0,
    output_size: tuple[int | None, int | None, int | None, int | None] | None = None,
    normalize_global_drift: bool = True,
    loess_frac: float = 0.25,
    use_cache: bool = True,
) -> torch.Tensor:
    if condition_dir is None:
        if condition_df is None:
            raise ValueError("condition_df is required when condition_dir is not provided")
        if selected_compound is None or selector_column is None or selected_concentration is None:
            raise ValueError(
                "selected_compound, selector_column, and selected_concentration are required "
                "when condition_dir is not provided"
            )

        condition_choices = select_condition_choices(
            condition_df,
            selected_compound=selected_compound,
            selector_column=selector_column,
            selected_concentration=selected_concentration,
            only_active=only_active,
        )
        if condition_choices.empty:
            raise ValueError(
                f"No matching condition folders for compound={selected_compound!r}, "
                f"{selector_column}={selected_concentration!r}, only_active={only_active}."
            )
        condition_dir = condition_choices.iloc[selected_condition_index]["image_condition_dir"]

    return _load_image_condition_tensor(
        condition_dir,
        output_size=output_size,
        normalize_global_drift=normalize_global_drift,
        loess_frac=loess_frac,
        use_cache=use_cache,
    )


def choose_sample_indices(n_total: int, n_samples: int = 10) -> list[int]:
    if n_total <= 0:
        return []
    if n_total <= n_samples:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()


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


def plot_midz_time_slices_from_tensor(
    tensor: torch.Tensor,
    n_columns: int = 5,
    title: str | None = None,
):
    if n_columns <= 0:
        raise ValueError(f"n_columns must be positive, got {n_columns}")

    n_timepoints = int(tensor.shape[0])
    z_index = int(tensor.shape[1] // 2)
    n_rows = int(np.ceil(n_timepoints / n_columns))

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(3 * n_columns, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, time_index in zip(axes, range(n_timepoints)):
        image = tensor[time_index, z_index].detach().cpu().numpy()
        ax.imshow(image, cmap="gray")
        ax.set_title(f"T={time_index}, Z={z_index}", fontsize=9)
        ax.axis("off")

    for ax in axes[n_timepoints:]:
        ax.axis("off")

    fig.suptitle(title or "Mid-Z Time Slices", fontsize=14)
    fig.tight_layout()
    return fig, axes
