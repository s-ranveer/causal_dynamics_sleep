"""
Learn dynamic structural causal models (SDCMs) from the windowed sleep-study
dataset using Tigramite's PCMCI+ algorithm.

1. Build one multivariate time series per patient.
2. Wrap those patient series in a Tigramite DataFrame with
   analysis_mode="multiple".
3. Run PCMCI.run_pcmciplus(...) over the six derived physiological nodes.
4. Save per-group edge tables, adjacency matrices, and run summaries.

The default configuration allows both contemporaneous edges inside the same
window and lagged edges across later windows by setting tau_min=0.

"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# These are the six compact derived nodes that we will use as the variables
# in the multivariate time series passed to Tigramite.
NODE_COLUMNS = [
    "ahe_proxy_fraction",
    "ifl_proxy_breath_fraction",
    "flow_limited_effort_fraction",
    "pulse_activation_fraction",
    "odi3_desaturation_fraction",
    "snoring_bout_fraction",
]

NODE_DISPLAY_NAMES = {
    "ahe_proxy_fraction": "AHE",
    "ifl_proxy_breath_fraction": "IFL",
    "odi3_desaturation_fraction": "DeSat",
    "flow_limited_effort_fraction": "FLRE",
    "snoring_bout_fraction": "Snoring",
    "pulse_activation_fraction": "Pulse",
}

# We keep aliases narrow here because the final windowed output has been
# intentionally reduced to one canonical six-feature set.
NODE_COLUMN_ALIASES = {
    "ahe_proxy_fraction": ["ahe_proxy_fraction"],
    "ifl_proxy_breath_fraction": ["ifl_proxy_breath_fraction"],
    "flow_limited_effort_fraction": ["flow_limited_effort_fraction"],
    "pulse_activation_fraction": ["pulse_activation_fraction"],
    "odi3_desaturation_fraction": ["odi3_desaturation_fraction"],
    "snoring_bout_fraction": ["snoring_bout_fraction"],
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "research_outputs" / "pcmci_plus"
DEFAULT_GROUPS = "all,sex:female,sex:male,age:young,age:old"
DEFAULT_TAU_GRID = "1"
DEFAULT_CI_TEST = "parcorr"

FORWARD_EDGE_MARKERS = {"-->", "o->", "x->"}
BACKWARD_EDGE_MARKERS = {"<--", "<-o", "<-x"}
UNDIRECTED_EDGE_MARKERS = {"o-o", "x-x"}


def parse_args():
    """Define the command-line interface for the PCMCI+ workflow.

    The arguments are grouped around four concerns:
    1. where to find the patient metadata and per-patient windowed data,
    2. which cohorts to analyze,
    3. which PCMCI+ hyperparameters to use,
    4. whether to create plots or run a simple tau sweep.
    """
    parser = argparse.ArgumentParser(
        description="Learn group-level SDCMs with Tigramite PCMCI+."
    )
    parser.add_argument(
        "--patient_info_path",
        type=str,
        default="full_patient_info.csv",
        help="Path to the patient metadata CSV.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="windowed_output",
        help="Directory containing per-patient windowed CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory where PCMCI+ outputs are written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help=(
            "Comma-separated list of groups to run. Supported values include "
            "'all', 'sex:female', 'sex:male', 'age:young', and 'age:old'."
        ),
    )
    parser.add_argument(
        "--edge_constraints_path",
        type=str,
        default="edge_constraints.json",
        help=(
            "Optional path to a JSON file defining allowed and forbidden edges "
            "for PCMCI+."
        ),
    )
    parser.add_argument(
        "--tau_min",
        type=int,
        default=0,
        help=(
            "Minimum lag passed to PCMCI+. Default: 0. "
            "Use 0 to allow contemporaneous edges as well as lagged edges."
        ),
    )
    parser.add_argument(
        "--tau_max",
        type=int,
        default=1,
        help="Maximum lag passed to PCMCI+. Default: 1.",
    )
    parser.add_argument(
        "--ci_test",
        type=str,
        choices=["parcorr", "robust_parcorr", "cmiknn", "regressionci", "gpdc"],
        default=DEFAULT_CI_TEST,
        help=(
            "Conditional independence test used inside PCMCI+. "
            "Choices: parcorr, robust_parcorr, cmiknn, regressionci, gpdc. "
            "Default: parcorr."
        ),
    )
    parser.add_argument(
        "--pc_alpha",
        type=float,
        default=0.05,
        help="PC alpha threshold used inside PCMCI+. Default: 0.05.",
    )
    parser.add_argument(
        "--alpha_level",
        type=float,
        default=0.05,
        help="Post-hoc p-value threshold for selecting reported edges. Default: 0.05.",
    )
    parser.add_argument(
        "--min_windows",
        type=int,
        default=10,
        help="Minimum number of windows required to keep a patient series. Default: 10.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Tigramite verbosity level. Default: 0.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip time-series and graph plotting.",
    )
    parser.add_argument(
        "--plot_max_patients",
        type=int,
        default=8,
        help=(
            "Maximum number of patient trajectories to overlay in the patient-level "
            "subplot figure. Default: 8."
        ),
    )
    parser.add_argument(
        "--tune_tau",
        action="store_true",
        help=(
            "Run a simple sweep over tau_max values and keep the best candidate. "
            "This is a practical heuristic, not a fully principled validation study."
        ),
    )
    parser.add_argument(
        "--tau_grid",
        type=str,
        default=DEFAULT_TAU_GRID,
        help=(
            "Comma-separated tau_max candidates used when --tune_tau is enabled. "
            f"Default: {DEFAULT_TAU_GRID}"
        ),
    )
    parser.add_argument(
        "--tau_tuning_metric",
        type=str,
        default="selected_directed_edge_count",
        choices=[
            "selected_directed_edge_count",
            "selected_adjacency_count",
            "mean_abs_mci_value",
        ],
        help=(
            "Heuristic metric used to rank tau candidates when --tune_tau is enabled. "
            "Default: selected_directed_edge_count."
        ),
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=10,
        help="Number of patient-level bootstrap PCMCI+ fits for Bootstrap-GMA. Default: 10.",
    )
    parser.add_argument(
        "--gma_threshold",
        type=float,
        default=0.5,
        help="Minimum target-local structure posterior retained by GMA. Default: 0.5.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for patient-level bootstrap sampling. Default: 0.",
    )
    return parser.parse_args()


@contextmanager
def numba_cache_disabled_for_import():
    """Temporarily disable numba decorator caching during fragile imports.

    Some third-party packages in the current environment request
    `cache=True` inside numba decorators at import time, which can fail with
    "no locator available" errors for site-packages files. We temporarily
    strip that flag only while importing those modules.
    """
    try:
        import numba
    except Exception:
        yield
        return

    original_njit = numba.njit
    original_jit = numba.jit
    original_guvectorize = numba.guvectorize

    def strip_cache_kwarg(kwargs: dict) -> dict:
        if "cache" not in kwargs:
            return kwargs
        patched = dict(kwargs)
        patched["cache"] = False
        return patched

    def patched_njit(*args, **kwargs):
        return original_njit(*args, **strip_cache_kwarg(kwargs))

    def patched_jit(*args, **kwargs):
        return original_jit(*args, **strip_cache_kwarg(kwargs))

    def patched_guvectorize(*args, **kwargs):
        return original_guvectorize(*args, **strip_cache_kwarg(kwargs))

    numba.njit = patched_njit
    numba.jit = patched_jit
    numba.guvectorize = patched_guvectorize
    try:
        yield
    finally:
        numba.njit = original_njit
        numba.jit = original_jit
        numba.guvectorize = original_guvectorize


def import_tigramite(ci_test_name: str):
    """Import the Tigramite pieces we need at runtime.

    Importing lazily instead of at module import time gives a cleaner error if the
    user has not activated the right environment yet.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        if ci_test_name == "parcorr":
            from tigramite.independence_tests.parcorr import ParCorr

            cond_ind_test = ParCorr(significance="analytic")
        elif ci_test_name == "robust_parcorr":
            from tigramite.independence_tests.robust_parcorr import RobustParCorr

            cond_ind_test = RobustParCorr(significance="analytic")
        elif ci_test_name == "cmiknn":
            from tigramite.independence_tests.cmiknn import CMIknn

            cond_ind_test = CMIknn(significance="shuffle_test")
        elif ci_test_name == "regressionci":
            from tigramite.independence_tests.regressionCI import RegressionCI

            cond_ind_test = RegressionCI()
        elif ci_test_name == "gpdc":
            with numba_cache_disabled_for_import():
                from tigramite.independence_tests.gpdc import GPDC

            cond_ind_test = GPDC(significance="analytic")
        else:
            raise ValueError(
                f"Unsupported ci_test '{ci_test_name}'. "
                "Supported values: parcorr, robust_parcorr, cmiknn, regressionci, gpdc."
            )
    except Exception as exc:
        raise ImportError(
            "PCMCI+ requires the tigramite package in your active Python "
            "environment."
        ) from exc
    return pp, cond_ind_test, PCMCI


def normalize_patient_id(value) -> str:
    """Normalize patient IDs so joins work across metadata and file names.

    The metadata file sometimes stores IDs as numeric-looking values like 48.0.
    The windowed CSVs use filenames like `48_windowed.csv`, so we coerce whole
    numbers back to integer-looking strings.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def load_patient_info(patient_info_path: str) -> pd.DataFrame:
    """Load and clean the patient metadata table.

    We also honor the `valid` column when present so the causal discovery stage
    only uses patients that passed the earlier data-quality checks.
    """
    patient_info = pd.read_csv(patient_info_path)
    if "patient_id" not in patient_info.columns:
        raise ValueError("patient_info CSV must contain a 'patient_id' column.")

    patient_info = patient_info.copy()
    patient_info["patient_id"] = patient_info["patient_id"].map(normalize_patient_id)
    patient_info = patient_info[patient_info["patient_id"] != ""].reset_index(drop=True)

    if "valid" in patient_info.columns:
        valid_mask = patient_info["valid"].fillna(False).astype(bool)
        patient_info = patient_info[valid_mask].reset_index(drop=True)

    return patient_info


def get_id_groups(patient_info: pd.DataFrame) -> dict[str, list[str]]:
    """Build the cohort splits requested in the original stub script.

    We keep the naming explicit as `sex:female`, `sex:male`, `age:young`,
    `age:old`, and `all:all` so each output folder is self-describing.
    """
    groups = {
        "all:all": patient_info["patient_id"].tolist(),
    }

    if "sex" in patient_info.columns:
        sex_series = patient_info["sex"].astype(str).str.strip().str.upper()
        groups["sex:female"] = patient_info.loc[sex_series == "F", "patient_id"].tolist()
        groups["sex:male"] = patient_info.loc[sex_series == "M", "patient_id"].tolist()

    if {"sex", "age"}.issubset(patient_info.columns):
        age_series = pd.to_numeric(patient_info["age"], errors="coerce")
        sex_series = patient_info["sex"].astype(str).str.strip().str.upper()

        male_young = patient_info.loc[
            (sex_series == "M") & (age_series < 40), "patient_id"
        ].tolist()
        male_old = patient_info.loc[
            (sex_series == "M") & (age_series >= 40), "patient_id"
        ].tolist()
        female_young = patient_info.loc[
            (sex_series == "F") & (age_series < 50), "patient_id"
        ].tolist()
        female_old = patient_info.loc[
            (sex_series == "F") & (age_series >= 50), "patient_id"
        ].tolist()

        groups["age:young"] = male_young + female_young
        groups["age:old"] = male_old + female_old

    return groups


def resolve_requested_groups(groups_arg: str, available_groups: dict[str, list[str]]) -> list[str]:
    """Parse the `--groups` string and validate that every requested cohort exists."""
    requested = []
    for raw_name in groups_arg.split(","):
        group_name = raw_name.strip()
        if not group_name:
            continue
        if group_name == "all":
            group_name = "all:all"
        if group_name not in available_groups:
            supported = ", ".join(sorted(available_groups))
            raise ValueError(f"Unknown group '{raw_name}'. Supported groups: {supported}")
        requested.append(group_name)
    if not requested:
        raise ValueError("No valid groups were requested.")
    return requested


def resolve_windowed_path(data_dir: Path, patient_id: str) -> Path | None:
    """Find the per-patient windowed CSV for a patient ID.

    The repository currently uses filenames like `1_windowed.csv`, but we also
    support an `anonymized_<id>_windowed.csv` fallback to keep the script robust
    against naming changes.
    """
    candidates = [
        data_dir / f"{patient_id}_windowed.csv",
        data_dir / f"anonymized_{patient_id}_windowed.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def impute_node_frame(node_frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing node values so Tigramite receives a finite matrix.

    Strategy:
    1. Interpolate within each patient series across time.
    2. Fill any remaining holes with the column median.
    3. If a whole column is missing, fall back to zeros.

    This is intentionally simple and transparent. If you later want a more
    rigorous missing-data strategy, this is one of the easiest places to swap in.
    """
    missing_before = int(node_frame.isna().sum().sum())
    imputed = node_frame.copy()

    # First use within-series interpolation so short gaps are filled using nearby
    # windows rather than a global constant.
    imputed = imputed.interpolate(axis=0, limit_direction="both")

    for column in imputed.columns:
        if imputed[column].isna().all():
            # If an entire node is missing for a patient, we fall back to zero so
            # the series remains usable. The patient-level summary will still tell
            # us how much imputation was needed.
            imputed[column] = 0.0
        else:
            imputed[column] = imputed[column].fillna(imputed[column].median())

    return imputed, missing_before


def build_canonical_node_frame(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical six-node frame from the final derived feature columns."""
    canonical_columns = {}
    missing_columns = []

    for canonical_name in NODE_COLUMNS:
        aliases = NODE_COLUMN_ALIASES.get(canonical_name, [canonical_name])
        matched_name = next((alias for alias in aliases if alias in patient_df.columns), None)
        if matched_name is None:
            missing_columns.append(canonical_name)
            continue
        canonical_columns[canonical_name] = pd.to_numeric(
            patient_df[matched_name],
            errors="coerce",
        )

    if missing_columns:
        raise ValueError(f"missing_columns:{'|'.join(missing_columns)}")

    return pd.DataFrame(canonical_columns)


def normalize_edge_constraints(
    constraints_payload: dict[str, object],
    tau_min: int,
    tau_max: int,
) -> dict[str, set[tuple[str, str, int]]]:
    """Validate and normalize edge constraints loaded from JSON.

    The JSON format is:

    {
      "allowed": [["source_node", "target_node", 1], ...],
      "forbidden": [["source_node", "target_node", 1], ...]
    }

    Current semantics:
    - `allowed` lists edges PCMCI+ is permitted to search over.
    - `forbidden` excludes an edge from consideration.
    - if `allowed` is empty, all non-forbidden edges remain admissible.

    Lags are specified as non-negative integers in the same convention used by
    PCMCI output tables: lag 1 means X(t-1) -> Y(t), lag 0 means a
    contemporaneous relation.
    """
    if "required" in constraints_payload:
        raise ValueError(
            "The 'required' constraint bucket is no longer supported. "
            "Use only 'allowed' and 'forbidden'."
        )

    normalized: dict[str, set[tuple[str, str, int]]] = {
        "allowed": set(),
        "forbidden": set(),
    }
    valid_nodes = set(NODE_COLUMNS)

    for bucket_name in normalized:
        raw_edges = constraints_payload.get(bucket_name, [])
        if raw_edges is None:
            continue
        if not isinstance(raw_edges, list):
            raise ValueError(
                f"Constraint bucket '{bucket_name}' must be a list of "
                "[source_node, target_node, lag] triples."
            )
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, list) or len(raw_edge) != 3:
                raise ValueError(
                    f"Every entry in '{bucket_name}' must have exactly three "
                    "items: [source_node, target_node, lag]."
                )
            source_name, target_name, lag_value = raw_edge
            if source_name not in valid_nodes:
                raise ValueError(
                    f"Unknown source node '{source_name}' in '{bucket_name}'. "
                    f"Supported nodes: {sorted(valid_nodes)}"
                )
            if target_name not in valid_nodes:
                raise ValueError(
                    f"Unknown target node '{target_name}' in '{bucket_name}'. "
                    f"Supported nodes: {sorted(valid_nodes)}"
                )
            lag = int(lag_value)
            if lag < tau_min or lag > tau_max:
                raise ValueError(
                    f"Edge ({source_name}, {target_name}, {lag}) in "
                    f"'{bucket_name}' falls outside tau range [{tau_min}, {tau_max}]."
                )
            if lag == 0 and source_name == target_name:
                raise ValueError(
                    f"Contemporaneous self-links are not valid: "
                    f"({source_name}, {target_name}, 0)."
                )
            normalized[bucket_name].add((source_name, target_name, lag))

    conflict_allowed_forbidden = normalized["allowed"] & normalized["forbidden"]
    if conflict_allowed_forbidden:
        raise ValueError(
            "The same edge cannot be both allowed and forbidden: "
            f"{sorted(conflict_allowed_forbidden)}"
        )

    return normalized


def load_edge_constraints(
    edge_constraints_path: str | None,
    tau_min: int,
    tau_max: int,
) -> dict[str, set[tuple[str, str, int]]] | None:
    """Load optional edge constraints from a JSON file."""
    if not edge_constraints_path:
        return None

    constraints_path = Path(edge_constraints_path)
    if not constraints_path.exists():
        raise FileNotFoundError(f"Edge constraints file not found: {constraints_path}")

    with constraints_path.open() as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Edge constraints JSON must be an object/dictionary.")

    normalized = normalize_edge_constraints(
        constraints_payload=payload,
        tau_min=tau_min,
        tau_max=tau_max,
    )

    if not any(normalized.values()):
        return None
    return normalized


def build_link_assumptions(
    edge_constraints: dict[str, set[tuple[str, str, int]]] | None,
    tau_min: int,
    tau_max: int,
) -> dict[int, dict[tuple[int, int], str]] | None:
    """Translate user-facing edge constraints into Tigramite link assumptions.

    Semantics:
    - If no constraints are provided, return None so PCMCI+ remains unconstrained.
    - If `allowed` is non-empty, only those edges are searched over.
    - If `allowed` is empty, all links in the tau range are admissible except
      those in the forbidden set.
    - Admissible lagged edges are represented as '-?>'.
    - Admissible contemporaneous edges are represented as 'o?o'.

    For contemporaneous forbidden edges, the source/target orientation in the
    JSON is treated as an adjacency-level exclusion because Tigramite's soft
    contemporaneous assumption is adjacency-based.
    """
    if edge_constraints is None:
        return None

    node_to_idx = {node_name: idx for idx, node_name in enumerate(NODE_COLUMNS)}
    allowed_edges = set(edge_constraints["allowed"])
    forbidden_edges = set(edge_constraints["forbidden"])

    if allowed_edges:
        candidate_edges = set(allowed_edges)
    else:
        candidate_edges = set()
        for source_name in NODE_COLUMNS:
            for target_name in NODE_COLUMNS:
                for lag in range(tau_min, tau_max + 1):
                    if lag == 0 and source_name == target_name:
                        continue
                    candidate_edges.add((source_name, target_name, lag))
    candidate_edges -= forbidden_edges

    if not candidate_edges:
        return None

    assumptions: dict[int, dict[tuple[int, int], str]] = {
        target_idx: {}
        for target_idx in range(len(NODE_COLUMNS))
    }

    for source_name, target_name, lag in sorted(candidate_edges):
        source_idx = node_to_idx[source_name]
        target_idx = node_to_idx[target_name]

        if lag == 0:
            if (source_name, target_name, 0) in forbidden_edges or (
                target_name,
                source_name,
                0,
            ) in forbidden_edges:
                continue

            assumptions[target_idx][(source_idx, 0)] = "o?o"
            assumptions[source_idx][(target_idx, 0)] = "o?o"
        else:
            if (source_name, target_name, lag) in forbidden_edges:
                continue
            assumptions[target_idx][(source_idx, -lag)] = "-?>"

    if all(not links for links in assumptions.values()):
        return None
    return assumptions


def extract_time_minutes(patient_df: pd.DataFrame) -> np.ndarray | None:
    """Extract a relative time axis in minutes when `time_s` is available."""
    if "time_s" not in patient_df.columns:
        return None
    time_seconds = pd.to_numeric(patient_df["time_s"], errors="coerce").to_numpy(dtype=float)
    if time_seconds.size == 0:
        return None
    finite_mask = np.isfinite(time_seconds)
    if not finite_mask.any():
        return None
    time_seconds = time_seconds[finite_mask]
    return time_seconds / 60.0


def load_patient_series(
    data_dir: Path,
    patient_id: str,
    min_windows: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    """Load one patient's windowed node table and convert it to a numpy series.

    PCMCI+ expects each patient to contribute a T x N matrix. This function is
    the bridge from a CSV file on disk to that matrix representation.
    """
    windowed_path = resolve_windowed_path(data_dir, patient_id)
    summary_row = {
        "patient_id": patient_id,
        "file_path": str(windowed_path) if windowed_path else "",
        "included": False,
        "skip_reason": "",
        "n_windows": 0,
        "n_missing_values_before_imputation": 0,
    }

    if windowed_path is None:
        summary_row["skip_reason"] = "missing_windowed_file"
        return None, None, summary_row

    patient_df = pd.read_csv(windowed_path)
    try:
        node_frame = build_canonical_node_frame(patient_df)
    except ValueError as exc:
        summary_row["skip_reason"] = str(exc)
        return None, None, summary_row

    if "time_s" in patient_df.columns:
        # Preserving the true within-patient temporal order is essential because
        # PCMCI+ reasons over lagged dependencies.
        patient_df = patient_df.sort_values("time_s").reset_index(drop=True)
    else:
        patient_df = patient_df.reset_index(drop=True)
    time_minutes = extract_time_minutes(patient_df)

    # Rebuild after sorting so the canonical node values stay aligned with the
    # time-ordered patient dataframe.
    node_frame = build_canonical_node_frame(patient_df)
    node_frame, missing_before = impute_node_frame(node_frame)

    summary_row["n_windows"] = int(len(node_frame))
    summary_row["n_missing_values_before_imputation"] = missing_before

    if len(node_frame) < min_windows:
        # Very short sequences are usually not informative enough for the lagged
        # conditional independence tests, so we drop them up front.
        summary_row["skip_reason"] = "too_few_windows"
        return None, None, summary_row

    series = node_frame.to_numpy(dtype=np.float64)
    if not np.isfinite(series).all():
        summary_row["skip_reason"] = "non_finite_values_after_imputation"
        return None, None, summary_row

    summary_row["included"] = True
    return series, time_minutes, summary_row


def build_group_series_dict(
    data_dir: Path,
    patient_ids: list[str],
    min_windows: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Load every usable patient series for one cohort.

    The returned dictionary is exactly what `pp.DataFrame(..., analysis_mode="multiple")`
    expects: one multivariate time series per patient/study.
    """
    series_dict: dict[str, np.ndarray] = {}
    time_axes_dict: dict[str, np.ndarray] = {}
    summary_rows = []

    patient_iterator = tqdm(
        patient_ids,
        desc="Loading patient series",
        unit="patient",
        leave=False,
    )
    for patient_id in patient_iterator:
        patient_iterator.set_postfix_str(patient_id)
        series, time_minutes, summary_row = load_patient_series(
            data_dir=data_dir,
            patient_id=patient_id,
            min_windows=min_windows,
        )
        summary_rows.append(summary_row)
        if series is not None:
            series_dict[patient_id] = series
            if time_minutes is not None and len(time_minutes) == series.shape[0]:
                time_axes_dict[patient_id] = time_minutes

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["included", "patient_id"], ascending=[False, True])
        summary_df = summary_df.reset_index(drop=True)
    return series_dict, time_axes_dict, summary_df


def normalize_graph_entry(entry) -> str:
    """Normalize Tigramite graph markers to plain Python strings."""
    if isinstance(entry, bytes):
        return entry.decode("utf-8")
    if entry is None:
        return ""
    return str(entry)


def parse_tau_grid(tau_grid_arg: str, tau_min: int) -> list[int]:
    """Parse and validate the tau sweep candidates supplied by the user."""
    tau_candidates = []
    for raw_value in tau_grid_arg.split(","):
        text = raw_value.strip()
        if not text:
            continue
        tau_value = int(text)
        if tau_value < tau_min:
            continue
        tau_candidates.append(tau_value)

    tau_candidates = sorted(set(tau_candidates))
    if not tau_candidates:
        raise ValueError(
            "No valid tau candidates remained after parsing --tau_grid. "
            "Make sure at least one candidate is >= --tau_min."
        )
    return tau_candidates


def learn_pcmci_plus_for_group(
    series_dict: dict[str, np.ndarray],
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    alpha_level: float,
    verbosity: int,
    ci_test_name: str,
    link_assumptions: dict[int, dict[tuple[int, int], str]] | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame, dict[str, object]]:
    """Run the actual PCMCI+ fit for one cohort.

    This is the main place where the script mirrors the Tigramite tutorial:
    - build a Tigramite `DataFrame`,
    - initialize `PCMCI` with the requested conditional independence test,
    - call `run_pcmciplus`,
    - unpack `graph`, `p_matrix`, and `val_matrix`.

    We return both human-readable tables and the raw result objects because the
    raw graph matrices are useful for plotting.
    """
    if not series_dict:
        raise ValueError("Cannot run PCMCI+ without at least one patient time series.")

    pp, cond_ind_test, PCMCI = import_tigramite(ci_test_name)

    dataframe = pp.DataFrame(
        series_dict,
        var_names=NODE_COLUMNS,
        analysis_mode="multiple",
    )
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity,
    )
    results = pcmci.run_pcmciplus(
        link_assumptions=link_assumptions,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
    )

    graph = results["graph"]
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    edge_rows = []
    for source_idx, source_node in enumerate(NODE_COLUMNS):
        for target_idx, target_node in enumerate(NODE_COLUMNS):
            for lag in range(tau_min, tau_max + 1):
                # Tigramite stores the discovered graph in a 3D matrix:
                # source x target x lag.
                graph_entry = normalize_graph_entry(graph[source_idx, target_idx, lag])
                p_value = float(p_matrix[source_idx, target_idx, lag])
                test_statistic = float(val_matrix[source_idx, target_idx, lag])

                # `graph_entry` tells us whether the adjacency exists and, if so,
                # whether it is oriented source -> target, target -> source, or
                # remains unresolved.
                is_adjacency = graph_entry in (
                    FORWARD_EDGE_MARKERS | BACKWARD_EDGE_MARKERS | UNDIRECTED_EDGE_MARKERS
                )
                is_forward = graph_entry in FORWARD_EDGE_MARKERS
                selected_adjacency = bool(is_adjacency and p_value <= alpha_level)
                selected_directed_edge = bool(is_forward and selected_adjacency)

                edge_rows.append(
                    {
                        "source_node": source_node,
                        "target_node": target_node,
                        "lag": int(lag),
                        "graph_entry": graph_entry,
                        "p_value": p_value,
                        "test_statistic": test_statistic,
                        "mci_value": test_statistic if selected_directed_edge else 0.0,
                        "abs_mci_value": float(abs(test_statistic)) if selected_directed_edge else 0.0,
                        "selected_adjacency": selected_adjacency,
                        "selected": selected_directed_edge,
                        "is_contemporaneous": bool(lag == 0),
                        "method": "pcmci_plus",
                    }
                )

    edge_table = pd.DataFrame(edge_rows).sort_values(
        ["lag", "target_node", "selected", "abs_mci_value"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    adjacency_tables = {}
    for lag in range(tau_min, tau_max + 1):
        adjacency = pd.DataFrame(
            np.zeros((len(NODE_COLUMNS), len(NODE_COLUMNS))),
            index=NODE_COLUMNS,
            columns=NODE_COLUMNS,
        )
        lag_edges = edge_table[(edge_table["lag"] == lag) & (edge_table["selected"])].copy()
        for _, row in lag_edges.iterrows():
            adjacency.loc[row["source_node"], row["target_node"]] = row["mci_value"]
        adjacency_tables[lag] = adjacency

    total_windows = int(sum(series.shape[0] for series in series_dict.values()))
    run_summary = pd.DataFrame(
        [
            {"metric": "n_patients", "value": int(len(series_dict))},
            {"metric": "n_windows", "value": total_windows},
            {"metric": "analysis_mode", "value": "multiple"},
            {"metric": "tau_min", "value": int(tau_min)},
            {"metric": "tau_max", "value": int(tau_max)},
            {"metric": "pc_alpha", "value": float(pc_alpha)},
            {"metric": "alpha_level", "value": float(alpha_level)},
            {"metric": "ci_test", "value": ci_test_name},
            {
                "metric": "selected_directed_edge_count",
                "value": int(edge_table["selected"].sum()),
            },
            {
                "metric": "selected_adjacency_count",
                "value": int(edge_table["selected_adjacency"].sum()),
            },
        ]
    )
    return edge_table, adjacency_tables, run_summary, results


def score_tau_candidate(edge_table: pd.DataFrame, metric: str) -> float:
    """Score one tau candidate using a simple, explicit heuristic.

    This is not a substitute for downstream predictive validation, but it gives
    us an automated way to compare multiple lag budgets in one run.
    """
    if metric == "selected_directed_edge_count":
        return float(edge_table["selected"].sum())
    if metric == "selected_adjacency_count":
        return float(edge_table["selected_adjacency"].sum())
    if metric == "mean_abs_mci_value":
        selected = edge_table[edge_table["selected"]].copy()
        if selected.empty:
            return 0.0
        return float(selected["abs_mci_value"].mean())
    raise ValueError(f"Unsupported tau tuning metric: {metric}")


def tune_tau_for_group(
    series_dict: dict[str, np.ndarray],
    tau_min: int,
    tau_candidates: list[int],
    pc_alpha: float,
    alpha_level: float,
    verbosity: int,
    ci_test_name: str,
    tuning_metric: str,
    link_assumptions: dict[int, dict[tuple[int, int], str]] | None = None,
) -> tuple[int, pd.DataFrame, dict[str, object]]:
    """Evaluate several tau_max candidates and keep the best one.

    We rank candidates by the requested heuristic score and break ties in favor
    of the smaller tau so the chosen model stays as simple as possible.
    """
    candidate_rows = []
    candidate_results = {}

    tau_iterator = tqdm(
        tau_candidates,
        desc="Evaluating tau candidates",
        unit="tau",
        leave=False,
    )
    for tau_candidate in tau_iterator:
        tau_iterator.set_postfix_str(f"tau_max={tau_candidate}")
        edge_table, adjacency_tables, run_summary, raw_results = learn_pcmci_plus_for_group(
            series_dict=series_dict,
            tau_min=tau_min,
            tau_max=tau_candidate,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            verbosity=verbosity,
            ci_test_name=ci_test_name,
            link_assumptions=link_assumptions,
        )
        score = score_tau_candidate(edge_table=edge_table, metric=tuning_metric)
        candidate_rows.append(
            {
                "tau_max": int(tau_candidate),
                "score": float(score),
                "selected_directed_edge_count": int(edge_table["selected"].sum()),
                "selected_adjacency_count": int(edge_table["selected_adjacency"].sum()),
                "mean_abs_mci_value": float(
                    edge_table.loc[edge_table["selected"], "abs_mci_value"].mean()
                )
                if edge_table["selected"].any()
                else 0.0,
            }
        )
        candidate_results[tau_candidate] = {
            "edge_table": edge_table,
            "adjacency_tables": adjacency_tables,
            "run_summary": run_summary,
            "raw_results": raw_results,
        }
        tau_iterator.set_postfix_str(
            f"tau_max={tau_candidate}, score={score:.3f}"
        )

    tuning_summary = pd.DataFrame(candidate_rows).sort_values(
        ["score", "tau_max"],
        ascending=[False, True],
    ).reset_index(drop=True)
    best_tau = int(tuning_summary.iloc[0]["tau_max"])
    return best_tau, tuning_summary, candidate_results[best_tau]


def serialize_edge(source_node: str, target_node: str, lag: int) -> str:
    """Serialize an edge compactly for diagnostic CSVs."""
    return f"{source_node}->{target_node}@lag{lag}"


def serialize_parent_edges(parent_edges: tuple[tuple[str, str, int], ...]) -> str:
    """Serialize a target-local parent edge set for structure diagnostics."""
    if not parent_edges:
        return "EMPTY"
    return ";".join(serialize_edge(source, target, lag) for source, target, lag in parent_edges)


def parse_parent_edges(parent_edges_text: str) -> list[tuple[str, str, int]]:
    """Parse serialized parent edges back into edge triples for plotting."""
    text = str(parent_edges_text).strip()
    if not text or text == "EMPTY":
        return []

    parsed_edges = []
    for raw_edge in text.split(";"):
        edge_text, lag_text = raw_edge.rsplit("@lag", 1)
        source_node, target_node = edge_text.split("->", 1)
        parsed_edges.append((source_node, target_node, int(lag_text)))
    return parsed_edges


def extract_parent_structures(edge_table: pd.DataFrame) -> dict[str, tuple[tuple[str, str, int], ...]]:
    """Extract one sorted parent-edge structure per target from selected edges."""
    structures = {}
    selected_edges = edge_table[edge_table["selected"]].copy()
    for target_node in NODE_COLUMNS:
        target_edges = selected_edges[selected_edges["target_node"] == target_node]
        parent_edges = tuple(
            sorted(
                (
                    str(row["source_node"]),
                    str(row["target_node"]),
                    int(row["lag"]),
                )
                for _, row in target_edges.iterrows()
            )
        )
        structures[target_node] = parent_edges
    return structures


def unrolled_edges(
    edges: set[tuple[str, str, int]],
    tau_max: int,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Expand rolled temporal edges onto time slices 0..tau_max."""
    expanded_edges = []
    for source_node, target_node, lag in edges:
        for target_time in range(lag, tau_max + 1):
            expanded_edges.append(
                (
                    (source_node, target_time - lag),
                    (target_node, target_time),
                )
            )
    return expanded_edges


def is_unrolled_acyclic(edges: set[tuple[str, str, int]], tau_max: int) -> bool:
    """Check temporal acyclicity without penalizing valid lagged feedback.

    With non-negative lags, edges with lag > 0 always point from an earlier
    slice to a later slice, so they cannot close a directed cycle in the
    unrolled graph. Only directed lag-0 edges can create a same-slice cycle.
    """
    del tau_max

    adjacency = {node_name: [] for node_name in NODE_COLUMNS}
    for source_node, target_node, lag in edges:
        if lag < 0:
            return False
        if lag > 0:
            continue
        if source_node == target_node:
            return False
        adjacency.setdefault(source_node, []).append(target_node)
        adjacency.setdefault(target_node, [])

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for child in adjacency.get(node, []):
            if not visit(child):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in list(adjacency))


def make_adjacency_tables_from_edges(
    edge_table: pd.DataFrame,
    tau_min: int,
    tau_max: int,
) -> dict[int, pd.DataFrame]:
    """Build lag-specific adjacency matrices from the final aggregated graph."""
    adjacency_tables = {}
    for lag in range(tau_min, tau_max + 1):
        adjacency = pd.DataFrame(
            np.zeros((len(NODE_COLUMNS), len(NODE_COLUMNS))),
            index=NODE_COLUMNS,
            columns=NODE_COLUMNS,
        )
        if not edge_table.empty:
            lag_edges = edge_table[(edge_table["lag"] == lag) & (edge_table["selected"])].copy()
            for _, row in lag_edges.iterrows():
                adjacency.loc[row["source_node"], row["target_node"]] = row["mci_value"]
        adjacency_tables[lag] = adjacency
    return adjacency_tables


def summarize_bootstrap_edge_mci(bootstrap_edge_table: pd.DataFrame) -> dict[tuple[str, str, int], dict[str, float | int]]:
    """Summarize selected-edge MCI values across bootstrap PCMCI+ fits."""
    if bootstrap_edge_table.empty:
        return {}

    edge_stats = {}
    grouped = bootstrap_edge_table.groupby(["source_node", "target_node", "lag"], dropna=False)
    for (source_node, target_node, lag), group in grouped:
        mci_values = pd.to_numeric(group["mci_value"], errors="coerce").dropna()
        if mci_values.empty:
            continue
        edge_stats[(str(source_node), str(target_node), int(lag))] = {
            "mean_mci_value": float(mci_values.mean()),
            "mean_abs_mci_value": float(mci_values.abs().mean()),
            "mci_bootstrap_count": int(len(mci_values)),
        }
    return edge_stats


def learn_bootstrap_gma_for_group(
    series_dict: dict[str, np.ndarray],
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    alpha_level: float,
    verbosity: int,
    ci_test_name: str,
    link_assumptions: dict[int, dict[tuple[int, int], str]] | None,
    n_bootstraps: int,
    gma_threshold: float,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Learn a final graph by bootstrapping PCMCI+ and aggregating parent sets."""
    if n_bootstraps <= 0:
        raise ValueError("--n_bootstraps must be positive.")
    if not 0.0 <= gma_threshold <= 1.0:
        raise ValueError("--gma_threshold must be between 0 and 1.")
    if not series_dict:
        raise ValueError("Cannot run Bootstrap-GMA without at least one patient time series.")

    rng = np.random.default_rng(bootstrap_seed)
    patient_ids = list(series_dict.keys())
    structure_counts: Counter[tuple[str, tuple[tuple[str, str, int], ...]]] = Counter()
    bootstrap_edge_tables = []
    bootstrap_summary_rows = []

    bootstrap_iterator = tqdm(
        range(1, n_bootstraps + 1),
        desc="Bootstrap-GMA fits",
        unit="bootstrap",
        leave=False,
    )
    for bootstrap_idx in bootstrap_iterator:
        sampled_patient_ids = [
            str(patient_id)
            for patient_id in rng.choice(patient_ids, size=len(patient_ids), replace=True).tolist()
        ]
        bootstrap_series_dict = {
            f"{patient_id}__bootstrap{bootstrap_idx}_draw{draw_idx}": series_dict[patient_id]
            for draw_idx, patient_id in enumerate(sampled_patient_ids)
        }

        edge_table, _, _, _ = learn_pcmci_plus_for_group(
            series_dict=bootstrap_series_dict,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            verbosity=verbosity,
            ci_test_name=ci_test_name,
            link_assumptions=link_assumptions,
        )

        selected_edges = edge_table[edge_table["selected"]].copy()
        if not selected_edges.empty:
            selected_edges.insert(0, "bootstrap_idx", bootstrap_idx)
            bootstrap_edge_tables.append(selected_edges)

        for target_node, parent_edges in extract_parent_structures(edge_table).items():
            structure_counts[(target_node, parent_edges)] += 1

        bootstrap_summary_rows.append(
            {
                "bootstrap_idx": int(bootstrap_idx),
                "sampled_patient_ids": ",".join(sampled_patient_ids),
                "unique_patient_count": int(len(set(sampled_patient_ids))),
                "selected_directed_edge_count": int(selected_edges.shape[0]),
                "selected_adjacency_count": int(edge_table["selected_adjacency"].sum()),
            }
        )
        bootstrap_iterator.set_postfix_str(f"selected_edges={selected_edges.shape[0]}")

    structure_rows = []
    for (target_node, parent_edges), count in structure_counts.items():
        posterior = float(count / n_bootstraps)
        structure_rows.append(
            {
                "target_node": target_node,
                "parent_edges": serialize_parent_edges(parent_edges),
                "parent_edge_count": int(len(parent_edges)),
                "bootstrap_count": int(count),
                "posterior": posterior,
                "retained": bool(posterior > gma_threshold),
                "accepted": False,
                "accepted_order": "",
            }
        )

    structure_table = pd.DataFrame(structure_rows)
    if structure_table.empty:
        structure_table = pd.DataFrame(
            columns=[
                "target_node",
                "parent_edges",
                "parent_edge_count",
                "bootstrap_count",
                "posterior",
                "retained",
                "accepted",
                "accepted_order",
            ]
        )

    retained_structures = []
    for (target_node, parent_edges), count in structure_counts.items():
        posterior = float(count / n_bootstraps)
        if posterior > gma_threshold:
            retained_structures.append(
                {
                    "target_node": target_node,
                    "parent_edges": parent_edges,
                    "parent_edges_text": serialize_parent_edges(parent_edges),
                    "bootstrap_count": int(count),
                    "posterior": posterior,
                }
            )

    retained_structures = sorted(
        retained_structures,
        key=lambda row: (
            -row["posterior"],
            -row["bootstrap_count"],
            row["target_node"],
            row["parent_edges_text"],
        ),
    )

    accepted_edges: set[tuple[str, str, int]] = set()
    accepted_structure_lookup: dict[tuple[str, str], int] = {}
    edge_support: dict[tuple[str, str, int], dict[str, float | int]] = {}
    for accepted_order, structure in enumerate(retained_structures, start=1):
        parent_edges = set(structure["parent_edges"])
        candidate_edges = accepted_edges | parent_edges
        if is_unrolled_acyclic(candidate_edges, tau_max=tau_max):
            accepted_edges = candidate_edges
            accepted_structure_lookup[
                (structure["target_node"], structure["parent_edges_text"])
            ] = accepted_order
            for edge in parent_edges:
                previous = edge_support.get(edge)
                if previous is None or structure["posterior"] > previous["posterior"]:
                    edge_support[edge] = {
                        "posterior": float(structure["posterior"]),
                        "bootstrap_count": int(structure["bootstrap_count"]),
                    }

    if not structure_table.empty:
        for row_idx, row in structure_table.iterrows():
            accepted_order = accepted_structure_lookup.get((row["target_node"], row["parent_edges"]))
            if accepted_order is not None:
                structure_table.loc[row_idx, "accepted"] = True
                structure_table.loc[row_idx, "accepted_order"] = str(accepted_order)
        structure_table = structure_table.sort_values(
            ["posterior", "bootstrap_count", "target_node", "parent_edges"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        structure_table.insert(0, "significance_rank", np.arange(1, len(structure_table) + 1))

    bootstrap_edge_table = (
        pd.concat(bootstrap_edge_tables, ignore_index=True)
        if bootstrap_edge_tables
        else pd.DataFrame()
    )
    edge_mci_stats = summarize_bootstrap_edge_mci(bootstrap_edge_table)

    final_edge_rows = []
    for source_node, target_node, lag in sorted(accepted_edges, key=lambda edge: (edge[2], edge[1], edge[0])):
        support = edge_support.get(
            (source_node, target_node, lag),
            {"posterior": 0.0, "bootstrap_count": 0},
        )
        posterior = float(support["posterior"])
        mci_stats = edge_mci_stats.get(
            (source_node, target_node, lag),
            {
                "mean_mci_value": np.nan,
                "mean_abs_mci_value": np.nan,
                "mci_bootstrap_count": 0,
            },
        )
        mean_mci_value = float(mci_stats["mean_mci_value"])
        mean_abs_mci_value = float(mci_stats["mean_abs_mci_value"])
        final_edge_rows.append(
            {
                "source_node": source_node,
                "target_node": target_node,
                "lag": int(lag),
                "graph_entry": "-->",
                "p_value": np.nan,
                "test_statistic": mean_mci_value,
                "mci_value": mean_mci_value,
                "abs_mci_value": mean_abs_mci_value,
                "selected_adjacency": True,
                "selected": True,
                "is_contemporaneous": bool(lag == 0),
                "method": "bootstrap_gma",
                "posterior": posterior,
                "bootstrap_count": int(support["bootstrap_count"]),
                "mci_bootstrap_count": int(mci_stats["mci_bootstrap_count"]),
            }
        )

    edge_table = pd.DataFrame(final_edge_rows)
    if edge_table.empty:
        edge_table = pd.DataFrame(
            columns=[
                "source_node",
                "target_node",
                "lag",
                "graph_entry",
                "p_value",
                "test_statistic",
                "mci_value",
                "abs_mci_value",
                "selected_adjacency",
                "selected",
                "is_contemporaneous",
                "method",
                "posterior",
                "bootstrap_count",
                "mci_bootstrap_count",
            ]
        )

    adjacency_tables = make_adjacency_tables_from_edges(edge_table, tau_min=tau_min, tau_max=tau_max)
    bootstrap_run_summary = pd.DataFrame(bootstrap_summary_rows)
    total_windows = int(sum(series.shape[0] for series in series_dict.values()))
    run_summary = pd.DataFrame(
        [
            {"metric": "n_patients", "value": int(len(series_dict))},
            {"metric": "n_windows", "value": total_windows},
            {"metric": "analysis_mode", "value": "multiple"},
            {"metric": "learning_algorithm", "value": "bootstrap_gma_pcmci_plus"},
            {"metric": "n_bootstraps", "value": int(n_bootstraps)},
            {"metric": "gma_threshold", "value": float(gma_threshold)},
            {"metric": "bootstrap_seed", "value": int(bootstrap_seed)},
            {"metric": "tau_min", "value": int(tau_min)},
            {"metric": "tau_max", "value": int(tau_max)},
            {"metric": "pc_alpha", "value": float(pc_alpha)},
            {"metric": "alpha_level", "value": float(alpha_level)},
            {"metric": "ci_test", "value": ci_test_name},
            {
                "metric": "retained_structure_count",
                "value": int(structure_table["retained"].sum()) if not structure_table.empty else 0,
            },
            {
                "metric": "accepted_structure_count",
                "value": int(structure_table["accepted"].sum()) if not structure_table.empty else 0,
            },
            {
                "metric": "selected_directed_edge_count",
                "value": int(edge_table["selected"].sum()) if not edge_table.empty else 0,
            },
            {
                "metric": "selected_adjacency_count",
                "value": int(edge_table["selected_adjacency"].sum()) if not edge_table.empty else 0,
            },
        ]
    )
    return (
        edge_table,
        adjacency_tables,
        run_summary,
        bootstrap_edge_table,
        structure_table,
        bootstrap_run_summary,
    )


def plot_group_mean_timeseries(
    series_dict: dict[str, np.ndarray],
    time_axes_dict: dict[str, np.ndarray],
    output_path: Path,
    group_name: str,
):
    """Plot the cohort-level multivariate time series with a readable legend.

    Each colored line is the mean node trajectory across patients as a function
    of relative time since the start of a study. The shaded band shows the
    interquartile range, which helps communicate between-patient variability.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Matplotlib is required for plotting.") from exc

    patient_ids = list(series_dict.keys())
    max_len = max(series.shape[0] for series in series_dict.values())
    padded = np.full((len(patient_ids), max_len, len(NODE_COLUMNS)), np.nan, dtype=float)
    padded_time = np.full((len(patient_ids), max_len), np.nan, dtype=float)

    for patient_idx, patient_id in enumerate(patient_ids):
        series = series_dict[patient_id]
        padded[patient_idx, : series.shape[0], :] = series
        patient_time = time_axes_dict.get(patient_id)
        if patient_time is not None and len(patient_time) == series.shape[0]:
            padded_time[patient_idx, : series.shape[0]] = patient_time
        else:
            padded_time[patient_idx, : series.shape[0]] = np.arange(series.shape[0], dtype=float)

    mean_values = np.nanmean(padded, axis=0)
    q25_values = np.nanquantile(padded, 0.25, axis=0)
    q75_values = np.nanquantile(padded, 0.75, axis=0)
    time_minutes = np.nanmedian(padded_time, axis=0)
    fallback_mask = ~np.isfinite(time_minutes)
    if fallback_mask.any():
        time_minutes[fallback_mask] = np.arange(max_len, dtype=float)[fallback_mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    for node_idx, node_name in enumerate(NODE_COLUMNS):
        ax.plot(
            time_minutes,
            mean_values[:, node_idx],
            linewidth=2,
            label=node_name,
        )
        ax.fill_between(
            time_minutes,
            q25_values[:, node_idx],
            q75_values[:, node_idx],
            alpha=0.15,
        )

    ax.set_title(f"Group mean multivariate time series: {group_name}")
    ax.set_xlabel("Relative time since study start (minutes)")
    ax.set_ylabel("Node value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_patient_overlay_timeseries(
    series_dict: dict[str, np.ndarray],
    time_axes_dict: dict[str, np.ndarray],
    output_path: Path,
    group_name: str,
    plot_max_patients: int,
):
    """Plot a small overlay of individual patient trajectories for inspection.

    This figure complements the group-mean plot above by showing what a handful
    of raw patient trajectories actually look like. We cap the patient count so
    the legend stays readable.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Matplotlib is required for plotting.") from exc

    # Pick the longest patient series first so the overlay plot is informative.
    ranked_items = sorted(
        series_dict.items(),
        key=lambda item: item[1].shape[0],
        reverse=True,
    )[:plot_max_patients]

    fig, axes = plt.subplots(
        len(NODE_COLUMNS),
        1,
        figsize=(13, 2.6 * len(NODE_COLUMNS)),
        sharex=False,
    )

    for node_idx, node_name in enumerate(NODE_COLUMNS):
        ax = axes[node_idx]
        for patient_id, series in ranked_items:
            patient_time = time_axes_dict.get(patient_id)
            if patient_time is not None and len(patient_time) == series.shape[0]:
                time_minutes = patient_time
            else:
                time_minutes = np.arange(series.shape[0], dtype=float)
            ax.plot(
                time_minutes,
                series[:, node_idx],
                alpha=0.8,
                linewidth=1.2,
                label=f"patient {patient_id}",
            )
        ax.set_title(node_name)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=True)

    axes[-1].set_xlabel("Relative time since study start (minutes)")
    fig.suptitle(f"Patient overlay multivariate time series: {group_name}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pcmci_graph(
    raw_results: dict[str, object],
    output_path: Path,
):
    """Plot the learned PCMCI+ graph if Tigramite plotting is available."""
    try:
        import matplotlib.pyplot as plt
        from tigramite import plotting as tp

        plotting_results = build_plotting_safe_results(raw_results)
        fig = plt.figure(figsize=(10, 8))
        tp.plot_graph(
            graph=plotting_results["graph"],
            val_matrix=plotting_results["val_matrix"],
            var_names=NODE_COLUMNS,
            link_colorbar_label="MCI test statistic",
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception as exc:
        print(
            f"Warning: could not save PCMCI graph plot to {output_path}: {exc}. "
            "This plot is optional and may require extra dependencies such as networkx."
        )
        return False


def build_plotting_safe_results(raw_results: dict[str, object]) -> dict[str, object]:
    """Create a plotting-safe copy of the Tigramite results.

    Tigramite's plotting utilities can be stricter than the learner itself,
    especially for contemporaneous lag-0 links. In practice the main failure is
    that the plotting helper expects lag-0 values to be symmetric across
    variable pairs. We do not change the learned graph structure here; we only
    make a safe copy of `val_matrix` for visualization.
    """
    safe_results = dict(raw_results)
    val_matrix = np.array(raw_results["val_matrix"], copy=True)

    if val_matrix.ndim == 3 and val_matrix.shape[2] > 0:
        lag0_matrix = val_matrix[:, :, 0].copy()
        num_nodes = lag0_matrix.shape[0]

        for i in range(num_nodes):
            lag0_matrix[i, i] = 0.0
            for j in range(i + 1, num_nodes):
                value_ij = lag0_matrix[i, j]
                value_ji = lag0_matrix[j, i]

                finite_ij = np.isfinite(value_ij)
                finite_ji = np.isfinite(value_ji)

                if finite_ij and finite_ji:
                    chosen_value = value_ij if abs(value_ij) >= abs(value_ji) else value_ji
                elif finite_ij:
                    chosen_value = value_ij
                elif finite_ji:
                    chosen_value = value_ji
                else:
                    chosen_value = 0.0

                lag0_matrix[i, j] = chosen_value
                lag0_matrix[j, i] = chosen_value

        val_matrix[:, :, 0] = lag0_matrix

    safe_results["val_matrix"] = val_matrix
    return safe_results


def plot_tigramite_time_series_graph(
    raw_results: dict[str, object],
    output_path: Path,
):
    """Plot Tigramite's rolled-out temporal graph across lags.

    This is the most direct visualization of the temporal graph because it shows
    copies of each variable across time slices and draws edges such as
    X(t-2) -> Y(t) explicitly. It is therefore easier to interpret than a single
    static graph when multiple lags are present.
    """
    try:
        from tigramite import plotting as tp
    except Exception as exc:
        print(f"Warning: Tigramite plotting import failed for {output_path}: {exc}")
        return False

    try:
        plotting_results = build_plotting_safe_results(raw_results)
        tp.plot_time_series_graph(
            graph=plotting_results["graph"],
            val_matrix=plotting_results["val_matrix"],
            var_names=NODE_COLUMNS,
            link_colorbar_label="MCI test statistic",
            save_name=str(output_path),
            figsize=(12, 8),
        )
        if Path(output_path).exists():
            return True
        print(f"Warning: Tigramite reported success but did not create {output_path}")
        return False
    except Exception as exc:
        print(f"Warning: could not save Tigramite time-series graph to {output_path}: {exc}")
        return False


def plot_unrolled_temporal_graph(
    edge_table: pd.DataFrame,
    output_path: Path,
    group_name: str,
    tau_min: int,
    tau_max: int,
    edge_abs_threshold: float = 0.0,
    node_display_names: dict[str, str] | None = None,
):
    """Plot a single unrolled temporal graph across time slices.

    This produces the style of visualization where each variable appears once
    per relative time slice, for example from t-5 up to t, and stationary edges
    are repeated across the columns wherever they fit.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize
    except Exception as exc:
        raise ImportError("Matplotlib is required for plotting.") from exc

    if node_display_names is None:
        node_display_names = NODE_DISPLAY_NAMES

    selected = edge_table[
        edge_table["selected"] & (edge_table["abs_mci_value"] > edge_abs_threshold)
    ].copy()

    time_slices = list(range(tau_max, -1, -1))
    column_positions = {lag_back: idx for idx, lag_back in enumerate(time_slices)}
    row_positions = {node_name: idx for idx, node_name in enumerate(NODE_COLUMNS)}

    fig_width = max(8, 2.2 * len(time_slices))
    fig_height = max(6, 1.2 * len(NODE_COLUMNS))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Draw node labels and nodes.
    for lag_back in time_slices:
        x_pos = column_positions[lag_back]
        if lag_back == 0:
            title = r"$t$"
        else:
            title = rf"$t-{lag_back}$"
        ax.text(x_pos, -0.8, title, ha="center", va="center", fontsize=12)

    for node_name in NODE_COLUMNS:
        y_pos = row_positions[node_name]
        label = node_display_names.get(node_name, node_name)
        ax.text(-0.45, y_pos, label, ha="right", va="center", fontsize=10)
        for lag_back in time_slices:
            x_pos = column_positions[lag_back]
            node = plt.Circle(
                (x_pos, y_pos),
                0.08,
                facecolor="lightgrey",
                edgecolor="lightgrey",
                zorder=3,
            )
            ax.add_patch(node)

    if selected.empty:
        ax.text(
            len(time_slices) / 2.0 - 0.5,
            len(NODE_COLUMNS) / 2.0,
            "No selected directed edges above threshold",
            ha="center",
            va="center",
            fontsize=13,
        )
        ax.set_xlim(-0.7, len(time_slices) - 0.3)
        ax.set_ylim(len(NODE_COLUMNS) - 0.5, -1.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    max_abs_weight = max(float(selected["abs_mci_value"].max()), 1e-8)
    norm = Normalize(vmin=-max_abs_weight, vmax=max_abs_weight)
    cmap = cm.get_cmap("RdBu_r")

    for _, row in selected.iterrows():
        source = row["source_node"]
        target = row["target_node"]
        lag = int(row["lag"])
        weight = float(row["mci_value"])
        color = cmap(norm(weight))
        source_y = row_positions[source]
        target_y = row_positions[target]

        # A lag-k edge can be drawn repeatedly from every column where the edge
        # still lands within the visible time window.
        for target_lag_back in range(tau_max - lag, -1, -1):
            source_lag_back = target_lag_back + lag
            if source_lag_back > tau_max:
                continue

            x_start = column_positions[source_lag_back]
            y_start = source_y
            x_end = column_positions[target_lag_back]
            y_end = target_y

            if lag == 0:
                if source == target:
                    loop = plt.Circle(
                        (x_start, y_start - 0.18),
                        0.12,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.2 + 2.8 * (abs(weight) / max_abs_weight),
                        zorder=2,
                    )
                    ax.add_patch(loop)
                    continue

                delta = 0.08 if source_y < target_y else -0.08
                ax.annotate(
                    "",
                    xy=(x_end, y_end - delta),
                    xytext=(x_start, y_start + delta),
                    arrowprops={
                        "arrowstyle": "->",
                        "linewidth": 1.2 + 2.8 * (abs(weight) / max_abs_weight),
                        "color": color,
                        "connectionstyle": "arc3,rad=0.28",
                        "alpha": 0.9,
                    },
                    zorder=1,
                )
            else:
                ax.annotate(
                    "",
                    xy=(x_end - 0.08, y_end),
                    xytext=(x_start + 0.08, y_start),
                    arrowprops={
                        "arrowstyle": "->",
                        "linewidth": 1.2 + 2.8 * (abs(weight) / max_abs_weight),
                        "color": color,
                        "connectionstyle": "arc3,rad=0.0",
                        "alpha": 0.9,
                    },
                    zorder=1,
                )

    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Mean bootstrap MCI value")

    title = f"Unrolled temporal graph: {group_name}"
    if edge_abs_threshold > 0:
        title = f"{title} (|mean MCI| > {edge_abs_threshold:g})"
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.7, len(time_slices) - 0.3)
    ax.set_ylim(len(NODE_COLUMNS) - 0.5, -1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_adjacency_heatmap(
    adjacency_tables: dict[int, pd.DataFrame],
    output_path: Path,
    group_name: str,
):
    """Plot one heatmap per lag as a robust fallback graph visualization."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Matplotlib is required for plotting.") from exc

    lags = sorted(adjacency_tables)
    fig, axes = plt.subplots(1, len(lags), figsize=(5 * len(lags), 5), squeeze=False)

    for axis_idx, lag in enumerate(lags):
        ax = axes[0, axis_idx]
        adjacency = adjacency_tables[lag]
        image = ax.imshow(adjacency.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
        ax.set_title(f"{group_name}\nLag {lag} adjacency")
        ax.set_xticks(range(len(NODE_COLUMNS)))
        ax.set_xticklabels(NODE_COLUMNS, rotation=45, ha="right")
        ax.set_yticks(range(len(NODE_COLUMNS)))
        ax.set_yticklabels(NODE_COLUMNS)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Mean bootstrap MCI value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_selected_edge_graphs(
    edge_table: pd.DataFrame,
    output_path: Path,
    group_name: str,
):
    """Plot the selected directed graph for each lag with nodes and arrows.

    This produces a more intuitive causal-graph style figure than a heatmap.
    Each panel corresponds to one lag, and only selected directed edges are
    shown. Edge labels display mean selected-bootstrap MCI values.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Matplotlib is required for plotting.") from exc

    selected = edge_table[edge_table["selected"]].copy()
    lags = sorted(edge_table["lag"].unique())
    if not lags:
        return

    fig, axes = plt.subplots(1, len(lags), figsize=(6 * len(lags), 6), squeeze=False)

    # Put the six nodes on a circle so edges are easy to compare across lags.
    node_count = len(NODE_COLUMNS)
    angles = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
    radius = 1.0
    positions = {
        node_name: (radius * np.cos(angle), radius * np.sin(angle))
        for node_name, angle in zip(NODE_COLUMNS, angles)
    }

    for axis_idx, lag in enumerate(lags):
        ax = axes[0, axis_idx]
        ax.set_title(f"{group_name}\nSelected graph, lag {lag} mean MCI")
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw nodes first.
        for node_name, (x_pos, y_pos) in positions.items():
            circle = plt.Circle(
                (x_pos, y_pos),
                0.14,
                facecolor="#d9ecff",
                edgecolor="#1f4e79",
                linewidth=2,
                zorder=3,
            )
            ax.add_patch(circle)
            ax.text(
                x_pos,
                y_pos,
                NODE_DISPLAY_NAMES.get(node_name, node_name),
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )

        lag_edges = selected[selected["lag"] == lag].copy()
        if lag_edges.empty:
            ax.text(0.0, -1.35, "No selected directed edges", ha="center", fontsize=10)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            continue

        max_abs_weight = max(float(lag_edges["abs_mci_value"].max()), 1e-8)

        for _, row in lag_edges.iterrows():
            source = row["source_node"]
            target = row["target_node"]
            weight = float(row["mci_value"])
            abs_weight = float(row["abs_mci_value"])

            x_start, y_start = positions[source]
            x_end, y_end = positions[target]

            if source == target:
                # Draw self-loops as small arcs above the node.
                loop = plt.Circle(
                    (x_start, y_start + 0.18),
                    0.12,
                    fill=False,
                    edgecolor="#b22222",
                    linewidth=1.5 + 3.0 * (abs_weight / max_abs_weight),
                    zorder=2,
                )
                ax.add_patch(loop)
                ax.text(
                    x_start,
                    y_start + 0.38,
                    f"{weight:.2f}",
                    color="#b22222",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                continue

            # Shorten the arrow so it starts/ends at the node boundary, not center.
            dx = x_end - x_start
            dy = y_end - y_start
            distance = max(np.hypot(dx, dy), 1e-8)
            shrink = 0.18
            x_arrow_start = x_start + shrink * dx / distance
            y_arrow_start = y_start + shrink * dy / distance
            x_arrow_end = x_end - shrink * dx / distance
            y_arrow_end = y_end - shrink * dy / distance

            line_width = 1.5 + 3.5 * (abs_weight / max_abs_weight)
            color = "#b22222" if weight >= 0 else "#005f73"
            ax.annotate(
                "",
                xy=(x_arrow_end, y_arrow_end),
                xytext=(x_arrow_start, y_arrow_start),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": line_width,
                    "color": color,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "alpha": 0.9,
                },
                zorder=1,
            )

            x_mid = (x_arrow_start + x_arrow_end) / 2
            y_mid = (y_arrow_start + y_arrow_end) / 2
            ax.text(
                x_mid,
                y_mid,
                f"{weight:.2f}",
                color=color,
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.4},
            )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_structure_graphs(
    structure_table: pd.DataFrame,
    output_dir: Path,
    group_name: str,
) -> pd.DataFrame:
    """Plot every target-local parent structure ordered by posterior support."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Matplotlib is required for plotting.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    if structure_table.empty:
        return pd.DataFrame()

    node_count = len(NODE_COLUMNS)
    angles = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
    radius = 1.0
    positions = {
        node_name: (radius * np.cos(angle), radius * np.sin(angle))
        for node_name, angle in zip(NODE_COLUMNS, angles)
    }

    ordered_structures = structure_table.sort_values(
        ["posterior", "bootstrap_count", "target_node", "parent_edges"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    for fallback_rank, row in ordered_structures.iterrows():
        rank = int(row.get("significance_rank", fallback_rank + 1))
        target_node = str(row["target_node"])
        parent_edges = parse_parent_edges(str(row["parent_edges"]))
        plot_name = f"structure_{rank:03d}_{target_node}.png"
        plot_path = output_dir / plot_name

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.set_title(
            f"{group_name}: structure {rank}\n"
            f"target={target_node}, posterior={float(row['posterior']):.3f}, "
            f"count={int(row['bootstrap_count'])}",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.axis("off")

        for node_name, (x_pos, y_pos) in positions.items():
            is_target = node_name == target_node
            circle = plt.Circle(
                (x_pos, y_pos),
                0.15 if is_target else 0.13,
                facecolor="#ffe8a3" if is_target else "#d9ecff",
                edgecolor="#7a4f00" if is_target else "#1f4e79",
                linewidth=2.4 if is_target else 1.8,
                zorder=3,
            )
            ax.add_patch(circle)
            ax.text(
                x_pos,
                y_pos,
                NODE_DISPLAY_NAMES.get(node_name, node_name).replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=9,
                zorder=4,
            )

        if not parent_edges:
            ax.text(0.0, -1.35, "No selected parents", ha="center", fontsize=10)
        else:
            for source_node, _, lag in parent_edges:
                x_start, y_start = positions[source_node]
                x_end, y_end = positions[target_node]

                if source_node == target_node:
                    loop = plt.Circle(
                        (x_start, y_start + 0.2),
                        0.13,
                        fill=False,
                        edgecolor="#b22222" if lag == 0 else "#005f73",
                        linewidth=2.0,
                        zorder=2,
                    )
                    ax.add_patch(loop)
                    ax.text(
                        x_start,
                        y_start + 0.42,
                        f"lag {lag}",
                        color="#b22222" if lag == 0 else "#005f73",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    continue

                dx = x_end - x_start
                dy = y_end - y_start
                distance = max(np.hypot(dx, dy), 1e-8)
                shrink = 0.2
                x_arrow_start = x_start + shrink * dx / distance
                y_arrow_start = y_start + shrink * dy / distance
                x_arrow_end = x_end - shrink * dx / distance
                y_arrow_end = y_end - shrink * dy / distance
                color = "#b22222" if lag == 0 else "#005f73"
                ax.annotate(
                    "",
                    xy=(x_arrow_end, y_arrow_end),
                    xytext=(x_arrow_start, y_arrow_start),
                    arrowprops={
                        "arrowstyle": "->",
                        "linewidth": 2.0,
                        "color": color,
                        "shrinkA": 0,
                        "shrinkB": 0,
                        "alpha": 0.9,
                    },
                    zorder=1,
                )
                ax.text(
                    (x_arrow_start + x_arrow_end) / 2,
                    (y_arrow_start + y_arrow_end) / 2,
                    f"lag {lag}",
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3},
                )

        status_parts = []
        if bool(row.get("retained", False)):
            status_parts.append("retained")
        if bool(row.get("accepted", False)):
            status_parts.append("accepted")
        ax.text(
            0.0,
            1.36,
            ", ".join(status_parts) if status_parts else "not retained",
            ha="center",
            fontsize=9,
        )
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.55)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        manifest_rows.append(
            {
                "significance_rank": rank,
                "target_node": target_node,
                "parent_edges": row["parent_edges"],
                "posterior": float(row["posterior"]),
                "bootstrap_count": int(row["bootstrap_count"]),
                "retained": bool(row["retained"]),
                "accepted": bool(row["accepted"]),
                "plot_path": str(plot_path),
            }
        )

    return pd.DataFrame(manifest_rows)


def slugify_group_name(group_name: str) -> str:
    return group_name.replace(":", "_")


def save_group_outputs(
    output_dir: Path,
    group_name: str,
    edge_table: pd.DataFrame,
    adjacency_tables: dict[int, pd.DataFrame],
    run_summary: pd.DataFrame,
    patient_series_summary: pd.DataFrame,
    bootstrap_edge_table: pd.DataFrame | None = None,
    structure_posterior_table: pd.DataFrame | None = None,
    bootstrap_run_summary: pd.DataFrame | None = None,
    tau_tuning_summary: pd.DataFrame | None = None,
):
    """Write all tabular outputs for one cohort to its own folder."""
    group_dir = output_dir / slugify_group_name(group_name)
    group_dir.mkdir(parents=True, exist_ok=True)

    edge_table.to_csv(group_dir / "edge_table.csv", index=False)
    run_summary.to_csv(group_dir / "run_summary.csv", index=False)
    patient_series_summary.to_csv(group_dir / "patient_series_summary.csv", index=False)
    if bootstrap_edge_table is not None:
        bootstrap_edge_table.to_csv(group_dir / "bootstrap_edge_table.csv", index=False)
    if structure_posterior_table is not None:
        structure_posterior_table.to_csv(group_dir / "structure_posterior_table.csv", index=False)
    if bootstrap_run_summary is not None:
        bootstrap_run_summary.to_csv(group_dir / "bootstrap_run_summary.csv", index=False)
    if tau_tuning_summary is not None:
        tau_tuning_summary.to_csv(group_dir / "tau_tuning_summary.csv", index=False)

    for lag, adjacency in adjacency_tables.items():
        adjacency.to_csv(group_dir / f"adjacency_matrix_lag{lag}.csv")

    return group_dir


def main():
    """Coordinate cohort loading, optional plotting, tau tuning, and PCMCI+ runs."""
    args = parse_args()

    if args.tau_min < 0:
        raise ValueError("--tau_min must be non-negative.")
    if args.tau_max < args.tau_min:
        raise ValueError("--tau_max must be greater than or equal to --tau_min.")
    if args.min_windows <= args.tau_max:
        raise ValueError("--min_windows must be larger than --tau_max.")
    if args.n_bootstraps <= 0:
        raise ValueError("--n_bootstraps must be positive.")
    if not 0.0 <= args.gma_threshold <= 1.0:
        raise ValueError("--gma_threshold must be between 0 and 1.")

    patient_info = load_patient_info(args.patient_info_path)
    id_groups = get_id_groups(patient_info)
    requested_groups = resolve_requested_groups(args.groups, id_groups)
    edge_constraints = load_edge_constraints(
        edge_constraints_path=args.edge_constraints_path,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )
    link_assumptions = build_link_assumptions(
        edge_constraints=edge_constraints,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []

    print("Starting Bootstrap-GMA PCMCI+ pipeline")
    print(f"Patient metadata: {args.patient_info_path}")
    print(f"Windowed data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Requested groups: {', '.join(requested_groups)}")
    print(f"tau_min={args.tau_min}, tau_max={args.tau_max}, pc_alpha={args.pc_alpha}, alpha_level={args.alpha_level}")
    print(
        f"Bootstrap-GMA: n_bootstraps={args.n_bootstraps}, "
        f"gma_threshold={args.gma_threshold}, bootstrap_seed={args.bootstrap_seed}"
    )
    print(f"CI test: {args.ci_test}")
    if args.edge_constraints_path:
        print(f"Edge constraints: {args.edge_constraints_path}")
    else:
        print("Edge constraints: none")
    if args.tau_min == 0:
        print("Contemporaneous edges are enabled (lag 0 included).")
    else:
        print("Only lagged edges are enabled (lag 0 excluded).")
    if args.tune_tau:
        print(
            "Tau tuning is ignored for Bootstrap-GMA v1; using the supplied "
            f"tau_max={args.tau_max} for every bootstrap."
        )

    group_iterator = tqdm(
        requested_groups,
        desc="Processing groups",
        unit="group",
    )
    for group_name in group_iterator:
        group_iterator.set_postfix_str(group_name)
        patient_ids = [normalize_patient_id(patient_id) for patient_id in id_groups[group_name]]
        patient_ids = [patient_id for patient_id in patient_ids if patient_id]
        patient_ids = sorted(set(patient_ids), key=lambda item: (len(item), item))

        print()
        print(f"Running PCMCI+ for group: {group_name}")
        print(f"Requested patients: {len(patient_ids)}")

        series_dict, time_axes_dict, patient_series_summary = build_group_series_dict(
            data_dir=data_dir,
            patient_ids=patient_ids,
            min_windows=args.min_windows,
        )

        if not series_dict:
            print("No usable patient series were found for this group. Skipping.")
            overall_rows.append(
                {
                    "group": group_name,
                    "requested_patients": int(len(patient_ids)),
                    "included_patients": 0,
                    "selected_edges": 0,
                    "status": "skipped_no_usable_series",
                }
            )
            save_group_outputs(
                output_dir=output_dir,
                group_name=group_name,
                edge_table=pd.DataFrame(),
                adjacency_tables={},
                run_summary=pd.DataFrame(
                    [
                        {"metric": "group", "value": group_name},
                        {"metric": "status", "value": "skipped_no_usable_series"},
                    ]
                ),
                patient_series_summary=patient_series_summary,
                bootstrap_edge_table=pd.DataFrame(),
                structure_posterior_table=pd.DataFrame(),
                bootstrap_run_summary=pd.DataFrame(),
                tau_tuning_summary=None,
            )
            continue

        included_count = int(patient_series_summary["included"].sum()) if not patient_series_summary.empty else 0
        skipped_count = int((~patient_series_summary["included"]).sum()) if not patient_series_summary.empty else 0
        total_windows = int(
            patient_series_summary.loc[patient_series_summary["included"], "n_windows"].sum()
        ) if not patient_series_summary.empty else 0
        print(
            f"Usable patient series: {included_count} included, {skipped_count} skipped, "
            f"{total_windows} total windows"
        )

        if not args.skip_plots:
            # Save the input-side plots first. These are useful regardless of which
            # tau value ends up winning.
            group_plot_dir = output_dir / slugify_group_name(group_name)
            group_plot_dir.mkdir(parents=True, exist_ok=True)
            print("Generating input time-series plots...")
            plot_group_mean_timeseries(
                series_dict=series_dict,
                time_axes_dict=time_axes_dict,
                output_path=group_plot_dir / "group_mean_multivariate_timeseries.png",
                group_name=group_name,
            )
            plot_patient_overlay_timeseries(
                series_dict=series_dict,
                time_axes_dict=time_axes_dict,
                output_path=group_plot_dir / "patient_overlay_multivariate_timeseries.png",
                group_name=group_name,
                plot_max_patients=args.plot_max_patients,
            )

        tau_tuning_summary = None
        print("Running Bootstrap-GMA fits...")
        (
            edge_table,
            adjacency_tables,
            run_summary,
            bootstrap_edge_table,
            structure_posterior_table,
            bootstrap_run_summary,
        ) = learn_bootstrap_gma_for_group(
            series_dict=series_dict,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            pc_alpha=args.pc_alpha,
            alpha_level=args.alpha_level,
            verbosity=args.verbosity,
            ci_test_name=args.ci_test,
            link_assumptions=link_assumptions,
            n_bootstraps=args.n_bootstraps,
            gma_threshold=args.gma_threshold,
            bootstrap_seed=args.bootstrap_seed,
        )

        run_summary = pd.concat(
            [
                pd.DataFrame(
                    [
                        {"metric": "group", "value": group_name},
                        {"metric": "patient_info_path", "value": args.patient_info_path},
                        {"metric": "data_dir", "value": str(data_dir)},
                        {"metric": "requested_patients", "value": int(len(patient_ids))},
                        {"metric": "min_windows", "value": int(args.min_windows)},
                        {"metric": "tau_tuned", "value": False},
                        {"metric": "tau_tuning_metric", "value": args.tau_tuning_metric},
                        {
                            "metric": "edge_constraints_path",
                            "value": args.edge_constraints_path or "",
                        },
                        {
                            "metric": "edge_constraints_enabled",
                            "value": bool(link_assumptions is not None),
                        },
                    ]
                ),
                run_summary,
            ],
            ignore_index=True,
        )

        print("Saving tables...")
        group_dir = save_group_outputs(
            output_dir=output_dir,
            group_name=group_name,
            edge_table=edge_table,
            adjacency_tables=adjacency_tables,
            run_summary=run_summary,
            patient_series_summary=patient_series_summary,
            bootstrap_edge_table=bootstrap_edge_table,
            structure_posterior_table=structure_posterior_table,
            bootstrap_run_summary=bootstrap_run_summary,
            tau_tuning_summary=tau_tuning_summary,
        )

        if not args.skip_plots:
            # Plot the aggregated graph after the fit so the visual matches the
            # saved edge tables and adjacency matrices. Raw Tigramite plots are
            # skipped because the final graph is not one raw PCMCI+ result.
            print("Generating graph visualizations...")
            plot_unrolled_temporal_graph(
                edge_table=edge_table,
                output_path=group_dir / "unrolled_temporal_graph.png",
                group_name=group_name,
                tau_min=args.tau_min,
                tau_max=args.tau_max,
                edge_abs_threshold=0.01,
                node_display_names=NODE_DISPLAY_NAMES,
            )
            plot_selected_edge_graphs(
                edge_table=edge_table,
                output_path=group_dir / "selected_edge_graphs.png",
                group_name=group_name,
            )
            plot_adjacency_heatmap(
                adjacency_tables=adjacency_tables,
                output_path=group_dir / "adjacency_heatmaps.png",
                group_name=group_name,
            )
            structure_plot_manifest = plot_structure_graphs(
                structure_table=structure_posterior_table,
                output_dir=group_dir / "structure_plots",
                group_name=group_name,
            )
            if not structure_plot_manifest.empty:
                structure_plot_manifest.to_csv(
                    group_dir / "structure_plots" / "structure_plot_manifest.csv",
                    index=False,
                )

        selected_edges = edge_table[edge_table["selected"]].copy()
        print(f"Included patients: {len(series_dict)}")
        print(f"Selected directed edges: {len(selected_edges)}")
        if not selected_edges.empty:
            print(
                selected_edges[
                    [
                        "source_node",
                        "target_node",
                        "lag",
                        "mci_value",
                        "p_value",
                        "graph_entry",
                    ]
                ].head(10)
            )
        print(f"Finished group: {group_name}")

        overall_rows.append(
            {
                "group": group_name,
                "requested_patients": int(len(patient_ids)),
                "included_patients": int(len(series_dict)),
                "selected_edges": int(len(selected_edges)),
                "status": "completed",
            }
        )

    pd.DataFrame(overall_rows).to_csv(output_dir / "overall_group_summary.csv", index=False)
    print()
    print(f"Saved PCMCI+ outputs to: {output_dir}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
