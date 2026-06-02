import pandas as pd
import argparse
import scipy
import os
import numpy as np
from tqdm import tqdm

SIGNAL_PREFIXES = ("flow", "snoring", "effort", "pulse", "saturation")


def parse_optional_float(value):
    """Allow numeric CLI values or the string 'none' to disable a bound."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", ""}:
        return None
    return float(value)


parser = argparse.ArgumentParser()
parser.add_argument("--patient_data_dir", type=str, default="csv_output", help="Path to patient data directory")
parser.add_argument("--output_path", type=str, default="full_patient_info.csv", help="Path to save the processed data")
parser.add_argument("--filtering_threshold", type=int, default=7200, help="Minimum number of seconds of data required for a patient to be included in the output")
parser.add_argument("--features_required", type=str, default="flow,snoring,effort,pulse,saturation", help="Comma-separated list of features required")
parser.add_argument("--windowed_output_dir", type=str, default="windowed_output", help="Directory to save per-patient windowed datasets")
parser.add_argument("--window_size", type=int, default=10, help="Window size in seconds for feature computation")
parser.add_argument(
    "--pulse_valid_min",
    type=parse_optional_float,
    default=25.0,
    help="Minimum valid pulse value. Use 'none' to disable the lower bound. Default: 25.",
)
parser.add_argument(
    "--pulse_valid_max",
    type=parse_optional_float,
    default=250.0,
    help="Maximum valid pulse value. Use 'none' to disable the upper bound. Default: 250.",
)
parser.add_argument(
    "--saturation_valid_min",
    type=parse_optional_float,
    default=50.0,
    help="Minimum valid saturation value. Use 'none' to disable the lower bound. Default: 50.",
)
parser.add_argument(
    "--saturation_valid_max",
    type=parse_optional_float,
    default=100.0,
    help="Maximum valid saturation value. Use 'none' to disable the upper bound. Default: 100.",
)
args = parser.parse_args()


def get_patient_info(patient_data_dir, filtering_threshold, features_required):
    patient_info_dict = {}
    patient_info_df = None
    pbar = tqdm(total=len(os.listdir(patient_data_dir)), desc="Processing patient data")
    for filename in os.listdir(patient_data_dir):
        if filename.endswith(".csv") and filename.startswith("anonymized_"):
            pbar.desc = f"Processing {filename}"
            pbar.update(1)
            patient_path = os.path.join(patient_data_dir, filename)
            header_df = pd.read_csv(patient_path, nrows=0)
            patient_id = filename.split("_")[1].split(".")[0]
            patient_info_dict[patient_id] = {}
            missing_features = []
            runtime_short = []
            header_cols = list(header_df.columns)
            header_cols_lower = [col.lower() for col in header_cols]
            lower_to_original = {col_lower: col for col, col_lower in zip(header_cols, header_cols_lower)}
            patient_columns = header_cols_lower
            for feature in features_required:
                if feature not in patient_columns:
                    missing_features.append(feature)
                else:
                    time_col = f"{feature}_time_s"
                    if time_col not in patient_columns:
                        missing_features.append(feature)
                        continue
                    time_col_original = lower_to_original[time_col]
                    time_df = pd.read_csv(patient_path, usecols=[time_col_original])
                    runtime = time_df[time_col_original].max() - time_df[time_col_original].min()
                    if runtime < filtering_threshold:
                        runtime_short.append(feature)
            patient_info_dict[patient_id]["missing_features"] = ", ".join(missing_features)
            patient_info_dict[patient_id]["runtime_short"] = ", ".join(runtime_short)
            patient_info_dict[patient_id]["valid"] = not missing_features and not runtime_short
        elif filename.endswith(".csv") and filename.startswith("patient_info"):
            patient_info_df = pd.read_csv(os.path.join(patient_data_dir, filename))
        else:
            print(f"Skipping unrecognized file: {filename}") 

    # Add the information to the patient_info_dict
    if patient_info_df is None:
        print("No patient_info.csv found in patient_data_dir; age/sex will be missing.")
        return patient_info_dict

    patient_info_df["patient_id"] = patient_info_df["patient_id"].astype(str).str.strip()
    for patient_id in patient_info_dict:
        patient_id_str = str(patient_id).strip()
        if patient_id_str in patient_info_df["patient_id"].values:
            patient_info = patient_info_df[patient_info_df["patient_id"] == patient_id_str].iloc[0]
            patient_info_dict[patient_id]["age"] = patient_info["age"]
            patient_info_dict[patient_id]["sex"] = patient_info["sex"]     
    return patient_info_dict

def create_windowed_features(
    patient_data: pd.DataFrame,
    window_size=30,
    pulse_valid_min=None,
    pulse_valid_max=None,
    saturation_valid_min=None,
    saturation_valid_max=None,
):
    # This method is used for creating the windowed features for the different patients. It takes the patient data as input, and returns a dataframe with the engineered features.
    df = patient_data.copy()
    df_windowed = pd.DataFrame()

    def get_value_cols(prefix: str) -> list:
        return [prefix] if prefix in df.columns else []

    def get_time_col(prefix: str) -> str:
        preferred = f"{prefix}_time_s"
        if preferred in df.columns:
            return preferred
        if "time_s" in df.columns:
            return "time_s"
        raise ValueError(f"Missing time column for {prefix}: expected '{preferred}' or 'time_s'.")

    def bin_by_seconds(value_cols: list, time_col: str) -> pd.core.groupby.generic.DataFrameGroupBy:
        temp = df[[time_col] + value_cols].copy()
        temp[time_col] = pd.to_numeric(temp[time_col], errors="coerce")
        temp = temp.dropna(subset=[time_col])
        temp["bin_start_s"] = (temp[time_col] // window_size) * window_size
        return temp.groupby("bin_start_s", sort=True)

    def flatten_stat_columns(windowed_df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(windowed_df.columns, pd.MultiIndex):
            windowed_df.columns = [f"{col}_{stat}" for col, stat in windowed_df.columns]
        return windowed_df

    def rolling_zero_crossing_rate_raw(window: np.ndarray) -> float:
        if window.size < 2:
            return 0.0
        return float(np.sum(window[1:] * window[:-1] < 0))

    def rolling_non_zero_fraction_raw(window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        return float(np.mean(window != 0))

    def slope_from_time_values(time_s: np.ndarray, values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        slope = np.polyfit(time_s, values, 1)[0]
        return float(slope)

    def rolling_rms_raw(window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(window ** 2)))

    def drop_from_start_to_min(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.min(values) - values[0])

    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0 or window <= 1:
            return values.astype(float, copy=True)
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(values, kernel, mode="same")

    def estimate_sampling_frequency(time_s: np.ndarray) -> float:
        if time_s.size < 2:
            return 1.0
        diffs = np.diff(time_s)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 1.0
        return float(1.0 / np.median(diffs))

    def get_time_and_values(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        value_cols = get_value_cols(prefix)
        if not value_cols:
            return np.array([], dtype=float), np.array([], dtype=float)
        time_col = get_time_col(prefix)
        temp = df[[time_col, value_cols[0]]].copy()
        temp[time_col] = pd.to_numeric(temp[time_col], errors="coerce")
        temp[value_cols[0]] = pd.to_numeric(temp[value_cols[0]], errors="coerce")
        temp = temp.dropna(subset=[time_col, value_cols[0]])
        return temp[time_col].to_numpy(dtype=float), temp[value_cols[0]].to_numpy(dtype=float)

    def series_by_second(time_s: np.ndarray, values: np.ndarray, reducer: str = "mean") -> pd.Series:
        if time_s.size == 0 or values.size == 0:
            return pd.Series(dtype=float)
        seconds = np.floor(time_s).astype(int)
        second_df = pd.DataFrame({"second": seconds, "value": values})
        grouped = second_df.groupby("second", sort=True)["value"]
        if reducer == "mean":
            series = grouped.mean()
        elif reducer == "max":
            series = grouped.max()
        elif reducer == "min":
            series = grouped.min()
        elif reducer == "rms":
            series = grouped.apply(lambda s: float(np.sqrt(np.mean(np.square(s.to_numpy())))))
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
        full_index = np.arange(int(series.index.min()), int(series.index.max()) + 1)
        return series.reindex(full_index)

    def rolling_baseline(series: pd.Series, window_s: int, reducer: str) -> pd.Series:
        if series.empty:
            return pd.Series(dtype=float)
        if reducer == "median":
            baseline = series.rolling(window_s, min_periods=max(5, window_s // 10)).median()
        elif reducer == "mean":
            baseline = series.rolling(window_s, min_periods=max(5, window_s // 10)).mean()
        else:
            raise ValueError(f"Unsupported baseline reducer: {reducer}")
        baseline = baseline.shift(1)
        return baseline.ffill().fillna(series.expanding().mean())

    def mask_segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        if mask.size == 0:
            return []
        padded = np.pad(mask.astype(int), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        segments = []
        for start, end in zip(starts, ends):
            if end - start >= min_len:
                segments.append((int(start), int(end)))
        return segments

    def mask_from_segments(length: int, segments: list[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(length, dtype=bool)
        for start, end in segments:
            mask[start:end] = True
        return mask

    def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        if mask.size == 0 or max_gap <= 0:
            return mask
        filled = mask.copy()
        false_segments = mask_segments(~mask, 1)
        for start, end in false_segments:
            if start == 0 or end == mask.size:
                continue
            if end - start <= max_gap and mask[start - 1] and mask[end]:
                filled[start:end] = True
        return filled

    def count_starts_per_window(times_s: np.ndarray) -> pd.Series:
        if times_s.size == 0:
            return pd.Series(dtype=float)
        bins = (np.floor(times_s / window_size) * window_size).astype(int)
        return pd.Series(1.0, index=bins).groupby(level=0).sum()

    def second_mask_to_window_features(
        mask_series: pd.Series,
        feature_prefix: str,
        start_times_s: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if mask_series.empty:
            return pd.DataFrame()
        temp = pd.DataFrame(
            {
                "window_start_s": (mask_series.index.to_numpy(dtype=int) // window_size) * window_size,
                "active": mask_series.to_numpy(dtype=float),
            }
        )
        grouped = temp.groupby("window_start_s", sort=True)["active"]
        feature_df = pd.DataFrame(index=grouped.mean().index)
        feature_df.index.name = "bin_start_s"
        feature_df[f"{feature_prefix}_binary"] = grouped.max()
        feature_df[f"{feature_prefix}_seconds"] = grouped.sum()
        feature_df[f"{feature_prefix}_fraction"] = grouped.mean()
        if start_times_s is not None:
            start_counts = count_starts_per_window(start_times_s)
            feature_df[f"{feature_prefix}_count"] = start_counts.reindex(feature_df.index, fill_value=0.0)
        return feature_df

    def build_ahe_proxy_support(
        flow_time_s: np.ndarray,
        flow_values: np.ndarray,
        saturation_series: pd.Series,
    ) -> tuple[pd.Series, np.ndarray]:
        if flow_time_s.size == 0 or flow_values.size == 0:
            return pd.Series(dtype=float), np.array([], dtype=float)
        flow_fs = max(1, int(round(estimate_sampling_frequency(flow_time_s))))
        centered_flow = flow_values - np.nanmedian(flow_values)
        flow_envelope = moving_average(np.abs(centered_flow), max(1, flow_fs))
        flow_series = series_by_second(flow_time_s, flow_envelope, reducer="mean")
        if flow_series.empty:
            return pd.Series(dtype=float), np.array([], dtype=float)

        baseline_flow = rolling_baseline(flow_series, window_s=120, reducer="median").clip(lower=1e-6)
        reduction_fraction = 1.0 - (flow_series / baseline_flow)
        apnea_segments = mask_segments((reduction_fraction >= 0.90).fillna(False).to_numpy(), min_len=10)
        apnea_mask = mask_from_segments(len(flow_series), apnea_segments)

        hypopnea_candidates = (reduction_fraction >= 0.30).fillna(False).to_numpy() & ~apnea_mask
        hypopnea_segments = mask_segments(hypopnea_candidates, min_len=10)

        qualifying_hypopneas = []
        if not saturation_series.empty:
            baseline_saturation = rolling_baseline(saturation_series, window_s=120, reducer="mean")
            desaturation_mask = ((baseline_saturation - saturation_series) >= 3.0).fillna(False).to_numpy()
            saturation_offset = int(saturation_series.index.min())
            saturation_len = len(saturation_series)
            for start, end in hypopnea_segments:
                start_second = int(flow_series.index[start])
                end_second = int(flow_series.index[end - 1])
                start_idx = max(0, start_second - saturation_offset)
                end_idx = min(saturation_len, end_second - saturation_offset + 31)
                if end_idx > start_idx and desaturation_mask[start_idx:end_idx].any():
                    qualifying_hypopneas.append((start, end))

        combined_segments = apnea_segments + qualifying_hypopneas
        combined_mask = mask_from_segments(len(flow_series), combined_segments)
        start_seconds = np.array([int(flow_series.index[start]) for start, _ in combined_segments], dtype=float)
        return pd.Series(combined_mask.astype(float), index=flow_series.index), start_seconds

    def detect_ahe_proxy(flow_time_s: np.ndarray, flow_values: np.ndarray, saturation_series: pd.Series) -> pd.DataFrame:
        ahe_mask_series, start_seconds = build_ahe_proxy_support(flow_time_s, flow_values, saturation_series)
        if ahe_mask_series.empty:
            return pd.DataFrame()
        return second_mask_to_window_features(
            ahe_mask_series,
            feature_prefix="ahe_proxy",
            start_times_s=start_seconds,
        )

    def build_odi3_support(
        saturation_time_s: np.ndarray,
        saturation_values: np.ndarray,
    ) -> tuple[pd.Series, np.ndarray, float]:
        if saturation_time_s.size == 0 or saturation_values.size == 0:
            return pd.Series(dtype=float), np.array([], dtype=float), 0.0
        saturation_series = series_by_second(saturation_time_s, saturation_values, reducer="mean")
        if saturation_series.empty:
            return pd.Series(dtype=float), np.array([], dtype=float), 0.0

        baseline_saturation = rolling_baseline(saturation_series, window_s=120, reducer="mean")
        desaturation_segments = mask_segments(
            ((baseline_saturation - saturation_series) >= 3.0).fillna(False).to_numpy(),
            min_len=10,
        )
        desaturation_mask = mask_from_segments(len(saturation_series), desaturation_segments)
        start_seconds = np.array([int(saturation_series.index[start]) for start, _ in desaturation_segments], dtype=float)
        total_hours = max(
            (float(saturation_series.index.max()) - float(saturation_series.index.min()) + 1.0) / 3600.0,
            1e-6,
        )
        events_per_hour = float(len(desaturation_segments) / total_hours)
        return pd.Series(desaturation_mask.astype(float), index=saturation_series.index), start_seconds, events_per_hour

    def detect_odi3_features(saturation_time_s: np.ndarray, saturation_values: np.ndarray) -> pd.DataFrame:
        desaturation_mask_series, start_seconds, events_per_hour = build_odi3_support(
            saturation_time_s,
            saturation_values,
        )
        if desaturation_mask_series.empty:
            return pd.DataFrame()
        feature_df = second_mask_to_window_features(
            desaturation_mask_series,
            feature_prefix="odi3_desaturation",
            start_times_s=start_seconds,
        )
        feature_df["odi3_events_per_hour"] = events_per_hour
        return feature_df

    def build_snoring_bout_support(
        snoring_time_s: np.ndarray,
        snoring_values: np.ndarray,
    ) -> tuple[pd.Series, np.ndarray]:
        if snoring_time_s.size == 0 or snoring_values.size == 0:
            return pd.Series(dtype=float), np.array([], dtype=float)
        snoring_fs = max(1, int(round(estimate_sampling_frequency(snoring_time_s))))
        snoring_energy = moving_average(np.square(snoring_values), max(1, int(round(snoring_fs * 0.2))))
        finite_energy = snoring_energy[np.isfinite(snoring_energy)]
        positive_energy = finite_energy[finite_energy > 0]
        if positive_energy.size == 0:
            return pd.Series(dtype=float), np.array([], dtype=float)

        baseline = float(np.median(finite_energy))
        mad = float(np.median(np.abs(finite_energy - baseline)))
        threshold = max(baseline + 3.0 * mad, float(np.quantile(positive_energy, 0.75)))
        active_mask = snoring_energy >= threshold
        active_mask = fill_short_false_gaps(active_mask, max_gap=max(1, int(round(2.0 * snoring_fs))))
        bout_segments = mask_segments(active_mask, min_len=max(1, int(round(1.0 * snoring_fs))))
        if not bout_segments:
            return pd.Series(dtype=float), np.array([], dtype=float)

        bout_mask = np.zeros_like(active_mask, dtype=bool)
        for start, end in bout_segments:
            bout_mask[start:end] = True
        second_mask = series_by_second(snoring_time_s, bout_mask.astype(float), reducer="max")
        start_seconds = np.array([float(snoring_time_s[start]) for start, _ in bout_segments], dtype=float)
        return second_mask.fillna(0.0), start_seconds

    def detect_snoring_bouts(snoring_time_s: np.ndarray, snoring_values: np.ndarray) -> pd.DataFrame:
        second_mask, start_seconds = build_snoring_bout_support(snoring_time_s, snoring_values)
        if second_mask.empty:
            return pd.DataFrame()
        return second_mask_to_window_features(second_mask.fillna(0.0), feature_prefix="snoring_bout", start_times_s=start_seconds)

    def build_ifl_proxy_support(
        flow_time_s: np.ndarray,
        flow_values: np.ndarray,
    ) -> tuple[pd.Series, np.ndarray, pd.Series]:
        if flow_time_s.size == 0 or flow_values.size == 0:
            return pd.Series(dtype=float), np.array([], dtype=float), pd.Series(dtype=float)
        flow_fs = max(1, int(round(estimate_sampling_frequency(flow_time_s))))
        smoothed_flow = moving_average(flow_values - np.nanmedian(flow_values), max(1, int(round(flow_fs * 0.1))))
        sign = np.sign(smoothed_flow)
        if sign.size == 0:
            return pd.Series(dtype=float), np.array([], dtype=float), pd.Series(dtype=float)

        nonzero_sign = sign.copy()
        for idx in range(1, nonzero_sign.size):
            if nonzero_sign[idx] == 0:
                nonzero_sign[idx] = nonzero_sign[idx - 1]
        if nonzero_sign[0] == 0:
            nonzero_sign[0] = 1

        zero_crossings = np.flatnonzero(nonzero_sign[1:] * nonzero_sign[:-1] < 0) + 1
        boundaries = np.concatenate(([0], zero_crossings, [len(smoothed_flow)]))
        candidate_flags = []
        candidate_starts = []
        total_breath_windows = []
        amplitude_floor = float(np.nanstd(np.sqrt(np.abs(smoothed_flow))))
        if amplitude_floor <= 0:
            amplitude_floor = 0.5

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment = smoothed_flow[start:end]
            if segment.size < 8:
                continue
            duration_s = float(flow_time_s[end - 1] - flow_time_s[start])
            if duration_s < 0.5 or duration_s > 10.0:
                continue
            transformed = np.sqrt(np.abs(segment))
            peak = float(np.max(transformed))
            if peak <= amplitude_floor:
                continue
            trim = max(1, int(round(segment.size * 0.25)))
            if segment.size <= 2 * trim + 1:
                continue
            mid_segment = transformed[trim:-trim]
            flattening_ratio = float(np.mean(mid_segment) / peak)
            candidate_flags.append(flattening_ratio >= 0.90)
            candidate_starts.append(float(flow_time_s[start]))
            total_breath_windows.append(int((flow_time_s[start] // window_size) * window_size))

        if not candidate_flags:
            return pd.Series(dtype=float), np.array([], dtype=float), pd.Series(dtype=float)

        candidate_flags = np.asarray(candidate_flags, dtype=bool)
        candidate_starts = np.asarray(candidate_starts, dtype=float)
        breath_fraction = pd.Series(candidate_flags.astype(float), index=total_breath_windows).groupby(level=0).mean()
        breath_fraction.index.name = "bin_start_s"
        limited_run_segments = []
        run_start = None
        for idx, flag in enumerate(candidate_flags):
            if flag and run_start is None:
                run_start = idx
            elif not flag and run_start is not None:
                if idx - run_start >= 4:
                    limited_run_segments.append((run_start, idx))
                run_start = None
        if run_start is not None and len(candidate_flags) - run_start >= 4:
            limited_run_segments.append((run_start, len(candidate_flags)))

        if not limited_run_segments:
            return pd.Series(dtype=float), np.array([], dtype=float), breath_fraction

        second_index = pd.Index(np.arange(int(np.floor(flow_time_s.min())), int(np.floor(flow_time_s.max())) + 1), dtype=int)
        second_mask = pd.Series(0.0, index=second_index)
        run_start_times = []
        for start, end in limited_run_segments:
            event_start = candidate_starts[start]
            event_end = candidate_starts[end - 1] + 1.0
            second_mask.loc[int(np.floor(event_start)) : int(np.floor(event_end))] = 1.0
            run_start_times.append(event_start)
        return second_mask, np.asarray(run_start_times, dtype=float), breath_fraction

    def detect_ifl_proxy(flow_time_s: np.ndarray, flow_values: np.ndarray) -> pd.DataFrame:
        second_mask, run_start_times, breath_fraction = build_ifl_proxy_support(flow_time_s, flow_values)
        if breath_fraction.empty:
            return pd.DataFrame()
        if second_mask.empty:
            feature_df = pd.DataFrame(index=breath_fraction.index)
            feature_df.index.name = "bin_start_s"
            feature_df["ifl_proxy_binary"] = 0.0
            feature_df["ifl_proxy_seconds"] = 0.0
            feature_df["ifl_proxy_fraction"] = 0.0
            feature_df["ifl_proxy_count"] = 0.0
            feature_df["ifl_proxy_breath_fraction"] = breath_fraction
            return feature_df

        feature_df = second_mask_to_window_features(
            second_mask,
            feature_prefix="ifl_proxy",
            start_times_s=run_start_times,
        )
        feature_df["ifl_proxy_breath_fraction"] = breath_fraction.reindex(feature_df.index, fill_value=0.0)
        return feature_df

    def detect_flow_limited_effort(
        flow_time_s: np.ndarray,
        flow_values: np.ndarray,
        effort_time_s: np.ndarray,
        effort_values: np.ndarray,
        pulse_time_s: np.ndarray,
        pulse_values: np.ndarray,
        ahe_mask_series: pd.Series,
        ifl_mask_series: pd.Series,
        odi_mask_series: pd.Series,
    ) -> pd.DataFrame:
        if (
            flow_time_s.size == 0
            or flow_values.size == 0
            or effort_time_s.size == 0
            or effort_values.size == 0
            or ifl_mask_series.empty
        ):
            return pd.DataFrame()

        flow_fs = max(1, int(round(estimate_sampling_frequency(flow_time_s))))
        effort_fs = max(1, int(round(estimate_sampling_frequency(effort_time_s))))
        centered_flow = flow_values - np.nanmedian(flow_values)
        flow_envelope = moving_average(np.abs(centered_flow), max(1, flow_fs))
        flow_series = series_by_second(flow_time_s, flow_envelope, reducer="mean")
        if flow_series.empty:
            return pd.DataFrame()
        flow_baseline = rolling_baseline(flow_series, window_s=120, reducer="median").clip(lower=1e-6)
        flow_recovery_ratio = flow_series / flow_baseline

        centered_effort = effort_values - np.nanmedian(effort_values)
        effort_energy = moving_average(np.abs(centered_effort), max(1, effort_fs))
        effort_series = series_by_second(effort_time_s, effort_energy, reducer="mean")
        if effort_series.empty:
            return pd.DataFrame()
        effort_baseline = rolling_baseline(effort_series, window_s=120, reducer="median").clip(lower=1e-6)
        effort_ratio = effort_series / effort_baseline
        effort_slope = effort_series.diff().rolling(5, min_periods=3).mean()

        common_index = flow_series.index.union(effort_series.index).union(ifl_mask_series.index)
        aligned = pd.DataFrame(index=common_index)
        aligned["flow_recovery_ratio"] = flow_recovery_ratio.reindex(common_index)
        aligned["effort_ratio"] = effort_ratio.reindex(common_index)
        aligned["effort_slope"] = effort_slope.reindex(common_index)
        aligned["ifl"] = ifl_mask_series.reindex(common_index, fill_value=0.0)
        aligned["ahe"] = ahe_mask_series.reindex(common_index, fill_value=0.0)
        aligned["odi"] = odi_mask_series.reindex(common_index, fill_value=0.0)

        if pulse_time_s.size > 0 and pulse_values.size > 0:
            pulse_series = series_by_second(pulse_time_s, pulse_values, reducer="mean")
            pulse_baseline = rolling_baseline(pulse_series, window_s=30, reducer="mean")
            aligned["pulse"] = pulse_series.reindex(common_index)
            aligned["pulse_baseline"] = pulse_baseline.reindex(common_index)
        else:
            aligned["pulse"] = np.nan
            aligned["pulse_baseline"] = np.nan

        candidate_mask = (
            (aligned["ifl"] > 0.0)
            & (aligned["effort_ratio"] >= 1.1)
            & (aligned["effort_slope"] > 0.0)
            & (aligned["ahe"] == 0.0)
            & (aligned["odi"] == 0.0)
        ).fillna(False).to_numpy()
        candidate_segments = mask_segments(candidate_mask, min_len=10)
        if not candidate_segments:
            return pd.DataFrame()

        qualifying_segments = []
        for start, end in candidate_segments:
            lookahead_end = min(len(aligned), end + 15)
            has_flow_recovery = bool((aligned["flow_recovery_ratio"].iloc[end:lookahead_end] >= 0.85).fillna(False).any())
            pulse_before = aligned["pulse_baseline"].iloc[max(0, end - 10):end].mean()
            pulse_after = aligned["pulse"].iloc[end:lookahead_end].max()
            has_pulse_rise = bool(
                np.isfinite(pulse_before)
                and np.isfinite(pulse_after)
                and (pulse_after - pulse_before >= 5.0)
            )
            if has_flow_recovery or has_pulse_rise:
                qualifying_segments.append((start, end))

        if not qualifying_segments:
            return pd.DataFrame()

        flow_limited_effort_mask = mask_from_segments(len(aligned), qualifying_segments)
        start_seconds = np.array([int(aligned.index[start]) for start, _ in qualifying_segments], dtype=float)
        return second_mask_to_window_features(
            pd.Series(flow_limited_effort_mask.astype(float), index=aligned.index),
            feature_prefix="flow_limited_effort",
            start_times_s=start_seconds,
        )

    def detect_pulse_activation(
        pulse_time_s: np.ndarray,
        pulse_values: np.ndarray,
    ) -> pd.DataFrame:
        if pulse_time_s.size == 0 or pulse_values.size == 0:
            return pd.DataFrame()

        pulse_series = series_by_second(pulse_time_s, pulse_values, reducer="mean")
        if pulse_series.empty:
            return pd.DataFrame()

        pulse_baseline = rolling_baseline(pulse_series, window_s=30, reducer="mean")
        pulse_delta = pulse_series - pulse_baseline
        pulse_slope = pulse_series.diff().rolling(3, min_periods=2).mean()

        activation_mask = (
            (pulse_delta >= 5.0)
            & (pulse_slope >= 0.0)
        ).fillna(False).to_numpy()
        activation_mask = fill_short_false_gaps(activation_mask, max_gap=2)
        activation_segments = mask_segments(activation_mask, min_len=3)
        if not activation_segments:
            return pd.DataFrame()

        second_mask = mask_from_segments(len(pulse_series), activation_segments)
        start_seconds = np.array(
            [int(pulse_series.index[start]) for start, _ in activation_segments],
            dtype=float,
        )
        return second_mask_to_window_features(
            pd.Series(second_mask.astype(float), index=pulse_series.index),
            feature_prefix="pulse_activation",
            start_times_s=start_seconds,
        )

    def apply_valid_range(value_cols: list, valid_min=None, valid_max=None) -> int:
        """Replace values outside a valid physiologic range with NaN."""
        replaced_count = 0
        for col in value_cols:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            invalid_mask = pd.Series(False, index=df.index)
            if valid_min is not None:
                invalid_mask = invalid_mask | (numeric_col < valid_min)
            if valid_max is not None:
                invalid_mask = invalid_mask | (numeric_col > valid_max)
            replaced_count += int(invalid_mask.sum())
            df[col] = numeric_col.mask(invalid_mask)
        return replaced_count

    flow_cols = get_value_cols("flow")
    snoring_cols = get_value_cols("snoring")
    effort_cols = get_value_cols("effort")
    pulse_cols = get_value_cols("pulse")
    saturation_cols = get_value_cols("saturation")

    # Clean obviously implausible pulse and saturation values before windowed
    # feature extraction so extreme artifacts do not dominate the downstream
    # summary statistics and SDCM node values.
    pulse_filtered_count = apply_valid_range(
        pulse_cols,
        valid_min=pulse_valid_min,
        valid_max=pulse_valid_max,
    )
    saturation_filtered_count = apply_valid_range(
        saturation_cols,
        valid_min=saturation_valid_min,
        valid_max=saturation_valid_max,
    )

    flow_time_s, flow_values = get_time_and_values("flow")
    snoring_time_s, snoring_values = get_time_and_values("snoring")
    effort_time_s, effort_values = get_time_and_values("effort")
    pulse_time_s, pulse_values = get_time_and_values("pulse")
    saturation_time_s, saturation_values = get_time_and_values("saturation")
    saturation_series = series_by_second(saturation_time_s, saturation_values, reducer="mean")

    flow_time_col = get_time_col("flow")
    flow_group = bin_by_seconds(flow_cols, flow_time_col)
    window_index = flow_group.size().index
    window_index.name = "time_s"

    ahe_mask_series, _ = build_ahe_proxy_support(flow_time_s, flow_values, saturation_series)
    ifl_mask_series, _, _ = build_ifl_proxy_support(flow_time_s, flow_values)
    odi_mask_series, _, _ = build_odi3_support(saturation_time_s, saturation_values)

    df_proxy_features = pd.concat(
        [
            detect_ahe_proxy(flow_time_s, flow_values, saturation_series),
            detect_ifl_proxy(flow_time_s, flow_values),
            detect_flow_limited_effort(
                flow_time_s,
                flow_values,
                effort_time_s,
                effort_values,
                pulse_time_s,
                pulse_values,
                ahe_mask_series,
                ifl_mask_series,
                odi_mask_series,
            ),
            detect_pulse_activation(pulse_time_s, pulse_values),
            detect_odi3_features(saturation_time_s, saturation_values),
            detect_snoring_bouts(snoring_time_s, snoring_values),
        ],
        axis=1,
        join="outer",
    ).sort_index()
    if not df_proxy_features.empty:
        df_proxy_features = df_proxy_features.loc[:, ~df_proxy_features.columns.duplicated()]

    df_final = pd.DataFrame(index=window_index)
    df_final["ahe_proxy_fraction"] = df_proxy_features.get("ahe_proxy_fraction", pd.Series(dtype=float)).reindex(window_index, fill_value=0.0).fillna(0.0)
    df_final["ifl_proxy_breath_fraction"] = df_proxy_features.get("ifl_proxy_breath_fraction", pd.Series(dtype=float)).reindex(window_index, fill_value=0.0).fillna(0.0)
    df_final["flow_limited_effort_fraction"] = df_proxy_features.get("flow_limited_effort_fraction", pd.Series(dtype=float)).reindex(window_index, fill_value=0.0).fillna(0.0)
    df_final["pulse_activation_fraction"] = df_proxy_features.get("pulse_activation_fraction", pd.Series(dtype=float)).reindex(window_index, fill_value=0.0).fillna(0.0)
    df_final["odi3_desaturation_fraction"] = df_proxy_features.get("odi3_desaturation_fraction", pd.Series(dtype=float)).reindex(window_index, fill_value=0.0).fillna(0.0)
    df_final["snoring_bout_fraction"] = df_proxy_features.get("snoring_bout_fraction", pd.Series(dtype=float)).reindex(window_index, fill_value=0.0).fillna(0.0)
    df_windowed = df_final.reset_index()
    df_windowed.attrs["pulse_filtered_count"] = pulse_filtered_count
    df_windowed.attrs["saturation_filtered_count"] = saturation_filtered_count
    return df_windowed


if __name__ == "__main__":
    patient_data_dir = args.patient_data_dir
    output_path = args.output_path
    filtering_threshold = args.filtering_threshold
    features_required = [s.strip() for s in args.features_required.split(",")]
    windowed_output_dir = args.windowed_output_dir
    window_size = args.window_size
    pulse_valid_min = args.pulse_valid_min
    pulse_valid_max = args.pulse_valid_max
    saturation_valid_min = args.saturation_valid_min
    saturation_valid_max = args.saturation_valid_max

    print(
        "Value filtering configuration: "
        f"pulse=[{pulse_valid_min}, {pulse_valid_max}], "
        f"saturation=[{saturation_valid_min}, {saturation_valid_max}]"
    )

    patient_info_dict = get_patient_info(patient_data_dir, filtering_threshold, features_required)
    patient_info_df = pd.DataFrame.from_dict(patient_info_dict, orient="index").reset_index().rename(columns={"index": "patient_id"})
    patient_info_df.to_csv(output_path, index=False)
    print(f"Combined patient data saved to {output_path}")

    os.makedirs(windowed_output_dir, exist_ok=True)
    valid_patients = patient_info_df[patient_info_df["valid"]]
    pbar = tqdm(total=len(valid_patients), desc="Windowing valid patients")
    for _, row in valid_patients.iterrows():
        patient_id = row["patient_id"]
        patient_file = os.path.join(patient_data_dir, f"anonymized_{patient_id}.csv")
        pbar.desc = f"Windowing patient {patient_id}"
        pbar.update(1)
        if not os.path.exists(patient_file):
            print(f"Missing patient file: {patient_file}")
            continue
        header_df = pd.read_csv(patient_file, nrows=0)
        header_cols = list(header_df.columns)
        header_cols_lower = [col.lower() for col in header_cols]
        required_column_names = {
            column_name
            for prefix in SIGNAL_PREFIXES
            for column_name in (prefix, f"{prefix}_time_s")
        }
        needed_cols = [
            col
            for col, col_lower in zip(header_cols, header_cols_lower)
            if col_lower == "time_s" or col_lower in required_column_names
        ]
        patient_df = pd.read_csv(patient_file, usecols=needed_cols)
        patient_df.columns = [col.lower() for col in patient_df.columns]
        windowed_df = create_windowed_features(
            patient_df,
            window_size=window_size,
            pulse_valid_min=pulse_valid_min,
            pulse_valid_max=pulse_valid_max,
            saturation_valid_min=saturation_valid_min,
            saturation_valid_max=saturation_valid_max,
        )
        pulse_filtered_count = int(windowed_df.attrs.get("pulse_filtered_count", 0))
        saturation_filtered_count = int(windowed_df.attrs.get("saturation_filtered_count", 0))
        if pulse_filtered_count or saturation_filtered_count:
            print(
                f"Patient {patient_id}: filtered {pulse_filtered_count} pulse values and "
                f"{saturation_filtered_count} saturation values outside the valid ranges."
            )
        windowed_path = os.path.join(windowed_output_dir, f"{patient_id}_windowed.csv")
        windowed_df.to_csv(windowed_path, index=False)
