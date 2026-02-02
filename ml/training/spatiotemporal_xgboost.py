"""
Spatio-temporal risk modeling with KDE target engineering and XGBoost.
Steps:
1) Load JSONL data and filter to Kucukcekmece bbox.
2) Temporal binning (4H/6H).
3) Spatial gridding (H3 or square grid).
4) KDE-based target risk score per grid/time.
5) Feature engineering (time, lags, neighbor lags, region clusters).
6) Train/test split by time and RMSE evaluation.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import box


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if BACKEND_ROOT.exists():
    sys.path.insert(0, str(BACKEND_ROOT))


try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import h3
except Exception:
    h3 = None


def get_kucukcekmece_bbox() -> Tuple[float, float, float, float]:
    try:
        from app.core.config import get_settings

        return get_settings().kucukcekmece_fallback_bbox
    except Exception:
        return (40.98, 28.70, 41.05, 28.80)


LAKE_BOUNDS = {
    "min_lat": 41.0,
    "max_lat": 41.02,
    "min_lng": 28.75,
    "max_lng": 28.78,
}


def is_in_lake(lat: float, lng: float) -> bool:
    return (
        LAKE_BOUNDS["min_lat"] <= lat <= LAKE_BOUNDS["max_lat"]
        and LAKE_BOUNDS["min_lng"] <= lng <= LAKE_BOUNDS["max_lng"]
    )


def haversine_matrix(
    grid_coords: np.ndarray,
    event_coords: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(grid_coords[:, 0])[:, None]
    lon1 = np.radians(grid_coords[:, 1])[:, None]
    lat2 = np.radians(event_coords[:, 0])[None, :]
    lon2 = np.radians(event_coords[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    earth_radius_m = 6371000.0
    return earth_radius_m * c


def gaussian_kernel(distances_m: np.ndarray, bandwidth_m: float) -> np.ndarray:
    if bandwidth_m <= 0:
        return np.zeros_like(distances_m)
    return np.exp(-0.5 * (distances_m / bandwidth_m) ** 2)


def load_jsonl(jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)
    df = df.rename(
        columns={
            "timestamp": "event_time",
            "latitude": "lat",
            "longitude": "lng",
        }
    )
    required = {"event_time", "lat", "lng"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["severity"] = pd.to_numeric(df.get("severity", 1.0), errors="coerce").fillna(1.0)
    df = df.dropna(subset=["event_time", "lat", "lng"])
    return df


def filter_to_bbox(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    min_lat, min_lng, max_lat, max_lng = bbox
    df = df[
        df["lat"].between(min_lat, max_lat)
        & df["lng"].between(min_lng, max_lng)
    ].copy()

    if gpd is not None and not df.empty:
        bbox_poly = box(min_lng, min_lat, max_lng, max_lat)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lng"], df["lat"]),
            crs="EPSG:4326",
        )
        gdf = gdf[gdf.within(bbox_poly)]
        df = pd.DataFrame(gdf.drop(columns="geometry"))

    df = df[~df.apply(lambda r: is_in_lake(r["lat"], r["lng"]), axis=1)]
    return df


@dataclass
class GridSpec:
    grid_ids: List[str]
    centers: Dict[str, Tuple[float, float]]
    neighbors: Dict[str, List[str]]


def build_h3_grid(bbox: Tuple[float, float, float, float], resolution: int) -> GridSpec:
    min_lat, min_lng, max_lat, max_lng = bbox
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [min_lng, min_lat],
            [max_lng, min_lat],
            [max_lng, max_lat],
            [min_lng, max_lat],
            [min_lng, min_lat],
        ]],
    }
    grid_ids = list(h3.polyfill(polygon, resolution, geo_json_conformant=True))
    centers = {gid: h3.h3_to_geo(gid) for gid in grid_ids}
    neighbors = {gid: list(h3.k_ring(gid, 1) - {gid}) for gid in grid_ids}
    return GridSpec(grid_ids=grid_ids, centers=centers, neighbors=neighbors)


def build_square_grid(
    bbox: Tuple[float, float, float, float],
    grid_size_m: float,
) -> GridSpec:
    min_lat, min_lng, max_lat, max_lng = bbox
    lat_step = grid_size_m / 111000.0
    avg_lat = (min_lat + max_lat) / 2.0
    lng_step = grid_size_m / (111000.0 * math.cos(math.radians(avg_lat)))

    grid_ids = []
    centers: Dict[str, Tuple[float, float]] = {}
    neighbors: Dict[str, List[str]] = {}
    rows = int(math.ceil((max_lat - min_lat) / lat_step))
    cols = int(math.ceil((max_lng - min_lng) / lng_step))

    for i in range(rows):
        for j in range(cols):
            center_lat = min_lat + (i + 0.5) * lat_step
            center_lng = min_lng + (j + 0.5) * lng_step
            if is_in_lake(center_lat, center_lng):
                continue
            grid_id = f"sq_{i}_{j}"
            grid_ids.append(grid_id)
            centers[grid_id] = (center_lat, center_lng)

    for grid_id in grid_ids:
        parts = grid_id.split("_")
        row = int(parts[1])
        col = int(parts[2])
        neighbor_ids = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                candidate = f"sq_{row + di}_{col + dj}"
                if candidate in centers:
                    neighbor_ids.append(candidate)
        neighbors[grid_id] = neighbor_ids

    return GridSpec(grid_ids=grid_ids, centers=centers, neighbors=neighbors)


def assign_grid_ids(
    df: pd.DataFrame,
    grid: GridSpec,
    bbox: Tuple[float, float, float, float],
    grid_size_m: float,
    use_h3: bool,
    h3_resolution: int,
) -> pd.DataFrame:
    if use_h3:
        df["grid_id"] = df.apply(
            lambda r: h3.geo_to_h3(r["lat"], r["lng"], h3_resolution), axis=1
        )
        return df

    min_lat, min_lng, max_lat, max_lng = bbox
    lat_step = grid_size_m / 111000.0
    avg_lat = (min_lat + max_lat) / 2.0
    lng_step = grid_size_m / (111000.0 * math.cos(math.radians(avg_lat)))

    def square_id(lat: float, lng: float) -> Optional[str]:
        if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
            return None
        row = int((lat - min_lat) / lat_step)
        col = int((lng - min_lng) / lng_step)
        grid_id = f"sq_{row}_{col}"
        return grid_id if grid_id in grid.centers else None

    df["grid_id"] = df.apply(lambda r: square_id(r["lat"], r["lng"]), axis=1)
    df = df.dropna(subset=["grid_id"]).copy()
    return df


def build_kde_targets(
    df: pd.DataFrame,
    grid: GridSpec,
    time_bin_hours: int,
    bandwidth_m: float,
) -> pd.DataFrame:
    grid_ids = grid.grid_ids
    grid_coords = np.array([grid.centers[gid] for gid in grid_ids])

    records = []
    for time_bin, group in df.groupby("time_bin"):
        event_coords = group[["lat", "lng"]].to_numpy()
        event_weights = (group["severity"] / 5.0).to_numpy()

        if len(event_coords) == 0:
            density = np.zeros(len(grid_ids))
        else:
            distances = haversine_matrix(grid_coords, event_coords)
            kernel = gaussian_kernel(distances, bandwidth_m)
            density = (kernel * event_weights).sum(axis=1)

        max_val = density.max() if density.size else 0.0
        if max_val > 0:
            density = density / max_val
        else:
            density = np.zeros_like(density)

        records.extend(
            {
                "grid_id": gid,
                "time_bin": time_bin,
                "risk_score": float(score),
            }
            for gid, score in zip(grid_ids, density)
        )

    return pd.DataFrame.from_records(records)


def build_full_grid_times(
    grid_ids: List[str],
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
    time_bin_hours: int,
) -> pd.DataFrame:
    time_bins = pd.date_range(
        start=min_time,
        end=max_time,
        freq=f"{time_bin_hours}H",
        tz=min_time.tz,
    )
    idx = pd.MultiIndex.from_product(
        [grid_ids, time_bins], names=["grid_id", "time_bin"]
    )
    return pd.DataFrame(index=idx).reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time_bin"].dt.hour
    df["day_of_week"] = df["time_bin"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df


def add_lag_features(df: pd.DataFrame, time_bin_hours: int) -> pd.DataFrame:
    df = df.sort_values(["grid_id", "time_bin"]).copy()
    periods_per_day = int(24 / time_bin_hours)
    df["lag_1d"] = df.groupby("grid_id")["risk_score"].shift(periods_per_day)
    df["lag_1w"] = df.groupby("grid_id")["risk_score"].shift(periods_per_day * 7)
    df["risk_lag1"] = df.groupby("grid_id")["risk_score"].shift(1)
    return df


def add_neighbor_lag(
    df: pd.DataFrame,
    neighbors: Dict[str, List[str]],
) -> pd.DataFrame:
    df = df.copy()
    lag_series = df.set_index(["grid_id", "time_bin"])["risk_lag1"]
    lag_lookup = lag_series.to_dict()
    neighbor_means = []
    for row in df.itertuples(index=False):
        neighbor_ids = neighbors.get(row.grid_id, [])
        values = [
            lag_lookup.get((neighbor_id, row.time_bin))
            for neighbor_id in neighbor_ids
        ]
        values = [v for v in values if v is not None and not np.isnan(v)]
        neighbor_means.append(float(np.mean(values)) if values else np.nan)
    df["neighbor_mean_lag1"] = neighbor_means
    return df


def add_region_clusters(
    df: pd.DataFrame,
    grid: GridSpec,
    region_clusters: int,
    random_state: int,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    centers = np.array([grid.centers[gid] for gid in grid.grid_ids])
    cluster_count = max(1, min(region_clusters, len(centers)))
    kmeans = KMeans(n_clusters=cluster_count, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(centers)
    region_map = {gid: int(label) for gid, label in zip(grid.grid_ids, labels)}

    df = df.copy()
    df["district_region"] = df["grid_id"].map(region_map)
    return df


def get_model():
    try:
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        return model, "xgboost"
    except Exception:
        try:
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.08,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            return model, "lightgbm"
        except Exception as exc:
            raise RuntimeError(
                "Missing xgboost or lightgbm. Install one to train the model."
            ) from exc


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_split_ratio: float,
) -> Tuple[float, object, List[str]]:
    from sklearn.metrics import mean_squared_error

    unique_times = sorted(df["time_bin"].unique())
    split_idx = int(len(unique_times) * time_split_ratio)
    cutoff_time = unique_times[split_idx - 1] if split_idx > 0 else unique_times[-1]

    train_mask = df["time_bin"] <= cutoff_time
    X = df[feature_cols]
    y = df["risk_score"]

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]

    model, model_name = get_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    print(f"Model: {model_name} | Test RMSE: {rmse:.4f}")
    return rmse, model, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spatio-temporal KDE XGBoost model")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "docs/dataset/crime_fixed.jsonl",
        help="Path to JSONL crime data",
    )
    parser.add_argument("--time-bin-hours", type=int, default=6)
    parser.add_argument("--h3-resolution", type=int, default=8)
    parser.add_argument("--grid-size-m", type=float, default=500.0)
    parser.add_argument("--bandwidth-m", type=float, default=450.0)
    parser.add_argument("--region-clusters", type=int, default=4)
    parser.add_argument("--no-h3", action="store_true")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=REPO_ROOT / "ml/models/spatiotemporal_model.pkl",
    )
    args = parser.parse_args()

    # Step 1: Load and filter data
    df = load_jsonl(args.input)
    bbox = get_kucukcekmece_bbox()
    df = filter_to_bbox(df, bbox)
    if df.empty:
        raise ValueError("No data left after filtering. Check bbox or input data.")

    # Step 2: Temporal binning
    df["time_bin"] = df["event_time"].dt.floor(f"{args.time_bin_hours}H")

    # Step 3: Grid creation
    use_h3 = (not args.no_h3) and (h3 is not None)
    if use_h3:
        grid = build_h3_grid(bbox, args.h3_resolution)
    else:
        if args.no_h3:
            print("H3 disabled, using square grid.")
        else:
            print("H3 not available, using square grid.")
        grid = build_square_grid(bbox, args.grid_size_m)

    df = assign_grid_ids(
        df, grid, bbox, args.grid_size_m, use_h3, args.h3_resolution
    )
    df = df.dropna(subset=["grid_id"])

    # Step 4: KDE target engineering
    kde_df = build_kde_targets(df, grid, args.time_bin_hours, args.bandwidth_m)

    # Step 5: Build full grid x time matrix
    full_df = build_full_grid_times(
        grid.grid_ids, kde_df["time_bin"].min(), kde_df["time_bin"].max(), args.time_bin_hours
    )
    full_df = full_df.merge(kde_df, on=["grid_id", "time_bin"], how="left")
    full_df["risk_score"] = full_df["risk_score"].fillna(0.0)
    full_df["center_lat"] = full_df["grid_id"].map(lambda gid: grid.centers[gid][0])
    full_df["center_lng"] = full_df["grid_id"].map(lambda gid: grid.centers[gid][1])

    # Step 6: feature 
    full_df = add_time_features(full_df)
    full_df = add_region_clusters(
        full_df, grid, args.region_clusters, random_state=42
    )
    full_df = add_lag_features(full_df, args.time_bin_hours)
    full_df = add_neighbor_lag(full_df, grid.neighbors)

    feature_cols = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "lag_1d",
        "lag_1w",
        "neighbor_mean_lag1",
        "district_region",
    ]

    feature_df = pd.get_dummies(
        full_df[feature_cols], columns=["district_region"], prefix="region"
    )
    full_df = full_df.join(feature_df)

    valid_mask = ~full_df[["lag_1d", "lag_1w", "neighbor_mean_lag1"]].isna().any(axis=1)
    train_df = full_df[valid_mask].copy()

    model_features = feature_df.columns.tolist()
    rmse, model, used_features = train_and_evaluate(
        train_df, model_features, time_split_ratio=0.8
    )

    # Step 7: Persist model and Feature list
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "time_bin_hours": args.time_bin_hours,
        "bandwidth_m": args.bandwidth_m,
        "region_clusters": args.region_clusters,
        "grid_size_m": args.grid_size_m,
        "use_hex": use_h3,
    }

    try:
        import joblib

        joblib.dump(
            {"model": model, "features": used_features, "metadata": metadata},
            args.model_out,
        )
    except Exception:
        import pickle

        with open(args.model_out, "wb") as f:
            pickle.dump(
                {"model": model, "features": used_features, "metadata": metadata}, f
            )

    print(f"Saved model to: {args.model_out}")
    print(f"Final RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()