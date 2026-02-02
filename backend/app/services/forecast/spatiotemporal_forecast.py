"""Spatio-temporal ML forecast using KDE-based features and a trained model."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from psycopg2.extras import DateTimeRange
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.risk_cell import RiskCell
from app.services.forecast.risk_cells import create_hex_grid, create_square_grid


settings = get_settings()
logger = logging.getLogger(__name__)


def get_geometry_center(geom: dict) -> Optional[Tuple[float, float]]:
    if not geom or not geom.get("coordinates"):
        return None
    try:
        if geom.get("type") == "Point":
            return (geom["coordinates"][1], geom["coordinates"][0])
        if geom.get("type") == "Polygon" and geom.get("coordinates", [])[0]:
            coords = geom["coordinates"][0]
            sum_lat = sum(c[1] for c in coords)
            sum_lng = sum(c[0] for c in coords)
            return (sum_lat / len(coords), sum_lng / len(coords))
    except Exception:
        return None
    return None


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


@dataclass
class GridIndex:
    grid_ids: List[str]
    centers: Dict[str, Tuple[float, float]]
    neighbors: Dict[str, List[str]]
    geom_map: Dict[str, dict]


def _coerce_feature_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return [str(item) for item in value.tolist()]
        except Exception:
            return None
    return None


def _get_model_feature_names(model: object) -> Optional[List[str]]:
    for attr in ("feature_names_in_", "feature_name_", "feature_names"):
        if hasattr(model, attr):
            names = getattr(model, attr)
            features = _coerce_feature_list(names)
            if features:
                return features
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            features = _coerce_feature_list(getattr(booster, "feature_names", None))
            if features:
                return features
        except Exception:
            return None
    return None


def _extract_features(payload: Any) -> Optional[List[str]]:
    if isinstance(payload, dict):
        for key in (
            "features",
            "feature_names",
            "feature_columns",
            "feature_cols",
            "model_features",
            "columns",
        ):
            features = _coerce_feature_list(payload.get(key))
            if features:
                return features
    return None


def _extract_metadata(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("metadata", "meta", "config", "settings"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def load_model_artifact(model_path: Path) -> Tuple[object, List[str], Dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        import joblib

        payload = joblib.load(model_path)
    except Exception:
        import pickle

        with open(model_path, "rb") as f:
            payload = pickle.load(f)

    model = None
    features = None
    metadata: Dict[str, Any] = {}

    if isinstance(payload, dict):
        model = payload.get("model") or payload.get("estimator") or payload.get("pipeline")
        features = _extract_features(payload)
        metadata = _extract_metadata(payload)
        for key in (
            "time_bin_hours",
            "bandwidth_m",
            "region_clusters",
            "neighbor_k",
            "grid_size_m",
            "use_hex",
            "min_risk_threshold",
        ):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
    elif isinstance(payload, (list, tuple)):
        if payload:
            model = payload[0]
        if len(payload) > 1:
            features = _coerce_feature_list(payload[1])
        if len(payload) > 2 and isinstance(payload[2], dict):
            metadata = dict(payload[2])
    else:
        model = payload

    if model is None:
        raise ValueError("Model artifact missing a model object.")
    if not features:
        features = _get_model_feature_names(model)
    if not features:
        raise ValueError("Model artifact missing feature names.")

    return model, list(features), metadata


def build_grid_index(
    db: Session,
    bbox: Tuple[float, float, float, float],
    grid_size_m: float,
    use_hex: bool,
    neighbor_k: int,
) -> GridIndex:
    if use_hex:
        grid_cells = create_hex_grid(db, bbox, grid_size_m)
    else:
        grid_cells = create_square_grid(db, bbox, grid_size_m)

    centers: Dict[str, Tuple[float, float]] = {}
    geom_map: Dict[str, dict] = {}
    grid_ids: List[str] = []

    min_lat, min_lng, max_lat, max_lng = bbox
    lat_step = grid_size_m / 111000.0
    avg_lat = (min_lat + max_lat) / 2.0
    lng_step = grid_size_m / (111000.0 * math.cos(math.radians(avg_lat)))

    for idx, cell in enumerate(grid_cells):
        geom = cell.get("geom")
        center_lat = cell.get("center_lat")
        center_lng = cell.get("center_lng")
        if center_lat is None or center_lng is None:
            center = get_geometry_center(geom)
            if center:
                center_lat, center_lng = center
        if center_lat is None or center_lng is None:
            continue

        if use_hex:
            grid_id = f"hex_{idx}"
        else:
            row = int((center_lat - min_lat) / lat_step)
            col = int((center_lng - min_lng) / lng_step)
            grid_id = f"sq_{row}_{col}"

        centers[grid_id] = (center_lat, center_lng)
        geom_map[grid_id] = geom
        grid_ids.append(grid_id)

    neighbors: Dict[str, List[str]] = {}
    if not grid_ids:
        return GridIndex(grid_ids=[], centers={}, neighbors={}, geom_map={})

    if use_hex:
        coords = np.array([centers[g] for g in grid_ids])
        coords_rad = np.radians(coords)
        try:
            from sklearn.neighbors import BallTree

            tree = BallTree(coords_rad, metric="haversine")
            _, indices = tree.query(coords_rad, k=min(neighbor_k + 1, len(grid_ids)))
        except Exception:
            indices = np.argsort(
                np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2), axis=1
            )[:, : min(neighbor_k + 1, len(grid_ids))]

        for idx, row in enumerate(indices):
            neighbor_ids = []
            for neighbor_idx in row:
                if neighbor_idx == idx:
                    continue
                neighbor_ids.append(grid_ids[neighbor_idx])
            neighbors[grid_ids[idx]] = neighbor_ids
    else:
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

    return GridIndex(
        grid_ids=grid_ids,
        centers=centers,
        neighbors=neighbors,
        geom_map=geom_map,
    )


def load_historical_events(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    crime_type: Optional[str] = None,
) -> pd.DataFrame:
    query = """
        SELECT
            event_time,
            severity,
            ST_Y(geom::geometry) as lat,
            ST_X(geom::geometry) as lng
        FROM crime_event
        WHERE event_time >= :start_time
          AND event_time <= :end_time
          AND is_within_kucukcekmece(geom)
    """
    params = {"start_time": start_time, "end_time": end_time}
    if crime_type:
        query += " AND crime_type = :crime_type"
        params["crime_type"] = crime_type

    results = db.execute(text(query), params).fetchall()

    records = [
        {
            "event_time": row.event_time,
            "severity": float(row.severity) if row.severity is not None else 1.0,
            "lat": float(row.lat),
            "lng": float(row.lng),
        }
        for row in results
        if row.lat is not None and row.lng is not None
    ]
    return pd.DataFrame.from_records(records)


def assign_grid_ids_for_events(
    events: pd.DataFrame,
    grid_index: GridIndex,
    bbox: Tuple[float, float, float, float],
    grid_size_m: float,
    use_hex: bool,
) -> pd.DataFrame:
    if events.empty:
        return events

    if use_hex:
        centers = np.array([grid_index.centers[g] for g in grid_index.grid_ids])
        centers_rad = np.radians(centers)
        event_coords = events[["lat", "lng"]].to_numpy()
        event_rad = np.radians(event_coords)
        try:
            from sklearn.neighbors import BallTree

            tree = BallTree(centers_rad, metric="haversine")
            _, idxs = tree.query(event_rad, k=1)
        except Exception:
            distances = np.linalg.norm(
                event_coords[:, None, :] - centers[None, :, :], axis=2
            )
            idxs = np.argmin(distances, axis=1).reshape(-1, 1)

        grid_ids = [grid_index.grid_ids[idx] for idx in idxs.flatten()]
        events = events.copy()
        events["grid_id"] = grid_ids
        return events

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
        return grid_id if grid_id in grid_index.centers else None

    events = events.copy()
    events["grid_id"] = events.apply(lambda r: square_id(r["lat"], r["lng"]), axis=1)
    events = events.dropna(subset=["grid_id"])
    return events


def assign_grid_id_for_point(
    lat: float,
    lng: float,
    grid_index: GridIndex,
    bbox: Tuple[float, float, float, float],
    grid_size_m: float,
    use_hex: bool,
) -> Optional[str]:
    if not grid_index.grid_ids:
        return None

    if use_hex:
        centers = np.array([grid_index.centers[g] for g in grid_index.grid_ids])
        centers_rad = np.radians(centers)
        point_rad = np.radians(np.array([[lat, lng]]))
        try:
            from sklearn.neighbors import BallTree

            tree = BallTree(centers_rad, metric="haversine")
            _, idxs = tree.query(point_rad, k=1)
            idx = int(idxs.flatten()[0])
        except Exception:
            distances = np.linalg.norm(centers - np.array([lat, lng]), axis=1)
            idx = int(np.argmin(distances))
        return grid_index.grid_ids[idx]

    min_lat, min_lng, max_lat, max_lng = bbox
    if not (min_lat <= lat <= max_lat and min_lng <= lng <= max_lng):
        return None
    lat_step = grid_size_m / 111000.0
    avg_lat = (min_lat + max_lat) / 2.0
    lng_step = grid_size_m / (111000.0 * math.cos(math.radians(avg_lat)))
    row = int((lat - min_lat) / lat_step)
    col = int((lng - min_lng) / lng_step)
    grid_id = f"sq_{row}_{col}"
    return grid_id if grid_id in grid_index.centers else None


def _resolve_model_path(model_path: Optional[Path] = None) -> Path:
    model_path = model_path or Path(settings.ml_model_path)
    if not model_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[4]
        model_path = repo_root / model_path
    return model_path


def _resolve_model_settings(
    metadata: Dict[str, Any],
    grid_size_m: Optional[float],
    use_hex: bool,
) -> Tuple[float, bool, int, int, float, int, float]:
    if metadata.get("use_hex") is not None:
        use_hex = bool(metadata.get("use_hex"))

    grid_size = grid_size_m or metadata.get("grid_size_m") or settings.default_grid_size_m
    neighbor_k = int(metadata.get("neighbor_k") or settings.ml_neighbor_k)
    time_bin_hours = int(metadata.get("time_bin_hours") or settings.ml_time_bin_hours)
    bandwidth_m = float(metadata.get("bandwidth_m") or settings.ml_bandwidth_m)
    region_clusters = int(metadata.get("region_clusters") or settings.ml_region_clusters)
    min_risk_threshold = float(
        metadata.get("min_risk_threshold") or settings.ml_min_risk_threshold
    )
    return (
        float(grid_size),
        use_hex,
        neighbor_k,
        time_bin_hours,
        bandwidth_m,
        region_clusters,
        min_risk_threshold,
    )


def _coerce_naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def build_feature_frame(
    events: pd.DataFrame,
    grid_index: GridIndex,
    time_bin_hours: int,
    bandwidth_m: float,
    region_clusters: int,
    max_time: datetime,
) -> pd.DataFrame:
    if events.empty or not grid_index.grid_ids:
        return pd.DataFrame()

    kde_df = build_kde_targets(
        events,
        grid_index,
        time_bin_hours=time_bin_hours,
        bandwidth_m=bandwidth_m,
        max_time=max_time,
    )
    if kde_df.empty:
        return pd.DataFrame()

    full_idx = pd.MultiIndex.from_product(
        [grid_index.grid_ids, kde_df["time_bin"].unique()],
        names=["grid_id", "time_bin"],
    )
    full_df = pd.DataFrame(index=full_idx).reset_index()
    full_df = full_df.merge(kde_df, on=["grid_id", "time_bin"], how="left")
    full_df["risk_score"] = full_df["risk_score"].fillna(0.0)

    full_df = add_time_features(full_df)
    full_df = add_region_clusters(full_df, grid_index, region_clusters)
    full_df = add_lag_features(full_df, time_bin_hours)
    full_df = add_neighbor_lag(full_df, grid_index.neighbors)

    full_df[["lag_1d", "lag_1w", "neighbor_mean_lag1"]] = full_df[
        ["lag_1d", "lag_1w", "neighbor_mean_lag1"]
    ].fillna(0.0)

    return full_df


def build_kde_targets(
    events: pd.DataFrame,
    grid_index: GridIndex,
    time_bin_hours: int,
    bandwidth_m: float,
    max_time: datetime,
) -> pd.DataFrame:
    if events.empty or not grid_index.grid_ids:
        return pd.DataFrame(columns=["grid_id", "time_bin", "risk_score"])

    grid_ids = grid_index.grid_ids
    grid_coords = np.array([grid_index.centers[g] for g in grid_ids])

    events = events.copy()
    events["time_bin"] = pd.to_datetime(events["event_time"]).dt.floor(f"{time_bin_hours}H")

    min_time = events["time_bin"].min()
    time_bins = pd.date_range(min_time, max_time, freq=f"{time_bin_hours}H")

    records = []
    grouped = events.groupby("time_bin")
    for time_bin in time_bins:
        group = grouped.get_group(time_bin) if time_bin in grouped.groups else None
        if group is None or group.empty:
            density = np.zeros(len(grid_ids))
        else:
            event_coords = group[["lat", "lng"]].to_numpy()
            event_weights = (group["severity"] / 5.0).to_numpy()
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


def add_neighbor_lag(df: pd.DataFrame, neighbors: Dict[str, List[str]]) -> pd.DataFrame:
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
    grid_index: GridIndex,
    region_clusters: int,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    centers = np.array([grid_index.centers[g] for g in grid_index.grid_ids])
    cluster_count = max(1, min(region_clusters, len(centers)))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(centers)
    region_map = {gid: int(label) for gid, label in zip(grid_index.grid_ids, labels)}

    df = df.copy()
    df["district_region"] = df["grid_id"].map(region_map)
    return df


def predict_risk_scores(
    model: object,
    model_features: List[str],
    df: pd.DataFrame,
) -> np.ndarray:
    if not model_features:
        raise ValueError("Model feature list is empty.")
    feature_df = pd.get_dummies(
        df,
        columns=["district_region"],
        prefix="region",
    )
    for col in model_features:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    feature_df = feature_df[model_features]

    preds = model.predict(feature_df)
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def geom_to_geography(db: Session, geom: dict) -> Optional[str]:
    if not geom:
        return None
    geom_json_str = json.dumps(geom) if isinstance(geom, dict) else str(geom)
    geom_ewkt = db.execute(
        text("""
            SELECT ST_AsEWKT(ST_GeomFromGeoJSON(CAST(:geom AS text))::geography) as geom_ewkt
        """),
        {"geom": geom_json_str},
    ).scalar()
    return geom_ewkt


def generate_spatiotemporal_forecast_cells(
    db: Session,
    time_window_start: datetime,
    time_window_end: datetime,
    bbox: Tuple[float, float, float, float],
    grid_size_m: Optional[float] = None,
    use_hex: bool = True,
) -> List[RiskCell]:
    model_path = _resolve_model_path()
    model, model_features, metadata = load_model_artifact(model_path)

    (
        grid_size,
        use_hex,
        neighbor_k,
        time_bin_hours,
        bandwidth_m,
        region_clusters,
        min_risk_threshold,
    ) = _resolve_model_settings(metadata, grid_size_m, use_hex)

    grid_index = build_grid_index(db, bbox, grid_size, use_hex, neighbor_k)
    if not grid_index.grid_ids:
        return []

    history_end = _coerce_naive(time_window_start)
    history_start = history_end - timedelta(days=settings.ml_history_days)

    events = load_historical_events(db, history_start, history_end)
    events = assign_grid_ids_for_events(events, grid_index, bbox, grid_size, use_hex)
    if events.empty:
        return []

    target_time = pd.to_datetime(_coerce_naive(time_window_start)).floor(
        f"{time_bin_hours}H"
    )
    full_df = build_feature_frame(
        events,
        grid_index,
        time_bin_hours=time_bin_hours,
        bandwidth_m=bandwidth_m,
        region_clusters=region_clusters,
        max_time=target_time,
    )
    if full_df.empty:
        return []

    target_df = full_df[full_df["time_bin"] == target_time].copy()
    if target_df.empty:
        return []

    feature_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "lag_1d",
        "lag_1w",
        "neighbor_mean_lag1",
        "district_region",
    ]
    preds = predict_risk_scores(model, model_features, target_df[feature_columns])
    target_df["predicted_risk"] = preds

    if time_window_start.tzinfo:
        time_window_start_naive = time_window_start.replace(tzinfo=None)
    else:
        time_window_start_naive = time_window_start
    if time_window_end.tzinfo:
        time_window_end_naive = time_window_end.replace(tzinfo=None)
    else:
        time_window_end_naive = time_window_end

    time_range = DateTimeRange(time_window_start_naive, time_window_end_naive, "[]")
    risk_cells: List[RiskCell] = []
    for row in target_df.itertuples(index=False):
        risk_score = float(row.predicted_risk)
        if risk_score < min_risk_threshold:
            continue
        geom = grid_index.geom_map.get(row.grid_id)
        geom_ewkt = geom_to_geography(db, geom)
        if not geom_ewkt:
            continue
        risk_cells.append(
            RiskCell(
                geom=geom_ewkt,
                time_window=time_range,
                risk_score=risk_score,
                confidence=0.7,
            )
        )

    if risk_cells:
        try:
            db.bulk_save_objects(risk_cells)
            db.commit()
        except Exception:
            db.rollback()

    return risk_cells


def _count_nearby_events(
    events: pd.DataFrame,
    lat: float,
    lng: float,
    radius_m: float = 1000.0,
) -> int:
    if events.empty:
        return 0
    coords = events[["lat", "lng"]].to_numpy()
    distances = haversine_matrix(np.array([[lat, lng]]), coords)
    return int(np.sum(distances[0] <= radius_m))


def predict_spatiotemporal_point(
    db: Session,
    lat: float,
    lng: float,
    forecast_time: datetime,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    grid_size_m: Optional[float] = None,
    use_hex: bool = True,
    crime_type: Optional[str] = None,
) -> Dict[str, Any]:
    forecast_time = _coerce_naive(forecast_time)
    model_path = _resolve_model_path()
    model, model_features, metadata = load_model_artifact(model_path)
    (
        grid_size,
        use_hex,
        neighbor_k,
        time_bin_hours,
        bandwidth_m,
        region_clusters,
        _,
    ) = _resolve_model_settings(metadata, grid_size_m, use_hex)

    from app.services.utils import get_kucukcekmece_bbox_from_polygon

    bbox = bbox or get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
    grid_index = build_grid_index(db, bbox, grid_size, use_hex, neighbor_k)
    grid_id = assign_grid_id_for_point(lat, lng, grid_index, bbox, grid_size, use_hex)
    if not grid_id:
        return {
            "forecast": 0.0,
            "confidence": 0.0,
            "nearby_events": 0,
            "grid_id": None,
            "time_bin_hours": time_bin_hours,
        }

    history_end = forecast_time
    history_start = history_end - timedelta(days=settings.ml_history_days)
    events_raw = load_historical_events(
        db,
        history_start,
        history_end,
        crime_type=crime_type,
    )
    if events_raw.empty:
        return {
            "forecast": 0.0,
            "confidence": 0.0,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
        }

    events = assign_grid_ids_for_events(events_raw, grid_index, bbox, grid_size, use_hex)
    if events.empty:
        return {
            "forecast": 0.0,
            "confidence": 0.0,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
        }

    target_time = pd.to_datetime(forecast_time).floor(f"{time_bin_hours}H")
    full_df = build_feature_frame(
        events,
        grid_index,
        time_bin_hours=time_bin_hours,
        bandwidth_m=bandwidth_m,
        region_clusters=region_clusters,
        max_time=target_time,
    )
    if full_df.empty:
        return {
            "forecast": 0.0,
            "confidence": 0.0,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
        }

    target_df = full_df[
        (full_df["grid_id"] == grid_id) & (full_df["time_bin"] == target_time)
    ].copy()
    if target_df.empty:
        return {
            "forecast": 0.0,
            "confidence": 0.0,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
        }

    feature_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "lag_1d",
        "lag_1w",
        "neighbor_mean_lag1",
        "district_region",
    ]
    preds = predict_risk_scores(model, model_features, target_df[feature_columns])
    forecast = float(preds[0]) if len(preds) else 0.0

    nearby_events = _count_nearby_events(events_raw, lat, lng)
    confidence = min(1.0, nearby_events / 10.0) if nearby_events else 0.0

    return {
        "forecast": forecast,
        "confidence": confidence,
        "nearby_events": int(nearby_events),
        "grid_id": grid_id,
        "time_bin_hours": time_bin_hours,
    }


def predict_spatiotemporal_timeseries(
    db: Session,
    lat: float,
    lng: float,
    start_time: datetime,
    end_time: datetime,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    grid_size_m: Optional[float] = None,
    use_hex: bool = True,
    crime_type: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = _coerce_naive(start_time)
    end_time = _coerce_naive(end_time)
    horizon_hours = int((end_time - start_time).total_seconds() / 3600)
    if horizon_hours <= 0:
        return {
            "forecast": [],
            "nearby_events": 0,
            "grid_id": None,
            "time_bin_hours": settings.ml_time_bin_hours,
            "historical_points": 0,
        }

    model_path = _resolve_model_path()
    model, model_features, metadata = load_model_artifact(model_path)
    (
        grid_size,
        use_hex,
        neighbor_k,
        time_bin_hours,
        bandwidth_m,
        region_clusters,
        _,
    ) = _resolve_model_settings(metadata, grid_size_m, use_hex)

    from app.services.utils import get_kucukcekmece_bbox_from_polygon

    bbox = bbox or get_kucukcekmece_bbox_from_polygon(db) or settings.kucukcekmece_fallback_bbox
    grid_index = build_grid_index(db, bbox, grid_size, use_hex, neighbor_k)
    grid_id = assign_grid_id_for_point(lat, lng, grid_index, bbox, grid_size, use_hex)
    if not grid_id:
        return {
            "forecast": [0.0] * horizon_hours,
            "nearby_events": 0,
            "grid_id": None,
            "time_bin_hours": time_bin_hours,
            "historical_points": 0,
        }

    history_end = start_time
    history_start = history_end - timedelta(days=settings.ml_history_days)
    events_raw = load_historical_events(
        db,
        history_start,
        history_end,
        crime_type=crime_type,
    )
    if events_raw.empty:
        return {
            "forecast": [0.0] * horizon_hours,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
            "historical_points": 0,
        }

    events = assign_grid_ids_for_events(events_raw, grid_index, bbox, grid_size, use_hex)
    if events.empty:
        return {
            "forecast": [0.0] * horizon_hours,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
            "historical_points": 0,
        }

    target_end = pd.to_datetime(end_time).floor(f"{time_bin_hours}H")
    target_start = pd.to_datetime(start_time).floor(f"{time_bin_hours}H")
    full_df = build_feature_frame(
        events,
        grid_index,
        time_bin_hours=time_bin_hours,
        bandwidth_m=bandwidth_m,
        region_clusters=region_clusters,
        max_time=target_end,
    )
    if full_df.empty:
        return {
            "forecast": [0.0] * horizon_hours,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
            "historical_points": int(len(events_raw)),
        }

    target_df = full_df[
        (full_df["grid_id"] == grid_id)
        & (full_df["time_bin"] >= target_start)
        & (full_df["time_bin"] <= target_end)
    ].copy()
    if target_df.empty:
        return {
            "forecast": [0.0] * horizon_hours,
            "nearby_events": 0,
            "grid_id": grid_id,
            "time_bin_hours": time_bin_hours,
            "historical_points": int(len(events_raw)),
        }

    target_df = target_df.sort_values("time_bin")
    feature_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "lag_1d",
        "lag_1w",
        "neighbor_mean_lag1",
        "district_region",
    ]
    preds = predict_risk_scores(model, model_features, target_df[feature_columns])
    pred_map = {
        pd.to_datetime(time_bin): float(pred)
        for time_bin, pred in zip(target_df["time_bin"], preds)
    }

    forecast = []
    for i in range(horizon_hours):
        ts = start_time + timedelta(hours=i)
        bin_time = pd.to_datetime(ts).floor(f"{time_bin_hours}H")
        forecast.append(float(pred_map.get(bin_time, 0.0)))

    nearby_events = _count_nearby_events(events_raw, lat, lng)

    return {
        "forecast": forecast,
        "nearby_events": int(nearby_events),
        "grid_id": grid_id,
        "time_bin_hours": time_bin_hours,
        "historical_points": int(len(events_raw)),
    }
