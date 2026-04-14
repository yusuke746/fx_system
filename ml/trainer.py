"""
LightGBM 学習・ウォークフォワード検証モジュール

■ 学習設計
  - 学習データ: 直近90日
  - ウォークフォワード: 訓練60日 / 検証15日
  - 週次再学習: 毎週日曜23:00 JST
  - ドリフト検知: 直近20トレード勝率<40% or PSI>0.2 → 緊急再学習
  - モデル管理: 通貨ペアごとに独立モデル（3モデル）
  - 保存世代数: 3世代

■ 時刻基準
  - 学習ログ: 保存基準（UTC）
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
from loguru import logger
from sklearn.metrics import balanced_accuracy_score

from core.time_manager import now_utc
from config.settings import get_trading_config
from ml.lgbm_model import LGBM_PARAMS, FEATURE_NAMES, get_lgbm_params


def _build_class_weight(y: np.ndarray, directional_boost: float) -> dict[int, float]:
    """クラス不均衡を補正しつつ、方向クラス(up/down)に重みを寄せる。"""
    if len(y) == 0:
        return {0: 1.0, 1: 1.0, 2: 1.0}

    counts = np.bincount(y, minlength=3).astype(float)
    total = float(counts.sum())
    class_weight: dict[int, float] = {}
    for cls in (0, 1, 2):
        denom = max(counts[cls], 1.0)
        class_weight[cls] = total / (3.0 * denom)

    boost = max(0.5, float(directional_boost))
    class_weight[0] *= boost
    class_weight[2] *= boost

    return class_weight


def save_model_metrics(pair: str, model_dir: str, metrics: dict) -> Path:
    """モデル精度メタデータを保存する。"""
    save_dir = Path(model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / f"lgbm_{pair}_metrics.json"
    payload = {"pair": pair, "updated_at_utc": now_utc().isoformat(), **metrics}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    logger.info(f"Model metrics saved: {metrics_path}")
    return metrics_path


def load_model_metrics(pair: str, model_dir: str = "models") -> dict | None:
    """モデル精度メタデータを読み込む。"""
    metrics_path = Path(model_dir) / f"lgbm_{pair}_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    pair: str,
    model_dir: str = "models",
    sample_weight: np.ndarray | None = None,
) -> lgb.LGBMClassifier:
    """
    ウォークフォワード検証付きで LightGBM モデルを学習する。

    Args:
        X: shape=(N, 35) の特徴量配列
        y: shape=(N,) のラベル配列 (0=up, 1=flat, 2=down)
        pair: 通貨ペア名
        model_dir: モデル保存ディレクトリ

    Returns:
        学習済み LGBMClassifier
    """
    logger.info(f"Training LightGBM for {pair}: {X.shape[0]} samples, {X.shape[1]} features")

    config = get_trading_config()
    ml_cfg = config.get("ml", {})
    directional_boost = float(ml_cfg.get("directional_class_boost", 1.0))
    class_weight = _build_class_weight(y, directional_boost)

    lgbm_params = get_lgbm_params(pair)
    model = lgb.LGBMClassifier(**lgbm_params, class_weight=class_weight)
    model.fit(
        X, y,
        feature_name=FEATURE_NAMES,
        sample_weight=sample_weight,
    )

    # モデル保存（世代管理）
    save_dir = Path(model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now_utc().strftime("%Y%m%d_%H%M%S")
    model_path = save_dir / f"lgbm_{pair}_{timestamp}.pkl"

    import joblib
    joblib.dump(model, model_path)

    # 最新モデルへのシンボリックリンク（Windows対応: コピー）
    latest_path = save_dir / f"lgbm_{pair}.pkl"
    import shutil
    shutil.copy2(model_path, latest_path)

    # 古い世代を削除（3世代保持）
    _cleanup_old_models(save_dir, pair, keep=3)

    logger.info(f"Model trained and saved: {model_path}")
    return model


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    signal_times: list[datetime] | None = None,
    train_days: int = 60,
    val_days: int = 15,
    samples_per_day: int = 96,  # signal_timesがNoneの場合のフォールバック
    pair: str | None = None,
) -> dict:
    """
    ウォークフォワード検証を実行する。

    signal_times が渡された場合はカレンダー日付ベースでフォールドを分割する。
    None の場合はサンプル数ベースのレガシーロジックで動作する。

    Returns:
        {"accuracy": float, "fold_results": list, "n_folds": int, "date_based": bool, "total_samples": int}
    """
    total_samples = len(X)
    config = get_trading_config()
    ml_cfg = config.get("ml", {})
    directional_boost = float(ml_cfg.get("directional_class_boost", 1.0))
    lgbm_params = get_lgbm_params(pair)

    if signal_times is not None:
        times = np.array(signal_times)
        order = np.argsort(times)
        times = times[order]
        X = X[order]
        y = y[order]
        if sample_weight is not None:
            sample_weight = sample_weight[order]
        fold_results = []
        fold = 0
        t_start = times[0]

        while True:
            train_start_dt = t_start + timedelta(days=fold * val_days)
            train_end_dt = train_start_dt + timedelta(days=train_days)
            val_end_dt = train_end_dt + timedelta(days=val_days)

            if val_end_dt > times[-1]:
                break

            train_mask = (times >= train_start_dt) & (times < train_end_dt)
            val_mask = (times >= train_end_dt) & (times < val_end_dt)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            w_train = sample_weight[train_mask] if sample_weight is not None else None

            _min_leaf = int(lgbm_params.get("min_child_samples", 20))
            if len(X_train) < _min_leaf:  # LightGBM の min_child_samples に満たない場合はスキップ
                fold += 1
                continue

            if len(X_val) == 0:
                fold += 1
                continue

            class_weight = _build_class_weight(y_train, directional_boost)
            model = lgb.LGBMClassifier(**lgbm_params, class_weight=class_weight)
            model.fit(X_train, y_train, feature_name=FEATURE_NAMES, sample_weight=w_train)
            y_pred = model.predict(X_val)
            accuracy = float(np.mean(y_pred == y_val))
            balanced_accuracy = float(balanced_accuracy_score(y_val, y_pred))
            class_counts = np.bincount(y_val, minlength=3)
            majority_baseline = float(class_counts.max() / len(y_val))

            fold_results.append({
                "fold": fold,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "majority_baseline": majority_baseline,
            })
            fold += 1

        if len(fold_results) == 0:
            logger.warning(
                "walk_forward_validate: no fold available, returning empty results."
            )
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "majority_baseline": 0.0,
                "fold_results": [],
                "n_folds": 0,
                "date_based": True,
                "total_samples": total_samples,
            }
        if len(fold_results) == 1:
            only = fold_results[0]
            logger.warning(
                "walk_forward_validate: only 1 fold available, using single-fold metrics."
            )
            return {
                "accuracy": float(only["accuracy"]),
                "balanced_accuracy": float(only["balanced_accuracy"]),
                "majority_baseline": float(only["majority_baseline"]),
                "fold_results": fold_results,
                "n_folds": 1,
                "date_based": True,
                "total_samples": total_samples,
            }

        avg_accuracy = float(np.mean([r["accuracy"] for r in fold_results]))
        avg_balanced_accuracy = float(np.mean([r["balanced_accuracy"] for r in fold_results]))
        avg_majority_baseline = float(np.mean([r["majority_baseline"] for r in fold_results]))
        logger.info(
            f"Walk-forward validation (date-based): {len(fold_results)} folds, "
            f"avg accuracy={avg_accuracy:.4f}, "
            f"avg balanced_accuracy={avg_balanced_accuracy:.4f}, "
            f"avg majority_baseline={avg_majority_baseline:.4f}"
        )
        return {
            "accuracy": avg_accuracy,
            "balanced_accuracy": avg_balanced_accuracy,
            "majority_baseline": avg_majority_baseline,
            "fold_results": fold_results,
            "n_folds": len(fold_results),
            "date_based": True,
            "total_samples": total_samples,
        }

    # フォールバック: サンプル数ベース（後方互換）
    logger.warning("walk_forward_validate: signal_times not provided, using sample count fallback.")
    train_size = train_days * samples_per_day
    val_size = val_days * samples_per_day

    fold_results = []
    start = 0
    fold = 0
    while start + train_size + val_size <= total_samples:
        train_end = start + train_size
        val_end = train_end + val_size

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        w_train = sample_weight[start:train_end] if sample_weight is not None else None

        class_weight = _build_class_weight(y_train, directional_boost)
        model = lgb.LGBMClassifier(**lgbm_params, class_weight=class_weight)
        model.fit(X_train, y_train, feature_name=FEATURE_NAMES, sample_weight=w_train)

        y_pred = model.predict(X_val)
        accuracy = float(np.mean(y_pred == y_val))
        balanced_accuracy = float(balanced_accuracy_score(y_val, y_pred))
        class_counts = np.bincount(y_val, minlength=3)
        majority_baseline = float(class_counts.max() / len(y_val))

        fold_results.append({
            "fold": fold,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "majority_baseline": majority_baseline,
        })

        start += val_size
        fold += 1

    avg_accuracy = float(np.mean([r["accuracy"] for r in fold_results])) if fold_results else 0.0
    avg_balanced_accuracy = float(np.mean([r["balanced_accuracy"] for r in fold_results])) if fold_results else 0.0
    avg_majority_baseline = float(np.mean([r["majority_baseline"] for r in fold_results])) if fold_results else 0.0

    logger.info(
        f"Walk-forward validation: {len(fold_results)} folds, "
        f"avg accuracy={avg_accuracy:.4f}, "
        f"avg balanced_accuracy={avg_balanced_accuracy:.4f}, "
        f"avg majority_baseline={avg_majority_baseline:.4f}"
    )

    return {
        "accuracy": avg_accuracy,
        "balanced_accuracy": avg_balanced_accuracy,
        "majority_baseline": avg_majority_baseline,
        "fold_results": fold_results,
        "n_folds": len(fold_results),
        "date_based": False,
        "total_samples": total_samples,
    }


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) を計算する。
    PSI > 0.2 → ドリフト検知。
    """
    eps = 1e-6
    expected_hist, bin_edges = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_hist / (len(expected) + eps)
    actual_pct = actual_hist / (len(actual) + eps)

    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _cleanup_old_models(model_dir: Path, pair: str, keep: int = 3) -> None:
    """古いモデルファイルを削除する（N世代保持）。"""
    pattern = f"lgbm_{pair}_*.pkl"
    model_files = sorted(model_dir.glob(pattern))

    if len(model_files) > keep:
        for old_file in model_files[:-keep]:
            old_file.unlink()
            logger.info(f"Deleted old model: {old_file}")
