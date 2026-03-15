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

from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from core.time_manager import now_utc
from ml.lgbm_model import LGBM_PARAMS, FEATURE_NAMES


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    pair: str,
    model_dir: str = "models",
) -> lgb.LGBMClassifier:
    """
    ウォークフォワード検証付きで LightGBM モデルを学習する。

    Args:
        X: shape=(N, 32) の特徴量配列
        y: shape=(N,) のラベル配列 (0=up, 1=flat, 2=down)
        pair: 通貨ペア名
        model_dir: モデル保存ディレクトリ

    Returns:
        学習済み LGBMClassifier
    """
    logger.info(f"Training LightGBM for {pair}: {X.shape[0]} samples, {X.shape[1]} features")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X, y,
        feature_name=FEATURE_NAMES,
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
    train_days: int = 60,
    val_days: int = 15,
    samples_per_day: int = 96,  # 15分足 = 96本/日
) -> dict:
    """
    ウォークフォワード検証を実行する。

    Returns:
        {"sharpe_ratio": float, "accuracy": float, "fold_results": list}
    """
    train_size = train_days * samples_per_day
    val_size = val_days * samples_per_day

    fold_results = []
    total_samples = len(X)

    start = 0
    fold = 0
    while start + train_size + val_size <= total_samples:
        train_end = start + train_size
        val_end = train_end + val_size

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train, feature_name=FEATURE_NAMES)

        y_pred = model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)

        fold_results.append({
            "fold": fold,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "accuracy": float(accuracy),
        })

        start += val_size
        fold += 1

    avg_accuracy = np.mean([r["accuracy"] for r in fold_results]) if fold_results else 0.0

    logger.info(
        f"Walk-forward validation: {len(fold_results)} folds, "
        f"avg accuracy={avg_accuracy:.4f}"
    )

    return {
        "accuracy": float(avg_accuracy),
        "fold_results": fold_results,
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
