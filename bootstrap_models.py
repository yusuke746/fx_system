"""
疎通テスト用: LightGBMダミーモデルを3通貨ペア分生成する。

本番運用向けではなく、Webhook→推論→発注フロー確認用。
"""

import numpy as np

from ml.lgbm_model import FEATURE_NAMES
from ml.trainer import train_model


def main() -> None:
    rng = np.random.default_rng(42)
    n_samples = 3000
    n_features = len(FEATURE_NAMES)

    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float64)

    score = 0.6 * X[:, 0] + 0.4 * X[:, 6] - 0.5 * X[:, 15]
    y = np.where(score > 0.2, 0, np.where(score < -0.2, 2, 1)).astype(np.int32)

    for pair in ["USDJPY", "EURUSD", "GBPJPY"]:
        train_model(X, y, pair=pair, model_dir="models")
        print(f"model generated: models/lgbm_{pair}.pkl")


if __name__ == "__main__":
    main()
