"""
ロギングモジュール

- ログファイルは保存基準（UTC）のタイムスタンプで記録。
- Discord 通知向けのフォーマットでは表示基準（JST）に変換して出力。
"""

import sys
from pathlib import Path

from loguru import logger

from core.time_manager import format_jst, now_utc


def setup_logger(log_dir: str = "logs") -> None:
    """loguru ロガーの初期設定。"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # デフォルトハンドラを除去
    logger.remove()

    # コンソール出力（JST 表示基準）
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        level="INFO",
    )

    # ファイル出力（UTC 保存基準、30日ローテーション）
    logger.add(
        str(log_path / "fx_system_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        compression="gz",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level="DEBUG",
        encoding="utf-8",
    )

    # エラー専用ファイル
    logger.add(
        str(log_path / "fx_system_error_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        compression="gz",
        level="ERROR",
        encoding="utf-8",
    )

    logger.info(f"Logger initialized. Log dir: {log_path.resolve()}")
    logger.info(f"System start time: {format_jst(now_utc())}")
