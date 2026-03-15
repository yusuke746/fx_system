"""
設定管理モジュール

.env からシークレット、config.json から動的パラメータを読み込む。
pydantic-settings で型チェック・SecretStr によるログ漏洩防止を提供。
"""

import json
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """環境変数（.env）から読み込む静的設定・シークレット。"""

    # OpenAI
    openai_api_key: SecretStr

    # Discord
    discord_webhook_url: str
    discord_webhook_critical_url: str

    # MT5
    mt5_login: int
    mt5_password: SecretStr
    mt5_server: str

    # DB
    db_path: str = "C:/fx_system/db/trading.db"
    db_backup_path: str = "C:/fx_system/db/backup/"

    # System
    log_dir: str = "C:/fx_system/logs/"
    model_dir: str = "C:/fx_system/models/"
    config_path: str = "C:/fx_system/config.json"
    demo_mode: bool = True

    # Webhook
    webhook_secret: SecretStr = SecretStr("")
    webhook_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def load_trading_config(config_path: str | None = None) -> dict:
    """
    config.json から動的トレーディングパラメータを読み込む。
    週末自動最適化で書き換えられるパラメータはこちらで管理。
    """
    if config_path is None:
        config_path = "config.json"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config.json が見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_trading_config(config: dict, config_path: str | None = None) -> None:
    """config.json を書き出す（週末最適化結果の反映用）。"""
    if config_path is None:
        config_path = "config.json"
    path = Path(config_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


# --- シングルトンアクセス ---
_settings: Settings | None = None
_trading_config: dict | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_trading_config() -> dict:
    global _trading_config
    if _trading_config is None:
        settings = get_settings()
        _trading_config = load_trading_config(settings.config_path)
    return _trading_config


def reload_trading_config() -> dict:
    """config.json を再読み込みする（最適化後の反映用）。"""
    global _trading_config
    settings = get_settings()
    _trading_config = load_trading_config(settings.config_path)
    return _trading_config
