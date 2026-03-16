"""
設定管理モジュール

.env からシークレット、config.json から動的パラメータを読み込む。
pydantic-settings で型チェック・SecretStr によるログ漏洩防止を提供。
"""

import json
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "trading.db"
DEFAULT_DB_BACKUP_PATH = PROJECT_ROOT / "db" / "backup"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


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
    db_path: str = str(DEFAULT_DB_PATH)
    db_backup_path: str = str(DEFAULT_DB_BACKUP_PATH)

    # System
    log_dir: str = str(DEFAULT_LOG_DIR)
    model_dir: str = str(DEFAULT_MODEL_DIR)
    config_path: str = str(DEFAULT_CONFIG_PATH)
    demo_mode: bool = True

    # Webhook
    webhook_secret: SecretStr = SecretStr("")
    webhook_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def _normalize_trading_config(config: dict) -> dict:
    """段階的リファクタリング中の設定差分を吸収する。"""
    risk = config.setdefault("risk", {})

    # 新しいダイナミック・エグジット戦略の既定値。
    risk.setdefault("exit_prob_threshold", 0.35)
    risk.setdefault("time_decay_minutes", 60)
    risk.setdefault("time_decay_min_profit_atr", 0.5)

    llm = config.setdefault("llm", {})
    llm.setdefault("model_diff", llm.get("model_instant", "gpt-5.2"))
    llm.setdefault("reasoning_effort_diff", llm.get("reasoning_effort_instant", "low"))
    llm.setdefault("hybrid_enabled", True)
    llm.setdefault("news_importance_escalation_threshold", 0.65)
    llm.setdefault("feature_news_importance_threshold", 0.55)

    return config


def load_trading_config(config_path: str | None = None) -> dict:
    """
    config.json から動的トレーディングパラメータを読み込む。
    週末自動最適化で書き換えられるパラメータはこちらで管理。
    """
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    path = Path(config_path)
    if not path.exists():
        fallback_path = DEFAULT_CONFIG_PATH
        if fallback_path.exists():
            path = fallback_path
        else:
            raise FileNotFoundError(
                f"config.json が見つかりません: requested={path} fallback={fallback_path}"
            )
    with open(path, encoding="utf-8") as f:
        return _normalize_trading_config(json.load(f))


def save_trading_config(config: dict, config_path: str | None = None) -> None:
    """config.json を書き出す（週末最適化結果の反映用）。"""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
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
