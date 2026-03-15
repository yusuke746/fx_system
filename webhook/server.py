"""
FastAPI Webhook サーバー

■ セキュリティ
    - JSON内共有トークン照合（TradingView制約に対応）
    - トークン不一致は 429 を返して破棄
    - 5分以内に3回以上の認証失敗で Discord 通知

■ 時刻基準
  - Webhook 受信時刻: 保存基準（UTC）で記録
  - ログ・Discord通知: 表示基準（JST）
"""

import time
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from loguru import logger

from config.settings import get_settings
from core.time_manager import now_utc, format_jst

app = FastAPI(title="FX Auto-Trading Webhook", version="2.2")

# 署名不一致のカウンター（IP別）
_failed_sig_counts: dict[str, list[float]] = defaultdict(list)
_FAIL_WINDOW_SEC = 300  # 5分
_FAIL_THRESHOLD = 3


@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    TradingView からの Webhook を受信する。

    期待される JSON ペイロード:
    {
        "pair": "USDJPY",
        "direction": "long",
        "fvg_4h_zone_active": true,
        "ob_4h_zone_active": false,
        "liq_sweep_1h": true,
        "bos_1h": false,
        "mtf_confluence": 2,
        "atr": 0.45,
        "close": 149.85
    }
    """
    settings = get_settings()

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 共有トークン検証（JSON内）
    secret = settings.webhook_secret.get_secret_value()
    provided_token = str(payload.get("webhook_token", ""))
    if secret and provided_token != secret:
        client_ip = request.client.host if request.client else "unknown"
        _record_sig_failure(client_ip)
        logger.warning(f"Webhook token verification failed from {client_ip}")
        raise HTTPException(status_code=429, detail="Webhook token verification failed")

    # 必須フィールドのバリデーション
    required = ["pair", "direction", "mtf_confluence", "atr", "close"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    pair = payload["pair"]
    if pair not in ("USDJPY", "EURUSD", "GBPJPY"):
        raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")

    # 受信時刻を UTC（保存基準）で付加
    payload["received_at_utc"] = now_utc().isoformat()

    logger.info(
        f"Webhook received: {pair} {payload['direction']} "
        f"confluence={payload['mtf_confluence']} "
        f"at {format_jst(now_utc())}"
    )

    # Orchestrator のキューに投入（main.py 側で設定）
    from webhook.signal_queue import signal_queue
    await signal_queue.put(payload)

    return {"status": "ok", "received_at": payload["received_at_utc"]}


@app.get("/health")
async def health():
    """ヘルスチェック用エンドポイント。"""
    return {
        "status": "healthy",
        "time_utc": now_utc().isoformat(),
        "time_jst": format_jst(now_utc()),
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """ブラウザの自動リクエストを 204 で返してログノイズを抑制。"""
    from fastapi.responses import Response
    return Response(status_code=204)


def _record_sig_failure(client_ip: str) -> None:
    """署名不一致を記録し、閾値を超えたら通知する。"""
    now = time.time()
    _failed_sig_counts[client_ip] = [
        t for t in _failed_sig_counts[client_ip]
        if now - t < _FAIL_WINDOW_SEC
    ]
    _failed_sig_counts[client_ip].append(now)

    if len(_failed_sig_counts[client_ip]) >= _FAIL_THRESHOLD:
        logger.critical(
            f"Webhook signature failure threshold exceeded: "
            f"{client_ip} ({len(_failed_sig_counts[client_ip])} failures in {_FAIL_WINDOW_SEC}s)"
        )
