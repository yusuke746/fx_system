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

import hmac
import ipaddress

from cachetools import TTLCache
from fastapi import FastAPI, Request, HTTPException
from loguru import logger

from config.settings import get_settings
from core.time_manager import now_utc, format_jst
from webhook.signal_queue import get_queue

app = FastAPI(title="FX Auto-Trading Webhook", version="2.2")

# 認証失敗のカウンター（IP別）
_FAIL_WINDOW_SEC = 300  # 5分
_FAIL_THRESHOLD = 3
_failed_token_counts: TTLCache = TTLCache(maxsize=1000, ttl=_FAIL_WINDOW_SEC)
_blocked_ip_cache: TTLCache = TTLCache(maxsize=1000, ttl=_FAIL_WINDOW_SEC)


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
    client_ip = _get_client_ip(request)
    if _blocked_ip_cache.get(client_ip):
        logger.warning(f"Blocked webhook request from {client_ip}")
        raise HTTPException(status_code=429, detail="Too many invalid auth attempts")

    settings = get_settings()
    if not _is_ip_allowed(client_ip, settings):
        logger.warning(f"Webhook rejected by IP allowlist: {client_ip}")
        raise HTTPException(status_code=403, detail="IP not allowed")

    try:
        payload = await request.json()
    except Exception as e:
        logger.warning(f"Failed to parse JSON from {client_ip}: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 共有トークン検証（JSON内）
    secret = settings.webhook_secret.get_secret_value()
    provided_token = str(payload.get("webhook_token", ""))
    if secret and not hmac.compare_digest(provided_token, secret):
        _record_auth_failure(client_ip)
        logger.warning(f"Webhook token verification failed from {client_ip}")
        raise HTTPException(status_code=429, detail="Webhook token verification failed")

    # 必須フィールドのバリデーション
    required = ["pair", "direction", "mtf_confluence", "atr", "close"]
    missing = [f for f in required if f not in payload]
    if missing:
        logger.warning(
            f"Webhook missing required fields from {client_ip}: {missing} "
            f"(received: {list(payload.keys())})"
        )
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
    signal_queue = get_queue()
    await signal_queue.put(payload)

    return {"status": "ok", "received_at": payload["received_at_utc"]}


@app.post("/webhook/mcp")
async def receive_mcp_webhook(request: Request):
    """
    TV MCP EA からのブレイクアウトシグナルを受信する。

    通常の /webhook と異なり、XAUUSD も受け付ける。
    必須フィールド: pair, direction, atr, close, breakout_score, pattern, signal_source
    """
    client_ip = _get_client_ip(request)
    if _blocked_ip_cache.get(client_ip):
        raise HTTPException(status_code=429, detail="Too many invalid auth attempts")

    settings = get_settings()
    if not _is_ip_allowed(client_ip, settings):
        raise HTTPException(status_code=403, detail="IP not allowed")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    secret = settings.webhook_secret.get_secret_value()
    provided_token = str(payload.get("webhook_token", ""))
    if secret and not hmac.compare_digest(provided_token, secret):
        _record_auth_failure(client_ip)
        raise HTTPException(status_code=429, detail="Webhook token verification failed")

    required = ["pair", "direction", "atr", "close", "breakout_score", "signal_source"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    pair = payload["pair"]
    if pair not in ("USDJPY", "EURUSD", "GBPJPY", "XAUUSD"):
        raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")

    payload["received_at_utc"] = now_utc().isoformat()

    logger.info(
        f"MCP Webhook received: {pair} {payload['direction']} "
        f"score={payload.get('breakout_score')} pattern={payload.get('pattern')} "
        f"at {format_jst(now_utc())}"
    )

    signal_queue = get_queue()
    await signal_queue.put(payload)

    return {"status": "ok", "received_at": payload["received_at_utc"]}


@app.post("/webhook/mcp_context")
async def receive_mcp_context(request: Request):
    """
    tv_mcp_ea からの Market Context 更新を受信する。

    5分毎に全ペアの S/R レベル・スイング高安・EMA 方向を受け取り、
    保有中ポジションのエグジット判断に活用する。
    エントリーは発生しない（キューに入るが signal_source="mcp_context" で分岐する）。
    """
    client_ip = _get_client_ip(request)
    if _blocked_ip_cache.get(client_ip):
        raise HTTPException(status_code=429, detail="Too many invalid auth attempts")

    settings = get_settings()
    if not _is_ip_allowed(client_ip, settings):
        raise HTTPException(status_code=403, detail="IP not allowed")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    secret = settings.webhook_secret.get_secret_value()
    provided_token = str(payload.get("webhook_token", ""))
    if secret and not hmac.compare_digest(provided_token, secret):
        _record_auth_failure(client_ip)
        raise HTTPException(status_code=429, detail="Webhook token verification failed")

    required = ["pair", "signal_source", "current_price", "atr"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    if payload.get("signal_source") != "mcp_context":
        raise HTTPException(status_code=400, detail="Invalid signal_source")

    pair = payload["pair"]
    if pair not in ("USDJPY", "EURUSD", "GBPJPY", "XAUUSD", "GOLD"):
        raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")

    payload["received_at_utc"] = now_utc().isoformat()

    logger.debug(
        f"MCP Context received: {pair} "
        f"res={payload.get('nearest_resistance')} sup={payload.get('nearest_support')} "
        f"bias={payload.get('htf_bias')}"
    )

    signal_queue = get_queue()
    await signal_queue.put(payload)

    return {"status": "ok", "received_at": payload["received_at_utc"]}


@app.post("/webhook/tv_alert")
async def receive_tv_alert(request: Request):
    """
    TradingView アラート webhook を受信する。

    tv_mcp_ea が設定した TV アラートが発火すると、TradingView が
    アラートメッセージの JSON 部分を body として POST する。
    ローカルの tv_mcp_ea からの直接 POST にも対応する。

    期待されるメッセージ形式:
    [MCP-EA] XAUUSD long @ 3050.00000
    {"signal_source": "tv_alert", "pair": "XAUUSD", "direction": "long", ...}
    """
    client_ip = _get_client_ip(request)
    if _blocked_ip_cache.get(client_ip):
        raise HTTPException(status_code=429, detail="Too many invalid auth attempts")

    settings = get_settings()

    try:
        body = await request.body()
        text = body.decode("utf-8").strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid body")

    # TV アラートは plain text で送信される場合がある — JSON 部分を抽出
    payload = None
    import json as _json

    # Case 1: 全体が JSON
    try:
        payload = _json.loads(text)
    except _json.JSONDecodeError:
        pass

    # Case 2: 複数行で 2行目以降に JSON
    if payload is None:
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = _json.loads(line)
                    break
                except _json.JSONDecodeError:
                    continue

    if payload is None:
        raise HTTPException(status_code=400, detail="No JSON payload found in alert message")

    # トークン認証（ローカル POST の場合）
    secret = settings.webhook_secret.get_secret_value()
    provided_token = str(payload.get("webhook_token", ""))
    if secret and provided_token:
        if not hmac.compare_digest(provided_token, secret):
            _record_auth_failure(client_ip)
            raise HTTPException(status_code=429, detail="Webhook token verification failed")

    if payload.get("signal_source") not in ("tv_alert", "exit_alert"):
        raise HTTPException(status_code=400, detail="Invalid signal_source for tv_alert endpoint")

    required = ["pair", "direction", "pattern_level", "atr"]
    missing = [f for f in required if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    pair = payload["pair"]
    if pair not in ("USDJPY", "EURUSD", "GBPJPY", "XAUUSD", "GOLD"):
        raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")

    # breakout_score がない場合のデフォルト（TV アラートには score がない）
    payload.setdefault("breakout_score", 7)
    payload.setdefault("close", payload["pattern_level"])
    # exit_alert の場合は signal_source をそのまま維持する
    if payload.get("signal_source") != "exit_alert":
        payload["signal_source"] = "tv_alert"
    payload["received_at_utc"] = now_utc().isoformat()

    source_label = payload["signal_source"]
    logger.info(
        f"TV Alert received [{source_label}]: {pair} {payload['direction']} "
        f"@ {payload.get('pattern_level')} pattern={payload.get('pattern')}"
    )

    signal_queue = get_queue()
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


@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon():
    """iOS/Safari の自動アイコン探索を 204 で処理して 404 ノイズを抑制。"""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_touch_icon_precomposed():
    """iOS の precomposed アイコン探索を 204 で処理してログを静かにする。"""
    from fastapi.responses import Response
    return Response(status_code=204)


def _get_client_ip(request: Request) -> str:
    """プロキシ環境を考慮してクライアントIPを取得する。"""
    settings = get_settings()
    direct_ip = request.client.host if request.client and request.client.host else "unknown"

    xff = request.headers.get("x-forwarded-for", "")
    if xff and _is_trusted_proxy(direct_ip, settings):
        first_ip = xff.split(",")[0].strip()
        if first_ip:
            return first_ip
    if direct_ip:
        return direct_ip
    return "unknown"


def _is_ip_allowed(client_ip: str, settings) -> bool:
    """allowlist が設定されている場合のみクライアントIPを制限する。"""
    allowed_networks = settings.webhook_allowed_networks
    if not allowed_networks:
        return True
    try:
        ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(ip_obj in network for network in allowed_networks)


def _is_trusted_proxy(client_ip: str, settings) -> bool:
    trusted_networks = settings.webhook_trusted_proxy_networks
    if not trusted_networks:
        return False
    try:
        ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(ip_obj in network for network in trusted_networks)


def _record_auth_failure(client_ip: str) -> None:
    """トークン認証失敗を記録し、閾値超過で一時ブロックする。"""
    fail_count = int(_failed_token_counts.get(client_ip, 0)) + 1
    _failed_token_counts[client_ip] = fail_count

    if fail_count >= _FAIL_THRESHOLD:
        _blocked_ip_cache[client_ip] = True
        logger.critical(
            f"Webhook auth failure threshold exceeded: "
            f"{client_ip} ({fail_count} failures in {_FAIL_WINDOW_SEC}s)"
        )
