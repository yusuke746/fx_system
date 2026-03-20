import argparse
import json
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv


def build_payload(pair: str, direction: str, token: str) -> dict:
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "pair": pair,
        "direction": direction,
        "webhook_token": token,
        "fvg_4h_zone_active": True,
        "ob_4h_zone_active": True,
        "liq_sweep_1h": True,
        "liq_sweep_qualified": True,
        "bos_1h": False,
        "choch_1h": True,
        "msb_15m_confirmed": True,
        "mtf_confluence": 4,
        "atr": 0.12,
        "close": 149.85,
        "atr_ratio": 1.15,
        "bb_width": 0.85,
        "rsi_14": 42.3,
        "rsi_zone": 0,
        "macd_histogram": 0.00021,
        "macd_signal_cross": 1,
        "stoch_k": 54.2,
        "stoch_d": 48.7,
        "momentum_3bar": 0.08,
        "close_vs_ema20_4h": -0.12,
        "close_vs_ema50_4h": 0.03,
        "high_low_range_15m": 0.09,
        "tp_swing_pips": 28.5,
        "tp_fvg_pips": 35.0,
        "ob_4h_distance_pips": 3.2,
        "fvg_4h_fill_ratio": 0.32,
        "liq_sweep_strength": 0.47,
        "prior_candle_body_ratio": 0.61,
        "consecutive_same_dir": 2,
        "pivot_proximity": 0.06,
        "sweep_pending_bars": 3,
        "sent_at_utc": now_iso,
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="FX webhook test sender")
    parser.add_argument("--url", default="http://127.0.0.1:80/webhook", help="Webhook URL")
    parser.add_argument("--pair", default="USDJPY", choices=["USDJPY", "EURUSD", "GBPJPY"])
    parser.add_argument("--direction", default="long", choices=["long", "short"])
    parser.add_argument("--token", default=os.getenv("WEBHOOK_SECRET", ""), help="Webhook token")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    payload = build_payload(args.pair, args.direction, args.token)

    print("=== Request ===")
    print(f"URL: {args.url}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    response = requests.post(args.url, json=payload, timeout=args.timeout)

    print("\n=== Response ===")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)


if __name__ == "__main__":
    main()
