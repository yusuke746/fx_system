"""
時刻管理モジュール — 保存基準・比較基準・表示基準の分離

■ 時刻基準の定義
  ┌─────────────┬──────────┬────────────────────────────────────────┐
  │ 基準         │ タイムゾーン │ 用途                                     │
  ├─────────────┼──────────┼────────────────────────────────────────┤
  │ 保存基準     │ UTC        │ DB保存・ログ記録・API通信のタイムスタンプ │
  │ 比較基準     │ UTC        │ 時間差計算・キャッシュ有効期限・Veto判定  │
  │ 表示基準     │ JST (UTC+9)│ Discord通知・ダッシュボード・ユーザー向け │
  │ ブローカー基準│ MT5サーバー │ MT5注文時刻・チャート時刻（EET: UTC+2/+3）│
  └─────────────┴──────────┴────────────────────────────────────────┘

■ ブローカー時刻について
  XMTrading MT5 サーバー時刻は EET（東欧時間）:
    - 冬時間: UTC+2 (11月第1日曜〜3月最終日曜)
    - 夏時間: UTC+3 (3月最終日曜〜11月第1日曜)
  MT5から取得した時刻は必ず to_utc() で UTC に変換してから保存・比較する。
"""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# ── タイムゾーン定数 ──────────────────────────────────
UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")             # UTC+9（固定、DST なし）
EET = ZoneInfo("Europe/Helsinki")         # XMTrading MT5 サーバー時刻（UTC+2 / UTC+3）


# ── 保存基準: 現在時刻を UTC で取得 ──────────────────
def now_utc() -> datetime:
    """保存基準・比較基準に使用する現在時刻（UTC aware）。"""
    return datetime.now(UTC)


# ── 表示基準: UTC → JST 変換 ─────────────────────────
def to_jst(dt: datetime) -> datetime:
    """UTC datetime を JST（表示基準）に変換する。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(JST)


def format_jst(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S JST") -> str:
    """UTC datetime を JST 文字列に変換して表示用に返す。"""
    return to_jst(dt).strftime(fmt)


# ── ブローカー基準: MT5 サーバー時刻 ↔ UTC ──────────
def mt5_server_to_utc(mt5_dt: datetime) -> datetime:
    """
    MT5サーバー時刻（EET: naive datetime）を UTC に変換する。

    MT5 Python API が返す datetime は timezone-naive で、サーバー側の
    ローカル時刻（EET: UTC+2 / 夏時間 UTC+3）として解釈する必要がある。
    """
    if mt5_dt.tzinfo is None:
        mt5_dt = mt5_dt.replace(tzinfo=EET)
    return mt5_dt.astimezone(UTC)


def utc_to_mt5_server(dt: datetime) -> datetime:
    """UTC datetime を MT5 サーバー時刻（EET）に変換する。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(EET)


def broker_day_start_utc(dt: datetime | None = None) -> datetime:
    """ブローカー時刻（EET）の当日00:00を UTC に変換して返す。"""
    if dt is None:
        dt = now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    broker_now = utc_to_mt5_server(dt)
    broker_midnight = broker_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return broker_midnight.astimezone(UTC)


# ── 比較基準: 時間差計算ユーティリティ ────────────────
def elapsed_seconds(since: datetime) -> float:
    """since（UTC aware）からの経過秒数を返す。"""
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    return (now_utc() - since).total_seconds()


def elapsed_minutes(since: datetime) -> float:
    """since（UTC aware）からの経過分数を返す。"""
    return elapsed_seconds(since) / 60.0


def is_within_buffer(
    event_utc: datetime,
    now: datetime | None = None,
    buffer_minutes: int = 30,
) -> bool:
    """
    イベント時刻の前後 buffer_minutes 分以内かどうかを判定する。
    Calendar Veto 判定用。比較基準（UTC）で計算する。
    """
    if now is None:
        now = now_utc()
    if event_utc.tzinfo is None:
        event_utc = event_utc.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    buf = timedelta(minutes=buffer_minutes)
    return (event_utc - buf) <= now <= (event_utc + buf)


# ── セッション判定 ──────────────────────────────────
def get_session(dt: datetime | None = None) -> str:
    """
    現在の取引セッションを判定する。

    Returns:
        "london"  : ロンドン（16:00〜01:00 JST / 07:00〜16:00 UTC）
        "ny"      : ニューヨーク（21:00〜06:00 JST / 12:00〜21:00 UTC）
        "overlap" : ロンドン・NY重複（21:00〜01:00 JST / 12:00〜16:00 UTC）
        "other"   : 深夜帯（除外推奨）
    """
    if dt is None:
        dt = now_utc()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    utc_hour = dt.astimezone(UTC).hour

    london = 7 <= utc_hour < 16
    ny = 12 <= utc_hour < 21

    if london and ny:
        return "overlap"
    if london:
        return "london"
    if ny:
        return "ny"
    return "other"


def get_session_flag(dt: datetime | None = None) -> int:
    """LightGBM 特徴量用のセッションフラグ（数値）を返す。"""
    mapping = {"london": 1, "ny": 2, "overlap": 3, "other": 0}
    return mapping[get_session(dt)]


def is_excluded_hours(dt: datetime | None = None) -> bool:
    """
    深夜帯（ブローカー時刻 00:00〜07:00）でエントリーを除外すべきか判定。

    ブローカーサーバー時刻（EET/EEST）を基準にすることで、
    DST切替時でも除外時間の意図を一定に保つ。
    """
    if dt is None:
        dt = now_utc()
    broker_hour = utc_to_mt5_server(dt).hour
    return 0 <= broker_hour < 7


def is_friday_close_window(dt: datetime | None = None) -> bool:
    """金曜22:00（ブローカー時刻）以降かどうかを時間ベース強制クローズ用に判定。"""
    if dt is None:
        dt = now_utc()
    broker_dt = utc_to_mt5_server(dt)
    return broker_dt.weekday() == 4 and broker_dt.hour >= 22


def is_broker_market_closed(dt: datetime | None = None) -> bool:
    """
        ブローカー基準で市場クローズ時間かどうかを判定する（EET/EEST基準）。

        Europe/Helsinki のタイムゾーン変換を使うため、
        夏時間（EEST）/冬時間（EET）のズレは自動補正される。

    運用ルール:
            - 金曜 22:00（サーバー時刻）以降はクローズ
            - 土曜・日曜はクローズ
            - 月曜 07:00（サーバー時刻）までクローズ
    """
    if dt is None:
        dt = now_utc()
    broker_dt = utc_to_mt5_server(dt)

    wd = broker_dt.weekday()  # Mon=0 ... Sun=6
    hour = broker_dt.hour

    if wd == 4 and hour >= 22:  # Fri late
        return True
    if wd in (5, 6):            # Sat/Sun
        return True
    if wd == 0 and hour < 7:    # Mon early
        return True
    return False
