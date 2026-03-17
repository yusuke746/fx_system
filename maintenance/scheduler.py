"""
自動メンテナンスモジュール

■ スケジュール
  毎日 01:00 JST:
    - ログローテーション（30日以前を圧縮）
    - sentiment キャッシュ切り詰め
    - DB バックアップ（直近7日保持）
    - 翌日カレンダーキャッシュ更新
    - Discord にポジションサマリー送信

  毎週日曜 23:00 JST:
        - training_samples クリーンアップ（label済み180日超を削除）
    - signals アーカイブ移動
    - VACUUM ANALYZE
    - api_call_log 週次集計
    - 週次レポート Discord 送信

  毎月月初 日曜 02:00 JST:
    - 古いモデルファイル削除（3世代保持）
    - trades テーブル月次集計
    - クラウドバックアップ
    - GPT-5.2 コストレポート

■ 時刻基準
  - スケジュール判定: JST（表示基準）で指定した時刻を UTC に変換して実行
  - ログ・Discord通知: 表示基準（JST）
  - DB操作: 保存基準（UTC）
"""

import gzip
import shutil
import sqlite3
from datetime import timedelta
from pathlib import Path

from loguru import logger

from core.database import check_integrity
from core.notifier import DiscordNotifier
from core.time_manager import now_utc, format_jst, is_broker_market_closed, broker_day_start_utc
from ml.retraining import run_weekly_retraining


async def daily_maintenance(
    db_conn: sqlite3.Connection,
    notifier: DiscordNotifier,
    log_dir: str,
    db_backup_path: str,
    calendar_veto=None,
) -> None:
    """毎日深夜01:00 JST（=16:00 UTC前日）実行のメンテナンスタスク。"""
    logger.info("=== Daily maintenance started ===")
    now = now_utc()

    # 1. ログローテーション（30日以前を圧縮）
    _compress_old_logs(log_dir, days=30)

    # 2. DB バックアップ（直近7日保持）
    _backup_db(db_conn, db_backup_path, keep_days=7)

    # 3. DB 整合性チェック
    if not check_integrity(db_conn):
        await notifier.send_critical("DB整合性チェック失敗！バックアップからの復元を検討してください。")

    # 4. カレンダーキャッシュ更新
    if calendar_veto:
        try:
            await calendar_veto.fetch_events_async()
            logger.info("Calendar cache updated")
        except Exception as e:
            await notifier.send_alert(f"カレンダー更新失敗: {e}")

    # 5. ポジションサマリー送信
    row = db_conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_jpy), 0) as total "
        "FROM trades WHERE close_time >= ?",
        ((now - timedelta(days=1)).isoformat(),),
    ).fetchone()

    await notifier.send(
        f"[日次サマリー] {format_jst(now)}\n"
        f"過去24時間: {row['cnt']}トレード / P&L: ¥{row['total']:,.0f}"
    )

    logger.info("=== Daily maintenance completed ===")


async def weekly_maintenance(
    db_conn: sqlite3.Connection,
    notifier: DiscordNotifier,
) -> None:
    """毎週日曜23:00 JST 実行の週次メンテナンスタスク。"""
    logger.info("=== Weekly maintenance started ===")
    now = now_utc()

    # 0. LightGBM 再学習（未ラベル→ラベル付け→再学習）
    # MT5 API 競合を避けるため、市場クローズ時のみ実行する。
    if is_broker_market_closed(now):
        retrain_result = run_weekly_retraining(db_conn)
    else:
        retrain_result = {
            "labeling": {"labeled": 0, "skipped": 0, "error": "skipped_market_open"},
            "retrain": {"trained_pairs": [], "skipped_pairs": {}, "validation": {}},
            "run_at_utc": now.isoformat(),
        }
        logger.warning("Weekly retraining skipped because broker market is open")

    # 1. training_samples クリーンアップ（label済み180日超）
    training_cutoff = (now - timedelta(days=180)).isoformat()
    cur = db_conn.execute(
        "DELETE FROM training_samples WHERE label IS NOT NULL AND signal_time < ?",
        (training_cutoff,),
    )
    deleted_training_samples = cur.rowcount if cur.rowcount is not None else 0
    db_conn.commit()
    logger.info(f"Old labeled training_samples cleaned up: {deleted_training_samples}")

    # 2. signals アーカイブ移動（30日以前）
    cutoff = (now - timedelta(days=30)).isoformat()
    db_conn.execute("DELETE FROM signals WHERE signal_time < ?", (cutoff,))
    db_conn.commit()
    logger.info("Old signals cleaned up")

    # 3. VACUUM ANALYZE
    db_conn.execute("VACUUM")
    logger.info("DB VACUUM completed")

    # 4. api_call_log 週次集計
    # 実行遅延による窓ズレを防ぐため、ブローカー日付の固定境界で集計する。
    week_end_utc = broker_day_start_utc(now)   # 当日00:00（broker）
    week_start_utc = week_end_utc - timedelta(days=7)
    week_start = week_start_utc.isoformat()
    week_end = week_end_utc.isoformat()
    row = db_conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(cost_usd), 0) as total_cost "
        "FROM api_call_log WHERE call_time >= ? AND call_time < ?",
        (week_start, week_end),
    ).fetchone()

    # 5. 週次レポート
    trades_row = db_conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_jpy), 0) as total_pnl, "
        "COALESCE(AVG(pnl_pips), 0) as avg_pips "
        "FROM trades WHERE close_time >= ? AND close_time < ?",
        (week_start, week_end),
    ).fetchone()

    win_trades = db_conn.execute(
        "SELECT COUNT(*) as cnt FROM trades WHERE close_time >= ? AND close_time < ? AND pnl_pips > 0",
        (week_start, week_end),
    ).fetchone()

    total_trades = trades_row["cnt"]
    winrate = (win_trades["cnt"] / total_trades * 100) if total_trades > 0 else 0

    report = (
        f"[週次レポート] {format_jst(now)}\n"
        f"トレード数: {total_trades}\n"
        f"勝率: {winrate:.1f}%\n"
        f"平均P&L: {trades_row['avg_pips']:.1f} pips\n"
        f"合計P&L: ¥{trades_row['total_pnl']:,.0f}\n"
        f"GPT API呼び出し: {row['cnt']}回 / ${row['total_cost']:.2f}\n"
        f"training_samples削除(180日超・label済): {deleted_training_samples}\n"
        f"再学習: labeled={retrain_result['labeling']['labeled']} / "
        f"trained={len(retrain_result['retrain']['trained_pairs'])}"
    )

    await notifier.send(report)
    logger.info("=== Weekly maintenance completed ===")


async def monthly_maintenance(
    db_conn: sqlite3.Connection,
    notifier: DiscordNotifier,
    model_dir: str,
) -> None:
    """毎月月初 日曜02:00 JST 実行の月次メンテナンスタスク。"""
    logger.info("=== Monthly maintenance started ===")
    now = now_utc()

    # 1. 古いモデルファイル削除（3世代保持）
    _cleanup_models(model_dir, keep_per_pair=3)

    # 2. 月次集計
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    prev_month_start = (month_start - timedelta(days=1)).replace(day=1)

    row = db_conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_jpy), 0) as total_pnl "
        "FROM trades WHERE close_time >= ? AND close_time < ?",
        (prev_month_start.isoformat(), month_start.isoformat()),
    ).fetchone()

    # 3. GPT-5.2 コストレポート
    cost_row = db_conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) as total_cost, COUNT(*) as cnt "
        "FROM api_call_log WHERE call_time >= ? AND call_time < ?",
        (prev_month_start.isoformat(), month_start.isoformat()),
    ).fetchone()

    report = (
        f"[月次レポート] {format_jst(now)}\n"
        f"前月トレード数: {row['cnt']}\n"
        f"前月P&L: ¥{row['total_pnl']:,.0f}\n"
        f"GPT-5.2コスト: ${cost_row['total_cost']:.2f} ({cost_row['cnt']}回)"
    )

    await notifier.send(report)
    logger.info("=== Monthly maintenance completed ===")


def _compress_old_logs(log_dir: str, days: int = 30) -> None:
    """N日以前のログを gzip 圧縮する。"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return

    cutoff = now_utc() - timedelta(days=days)
    for f in log_path.glob("*.log"):
        try:
            mtime = f.stat().st_mtime
            from datetime import datetime, timezone
            file_time = datetime.fromtimestamp(mtime, tz=timezone.utc)
            if file_time < cutoff:
                with open(f, "rb") as f_in:
                    with gzip.open(f"{f}.gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                f.unlink()
                logger.debug(f"Compressed old log: {f}")
        except OSError as e:
            logger.warning(f"Log compression failed for {f}: {e}")


def _backup_db(db_conn: sqlite3.Connection, backup_dir: str, keep_days: int = 7) -> None:
    """DB をバックアップし、古いバックアップを削除する。"""
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    timestamp = now_utc().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_path / f"trading_{timestamp}.db"

    try:
        # SQLite のオンラインバックアップ
        backup_conn = sqlite3.connect(str(backup_file))
        db_conn.backup(backup_conn)
        backup_conn.close()
        logger.info(f"DB backup created: {backup_file}")
    except Exception as e:
        logger.error(f"DB backup failed: {e}")
        return

    # 古いバックアップ削除
    cutoff = now_utc() - timedelta(days=keep_days)
    for f in sorted(backup_path.glob("trading_*.db")):
        try:
            mtime = f.stat().st_mtime
            from datetime import datetime, timezone
            file_time = datetime.fromtimestamp(mtime, tz=timezone.utc)
            if file_time < cutoff:
                f.unlink()
                logger.debug(f"Deleted old backup: {f}")
        except OSError:
            pass


def _cleanup_models(model_dir: str, keep_per_pair: int = 3) -> None:
    """古いモデルファイルを削除する（ペアごとに N 世代保持）。"""
    model_path = Path(model_dir)
    if not model_path.exists():
        return

    for pair in ["USDJPY", "EURUSD", "GBPJPY"]:
        pattern = f"lgbm_{pair}_*.pkl"
        files = sorted(model_path.glob(pattern))
        if len(files) > keep_per_pair:
            for old in files[:-keep_per_pair]:
                old.unlink()
                logger.info(f"Deleted old model: {old}")
