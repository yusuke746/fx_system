"""
手動ストレージ掃除CLI。

用途:
  - 日次メンテに組み込まれたアーティファクト掃除を手動実行
  - いま容量を即時確保したい時に利用

実行例:
  python -m maintenance.cleanup_storage
"""

from maintenance.scheduler import _cleanup_storage_artifacts


def main() -> None:
    result = _cleanup_storage_artifacts()
    print(
        f"deleted_files={result['deleted_files']} "
        f"freed_mb={result['freed_mb']:.2f}"
    )


if __name__ == "__main__":
    main()
