#!/usr/bin/env python
"""jpy_macro_score の3ペアへの推測可能性を分析"""

import sqlite3
from pathlib import Path
from ml.lgbm_model import FEATURE_NAMES

print("=" * 70)
print("分析: jpy_macro_score の3ペアへの効果")
print("=" * 70)

# FEATURE_NAMES 内での jpy_macro_score の位置
jpy_idx = FEATURE_NAMES.index("jpy_macro_score")
print(f"\n✓ jpy_macro_score の特徴量インデックス: {jpy_idx} / {len(FEATURE_NAMES)}")

# 各ペアの特徴量の構成
print("\n" + "=" * 70)
print("通貨ペア別の macro 特徴量")
print("=" * 70)

pairs_macro_map = {
    "USDJPY": {
        "base": "USD",
        "quote": "JPY",
        "relevant_features": ["usd_macro_score", "jpy_macro_score"],
        "relationship": "USD強 + JPY弱 = 上昇",
    },
    "EURUSD": {
        "base": "EUR",
        "quote": "USD",
        "relevant_features": ["usd_macro_score"],
        "relationship": "USD強 = 下落（JPYなし）",
        "jpy_impact": "⚠️ 間接的（理論上 JPY 情報は不要）",
    },
    "GBPJPY": {
        "base": "GBP",
        "quote": "JPY",
        "relevant_features": ["usd_macro_score", "jpy_macro_score"],
        "relationship": "JPY弱 + GBP強 = 上昇",
    },
}

for pair, info in pairs_macro_map.items():
    print(f"\n{pair}: {info['base']}/{info['quote']}")
    print(f"  方向性: {info['relationship']}")
    print(f"  関連特徴量: {', '.join(info['relevant_features'])}")
    
    if "jpy_impact" in info:
        print(f"  {info['jpy_impact']}")
    else:
        print(f"  ✓ JPY要因が重要")

# 統計分析
print("\n" + "=" * 70)
print("推測可能性の評価")
print("=" * 70)

analysis = {
    "USDJPY": {
        "rating": "✅ HIGH",
        "reason": "JPY 要因が明示的。jpy_macro_score は直接的に有効",
        "recommendation": "jpy_macro_score を通常に使用",
    },
    "EURUSD": {
        "rating": "⚠️ MEDIUM",
        "reason": "JPY 要因がない。jpy_macro_score + usd_macro_score の複合で動く可能性あり",
        "recommendation": "jpy_macro_score の重みを 50% に減衰し、usd_macro_score 主体で推測",
    },
    "GBPJPY": {
        "rating": "✅ HIGH",
        "reason": "JPY 要因が明示的。jpy_macro_score は直接的に有効",
        "recommendation": "jpy_macro_score を通常に使用",
    },
}

for pair, eval_info in analysis.items():
    print(f"\n{pair}: {eval_info['rating']}")
    print(f"  理由: {eval_info['reason']}")
    print(f"  推奨: {eval_info['recommendation']}")

# 改善案
print("\n" + "=" * 70)
print("推奨改善: コンフィグ追加")
print("=" * 70)

improvement = """
"ml": {
  ...
  "llm_macro_scaling_per_pair": {
    "USDJPY": {
      "jpy_macro_score": 1.0,      # 100% 使用（デフォルト）
      "usd_macro_score": 1.0
    },
    "EURUSD": {
      "jpy_macro_score": 0.5,      # 50% に減衰（ノイズ低減）
      "usd_macro_score": 1.0
    },
    "GBPJPY": {
      "jpy_macro_score": 1.0,      # 100% 使用（デフォルト）
      "usd_macro_score": 0.8       # GBP macro も関連
    }
  }
}

実装上は build_features(..., pair=pair) へ pair パラメータを追加し、
LLM 特徴量の値にスケーリング係数を適用する。
"""

print(improvement)

print("\n" + "=" * 70)
print("現在の実装: 全ペアに同じ jpy_macro_score を使用")
print("改善後: ペア別にスケーリングして最適化")
print("=" * 70)
