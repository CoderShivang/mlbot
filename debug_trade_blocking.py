#!/usr/bin/env python3
"""
Debug script to trace why trades are being blocked

Simulates the decision flow to identify where trades get rejected
"""

print("\n" + "="*80)
print("TRADE BLOCKING DIAGNOSTIC")
print("="*80 + "\n")

print("Simulating ML prediction logic:")
print("-" * 80)

# Simulate various ML prediction scenarios
scenarios = [
    {"success_prob": 0.70, "confidence": 0.75, "name": "High quality trade"},
    {"success_prob": 0.65, "confidence": 0.70, "name": "Good trade"},
    {"success_prob": 0.60, "confidence": 0.65, "name": "Borderline trade (at threshold)"},
    {"success_prob": 0.55, "confidence": 0.60, "name": "Below success threshold"},
    {"success_prob": 0.50, "confidence": 0.50, "name": "50/50 trade"},
    {"success_prob": 0.45, "confidence": 0.50, "name": "Low success prob"},
    {"success_prob": 0.40, "confidence": 0.45, "name": "Very low success prob"},
    {"success_prob": 0.35, "confidence": 0.40, "name": "Extremely low"},
    {"success_prob": 0.30, "confidence": 0.35, "name": "Minimum threshold"},
]

min_confidence = 0.30  # Default from ResearchBackedTrendBot

print(f"\nConfiguration:")
print(f"  min_confidence (from ResearchBackedTrendBot): {min_confidence:.0%}")
print(f"  ML model success_prob threshold: 60%")
print(f"  should_enter_trade local threshold: 30%")
print()

for scenario in scenarios:
    success_prob = scenario["success_prob"]
    confidence = scenario["confidence"]
    name = scenario["name"]

    # Model decision (from model_factory.py line 240-243)
    ml_should_trade = (
        success_prob >= 0.60 and  # ML model threshold
        confidence >= min_confidence
    )

    # should_enter_trade decision (trend_strategy_v2.py line 409-413)
    passes_local_check = success_prob >= 0.30
    final_decision = passes_local_check and ml_should_trade

    # Determine rejection reason
    if not passes_local_check:
        reason = f"Rejected at local check: {success_prob:.0%} < 30%"
    elif not ml_should_trade:
        if success_prob < 0.60:
            reason = f"Rejected by ML: success_prob {success_prob:.0%} < 60%"
        elif confidence < min_confidence:
            reason = f"Rejected by ML: confidence {confidence:.0%} < {min_confidence:.0%}"
        else:
            reason = "Rejected by ML: unknown reason"
    else:
        reason = "✅ TRADE APPROVED"

    status = "✅" if final_decision else "❌"
    print(f"{status} {name:30s} | Prob: {success_prob:.0%} | Conf: {confidence:.0%} | {reason}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("""
CRITICAL ISSUE FOUND: Mismatched thresholds!

Location 1: trend_strategy_v2.py line 409
  if ml_prediction['success_probability'] < 0.30:  # Recently lowered
      return None

Location 2: model_factory.py line 241
  should_trade = (
      success_prob >= 0.60 and  # STILL AT 60%! ⚠️
      confidence >= self.min_confidence
  )

The Problem:
- Thresholds were lowered in trend_strategy_v2.py (0.45 → 0.30)
- But NOT lowered in model_factory.py (still at 0.60)
- Even if a trade passes the 30% local check, it gets rejected at 60%

This means:
- Trades need 60% predicted success to pass (not 30% as intended)
- This is WAY too conservative for trend following
- Result: 0 trades (or very few)

Solution:
Change model_factory.py line 241:
  FROM: success_prob >= 0.60
  TO:   success_prob >= 0.40  (or lower)

Expected Impact:
- Trades with 40-60% success prob will now be allowed
- Should increase trade frequency by 10-30x
- More statistical significance with more trades
""")
print("="*80 + "\n")
