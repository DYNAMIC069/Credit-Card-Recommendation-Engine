"""
Data Validation Pipeline — India Credit Card Dataset
Uses Pydantic for schema validation + custom business rule checks
"""
import pandas as pd
import numpy as np
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json, warnings
warnings.filterwarnings('ignore')

# ── Pydantic Schema ─────────────────────────────────────────────────────────────
VALID_NETWORKS  = {'Visa','Mastercard','RuPay','Amex','Diners Club','Visa/Mastercard',
                   'RuPay/Mastercard','Mastercard/Visa','RuPay/Visa'}
VALID_SEGMENTS  = {'Entry','Entry-Mid','Mid','Mid-Premium','Premium','Super Premium'}
VALID_REWARDS   = {'Cashback','Reward Points','NeuCoins','Air Miles','KrisFlyer Miles',
                   'EDGE Miles','Fuel Points','Scapia Coins','Membership Rewards',
                   'Travel Credits','Reward Points/Cashback'}

class CreditCard(BaseModel):
    bank:             str
    name:             str
    network:          str
    joining_fee:      float
    annual_fee:       float
    reward_type:      str
    base_reward_rate: float
    best_reward_rate: float
    best_category:    str
    lounge_domestic:  bool
    lounge_international: bool
    forex_markup:     float
    min_income_lpa:   float
    segment:          str
    rating:           float
    best_for:         str
    lifetime_free:    bool

    @field_validator('network')
    @classmethod
    def check_network(cls, v):
        if v not in VALID_NETWORKS:
            raise ValueError(f"Unknown network: {v}")
        return v

    @field_validator('segment')
    @classmethod
    def check_segment(cls, v):
        if v not in VALID_SEGMENTS:
            raise ValueError(f"Unknown segment: {v}")
        return v

    @field_validator('rating')
    @classmethod
    def check_rating(cls, v):
        if not (0 <= v <= 5):
            raise ValueError(f"Rating must be 0-5, got {v}")
        return v

    @field_validator('forex_markup')
    @classmethod
    def check_forex(cls, v):
        if not (0 <= v <= 5):
            raise ValueError(f"Forex markup must be 0-5%, got {v}")
        return v

    @field_validator('base_reward_rate','best_reward_rate')
    @classmethod
    def check_reward_rates(cls, v):
        if not (0 <= v <= 100):
            raise ValueError(f"Reward rate {v}% is out of range 0-100")
        return v

    @model_validator(mode='after')
    def check_fee_logic(self):
        if self.lifetime_free and self.annual_fee > 0:
            raise ValueError("lifetime_free=True but annual_fee > 0")
        if self.best_reward_rate < self.base_reward_rate:
            raise ValueError("best_reward_rate must be >= base_reward_rate")
        if self.lounge_international and not self.lounge_domestic:
            raise ValueError("Card has intl lounge but not domestic — unlikely")
        return self

# ── Business Rules ──────────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    rule:     str
    passed:   bool
    details:  str
    severity: str  # 'ERROR' | 'WARNING' | 'INFO'

def run_business_rules(df: pd.DataFrame) -> list:
    results = []

    # Rule 1: No duplicate cards
    dupes = df[df.duplicated(subset=['bank','name'], keep=False)]
    results.append(ValidationResult(
        "No Duplicate Cards", len(dupes) == 0,
        f"{len(dupes)} duplicate rows found" if len(dupes) else "All cards are unique",
        "ERROR"
    ))

    # Rule 2: Fee consistency
    mismatch = df[(df['lifetime_free']==True) & (df['annual_fee'] > 0)]
    results.append(ValidationResult(
        "Fee Consistency", len(mismatch) == 0,
        f"{len(mismatch)} lifetime-free cards have non-zero fee" if len(mismatch) else "Fee data consistent",
        "ERROR"
    ))

    # Rule 3: Reward rate ordering
    bad_rate = df[df['best_reward_rate'] < df['base_reward_rate']]
    results.append(ValidationResult(
        "Reward Rate Ordering", len(bad_rate) == 0,
        f"{len(bad_rate)} cards have best_rate < base_rate" if len(bad_rate) else "All reward rates ordered correctly",
        "ERROR"
    ))

    # Rule 4: Rating range
    bad_rating = df[(df['rating'] < 0) | (df['rating'] > 5)]
    results.append(ValidationResult(
        "Rating Range (0-5)", len(bad_rating) == 0,
        f"{len(bad_rating)} cards with invalid ratings" if len(bad_rating) else "All ratings in valid range",
        "ERROR"
    ))

    # Rule 5: Forex markup range
    bad_fx = df[(df['forex_markup'] < 0) | (df['forex_markup'] > 5)]
    results.append(ValidationResult(
        "Forex Markup Range (0-5%)", len(bad_fx) == 0,
        f"{len(bad_fx)} cards with invalid forex" if len(bad_fx) else "All forex values valid",
        "ERROR"
    ))

    # Rule 6: Premium cards should have lounge
    premium = df[df['segment'].isin(['Premium','Super Premium'])]
    no_lounge = premium[~premium['lounge_domestic']]
    results.append(ValidationResult(
        "Premium Cards Have Lounge", len(no_lounge) == 0,
        f"{len(no_lounge)} premium cards without domestic lounge: {list(no_lounge['name'][:3])}" if len(no_lounge) else "All premium cards have lounge",
        "WARNING"
    ))

    # Rule 7: Min income LPA reasonable
    bad_income = df[df['min_income_lpa'] > 100]
    results.append(ValidationResult(
        "Min Income LPA Reasonable (<=100)", len(bad_income) == 0,
        f"{len(bad_income)} cards with unusually high min income (invite-only cards)" if len(bad_income) else "All income requirements reasonable",
        "INFO"
    ))

    # Rule 8: Completeness — no nulls in critical cols
    critical = ['bank','name','annual_fee','best_reward_rate','rating','segment']
    null_count = df[critical].isnull().sum().sum()
    results.append(ValidationResult(
        "No Nulls in Critical Columns", null_count == 0,
        f"{null_count} nulls found in critical columns" if null_count else "No missing values in critical columns",
        "ERROR"
    ))

    # Rule 9: Banks have at least 1 card
    bank_counts = df['bank'].value_counts()
    results.append(ValidationResult(
        "Bank Coverage", len(bank_counts) >= 10,
        f"Only {len(bank_counts)} banks in dataset" if len(bank_counts) < 10 else f"{len(bank_counts)} banks covered",
        "INFO"
    ))

    # Rule 10: Reward rate correlation check
    corr = df['annual_fee'].corr(df['best_reward_rate'])
    results.append(ValidationResult(
        "Fee-Reward Rate Positive Correlation", corr > 0,
        f"Fee-reward correlation = {corr:.3f} (expected positive)" if corr <= 0 else f"Fee-reward correlation = {corr:.3f} (healthy)",
        "WARNING"
    ))

    return results

# ── Run Validation ──────────────────────────────────────────────────────────────
def validate_dataset(path: str):
    print("=" * 65)
    print("  DATA VALIDATION REPORT — India Credit Card Dataset")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    df = pd.read_csv(path)
    print(f"\n✦ Loaded {len(df)} rows × {len(df.columns)} columns\n")

    # Schema validation
    print("── 1. Pydantic Schema Validation ────────────────────────────")
    schema_errors, schema_ok = [], 0
    for idx, row in df.iterrows():
        try:
            CreditCard(**{k: row[k] for k in CreditCard.model_fields if k in row})
            schema_ok += 1
        except Exception as e:
            schema_errors.append(f"Row {idx} ({row.get('name','?')}): {e}")

    print(f"  ✅ Passed : {schema_ok}/{len(df)} rows")
    if schema_errors:
        print(f"  ❌ Errors : {len(schema_errors)}")
        for e in schema_errors[:5]: print(f"    {e}")
    else:
        print("  ✅ All rows pass schema validation")

    # Business rules
    print("\n── 2. Business Rule Checks ──────────────────────────────────")
    results = run_business_rules(df)
    errors = warnings_list = infos = 0
    for r in results:
        icon = "✅" if r.passed else ("❌" if r.severity=="ERROR" else "⚠️")
        print(f"  {icon} [{r.severity:7s}] {r.rule}")
        print(f"           → {r.details}")
        if not r.passed:
            if r.severity == "ERROR": errors += 1
            elif r.severity == "WARNING": warnings_list += 1
            else: infos += 1

    print(f"\n── 3. Data Quality Summary ──────────────────────────────────")
    print(f"  Total rules checked : {len(results)}")
    print(f"  Errors              : {errors}")
    print(f"  Warnings            : {warnings_list}")
    print(f"  Info                : {infos}")
    print(f"\n── 4. Statistical Profile ───────────────────────────────────")
    print(f"  Cards              : {len(df)}")
    print(f"  Banks              : {df['bank'].nunique()}")
    print(f"  Lifetime Free      : {df['lifetime_free'].sum()} ({df['lifetime_free'].mean():.1%})")
    print(f"  Dom. Lounge        : {df['lounge_domestic'].sum()}")
    print(f"  Intl. Lounge       : {df['lounge_international'].sum()}")
    print(f"  Null values        : {df.isnull().sum().sum()}")
    print(f"  Avg Annual Fee     : ₹{df['annual_fee'].mean():.0f}")
    print(f"  Avg Best RR        : {df['best_reward_rate'].mean():.2f}%")
    print(f"  Avg Rating         : {df['rating'].mean():.2f}/5")

    # Save report
    report = {
        "run_at": datetime.now().isoformat(),
        "dataset": path,
        "total_rows": len(df),
        "schema_pass": schema_ok,
        "schema_errors": len(schema_errors),
        "business_rules": [
            {"rule": r.rule, "passed": r.passed,
             "details": r.details, "severity": r.severity}
            for r in results
        ],
        "overall_status": "PASS" if errors == 0 else "FAIL",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_report.json')
    with open(out,'w') as f: json.dump(report, f, indent=2, default=str)
    print(f"\n  📄 Full report saved → {out}")
    print(f"\n  Overall Status: {'✅ PASS' if errors==0 else '❌ FAIL'}")
    print("=" * 65)
    return report

if __name__ == "__main__":
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(HERE, '..', 'data', 'cards_features.csv')
    validate_dataset(os.path.normpath(data_path))
