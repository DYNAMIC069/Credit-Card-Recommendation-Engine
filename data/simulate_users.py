import pandas as pd
import numpy as np
import os

np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
cards_path = os.path.join(HERE, 'cards.csv')
df = pd.read_csv(cards_path)

CATEGORIES = ['Shopping','Travel','Fuel','Food','Utility','Health']
PERKS      = ['Cashback','Rewards','Lounge','Zero Fee','Premium','Forex']
FEE_PREFS  = ['Free','Low (<1500)','Mid (1.5-5K)','Premium (5K+)']

def simulate_preference(row, spend_cat, perk_pref, fee_pref, monthly_spend):
    score = 0
    bf = str(row['best_for']).lower()
    
    # Category match
    cat_map = {'Shopping':'shopping','Travel':'travel','Fuel':'fuel',
               'Food':'food','Utility':'utility','Health':'health'}
    if cat_map.get(spend_cat,'') in bf: score += 40
    
    # Perk match
    if perk_pref == 'Cashback' and row['is_cashback']: score += 30
    if perk_pref == 'Lounge' and row['has_both_lounge']: score += 30
    elif perk_pref == 'Lounge' and row['lounge_domestic']: score += 15
    if perk_pref == 'Zero Fee' and row['lifetime_free']: score += 35
    if perk_pref == 'Forex' and row['forex_low']: score += 30
    if perk_pref == 'Premium' and row['segment_encoded'] >= 4: score += 30
    if perk_pref == 'Rewards' and not row['is_cashback'] and not row['is_miles']: score += 25
    
    # Fee match
    af = row['annual_fee']
    if fee_pref == 'Free' and af == 0: score += 25
    elif fee_pref == 'Low (<1500)' and af <= 1500: score += 20
    elif fee_pref == 'Mid (1.5-5K)' and 1500 < af <= 5000: score += 15
    elif fee_pref == 'Premium (5K+)' and af > 5000: score += 20
    
    # Reward rate bonus
    score += min(20, row['best_reward_rate'] * 0.8)
    
    # Rating
    score += (row['rating'] - 3) * 5
    
    # Monthly spend vs card segment
    if monthly_spend >= 50000 and row['segment_encoded'] >= 3: score += 10
    if monthly_spend < 15000 and row['segment_encoded'] <= 1: score += 8
    
    # Add noise
    score += np.random.normal(0, 6)
    return max(0, min(100, score))

records = []
n_users = 900

for uid in range(n_users):
    cat   = np.random.choice(CATEGORIES, p=[0.28,0.18,0.12,0.18,0.12,0.12])
    perk  = np.random.choice(PERKS,      p=[0.25,0.20,0.18,0.15,0.12,0.10])
    fee   = np.random.choice(FEE_PREFS,  p=[0.20,0.30,0.35,0.15])
    spend = np.random.choice([5000,10000,20000,35000,60000,100000],
                              p=[0.10,0.25,0.30,0.20,0.10,0.05])
    age   = np.random.randint(22, 55)
    
    scores = df.apply(lambda r: simulate_preference(r, cat, perk, fee, spend), axis=1)
    
    # Top-1 choice (clicked/applied)
    top_idx  = scores.idxmax()
    chosen   = df.loc[top_idx, 'name']
    
    # Top-5 positive interactions
    top5 = scores.nlargest(5).index.tolist()
    
    for idx, row in df.iterrows():
        liked = 1 if idx in top5 else 0
        records.append({
            'user_id': uid,
            'age': age,
            'monthly_spend': spend,
            'spend_category': cat,
            'pref_perk': perk,
            'fee_preference': fee,
            'card_name': row['name'],
            'bank': row['bank'],
            'card_annual_fee': row['annual_fee'],
            'card_best_reward_rate': row['best_reward_rate'],
            'card_has_lounge': int(row['lounge_domestic'] or row['lounge_international']),
            'card_is_cashback': row['is_cashback'],
            'card_lifetime_free': row['is_lifetime_free'],
            'card_forex_low': row['forex_low'],
            'card_segment_encoded': row['segment_encoded'],
            'card_rating': row['rating'],
            'preference_score': round(scores[idx], 2),
            'liked': liked,
            'chosen': int(row['name'] == chosen),
        })

interactions = pd.DataFrame(records)
interactions_path = os.path.join(HERE, 'user_interactions.csv')
interactions.to_csv(interactions_path, index=False)

# Also save user profiles separately
users = interactions.drop_duplicates('user_id')[['user_id','age','monthly_spend','spend_category','pref_perk','fee_preference']]
profiles_path = os.path.join(HERE, 'user_profiles.csv')
users.to_csv(profiles_path, index=False)

print(f"Generated {n_users} users × {len(df)} cards = {len(interactions):,} interaction records")
print(f"Positive interactions (liked=1): {interactions['liked'].sum():,}")
print(f"Class balance: {interactions['liked'].mean():.1%} positive")
