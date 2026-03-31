"""
India Credit Card Recommender — ML Training Pipeline
Models: Logistic Regression, Random Forest, Content-Based Filtering
Evaluation: Accuracy, Precision, Recall, F1, NDCG@5, MAP@5
"""

import pandas as pd
import numpy as np
import pickle, json, os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, roc_auc_score,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Paths — relative to this file, works on Windows / Mac / Linux ─────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(HERE)          # project root
DATA_DIR   = os.path.join(ROOT, 'data')
MODELS_DIR = HERE                            # models/ is where this script lives

# ── 1. Load Data ──────────────────────────────────────────────────────────────
cards        = pd.read_csv(os.path.join(DATA_DIR, 'cards.csv'))
interactions = pd.read_csv(os.path.join(DATA_DIR, 'user_interactions.csv'))

print("=" * 60)
print("INDIA CREDIT CARD RECOMMENDER — MODEL TRAINING")
print("=" * 60)
print(f"Cards in dataset : {len(cards)}")
print(f"Interaction rows : {len(interactions):,}")
print(f"Unique users     : {interactions['user_id'].nunique()}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
le_cat  = LabelEncoder()
le_perk = LabelEncoder()
le_fee  = LabelEncoder()

interactions['cat_enc']  = le_cat.fit_transform(interactions['spend_category'])
interactions['perk_enc'] = le_perk.fit_transform(interactions['pref_perk'])
interactions['fee_enc']  = le_fee.fit_transform(interactions['fee_preference'])
interactions['spend_log'] = np.log1p(interactions['monthly_spend'])

FEATURES = [
    'cat_enc','perk_enc','fee_enc','spend_log','age',
    'card_annual_fee','card_best_reward_rate','card_has_lounge',
    'card_is_cashback','card_lifetime_free','card_forex_low',
    'card_segment_encoded','card_rating',
]

X = interactions[FEATURES].values
y = interactions['liked'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 3. Train Models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42),
}

results = {}
print("\n── Model Training Results ─────────────────────────────────")

for name, model in models.items():
    X_tr = X_train_s if name == "Logistic Regression" else X_train
    X_te = X_test_s  if name == "Logistic Regression" else X_test

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv,
                                 scoring='f1', n_jobs=-1)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()

    results[name] = dict(accuracy=acc, precision=prec, recall=rec,
                         f1=f1, auc_roc=auc, cv_f1_mean=cv_mean, cv_f1_std=cv_std)

    print(f"\n{name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  CV F1     : {cv_mean:.4f} ± {cv_std:.4f}")

# ── 4. NDCG@5 and MAP@5 ───────────────────────────────────────────────────────
def ndcg_at_k(rel, k=5):
    rel = np.array(rel[:k], dtype=float)
    if not rel.any(): return 0.0
    dcg  = np.sum(rel / np.log2(np.arange(2, len(rel)+2)))
    idcg = np.sum(np.sort(rel)[::-1] / np.log2(np.arange(2, len(rel)+2)))
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(rel, k=5):
    rel = np.array(rel[:k])
    if not rel.any(): return 0.0
    precisions = [rel[:i+1].mean() for i in range(k) if rel[i]]
    return np.mean(precisions) if precisions else 0.0

# Use RF for ranking eval (best model)
best_model = models["Random Forest"]
X_all_s = scaler.transform(X) if False else X  # RF doesn't need scaling
probs = best_model.predict_proba(X)[:, 1]
interactions['pred_prob'] = probs

ndcg_scores, map_scores = [], []
for uid in interactions['user_id'].unique()[:200]:
    u_df = interactions[interactions['user_id'] == uid].sort_values('pred_prob', ascending=False)
    rel = u_df['liked'].values
    ndcg_scores.append(ndcg_at_k(rel, k=5))
    map_scores.append(map_at_k(rel, k=5))

ndcg5 = np.mean(ndcg_scores)
map5  = np.mean(map_scores)
results['Random Forest']['ndcg_at_5'] = ndcg5
results['Random Forest']['map_at_5']  = map5

print(f"\n── Ranking Metrics (Random Forest) ──────────────────────")
print(f"  NDCG@5 : {ndcg5:.4f}")
print(f"  MAP@5  : {map5:.4f}")

# ── 5. Content-Based Filtering ────────────────────────────────────────────────
cards['content'] = (
    cards['bank'] + ' ' + cards['name'] + ' ' +
    cards['reward_type'] + ' ' + cards['best_for'] + ' ' +
    cards['segment'] + ' ' + cards['best_category']
)
tfidf   = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(cards['content'])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(card_name, n=5):
    idx = cards[cards['name'] == card_name].index
    if len(idx) == 0: return []
    i = idx[0]
    sim_scores = list(enumerate(cosine_sim[i]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    card_indices = [s[0] for s in sim_scores]
    return cards.iloc[card_indices][['bank','name','best_reward_rate','annual_fee']].to_dict('records')

print("\n── Content-Based: Similar to 'Cashback SBI Credit Card' ──")
recs = get_content_recommendations('Cashback SBI Credit Card', n=5)
for r in recs:
    print(f"  → {r['bank']} | {r['name']} | {r['best_reward_rate']}% | ₹{r['annual_fee']}")

# ── 6. Feature Importance (RF) ────────────────────────────────────────────────
feat_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n── Feature Importances (Random Forest) ──────────────────")
for _, row in feat_imp.iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

# ── 7. Save Everything ────────────────────────────────────────────────────────
pickle.dump(best_model, open(os.path.join(MODELS_DIR, 'rf_model.pkl'),    'wb'))
pickle.dump(models["Logistic Regression"], open(os.path.join(MODELS_DIR, 'lr_model.pkl'),  'wb'))
pickle.dump(scaler,     open(os.path.join(MODELS_DIR, 'scaler.pkl'),      'wb'))
pickle.dump(tfidf,      open(os.path.join(MODELS_DIR, 'tfidf.pkl'),       'wb'))
pickle.dump(cosine_sim, open(os.path.join(MODELS_DIR, 'cosine_sim.pkl'),  'wb'))
pickle.dump({'cat':le_cat,'perk':le_perk,'fee':le_fee},
            open(os.path.join(MODELS_DIR, 'encoders.pkl'), 'wb'))

# Save results JSON
with open(os.path.join(MODELS_DIR, 'eval_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

feat_imp.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)
cards.to_csv(os.path.join(DATA_DIR, 'cards_features.csv'), index=False)

print("\n✅ All models & artefacts saved.")
print(f"   rf_model.pkl | lr_model.pkl | scaler.pkl | tfidf.pkl | cosine_sim.pkl")
