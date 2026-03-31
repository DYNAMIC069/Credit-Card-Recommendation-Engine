"""
India Credit Card Recommender — FastAPI Backend
Endpoints: /recommend, /cards, /cards/{id}, /similar/{name}, /health
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle, json, os

# ── Load Artefacts ─────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR   = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')

cards_df   = pd.read_csv(os.path.join(DATA_DIR,   'cards_features.csv'))
rf_model   = pickle.load(open(os.path.join(MODELS_DIR, 'rf_model.pkl'),   'rb'))
encoders   = pickle.load(open(os.path.join(MODELS_DIR, 'encoders.pkl'),   'rb'))
cosine_sim = pickle.load(open(os.path.join(MODELS_DIR, 'cosine_sim.pkl'), 'rb'))
eval_res   = json.load(open(os.path.join(MODELS_DIR,   'eval_results.json')))

app = FastAPI(
    title="India Credit Card Recommender API",
    description="""
## 🇮🇳 India Credit Card Recommender API

A machine-learning-powered REST API that recommends the best Indian credit cards 
based on user spending profile, preferences, and financial goals.

**Dataset:** 111 cards · 20 banks · scraped via Apify from CardInsider  
**Model:** Random Forest Classifier (AUC-ROC: 0.937, NDCG@5: 0.877)
    """,
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Schemas ────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    spend_category: str = Field(..., description="Primary spend: Shopping|Travel|Fuel|Food|Utility|Health")
    pref_perk: str      = Field(..., description="Preferred perk: Cashback|Rewards|Lounge|Zero Fee|Premium|Forex")
    fee_preference: str = Field(..., description="Fee tolerance: Free|Low (<1500)|Mid (1.5-5K)|Premium (5K+)")
    monthly_spend: int  = Field(20000, ge=1000, le=500000, description="Monthly spend in INR")
    age: int            = Field(28, ge=18, le=70)
    top_n: int          = Field(5, ge=1, le=20, description="Number of recommendations")

    @field_validator('spend_category')
    @classmethod
    def validate_cat(cls, v):
        valid = ['Shopping','Travel','Fuel','Food','Utility','Health']
        if v not in valid: raise ValueError(f"Must be one of {valid}")
        return v

    @field_validator('pref_perk')
    @classmethod
    def validate_perk(cls, v):
        valid = ['Cashback','Rewards','Lounge','Zero Fee','Premium','Forex']
        if v not in valid: raise ValueError(f"Must be one of {valid}")
        return v

    @field_validator('fee_preference')
    @classmethod
    def validate_fee(cls, v):
        valid = ['Free','Low (<1500)','Mid (1.5-5K)','Premium (5K+)']
        if v not in valid: raise ValueError(f"Must be one of {valid}")
        return v

class CardResponse(BaseModel):
    rank: int
    bank: str
    name: str
    network: str
    annual_fee: float
    reward_type: str
    base_reward_rate: float
    best_reward_rate: float
    best_category: str
    lounge_domestic: bool
    lounge_international: bool
    forex_markup: float
    segment: str
    rating: float
    lifetime_free: bool
    match_score: float
    reason: str

# ── Helper ─────────────────────────────────────────────────────────────────────
def build_feature_row(req: RecommendRequest, card: pd.Series) -> np.ndarray:
    try:
        cat_enc  = encoders['cat'].transform([req.spend_category])[0]
        perk_enc = encoders['perk'].transform([req.pref_perk])[0]
        fee_enc  = encoders['fee'].transform([req.fee_preference])[0]
    except Exception:
        cat_enc = perk_enc = fee_enc = 0

    return np.array([[
        cat_enc, perk_enc, fee_enc,
        np.log1p(req.monthly_spend), req.age,
        card['annual_fee'], card['best_reward_rate'],
        int(card['lounge_domestic'] or card['lounge_international']),
        card.get('is_cashback', 0), card.get('is_lifetime_free', 0),
        card.get('forex_low', 0), card.get('segment_encoded', 0),
        card['rating'],
    ]])

def generate_reason(card: pd.Series, req: RecommendRequest) -> str:
    parts = []
    cat_l = req.spend_category.lower()
    if cat_l in str(card['best_for']).lower():
        parts.append(f"Best suited for {req.spend_category} spends")
    if req.pref_perk == 'Cashback' and card.get('is_cashback'):
        parts.append(f"Offers direct cashback up to {card['best_reward_rate']}%")
    if req.pref_perk == 'Lounge' and card['lounge_domestic']:
        parts.append("Complimentary airport lounge access")
    if req.pref_perk == 'Zero Fee' and card['lifetime_free']:
        parts.append("Zero lifetime annual fee")
    if req.pref_perk == 'Forex' and card['forex_markup'] <= 1.5:
        parts.append(f"Ultra-low {card['forex_markup']}% forex markup")
    if not parts:
        parts.append(f"Strong {card['best_reward_rate']}% rewards on {card['best_category']}")
    return " · ".join(parts[:2])

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "cards_loaded": len(cards_df),
        "model": "Random Forest",
        "auc_roc": round(eval_res['Random Forest']['auc_roc'], 4),
        "ndcg_at_5": round(eval_res['Random Forest'].get('ndcg_at_5', 0), 4),
    }

@app.post("/recommend", response_model=List[CardResponse], summary="Get personalised recommendations")
def recommend(req: RecommendRequest):
    """
    Recommend top-N credit cards based on user profile using the Random Forest model.
    
    - **spend_category**: Where you spend most (Shopping/Travel/Fuel/Food/Utility/Health)
    - **pref_perk**: What benefit matters most to you
    - **fee_preference**: Annual fee you're comfortable with
    - **monthly_spend**: Your monthly credit card spend in INR
    """
    scores = []
    for _, card in cards_df.iterrows():
        features = build_feature_row(req, card)
        prob = rf_model.predict_proba(features)[0][1]
        scores.append((prob, card))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:req.top_n]

    results = []
    for rank, (score, card) in enumerate(top, 1):
        results.append(CardResponse(
            rank=rank,
            bank=str(card['bank']),
            name=str(card['name']),
            network=str(card['network']),
            annual_fee=float(card['annual_fee']),
            reward_type=str(card['reward_type']),
            base_reward_rate=float(card['base_reward_rate']),
            best_reward_rate=float(card['best_reward_rate']),
            best_category=str(card['best_category']),
            lounge_domestic=bool(card['lounge_domestic']),
            lounge_international=bool(card['lounge_international']),
            forex_markup=float(card['forex_markup']),
            segment=str(card['segment']),
            rating=float(card['rating']),
            lifetime_free=bool(card['lifetime_free']),
            match_score=round(float(score) * 100, 1),
            reason=generate_reason(card, req),
        ))
    return results

@app.get("/cards", summary="List all cards with optional filters")
def list_cards(
    bank: Optional[str] = Query(None),
    segment: Optional[str] = Query(None),
    lifetime_free: Optional[bool] = Query(None),
    lounge: Optional[bool] = Query(None),
    min_reward_rate: Optional[float] = Query(None),
    max_annual_fee: Optional[int] = Query(None),
    limit: int = Query(20, le=111),
):
    """Filter and list credit cards from the dataset."""
    df = cards_df.copy()
    if bank:           df = df[df['bank'].str.contains(bank, case=False)]
    if segment:        df = df[df['segment'] == segment]
    if lifetime_free is not None: df = df[df['lifetime_free'] == lifetime_free]
    if lounge is not None: df = df[df['lounge_domestic'] == lounge]
    if min_reward_rate: df = df[df['best_reward_rate'] >= min_reward_rate]
    if max_annual_fee is not None: df = df[df['annual_fee'] <= max_annual_fee]
    df = df.sort_values('rating', ascending=False).head(limit)
    return df[['bank','name','network','annual_fee','reward_type','best_reward_rate',
               'segment','rating','lifetime_free','lounge_domestic','lounge_international']].to_dict('records')

@app.get("/cards/{card_name}", summary="Get full details of a specific card")
def get_card(card_name: str):
    """Fetch all details for a specific card by name (partial match)."""
    matches = cards_df[cards_df['name'].str.contains(card_name, case=False)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Card '{card_name}' not found")
    return matches.iloc[0].to_dict()

@app.get("/similar/{card_name}", summary="Find similar cards using content-based filtering")
def similar_cards(card_name: str, top_n: int = Query(5, le=10)):
    """Get similar cards to a given card using TF-IDF cosine similarity."""
    idx = cards_df[cards_df['name'].str.contains(card_name, case=False)].index
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="Card not found")
    i = idx[0]
    sim_scores = sorted(enumerate(cosine_sim[i]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    card_indices = [s[0] for s in sim_scores]
    result = cards_df.iloc[card_indices][['bank','name','best_reward_rate','annual_fee','segment','rating']]
    return result.to_dict('records')

@app.get("/stats", summary="Market-level statistics")
def market_stats():
    """Aggregate statistics on the Indian credit card market."""
    return {
        "total_cards": len(cards_df),
        "total_banks": cards_df['bank'].nunique(),
        "lifetime_free_cards": int(cards_df['lifetime_free'].sum()),
        "cards_with_domestic_lounge": int(cards_df['lounge_domestic'].sum()),
        "cards_with_intl_lounge": int(cards_df['lounge_international'].sum()),
        "avg_annual_fee": round(float(cards_df['annual_fee'].mean()), 0),
        "median_annual_fee": round(float(cards_df['annual_fee'].median()), 0),
        "avg_best_reward_rate": round(float(cards_df['best_reward_rate'].mean()), 2),
        "max_best_reward_rate": round(float(cards_df['best_reward_rate'].max()), 2),
        "avg_rating": round(float(cards_df['rating'].mean()), 2),
        "model_metrics": eval_res.get('Random Forest', {}),
        "banks": cards_df['bank'].value_counts().to_dict(),
        "segments": cards_df['segment'].value_counts().to_dict(),
    }

@app.get("/model/metrics", summary="ML model evaluation metrics")
def model_metrics():
    """Return evaluation metrics for all trained models."""
    return eval_res
