# 🇮🇳 India Credit Card Recommendation Engine

> **End-to-end ML project** — Web scraping → Data engineering → EDA → ML models → REST API  
> Built with Apify, Python, scikit-learn, FastAPI, Pandas, and Matplotlib

---

## 📌 Project Overview

A production-ready recommendation system for Indian credit cards that combines **live web scraping**, **exploratory data analysis**, **machine learning**, and a **REST API backend** — designed to help users find the best credit card for their spending profile.

---

## 🏗️ Project Structure

```
creditcard_project/
├── data/
│   ├── cards.csv                  # 111 Indian credit cards (scraped via Apify)
│   ├── cards_features.csv         # Engineered features dataset
│   ├── user_interactions.csv      # 99,900 simulated user-card interactions
│   └── user_profiles.csv          # 900 simulated user profiles
│
├── notebooks/
│   └── EDA_CreditCard_India.ipynb # Full EDA with 9 visualisation sections
│
├── models/
│   ├── train_model.py             # Model training pipeline
│   ├── rf_model.pkl               # Random Forest (best model)
│   ├── lr_model.pkl               # Logistic Regression
│   ├── scaler.pkl                 # Feature scaler
│   ├── tfidf.pkl                  # TF-IDF vectorizer (content-based)
│   ├── cosine_sim.pkl             # Cosine similarity matrix
│   ├── encoders.pkl               # Label encoders
│   ├── feature_importance.csv     # RF feature importances
│   └── eval_results.json          # All model metrics
│
├── api/
│   └── main.py                    # FastAPI REST backend
│
├── validation/
│   ├── data_validation.py         # Pydantic + business rule validation
│   └── validation_report.json     # Latest validation run report
│
└── outputs/
    ├── 01_cards_per_bank.png
    ├── 02_fee_distribution.png
    ├── 03_reward_analysis.png
    ├── 04_lounge_access.png
    ├── 05_reward_types.png
    ├── 06_forex_analysis.png
    ├── 07_ratings.png
    ├── 08_correlation_heatmap.png
    └── 09_model_results.png
```

---

## 📊 Dataset

| Feature           | Details                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Source**        | CardInsider.com scraped via Apify RAG Web Browser                                                                              |
| **Cards**         | 111 across 19 Indian banks                                                                                                     |
| **Features**      | 25 per card (fees, rewards, lounge, forex, segment, rating…)                                                                   |
| **User data**     | 900 synthetic users × 111 cards = 99,900 interaction rows                                                                      |
| **Banks covered** | HDFC, SBI, Axis, ICICI, IDFC FIRST, AU Bank, IndusInd, Amex, HSBC, Kotak, SCB, Yes Bank, RBL, Federal, BoB, PNB, Union, Canara |

---

## 🤖 ML Models

Three models trained and evaluated:

| Model               | Accuracy  | Precision | Recall    | F1        | AUC-ROC   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 75.1%     | 11.1%     | 64.9%     | 19.0%     | 0.775     |
| **Random Forest**   | **88.8%** | **26.3%** | **81.9%** | **39.8%** | **0.937** |
| Gradient Boosting   | 96.6%     | 81.2%     | 30.7%     | 44.5%     | 0.951     |

**Ranking Metrics (Random Forest):**

- **NDCG@5 = 0.877** (Normalised Discounted Cumulative Gain)
- **MAP@5 = 0.813** (Mean Average Precision)

**Top Feature Importances:**

1. `card_best_reward_rate` (27%) — reward maximisation is primary driver
2. `perk_enc` (14%) — preferred perk type strongly influences choice
3. `card_rating` (11%) — user ratings matter significantly
4. `card_annual_fee` (11%) — fee is always a factor
5. `cat_enc` (9%) — spending category alignment

**Content-Based Filtering:**

- TF-IDF on card metadata (bank + name + reward type + segment + category)
- Cosine similarity matrix for `/similar/{card_name}` endpoint

---

## 🛠️ Tech Stack

| Layer         | Technology                         |
| ------------- | ---------------------------------- |
| Scraping      | Apify RAG Web Browser              |
| Data          | Pandas, NumPy                      |
| ML            | scikit-learn (RF, LR, GBM)         |
| Validation    | Pydantic v2, custom business rules |
| Visualisation | Matplotlib, Seaborn                |
| API           | FastAPI, Uvicorn                   |
| Notebook      | Jupyter                            |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python models/train_model.py

# 3. Run data validation
python validation/data_validation.py

# 4. Start API server
uvicorn api.main:app --reload --port 8000

# 5. View API docs
open http://localhost:8000/docs
```

---

## 🌐 API Endpoints

| Method | Endpoint          | Description                        |
| ------ | ----------------- | ---------------------------------- |
| GET    | `/health`         | System health + model metrics      |
| POST   | `/recommend`      | Get personalised recommendations   |
| GET    | `/cards`          | List/filter all cards              |
| GET    | `/cards/{name}`   | Get full card details              |
| GET    | `/similar/{name}` | Find similar cards (content-based) |
| GET    | `/stats`          | Market-level statistics            |
| GET    | `/model/metrics`  | All model evaluation metrics       |

**Example recommendation request:**

```json
POST /recommend
{
  "spend_category": "Shopping",
  "pref_perk": "Cashback",
  "fee_preference": "Mid (1.5-5K)",
  "monthly_spend": 25000,
  "age": 28,
  "top_n": 5
}
```

---

## ✅ Data Validation

10 automated checks on every dataset update:

- Schema validation via Pydantic v2 (type checking, range validation)
- Fee consistency (lifetime_free ↔ annual_fee)
- Reward rate ordering (best_rate ≥ base_rate)
- Rating range enforcement (0–5)
- Forex markup range (0–5%)
- Premium card lounge requirement
- No null values in critical columns
- No duplicate cards
- Bank coverage threshold
- Fee-reward correlation health check

**Latest run: 10/10 checks PASSED ✅**

---

## 💡 Key Insights from EDA

1. **HDFC & SBI dominate** — 20+ cards each, reflecting India's credit card market
2. **42% of entry-segment cards are lifetime free** — accessibility is improving
3. **Reward rates up to 33%** exist (HDFC SmartBuy) but require specific redemption
4. **Lounge access is premium-gated** — only ~18% of entry cards offer it
5. **60%+ of cards charge standard 3.5% forex** — Scapia (0%) and IDFC Ashva (1%) are outliers
6. **Random Forest achieves 0.937 AUC-ROC** and NDCG@5 = 0.877

---

## 📈 Resume Bullet Points

```
• Scraped 111 Indian credit cards (25 features/card) from CardInsider
  via Apify, engineering a clean dataset across 19 banks

• Simulated 99,900 user-card interactions and trained 3 ML models
  (LR, RF, GBM) achieving AUC-ROC 0.937 and NDCG@5 0.877 with Random Forest

• Validated dataset with 10 automated Pydantic + business rule checks;
  100% pass rate on schema, nulls, consistency, and range checks

• Built FastAPI REST backend (7 endpoints) with content-based filtering
  via TF-IDF cosine similarity for card-to-card recommendations

• Conducted 9-section EDA generating correlation heatmaps, reward rate
  analysis, lounge access breakdowns, and model performance comparisons
```
