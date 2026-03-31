# SETUP GUIDE — India Credit Card Recommender
Run locally → Push to GitHub → Host online

---

## WHAT YOU WILL HAVE AT THE END

Frontend live on GitHub Pages (free)
API live on Render.com (free)
Full repo on GitHub with README, EDA notebook, models, validation

---

## STEP 0 — PREREQUISITES

Install these first:

Python 3.10+  →  https://www.python.org/downloads/
Git           →  https://git-scm.com/downloads
VS Code       →  https://code.visualstudio.com/

Check they work:
  python --version
  git --version

---

## STEP 1 — ORGANIZE YOUR FILES

Create a folder and copy all the downloaded files into it:

  mkdir creditcard-recommender
  cd creditcard-recommender

Final folder structure needed:
  creditcard-recommender/
  ├── frontend/
  │   └── index.html
  ├── api/
  │   └── main.py
  ├── models/
  │   └── train_model.py
  ├── data/
  │   ├── cards.csv
  │   └── cards_features.csv
  ├── notebooks/
  │   └── EDA_CreditCard_India.ipynb
  ├── validation/
  │   └── data_validation.py
  ├── requirements.txt
  └── README.md

---

## STEP 2 — PYTHON ENVIRONMENT

  # Create virtual environment
  python -m venv venv

  # Activate it
  # Windows:
  venv\Scripts\activate
  # Mac/Linux:
  source venv/bin/activate

  # Install packages
  pip install -r requirements.txt

You will see (venv) at the start of your terminal line now.

---

## STEP 3 — TRAIN THE MODEL

  python data/create_dataset.py
  python data/simulate_users.py
  python models/train_model.py

Expected output at the end:
  Random Forest  AUC-ROC: 0.9372   NDCG@5: 0.8765
  All models saved.

You will now have rf_model.pkl, scaler.pkl, encoders.pkl etc. in models/

---

## STEP 4 — DATA VALIDATION

  python validation/data_validation.py

Expected:
  Overall Status: PASS  (10/10 checks)

---

## STEP 5 — START THE API

  uvicorn api.main:app --reload --port 8000

Open in browser:
  http://localhost:8000/docs

You should see the Swagger docs page with all 7 endpoints.

Test with curl (open a second terminal, keep uvicorn running):
  curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"spend_category":"Shopping","pref_perk":"Cashback","fee_preference":"Mid (1.5-5K)","monthly_spend":25000,"age":28,"top_n":3}'

---

## STEP 6 — OPEN THE FRONTEND

Open a new terminal:
  cd frontend
  python -m http.server 3000

Open browser:
  http://localhost:3000

Click "Run Model". It calls your local API and shows real recommendations.
Everything is working locally now.

---

## STEP 7 — PUSH TO GITHUB

7a. Create GitHub account at https://github.com (free)

7b. Create a new repository:
  - Go to https://github.com/new
  - Name: creditcard-recommender
  - Set to Public
  - Do NOT add README (you already have one)
  - Click Create repository

7c. Push your code (run in your project folder):

  git init

  # Create .gitignore
  echo "venv/" >> .gitignore
  echo "__pycache__/" >> .gitignore
  echo "*.pyc" >> .gitignore
  echo ".DS_Store" >> .gitignore
  echo "data/user_interactions.csv" >> .gitignore
  echo "models/rf_model.pkl" >> .gitignore
  echo "models/cosine_sim.pkl" >> .gitignore

  git add .
  git commit -m "Initial commit: ML credit card recommender"

  # Replace YOUR_USERNAME with your actual GitHub username
  git remote add origin https://github.com/YOUR_USERNAME/creditcard-recommender.git
  git branch -M main
  git push -u origin main

Refresh your GitHub page. Your files are now online.

---

## STEP 8 — HOST FRONTEND ON GITHUB PAGES (FREE)

GitHub Pages serves your HTML for free.

1. Go to your repo on GitHub
2. Click Settings (top menu bar)
3. Click Pages (left sidebar)
4. Source: Deploy from a branch
5. Branch: main   Folder: /frontend
6. Click Save

Wait 3 minutes. Your site goes live at:
  https://YOUR_USERNAME.github.io/creditcard-recommender/

The frontend works in static mode without the API — it uses built-in data.

---

## STEP 9 — HOST API ON RENDER.COM (FREE)

9a. Create account at https://render.com
    Sign up with your GitHub account.

9b. Add a Procfile to your repo.
    In your project folder, create a file called Procfile (no extension):

    echo "web: uvicorn api.main:app --host 0.0.0.0 --port $PORT" > Procfile
    git add Procfile
    git commit -m "Add Procfile for Render"
    git push

9c. Deploy on Render:
  1. Go to https://dashboard.render.com
  2. Click New → Web Service
  3. Connect GitHub → select creditcard-recommender
  4. Fill in:
       Name:           creditcard-recommender-api
       Environment:    Python 3
       Build Command:  pip install -r requirements.txt && python data/create_dataset.py && python data/simulate_users.py && python models/train_model.py
       Start Command:  uvicorn api.main:app --host 0.0.0.0 --port $PORT
  5. Plan: Free
  6. Click Create Web Service

Build takes 5-10 minutes. Your API goes live at:
  https://creditcard-recommender-api.onrender.com

Test:
  https://creditcard-recommender-api.onrender.com/health
  https://creditcard-recommender-api.onrender.com/docs

---

## STEP 10 — CONNECT FRONTEND TO LIVE API

Open frontend/index.html and find this line near the top of the script tag:

  const API = 'http://localhost:8000';

Change it to your Render URL:

  const API = 'https://creditcard-recommender-api.onrender.com';

Push the change:
  git add frontend/index.html
  git commit -m "Connect to live API"
  git push

GitHub Pages updates in ~2 minutes.

Your project is now fully live:
  App:     https://YOUR_USERNAME.github.io/creditcard-recommender/
  API:     https://creditcard-recommender-api.onrender.com/docs
  GitHub:  https://github.com/YOUR_USERNAME/creditcard-recommender

---

## TROUBLESHOOTING

ModuleNotFoundError
  → Activate venv first: source venv/bin/activate

GitHub Pages shows 404
  → Wait 5 minutes. Check Settings → Pages that folder is /frontend

Frontend shows "API offline"
  → Free Render plan sleeps after 15 min. First request wakes it (~30 sec wait).
  → Normal behavior, not a bug.

Render build fails
  → Check build logs on Render dashboard. Usually a package issue.

CORS error in browser
  → API has CORS open for all origins. Clear browser cache and retry.

---

## QUICK COMMAND REFERENCE

# Activate environment
source venv/bin/activate       (Mac/Linux)
venv\Scripts\activate          (Windows)

# Run everything locally
python models/train_model.py
uvicorn api.main:app --reload --port 8000
cd frontend && python -m http.server 3000

# Push to GitHub
git add .
git commit -m "update"
git push

---

## RESUME LINKS TO ADD

GitHub:   github.com/YOUR_USERNAME/creditcard-recommender
Live App: YOUR_USERNAME.github.io/creditcard-recommender
API Docs: creditcard-recommender-api.onrender.com/docs
