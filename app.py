import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from scipy.sparse import hstack

# 🤫 Charge les variables d'environnement (pour le local et le cloud)
load_dotenv()

# Téléchargement du dictionnaire VADER (silencieux)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API sécurisée pour analyser le sentiment des avis clients",
    version="2.0"
)

# 🛡️ 1. Configuration de la sécurité (LE CHANGEMENT EST ICI)
# On va chercher la clé secrète configurée sur Render !
API_KEY = os.getenv("API_KEY") 

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Accès refusé. Clé API invalide.")

# 🧠 2. Chargement du TF-IDF et du Modèle
try:
    chargement = joblib.load("model_robust.pkl")
    if isinstance(chargement, dict):
        tfidf = chargement.get("tfidf")
        model = chargement.get("model")
        print("📦 Modèle et TF-IDF chargés avec succès !")
    else:
        model = chargement
        tfidf = None
except Exception as e:
    model = None
    tfidf = None
    print(f"❌ Erreur lors du chargement : {e}")

class Review(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps sécurisée. Allez sur /docs pour tester."}

# 🔒 3. Route de prédiction
@app.post("/predict")
def predict_sentiment(review: Review, api_key: str = Depends(get_api_key)):
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Le modèle ou le TF-IDF n'est pas chargé.")
    
    # ÉTAPE A : TF-IDF (On récupère les 500 colonnes)
    texte_vectorise = tfidf.transform([review.text])
    
    # ÉTAPE B : VADER (On calcule la 501ème colonne)
    vader_score = sia.polarity_scores(review.text)['compound']
    vader_feature = np.array([[vader_score]])
    
    # ÉTAPE C : On colle les deux ensemble (500 + 1 = 501 colonnes)
    features_finales = hstack([texte_vectorise, vader_feature])
    
    # ÉTAPE D : On fait la prédiction avec le Modèle
    prediction = model.predict(features_finales)[0]
    
    # Calcul de la confiance
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_finales)[0]
        confidence = round(max(proba) * 100, 2)
    else:
        confidence = 100.0

    sentiment = "Positif" if prediction == 1 else "Négatif"
    
    return {
        "texte": review.text,
        "sentiment": sentiment,
        "prediction_score": f"{confidence}%",
        "vader_compound_calcule": vader_score # Petit bonus pour voir ce qu'il a calculé
    }