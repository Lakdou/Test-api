import joblib
import re
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Config NLTK (Toujours nécessaire)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# 2. Fonctions utilitaires (Les mêmes que dans train_robust.py)
def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", str(text).lower())

def get_vader_score(text):
    return sia.polarity_scores(str(text))['compound']

# 3. Chargement du Kit Robuste
print("📦 Chargement du kit modèle...")
try:
    pack = joblib.load("model_robust.pkl")
    tfidf = pack["tfidf"]   # On récupère le vectoriseur
    model = pack["model"]   # On récupère le Random Forest
    print("✅ Kit chargé et prêt !")
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    exit()

# 4. L'Application API
class ReviewInput(BaseModel):
    text: str

app = FastAPI(title="Trustpilot API (Version Robuste)")

@app.post("/predict")
def predict_sentiment(review: ReviewInput):
    try:
        # A. Nettoyage manuel
        cleaned = clean_text(review.text)
        
        # B. Transformation TF-IDF
        # Attention : transform attend une liste, donc on met [cleaned]
        # .toarray() est important pour avoir un format "dense" compatible avec numpy
        t_tfidf = tfidf.transform([cleaned]).toarray()
        
        # C. Calcul VADER
        vader_score = get_vader_score(review.text)
        t_vader = np.array([[vader_score]]) # On le met au format (1, 1)
        
        # D. Fusion (Comme dans l'entraînement)
        t_final = np.hstack((t_tfidf, t_vader))
        
        # E. Prédiction
        prediction = model.predict(t_final)[0]
        
        # Logique métier pour l'affichage
        sentiment = "Positif"
        if prediction < 3:
            sentiment = "Négatif"
        elif prediction == 3:
            sentiment = "Neutre"
            
        return {
            "text": review.text,
            "prediction_score": int(prediction),
            "sentiment": sentiment,
            "debug_info": {
                "vader_score": vader_score,
                "input_clean": cleaned
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)