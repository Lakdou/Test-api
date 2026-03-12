import os
import json
import secrets
import hashlib
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from fastapi import Header
from datetime import date

# 🤫 Charge les variables d'environnement
load_dotenv()

# 🚀 INITIALISATION DE L'API (Doit être en haut !)
app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API sécurisée avec gestion des utilisateurs",
    version="3.0"
)

# --- 1. MODÈLES DE DONNÉES ---
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"  

class UserLogin(BaseModel):
    username: str
    password: str

class Review(BaseModel):
    text: str

# --- 2. GESTION DU FICHIER JSON (Fausse BDD) ---
USERS_FILE = "users.json"

def get_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# --- 3. SÉCURITÉ (Le nouveau Vigile) ---
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    users = get_users()
    # On cherche si cette clé appartient à un utilisateur enregistré
    for user_data in users.values():
        if user_data.get("api_key") == api_key:
            return api_key # La clé est valide !
            
    # Si la boucle se termine sans trouver la clé :
    raise HTTPException(status_code=403, detail="Accès refusé. Clé API invalide ou inexistante.")

# --- 4. CHARGEMENT DU MODÈLE ---
try:
    model = joblib.load("trustpilot_lgbm_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Modèle LightGBM et TF-IDF chargés avec succès !")
except Exception as e:
    model = None
    tfidf = None
    print(f"❌ Erreur lors du chargement : {e}")

# --- 5. LES ROUTES ---

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps sécurisée. Allez sur /docs pour tester."}

@app.post("/login")
def create_user(user: UserCreate):
    users = get_users()
    if user.username in users:
        raise HTTPException(status_code=400, detail="Cet utilisateur existe déjà.")
    
    users[user.username] = {
        "password": hash_password(user.password),
        "role": user.role,
        "api_key": None
    }
    save_users(users)
    return {"message": f"Utilisateur '{user.username}' créé avec succès."}

@app.post("/token_API")
def generate_token(user: UserLogin):
    users = get_users()
    if user.username not in users or users[user.username]["password"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")
    
    new_token = secrets.token_hex(32)
    users[user.username]["api_key"] = new_token
    save_users(users)
    
    return {
        "access_token": new_token, 
        "role": users[user.username]["role"],
        "message": "Conservez ce token précieusement."
    }


# 🚪 La porte de sécurité de Nginx (qui vérifie les quotas maintenant !)
@app.get("/verify_admin")
def verify_admin(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Clé API manquante")
    
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == x_api_key:
            role = user_data.get("role", "user")
            
            # 👑 L'admin passe toujours en illimité
            if role == "admin":
                return {"message": "Autorisé"}
            
            # 🧑‍💻 Gestion du Quota pour les utilisateurs standards (5 par jour)
            today = date.today().isoformat() # ex: '2026-03-12'
            last_date = user_data.get("last_request_date")
            count = user_data.get("daily_count", 0)
            
            # Si c'est un nouveau jour, on fait comme si le compteur était à zéro
            if last_date != today:
                count = 0
            
            # Si le quota est atteint, on dit à Nginx de bloquer !
            if count >= 5:
                raise HTTPException(status_code=403, detail="Quota de 5 prédictions atteint.")
                
            return {"message": "Autorisé"}
            
    raise HTTPException(status_code=401, detail="Clé API invalide")

# 🧠 La route de prédiction (qui valide le ticket)
@app.post("/predict")
def predict_sentiment(review: Review, api_key: str = Depends(get_api_key)):
    if model is None or tfidf is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")
    
    # ÉTAPE A : Prédiction
    texte_vectorise = tfidf.transform([review.text])
    prediction = model.predict(texte_vectorise)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texte_vectorise)[0]
        confidence = round(max(proba) * 100, 2)
    else:
        confidence = 100.0

    labels = {0: "Négatif", 1: "Neutre", 2: "Positif"}
    sentiment = labels.get(int(prediction), "Inconnu")
    
    # ÉTAPE B : La prédiction a réussi, on consomme 1 crédit pour les 'user' !
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("api_key") == api_key:
            if user_data.get("role") != "admin":
                today = date.today().isoformat()
                
                # Mise à zéro du compteur si c'est le premier de la journée
                if user_data.get("last_request_date") != today:
                    user_data["daily_count"] = 0
                    user_data["last_request_date"] = today
                
                # On ajoute +1
                user_data["daily_count"] = user_data.get("daily_count", 0) + 1
                save_users(users)
            break
    
    return {
        "texte": review.text,
        "sentiment": sentiment,
        "prediction_score": f"{confidence}%",
        "class_id": int(prediction)
    }