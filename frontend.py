import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Charge les variables d'environnement (pour le local)
load_dotenv()

# ⚠️ Remplace par l'URL de ton API sur Render quand elle sera en ligne
API_URL = "https://test-api-bcgp.onrender.com/predict" 

# On récupère la clé API (via le .env en local, ou st.secrets sur Streamlit Cloud)
API_KEY = os.getenv("API_KEY") 

st.title("🌟 Analyseur d'Avis Clients (LightGBM)")
st.write("Entrez un avis ci-dessous pour savoir s'il est Positif, Neutre ou Négatif.")

# Zone de texte pour l'utilisateur
user_input = st.text_area("Avis client :", placeholder="Le produit est génial mais la livraison était en retard...")

if st.button("Analyser le sentiment"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte avant d'analyser.")
    else:
        # 1. On prépare la requête pour l'API
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {"text": user_input}

        # 2. On appelle ton API (le moteur)
        try:
            with st.spinner("Analyse en cours par l'IA..."):
                response = requests.post(API_URL, json=payload, headers=headers)
            
            # 3. On affiche le résultat selon la réponse
            if response.status_code == 200:
                data = response.json()
                sentiment = data["sentiment"]
                score = data["prediction_score"]
                
                # Affichage dynamique selon la classe !
                if sentiment == "Positif":
                    st.success(f"🟢 **Sentiment : {sentiment}** (Confiance : {score})")
                elif sentiment == "Négatif":
                    st.error(f"🔴 **Sentiment : {sentiment}** (Confiance : {score})")
                else:
                    st.info(f"⚪ **Sentiment : {sentiment}** (Confiance : {score})")
                    
            elif response.status_code == 403:
                st.error("⛔ Accès refusé : Vérifiez votre clé API.")
            else:
                st.error(f"⚠️ Erreur de l'API : {response.status_code}")
                
        except Exception as e:
            st.error(f"🔌 Impossible de se connecter à l'API. Est-elle bien lancée ? Détails : {e}")