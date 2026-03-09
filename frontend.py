import streamlit as st
import requests
import os
from dotenv import load_dotenv

# On charge le mot de passe depuis les variables d'environnement
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 🚨 Ton API en ligne qui tourne déjà sur Render !
API_URL = "https://test-api-bcgp.onrender.com/predict"

# Design de la page
st.set_page_config(page_title="Analyseur Trustpilot", page_icon="⭐")
st.title("⭐ Analyseur d'Avis Clients")
st.markdown("Testez notre IA. La sécurité est gérée de manière invisible !")

# Ce que l'utilisateur voit
user_input = st.text_area("Tapez l'avis client ici :", "Le service est vraiment top !")

if st.button("Analyser le sentiment 🚀"):
    if user_input:
        with st.spinner("L'IA réfléchit..."):
            
            # Le frontend prépare le badge VIP
            headers = {"X-API-Key": API_KEY}
            payload = {"text": user_input}
            
            try:
                # Le frontend envoie le texte ET le mot de passe à ton API Render
                response = requests.post(API_URL, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    st.divider()
                    if result["sentiment"] == "Positif":
                        st.success(f"### Résultat : {result['sentiment']} 😊")
                    else:
                        st.error(f"### Résultat : {result['sentiment']} 😡")
                    
                    st.metric(label="Confiance", value=result["prediction_score"])
                else:
                    st.error(f"Erreur API : Accès refusé ou problème serveur (Code {response.status_code})")
                    
            except Exception as e:
                st.error(f"Impossible de joindre l'API : {e}")
    else:
        st.warning("Veuillez entrer un texte.")