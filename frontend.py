import streamlit as st
import requests

# 🌍 L'adresse de ton API à travers le Vigile Nginx
API_URL = "http://nginx:80"

st.set_page_config(page_title="Analyse de Sentiment", page_icon="🧠")
st.title("Analyse de Sentiment Trustpilot 🧠")

# --- 🧠 MÉMOIRE DE STREAMLIT ---
if "token" not in st.session_state:
    st.session_state["token"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None

# --- 🔒 BARRE LATÉRALE : INSCRIPTION & CONNEXION ---
with st.sidebar:
    if st.session_state["token"] is None:
        
        # Le sélecteur pour choisir le mode
        choix = st.radio("Que voulez-vous faire ?", ["Connexion", "Inscription"])
        st.divider() # Petite ligne de séparation
        
        if choix == "Inscription":
            st.header("📝 Créer un compte")
            new_username = st.text_input("Nouveau pseudo")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            
            if st.button("S'inscrire"):
                if new_username and new_password:
                    # On n'envoie plus le rôle ! L'API mettra "user" par défaut.
                    payload = {"username": new_username, "password": new_password}
                    res_creation = requests.post(f"{API_URL}/login", json=payload)
                    
                    if res_creation.status_code == 200:
                        st.success("✅ Compte créé avec succès ! Vous avez le rôle standard 'user'.")
                    elif res_creation.status_code == 400:
                        st.error("❌ Ce nom d'utilisateur existe déjà.")
                    else:
                        st.error("❌ Erreur lors de la création du compte.")
                else:
                    st.warning("Veuillez remplir tous les champs.")

        elif choix == "Connexion":
            st.header("🔒 Connexion")
            username = st.text_input("Pseudo")
            password = st.text_input("Mot de passe", type="password")
            
            if st.button("Se connecter"):
                # On demande le badge (Token)
                response = requests.post(f"{API_URL}/token_API", json={"username": username, "password": password})
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["role"] = data["role"]
                    st.success("Connexion réussie !")
                    st.rerun()
                else:
                    st.error("❌ Identifiants incorrects.")
    else:
        # L'utilisateur est connecté
        st.success(f"Connecté en tant que : {st.session_state['role'].upper()}")
        if st.button("Se déconnecter"):
            st.session_state["token"] = None
            st.session_state["role"] = None
            st.rerun()

# --- 🚀 PAGE PRINCIPALE : L'APPLICATION ---
if st.session_state["token"]:
    
    st.write("Entrez un avis client ci-dessous pour analyser son sentiment.")
    
    user_input = st.text_area("Avis client :", placeholder="J'ai adoré ce produit, la livraison était rapide !")
    
    if st.button("Analyser l'avis"):
        if user_input:
            headers = {"X-API-Key": st.session_state["token"]}
            payload = {"text": user_input}
            
            with st.spinner("Analyse en cours..."):
                res = requests.post(f"{API_URL}/predict", headers=headers, json=payload)
                
                if res.status_code == 200:
                    resultat = res.json()
                    sentiment = resultat["sentiment"]
                    score = resultat["prediction_score"]
                    
                    if sentiment == "Positif":
                        st.success(f"✅ Sentiment : {sentiment} (Confiance : {score})")
                    elif sentiment == "Négatif":
                        st.error(f"❌ Sentiment : {sentiment} (Confiance : {score})")
                    else:
                        st.warning(f"😐 Sentiment : {sentiment} (Confiance : {score})")
                        
                elif res.status_code == 403:
                    st.error("🛑 Nginx a bloqué la requête : Quota quotidien atteint (5/5) ou accès non autorisé.")
                elif res.status_code == 503:
                    st.error("🐌 Nginx a bloqué la requête : Vous allez trop vite (Anti-Spam) !")
                else:
                    st.error(f"Erreur technique : {res.status_code}")
        else:
            st.warning("Veuillez entrer un texte à analyser.")
else:
    st.info("👈 Veuillez vous connecter ou créer un compte dans le menu de gauche pour accéder à l'application.")