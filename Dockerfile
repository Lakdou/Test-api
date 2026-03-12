# 1. Image de base : On part d'un Python léger (Linux)
FROM python:3.12-slim

# 2. On définit le dossier de travail dans le conteneur
WORKDIR /app

# --- 🛠️ LA LIGNE MAGIQUE POUR LIGHTGBM ---
# On réinstalle le composant Linux manquant pour les calculs
RUN apt-get update && apt-get install -y libgomp1
# -----------------------------------------

# 3. On copie le fichier requirements pour installer les libs
COPY requirements.txt .

# 4. Installation des dépendances (sans garder de cache pour alléger)
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout ton code (app.py, model_robust.pkl) dans le conteneur
COPY . .

# 6. Commande de démarrage
# On lance uvicorn sur le port 80 à l'intérieur du conteneur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]