# Librairies
from joblib import load
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
X = pd.read_csv('X_test_init_sample_saved.csv')
    


# Variables sélectionnées
df_vars_selected = pd.read_csv('df_vars_selected_saved.csv')


vars_selected = df_vars_selected['feature'].to_list()
vars_selected.insert(0, 'SK_ID_CURR') # Ajout de l'identifiant aux features

# Chargement du modèle
from joblib import dump, load
pipeline_loaded = load('pipeline_credit.joblib')
pipeline = pipeline_loaded

# Création d'une nouvelle instance fastAPI
app = FastAPI()

# Définir un objet ( une classe) pour réaliser des requêtes
# dot notation (.)
class request_body(BaseModel):
    SK_ID_CURR : float



# Définition du chemin du point de terminaison (API)
@app.post("/predict ")# local : http://127.0.0.1:8000/predict

# Définition de la fonction de prédiction

def predict_proba( ID  : request_body):
    # Nouvelles données sur lesquelles on fait la prédiction
  
    #donnees_client = X[vars_selected][X[vars_selected]['SK_ID_CURR']==ID] 
    new_ID = ID
    donnees_client = X[vars_selected][X[vars_selected]['SK_ID_CURR']==new_ID]

            
    # Prédiction 
    prevision = pipeline.predict_proba(donnees_client.drop(['SK_ID_CURR'],axis=1))
    prevision = prevision[:, 1]


    # je retourne le sens de la prédiction yes ou now
    return {'reponse' :prevision}
    
