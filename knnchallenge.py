# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:59:09 2024

@author: DELL
"""

#code knn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Charger les données d'entraînement et de test
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Vérification et remplacement des valeurs manquantes
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

# Supposons que la première colonne est 'Id', que nous devons exclure
X_train = df_train.iloc[:, 1:-1].values  # Caractéristiques d'entraînement (exclure 'Id' et la cible)
y_train = df_train.iloc[:, -1].values    # Colonne cible (dernière colonne)
X_test = df_test.iloc[:, 1:].values      # Caractéristiques de test (exclure 'Id')

# Standardisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optimisation des hyperparamètres avec GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Tester un large éventail de valeurs pour k
    'weights': ['uniform', 'distance'],  # Pondération uniforme ou par distance
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Différentes métriques
    'p': [1, 2]  # Distance Manhattan (p=1) et Euclidienne (p=2)
}
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=10,  # Validation croisée à 10 plis
    scoring='accuracy',
    n_jobs=-1  # Utilise tous les cœurs disponibles pour accélérer
)
grid_search.fit(X_train, y_train)

# Meilleurs paramètres obtenus
best_knn = grid_search.best_estimator_
print("Meilleurs paramètres :", grid_search.best_params_)

# Entraîner le meilleur modèle avec toutes les données d'entraînement
best_knn.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = best_knn.predict(X_test)

# Préparer les résultats pour la soumission
test_ids = df_test['Id']
results = pd.DataFrame({
    'Id': test_ids,
    'label': y_pred
})

# Sauvegarder les résultats
results.to_excel('predictions.xlsx', index=False)
results.to_csv('predictions.csv', index=False)

# Message de confirmation
print("Les résultats ont été sauvegardés dans 'predictions.xlsx' et 'predictions.csv'.")