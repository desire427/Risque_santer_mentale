# Maternal Health Risk — Advanced ML Dashboard

Une solution interactive d'analyse prédictive conçue pour aider à identifier les niveaux de risque pour la santé maternelle à l'aide d'algorithmes d'apprentissage automatique supervisé.

---

## Vue d'ensemble du projet
Ce projet propose un pipeline complet, allant de l'ingestion automatisée des données à l'évaluation comparative de modèles de classification. L'application est construite avec **Streamlit** pour offrir une interface utilisateur moderne, réactive et intuitive aux professionnels de santé ou aux data scientists.

## Fonctionnalités clés

1.  **Exploration des Données (EDA) :** Visualisations statistiques dynamiques, répartition des classes et statistiques descriptives.
2.  **Analyse des Variables :** Matrices de corrélation, distributions par densité et boxplots pour comprendre les facteurs d'influence.
3.  **Entraînement Configurable :** Paramétrage manuel des hyperparamètres, taille des jeux de test et validation croisée (K-Fold).
4.  **Benchmark de Modèles :** Comparaison automatique de 6 algorithmes différents basée sur l'Accuracy, le F1-Score et la précision.
5.  **Outil de Prédiction Clinique :** Interface de saisie en temps réel pour prédire le risque patient avec affichage de l'indice de confiance.

## Spécifications du Dataset

Les données proviennent de la plateforme **Kaggle** (Lien vers le dataset).

### Caractéristiques (Features)
| Feature | Description |
| :--- | :--- |
| **Age** | Âge de la patiente (années) |
| **SystolicBP** | Pression artérielle systolique (mmHg) |
| **DiastolicBP** | Pression artérielle diastolique (mmHg) |
| **BS** | Glycémie (mmol/L) |
| **BodyTemp** | Température corporelle (°F) |
| **HeartRate** | Fréquence cardiaque (bpm) |

### Variable Cible (Target)
Le modèle classifie les patientes en trois catégories :
*   `low risk` (Risque faible)
*   `mid risk` (Risque modéré)
*   `high risk` (Risque élevé)

## Stack Technique

*   **Interface :** Streamlit
*   **Analyse de données :** Pandas, NumPy
*   **Visualisation :** Matplotlib, Seaborn
*   **Machine Learning (Scikit-Learn) :**
    *   Random Forest Classifier
    *   Gradient Boosting Classifier
    *   Logistic Regression
    *   Support Vector Machine (SVM)
    *   K-Nearest Neighbors (KNN)
    *   Decision Tree
*   **Prétraitement :** StandardScaler, LabelEncoder

## Installation et Configuration

### Prérequis
*   Python 3.8+
*   Un compte Kaggle (pour le téléchargement automatique via `opendatasets`)

### Étapes d'installation

1. **Cloner le dépôt :**
   ```bash
   git clone [URL_DU_DEPOT]
   cd "Maternal Health Risk Data"
   ```

2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application :**
   ```bash
   streamlit run model.py
   ```

> **Note :** Lors du premier lancement, vous devrez saisir votre nom d'utilisateur et votre clé API Kaggle pour télécharger le dataset.

## Utilisation

### 1. Navigation
Utilisez la barre latérale gauche pour naviguer entre les différents modules de l'application. Vous pouvez également importer votre propre fichier CSV pour tester le modèle sur de nouvelles données.

### 2. Entraînement
Dans la section "Model Training", ajustez les paramètres de l'algorithme choisi pour observer l'impact sur les performances (Matrice de confusion, rapport de classification).

### 3. Prédiction
Saisissez les paramètres vitaux d'une patiente dans l'outil de prédiction pour obtenir un diagnostic instantané accompagné d'une analyse des seuils cliniques (alertes sur l'hypertension ou l'hyperglycémie).

## Aperçu des Performances
L'application intègre un module de comparaison permettant d'identifier le meilleur modèle. Actuellement, le modèle **Random Forest** offre les meilleures performances globales sur ce jeu de données.

---
*Projet développé dans le cadre d'une analyse de données de santé maternelle.*
```

## Fonctionnalités

- **Overview & EDA** : statistiques descriptives, distribution des classes
- **Feature Analysis** : distributions par risque, corrélations, boxplots
- **Model Training** : entraînement configurable + validation croisée + matrice de confusion
- **Model Comparison** : benchmark automatique de 6 algorithmes
- **Prediction Tool** : prédiction patient en temps réel avec probabilités

## Algorithmes disponibles

- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM (RBF kernel)
- K-Nearest Neighbors
- Decision Tree