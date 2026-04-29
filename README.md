# Maternal Health Risk — ML Dashboard

Application Streamlit complète pour la prédiction du risque santé maternelle par apprentissage automatique supervisé.

## Dataset

**Source** : [Kaggle — Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)

| Variable | Description |
|---|---|
| Age | Âge de la patiente |
| SystolicBP | Pression artérielle systolique (mmHg) |
| DiastolicBP | Pression artérielle diastolique (mmHg) |
| BS | Glycémie (mmol/L) |
| BodyTemp | Température corporelle (°F) |
| HeartRate | Fréquence cardiaque (bpm) |
| RiskLevel | **Cible** : low risk / mid risk / high risk |

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
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