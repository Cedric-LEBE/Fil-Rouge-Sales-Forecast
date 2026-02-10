# 🛒 Prévision des Ventes E-commerce

## 📋 Description

Application interactive de prévision des ventes e-commerce basée sur l'apprentissage automatique et l'analyse de séries temporelles. Ce projet permet d'anticiper les ventes futures en exploitant des données transactionnelles historiques complexes pour optimiser la gestion des stocks, la planification logistique et les stratégies marketing.

## 🎯 Objectifs

L'objectif principal est de concevoir une application interactive permettant de :

1. **Prédire les ventes futures** sur différentes granularités temporelles (journalière, mensuelle)
2. **Fournir des indicateurs clairs** et interprétables pour la prise de décision
3. **Rendre les résultats accessibles** via une application interactive Streamlit

### Objectifs détaillés

#### 1️⃣ Compréhension et préparation des données
- Explorer et comprendre les différentes tables du dataset e-commerce
- Analyser la qualité des données (valeurs manquantes, incohérences, distributions)
- Fusionner les tables pertinentes pour construire des datasets analytiques cohérents

#### 2️⃣ Analyse Exploratoire des Données (EDA)
- Étudier l'évolution des ventes dans le temps
- Identifier les tendances, saisonnalités et cycles récurrents
- Analyser les ventes par type de produit, région géographique, mode de paiement
- Mettre en évidence les corrélations entre variables

#### 3️⃣ Feature Engineering orienté séries temporelles
- Créer des variables temporelles pertinentes (jour, mois, trimestre, année)
- Générer des features de séries temporelles (lag features, moyennes glissantes)
- Construire des variables métier utiles à la prévision

#### 4️⃣ Modélisation et prévision des ventes
- Modèles de Machine Learning : Régression, Random Forest, XGBoost, SARIMAX, Prophet, Holt-Winters
- Évaluation avec métriques adaptées (MAE, RMSE, MAPE)

#### 5️⃣ Restitution et déploiement
- Concevoir des visualisations interactives
- Déployer via une application Streamlit avec filtres dynamiques
- Intrégrer un Chatbot permettant aux utilisateurs de discuter avec les données. 

## 📊 Dataset

Le projet utilise des données e-commerce multi-tables incluant :

- **df_Orders.csv** - Commandes avec statuts et dates
- **df_OrderItems.csv** - Détails des articles commandés
- **df_Customers.csv** - Informations clients
- **df_Payments.csv** - Modes et montants de paiement
- **df_Products.csv** - Catalogue de produits

Les données sont organisées en deux ensembles :
- `data/raw/` - Données brutes
- `data/interim/` - Données prétraiter (fusion/clean/features)
- `data/processed/` - Données final prêt pour l'entrainement

## 🚀 Installation

### Prérequis

- Python 3.10 à 3.12
- pip ou uv pour la gestion des dépendances

### Installation des dépendances

```bash
# Cloner le repository
git remote add origin https://github.com/Cedric-LEBE/Fil-Rouge-Sales-Forecast.git
cd Fil-Rouge-Sales-Forecast

# Créer un environnement virtuel
python3 -m venv .venv

# Activer l'environnement virtuel
# Sur Linux/Mac:
source .venv/bin/activate
# Sur Windows:
.venv\Scripts\activate

# Installer les dépendances
pip install -e ".[prophet]" 
```
### Workflow (Quickstart)
```
# 1) Préparer le dataset & entrainer le model 

python scripts/make_dataset.py
python scripts/train_ml_global.py
python scripts/train_ml_region.py
python scripts/train_ts_region.py

# 2) Lancer l'application streamlit

streamlit run app/app.py
```

## 📁 Structure du Projet

```
Fil-Rouge-Sales-Forecast/
├── app/
│   └── app.py                 # Streamlit (Dashboard + Prévisions + Chatbot placeholder)
├── artefacts
│   └── runs_ml/
│   └── runs_ts/
├── scripts/
│   └── benchmark.py           # Lance un benchmark d'entraînement (leaderboard + best model)
├── data/
│   ├── raw/                   # Données sources (CSV)
│   ├── interim/               # Données intermédiaires (parquet) 
│   └── interim/               # Données prête pour le ML (parquet) 
├── model_store/
│   └── latest/                # Pointeur/artefacts du dernier run
│   └── runs/
├── fil_rouge/                 # Package Python (config, io, train, etc.)
├── reports/
├── scripts/
├── pyproject.toml
├── requirements.txt           # Dependances pour le déploiement streamlit
├── Readme.md
└── .gitignore
```

## 🛠️ Technologies Utilisées

### Analyse et Traitement de Données
- **pandas** - Manipulation de données
- **numpy** - Calculs numériques

### Visualisation
- **matplotlib** - Graphiques statiques
- **seaborn** - Visualisations statistiques
- **plotly** - Graphiques interactifs

### Machine Learning
- **scikit-learn** (v1.5.0) - Modèles ML classiques
- **xgboost** - Gradient boosting
- **statsmodels** - Modèles statistiques (SARIMAX)
- **prophet** - Prévisions de séries temporelles 
- **Holt-Winters** - Prévisions de séries temporelles 

### Déploiement
- **streamlit** - Application web interactive
- **python-dotenv** - Gestion de configuration

### Utilitaires
- **joblib** - Sérialisation de modèles

## 🎯 Fonctionnalités actuelles de l'app

### Pages Streamlit

- **📊 Dashboard** : KPIs, ventes au fil du temps, régions, catégories, paiements, clients
- **🔮 Prévision** : prévision des ventes (globale/région)
- **💬 Chatbot** : front-end uniquement (placeholder), backend à brancher plus tard (Text-to-SQL/RAG)

### En développement
- 🚧 Developpement du backend du Chatbot (Chat with data)

## 🌐 Déploiement

L’application est actuellement déployée en version bêta et accessible en ligne.

URL de l’application (beta) :

🔗 https://sales-forecasts.streamlit.app/

## 📝 Contexte Métier

Dans un contexte e-commerce, la capacité à anticiper les ventes futures est un levier stratégique majeur pour :
1. Optimiser la gestion des stocks
2. Améliorer la planification logistique
3. Ajuster les stratégies marketing et promotionnelles
4. Maximiser le chiffre d'affaires tout en réduisant les coûts opérationnels

Les ventes e-commerce sont influencées par de nombreux facteurs : saisonnalité, comportements clients, types de produits, modes de paiement, localisation géographique, et conditions de livraison.

---

**Note** : Ce projet est en cours de développement. Certaines fonctionnalités sont encore en phase d'implémentation.
