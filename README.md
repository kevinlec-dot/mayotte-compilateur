# IDEPHI Mayotte - Agent IA de Gestion de Projets

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## 🌟 À propos

**IDEPHI Mayotte** est un agent IA avancé pour la gestion de projets, inspiré des appels à projets France 2030 à Mayotte. Cet outil vise à accélérer le développement de produits et services innovants à haute valeur ajoutée, particulièrement dans les domaines de la construction, de l'urbanisme et des énergies renouvelables.

### 🎯 Objectifs du Projet

- **Innovation en gestion de projets complexes** : Outils numériques pour anticiper les mutations
- **Soutien aux projets structurants** : Compétitivité et création d'emplois (budget 50k-200k€)
- **Anticipation des mutations** : Formations et ingénieries professionnelles
- **Automatisation intelligente** : Suivi en temps réel et complétion automatisée

## ✨ Fonctionnalités

### 🤖 Agent IA Intelligent
- **Suivi automatisé** des tâches et projets en temps réel
- **Completion automatique** de tâches via des modèles d'IA
- **Prédiction de retards** avec machine learning (scikit-learn)
- **Optimisation automatique** des plannings

### 📊 Gestion Avancée
- **Architecture modulaire** avec classes spécialisées
- **Gestion des délais** et alertes automatiques
- **Persistence des données** (SQLite)
- **Génération de rapports** automatisée

### 🔗 Intégrations
- **APIs externes** : Prêt pour Trello, Jira, Google Calendar
- **Notifications** : Système d'alertes multi-canal
- **Interface CLI** avec Click
- **Prêt pour interface web** (Flask/FastAPI)

### 🧠 Intelligence Artificielle
- **Analyse des risques** en temps réel
- **Apprentissage automatique** pour la prédiction
- **Recommandations intelligentes**
- **Détection automatique** des goulots d'étranglement

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/votre-username/idephi-mayotte.git
cd idephi-mayotte

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur Linux/Mac :
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Installation pour le développement

```bash
# Installation avec les dépendances de développement
pip install -r requirements.txt
pip install -e .

# Installation des hooks pre-commit (optionnel)
pre-commit install
```

## 📖 Utilisation

### Interface en Ligne de Commande (CLI)

#### Créer un nouveau projet

```bash
python idephi_agent.py create-project \
  --name "Énergie Solaire Mayotte" \
  --budget 120000 \
  --manager "Jean Dupont" \
  --duration 90
```

#### Analyser un projet existant

```bash
python idephi_agent.py analyze-project PROJECT_ID
```

#### Prédire les risques

```bash
python idephi_agent.py predict-risks PROJECT_ID
```

### Utilisation Programmatique

```python
import asyncio
from idephi_agent import IDEPHIAgent, Project, ProjectStatus

async def main():
    # Initialiser l'agent
    agent = IDEPHIAgent()
    await agent.initialize()
    
    # Créer un projet
    project = Project(
        id="renewable-energy-001",
        name="Projet Énergies Renouvelables",
        description="Développement de solutions énergétiques durables",
        budget=150000.0,
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=180),
        status=ProjectStatus.EN_COURS,
        manager="Responsable Projet",
        team_members=["Dev1", "Dev2", "Expert Énergie"]
    )
    
    # Ajouter le projet
    agent.add_project(project)
    
    # Générer un rapport
    report = agent.generate_comprehensive_report(project.id)
    print(report)
    
    # Analyser les risques
    risks = agent.predict_project_risks(project.id)
    print(f"Risques détectés: {risks}")

# Lancer l'exemple
asyncio.run(main())
```

### Mode Démonstration

Pour voir toutes les fonctionnalités en action :

```bash
python idephi_agent.py
```

## 🏗️ Architecture

### Structure du Projet

```
idephi-mayotte/
├── idephi_agent.py          # Module principal
├── test_idephi_agent.py     # Tests unitaires
├── requirements.txt         # Dépendances
