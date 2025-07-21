# IDEPHI Mayotte - Agent IA de Gestion de Projets
# Inspiré des appels à projets France 2030 pour Mayotte

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from pathlib import Path

# Imports pour les fonctionnalités avancées
import schedule
import click
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('idephi_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """États possibles d'un projet."""
    PLANIFIE = "planifié"
    EN_COURS = "en_cours"
    EN_RETARD = "en_retard"
    TERMINE = "terminé"
    SUSPENDU = "suspendu"


class TaskPriority(Enum):
    """Niveaux de priorité des tâches."""
    FAIBLE = 1
    MOYENNE = 2
    HAUTE = 3
    CRITIQUE = 4


@dataclass
class Task:
    """Représente une tâche dans un projet."""
    id: str
    name: str
    description: str
    assignee: str
    due_date: datetime
    priority: TaskPriority
    status: str = "à_faire"
    progress: float = 0.0
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    def to_dict(self) -> Dict:
        """Convertit la tâche en dictionnaire."""
        data = asdict(self)
        data['due_date'] = self.due_date.isoformat()
        data['priority'] = self.priority.value
        return data


@dataclass
class Project:
    """Représente un projet avec ses métadonnées."""
    id: str
    name: str
    description: str
    budget: float
    start_date: datetime
    end_date: datetime
    status: ProjectStatus
    manager: str
    team_members: List[str]
    tasks: List[Task] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
    
    def get_completion_rate(self) -> float:
        """Calcule le taux de completion du projet."""
        if not self.tasks:
            return 0.0
        return sum(task.progress for task in self.tasks) / len(self.tasks)
    
    def is_delayed(self) -> bool:
        """Vérifie si le projet est en retard."""
        return datetime.now() > self.end_date and self.status != ProjectStatus.TERMINE


class DatabaseManager:
    """Gestionnaire de base de données pour la persistence."""
    
    def __init__(self, db_path: str = "idephi_projects.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    budget REAL,
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT,
                    manager TEXT,
                    team_members TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    assignee TEXT,
                    due_date TEXT,
                    priority INTEGER,
                    status TEXT,
                    progress REAL,
                    estimated_hours REAL,
                    actual_hours REAL,
                    dependencies TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    metric_date TEXT,
                    completion_rate REAL,
                    budget_used REAL,
                    team_efficiency REAL,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)
    
    def save_project(self, project: Project):
        """Sauvegarde un projet en base."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO projects 
                (id, name, description, budget, start_date, end_date, status, manager, team_members)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project.id, project.name, project.description, project.budget,
                project.start_date.isoformat(), project.end_date.isoformat(),
                project.status.value, project.manager, json.dumps(project.team_members)
            ))
    
    def load_project(self, project_id: str) -> Optional[Project]:
        """Charge un projet depuis la base."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            )
            row = cursor.fetchone()
            if row:
                return Project(
                    id=row[0], name=row[1], description=row[2], budget=row[3],
                    start_date=datetime.fromisoformat(row[4]),
                    end_date=datetime.fromisoformat(row[5]),
                    status=ProjectStatus(row[6]), manager=row[7],
                    team_members=json.loads(row[8])
                )
        return None


class DelayPredictor:
    """Modèle d'apprentissage automatique pour prédire les retards."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, projects: List[Project]) -> pd.DataFrame:
        """Prépare les features pour l'entraînement."""
        features = []
        for project in projects:
            if project.tasks:
                avg_task_complexity = np.mean([len(task.dependencies) for task in project.tasks])
                total_estimated_hours = sum(task.estimated_hours for task in project.tasks)
                team_size = len(project.team_members)
                budget_per_day = project.budget / max((project.end_date - project.start_date).days, 1)
                
                features.append({
                    'budget': project.budget,
                    'team_size': team_size,
                    'task_count': len(project.tasks),
                    'avg_task_complexity': avg_task_complexity,
                    'total_estimated_hours': total_estimated_hours,
                    'budget_per_day': budget_per_day,
                    'project_duration_days': (project.end_date - project.start_date).days
                })
        
        return pd.DataFrame(features)
    
    def train(self, historical_projects: List[Project]):
        """Entraîne le modèle sur des projets historiques."""
        if len(historical_projects) < 5:
            logger.warning("Pas assez de données pour entraîner le modèle de prédiction")
            return
        
        features_df = self.prepare_features(historical_projects)
        
        # Calcul des retards comme target
        delays = []
        for project in historical_projects:
            if project.status == ProjectStatus.TERMINE:
                # Calculer le retard réel
                planned_duration = (project.end_date - project.start_date).days
                # Simulation d'un retard pour l'exemple
                actual_delay = max(0, planned_duration * 0.1)  # 10% de retard moyen
                delays.append(actual_delay)
            else:
                delays.append(0)
        
        if len(delays) == len(features_df):
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, delays, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Évaluation simple
            score = self.model.score(X_test, y_test)
            logger.info(f"Modèle entraîné avec un score R² de {score:.3f}")
    
    def predict_delay(self, project: Project) -> float:
        """Prédit le retard potentiel d'un projet."""
        if not self.is_trained:
            return 0.0
        
        features_df = self.prepare_features([project])
        if not features_df.empty:
            prediction = self.model.predict(features_df)[0]
            return max(0, prediction)
        return 0.0


class NotificationManager:
    """Gestionnaire de notifications et alertes."""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
    
    async def send_notification(self, recipient: str, message: str, priority: str = "info"):
        """Envoie une notification (simulation)."""
        logger.info(f"NOTIFICATION [{priority.upper()}] pour {recipient}: {message}")
        # Ici on pourrait intégrer Slack, email, SMS, etc.
    
    async def check_deadlines(self, projects: List[Project]):
        """Vérifie les échéances et envoie des alertes."""
        for project in projects:
            if project.status in [ProjectStatus.EN_COURS, ProjectStatus.PLANIFIE]:
                days_until_deadline = (project.end_date - datetime.now()).days
                
                if days_until_deadline <= 3:
                    await self.send_notification(
                        project.manager,
                        f"⚠️ Projet '{project.name}' : échéance dans {days_until_deadline} jours!",
                        "warning"
                    )
                
                for task in project.tasks:
                    task_days_left = (task.due_date - datetime.now()).days
                    if task_days_left <= 1 and task.status != "terminé":
                        await self.send_notification(
                            task.assignee,
                            f"🚨 Tâche '{task.name}' due demain!",
                            "urgent"
                        )


class ReportGenerator:
    """Générateur de rapports automatisés."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def generate_project_summary(self, project: Project) -> str:
        """Génère un résumé de projet."""
        completion_rate = project.get_completion_rate()
        total_tasks = len(project.tasks)
        completed_tasks = len([t for t in project.tasks if t.status == "terminé"])
        
        summary = f"""
# Rapport de Projet : {project.name}

## Vue d'ensemble
- **Statut**: {project.status.value}
- **Gestionnaire**: {project.manager}
- **Budget**: {project.budget:,.2f} €
- **Période**: {project.start_date.strftime('%d/%m/%Y')} - {project.end_date.strftime('%d/%m/%Y')}

## Progression
- **Taux de completion**: {completion_rate:.1f}%
- **Tâches terminées**: {completed_tasks}/{total_tasks}
- **Équipe**: {len(project.team_members)} membres

## Analyse
"""
        
        if project.is_delayed():
            summary += "⚠️ **Projet en retard** - Actions correctives nécessaires\n"
        elif completion_rate > 80:
            summary += "✅ **Projet sur la bonne voie** - Progression satisfaisante\n"
        else:
            summary += "📊 **Suivi requis** - Attention aux échéances\n"
        
        return summary
    
    def generate_team_performance_report(self, projects: List[Project]) -> str:
        """Génère un rapport de performance d'équipe."""
        team_stats = {}
        
        for project in projects:
            for task in project.tasks:
                if task.assignee not in team_stats:
                    team_stats[task.assignee] = {
                        'total_tasks': 0,
                        'completed_tasks': 0,
                        'total_hours': 0,
                        'efficiency': 0
                    }
                
                team_stats[task.assignee]['total_tasks'] += 1
                if task.status == "terminé":
                    team_stats[task.assignee]['completed_tasks'] += 1
                team_stats[task.assignee]['total_hours'] += task.actual_hours
        
        # Calcul de l'efficacité
        for member, stats in team_stats.items():
            if stats['total_tasks'] > 0:
                stats['efficiency'] = stats['completed_tasks'] / stats['total_tasks'] * 100
        
        report = "# Rapport de Performance d'Équipe\n\n"
        for member, stats in team_stats.items():
            report += f"## {member}\n"
            report += f"- Tâches: {stats['completed_tasks']}/{stats['total_tasks']}\n"
            report += f"- Efficacité: {stats['efficiency']:.1f}%\n"
            report += f"- Heures travaillées: {stats['total_hours']:.1f}h\n\n"
        
        return report


class IDEPHIAgent:
    """Agent IA principal pour la gestion de projets IDEPHI Mayotte."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.delay_predictor = DelayPredictor()
        self.notification_manager = NotificationManager()
        self.report_generator = ReportGenerator(self.db_manager)
        self.projects: Dict[str, Project] = {}
        
        logger.info("Agent IDEPHI initialisé avec succès")
    
    async def initialize(self):
        """Initialise l'agent et démarre les services."""
        # Démarrage du scheduler pour les notifications
        self.notification_manager.scheduler.start()
        
        # Programmation des vérifications périodiques
        self.notification_manager.scheduler.add_job(
            self._periodic_check,
            'interval',
            hours=6,
            id='periodic_check'
        )
        
        logger.info("Services de l'agent démarrés")
    
    async def _periodic_check(self):
        """Vérification périodique des projets."""
        await self.notification_manager.check_deadlines(list(self.projects.values()))
        logger.info("Vérification périodique effectuée")
    
    def add_project(self, project: Project):
        """Ajoute un nouveau projet."""
        self.projects[project.id] = project
        self.db_manager.save_project(project)
        logger.info(f"Projet '{project.name}' ajouté avec succès")
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Récupère un projet par son ID."""
        if project_id in self.projects:
            return self.projects[project_id]
        
        # Tentative de chargement depuis la base
        project = self.db_manager.load_project(project_id)
        if project:
            self.projects[project_id] = project
        return project
    
    async def complete_task_automatically(self, task_id: str, project_id: str):
        """Tente de compléter automatiquement une tâche simple."""
        project = self.get_project(project_id)
        if not project:
            return False
        
        task = next((t for t in project.tasks if t.id == task_id), None)
        if not task:
            return False
        
        # Simulation de completion automatique pour certains types de tâches
        if "rapport" in task.name.lower() and task.status == "en_cours":
            # Génération automatique de rapport
            report_content = self.report_generator.generate_project_summary(project)
            
            task.status = "terminé"
            task.progress = 100.0
            
            await self.notification_manager.send_notification(
                task.assignee,
                f"✅ Tâche '{task.name}' complétée automatiquement",
                "success"
            )
            
            logger.info(f"Tâche '{task.name}' complétée automatiquement")
            return True
        
        return False
    
    def predict_project_risks(self, project_id: str) -> Dict[str, Any]:
        """Analyse les risques d'un projet."""
        project = self.get_project(project_id)
        if not project:
            return {}
        
        risks = {
            'delay_risk': 'low',
            'budget_risk': 'low',
            'resource_risk': 'low',
            'predicted_delay_days': 0,
            'recommendations': []
        }
        
        # Prédiction de retard
        predicted_delay = self.delay_predictor.predict_delay(project)
        risks['predicted_delay_days'] = predicted_delay
        
        if predicted_delay > 7:
            risks['delay_risk'] = 'high'
            risks['recommendations'].append("Revoir la planification et allouer plus de ressources")
        elif predicted_delay > 3:
            risks['delay_risk'] = 'medium'
            risks['recommendations'].append("Surveiller de près les échéances critiques")
        
        # Analyse des ressources
        overloaded_members = []
        for member in project.team_members:
            member_tasks = [t for t in project.tasks if t.assignee == member and t.status != "terminé"]
            if len(member_tasks) > 5:
                overloaded_members.append(member)
        
        if overloaded_members:
            risks['resource_risk'] = 'high'
            risks['recommendations'].append(f"Redistribuer les tâches pour: {', '.join(overloaded_members)}")
        
        # Analyse budgétaire (simulation)
        completion_rate = project.get_completion_rate()
        days_elapsed = (datetime.now() - project.start_date).days
        total_days = (project.end_date - project.start_date).days
        
        if days_elapsed > 0:
            expected_completion = (days_elapsed / total_days) * 100
            if completion_rate < expected_completion * 0.8:
                risks['budget_risk'] = 'medium'
                risks['recommendations'].append("Revoir l'allocation budgétaire")
        
        return risks
    
    async def optimize_project_schedule(self, project_id: str):
        """Optimise automatiquement le planning d'un projet."""
        project = self.get_project(project_id)
        if not project:
            return
        
        # Tri des tâches par priorité et dépendances
        sorted_tasks = sorted(project.tasks, 
                            key=lambda t: (t.priority.value, len(t.dependencies)), 
                            reverse=True)
        
        # Redistribution intelligente des échéances
        current_date = datetime.now()
        for i, task in enumerate(sorted_tasks):
            if task.status not in ["terminé", "en_cours"]:
                # Calcul d'une nouvelle date optimale
                optimal_date = current_date + timedelta(days=(i + 1) * 2)
                if optimal_date != task.due_date:
                    old_date = task.due_date
                    task.due_date = optimal_date
                    
                    await self.notification_manager.send_notification(
                        task.assignee,
                        f"📅 Tâche '{task.name}' reprogrammée du {old_date.strftime('%d/%m')} au {optimal_date.strftime('%d/%m')}",
                        "info"
                    )
        
        logger.info(f"Planning du projet '{project.name}' optimisé")
    
    def generate_comprehensive_report(self, project_id: str) -> str:
        """Génère un rapport complet de projet."""
        project = self.get_project(project_id)
        if not project:
            return "Projet non trouvé"
        
        # Rapport de base
        report = self.report_generator.generate_project_summary(project)
        
        # Analyse des risques
        risks = self.predict_project_risks(project_id)
        report += "\n## Analyse des Risques\n"
        report += f"- **Risque de retard**: {risks['delay_risk']}\n"
        report += f"- **Risque budgétaire**: {risks['budget_risk']}\n"
        report += f"- **Risque ressources**: {risks['resource_risk']}\n"
        
        if risks['predicted_delay_days'] > 0:
            report += f"- **Retard prédit**: {risks['predicted_delay_days']:.1f} jours\n"
        
        if risks['recommendations']:
            report += "\n## Recommandations\n"
            for rec in risks['recommendations']:
                report += f"- {rec}\n"
        
        return report


# Interface CLI avec Click
@click.group()
def cli():
    """IDEPHI Mayotte - Agent IA de Gestion de Projets"""
    pass


@cli.command()
@click.option('--name', required=True, help='Nom du projet')
@click.option('--budget', type=float, required=True, help='Budget en euros')
@click.option('--manager', required=True, help='Gestionnaire de projet')
@click.option('--duration', type=int, default=30, help='Durée en jours')
def create_project(name, budget, manager, duration):
    """Crée un nouveau projet."""
    import uuid
    
    project = Project(
        id=str(uuid.uuid4()),
        name=name,
        description=f"Projet créé dans le cadre du programme IDEPHI Mayotte",
        budget=budget,
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=duration),
        status=ProjectStatus.PLANIFIE,
        manager=manager,
        team_members=[manager]
    )
    
    agent = IDEPHIAgent()
    agent.add_project(project)
    
    click.echo(f"✅ Projet '{name}' créé avec l'ID: {project.id}")
    click.echo(f"Budget: {budget:,.2f} €")
    click.echo(f"Gestionnaire: {manager}")


@cli.command()
@click.argument('project_id')
def analyze_project(project_id):
    """Analyse un projet et génère un rapport."""
    agent = IDEPHIAgent()
    project = agent.get_project(project_id)
    
    if not project:
        click.echo("❌ Projet non trouvé")
        return
    
    report = agent.generate_comprehensive_report(project_id)
    click.echo(report)


@cli.command()
@click.argument('project_id')
def predict_risks(project_id):
    """Prédit les risques d'un projet."""
    agent = IDEPHIAgent()
    risks = agent.predict_project_risks(project_id)
    
    if not risks:
        click.echo("❌ Projet non trouvé")
        return
    
    click.echo("🔍 Analyse des Risques:")
    click.echo(f"  Retard: {risks['delay_risk']}")
    click.echo(f"  Budget: {risks['budget_risk']}")
    click.echo(f"  Ressources: {risks['resource_risk']}")
    
    if risks['predicted_delay_days'] > 0:
        click.echo(f"  Retard prédit: {risks['predicted_delay_days']:.1f} jours")
    
    if risks['recommendations']:
        click.echo("\n💡 Recommandations:")
        for rec in risks['recommendations']:
            click.echo(f"  - {rec}")


async def main():
    """Fonction principale pour démonstration."""
    # Initialisation de l'agent
    agent = IDEPHIAgent()
    await agent.initialize()
    
    # Création d'un projet d'exemple inspiré de France 2030 Mayotte
    sample_project = Project(
        id="mayotte-renewable-001",
        name="Développement Énergies Renouvelables Mayotte",
        description="Projet d'innovation pour le développement de solutions énergétiques durables à Mayotte, dans le cadre de France 2030",
        budget=150000.0,  # Budget typique 50k-200k€
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=180),
        status=ProjectStatus.EN_COURS,
        manager="Jean Dupont",
        team_members=["Jean Dupont", "Marie Martin", "Pierre Durand"]
    )
    
    # Ajout de tâches d'exemple
    sample_project.tasks = [
        Task(
            id="task-001",
            name="Étude de faisabilité énergétique",
            description="Analyse des potentiels énergétiques renouvelables",
            assignee="Marie Martin",
            due_date=datetime.now() + timedelta(days=30),
            priority=TaskPriority.HAUTE,
            status="en_cours",
            progress=60.0,
            estimated_hours=80.0,
            actual_hours=50.0
        ),
        Task(
            id="task-002",
            name="Rapport d'impact environnemental",
            description="Évaluation de l'impact environnemental du projet",
            assignee="Pierre Durand",
            due_date=datetime.now() + timedelta(days=45),
            priority=TaskPriority.MOYENNE,
            status="à_faire",
            progress=0.0,
            estimated_hours=60.0,
            actual_hours=0.0
        )
    ]
    
    # Démonstration des fonctionnalités
    agent.add_project(sample_project)
    
    # Génération de rapport
    print("=== RAPPORT DE PROJET ===")
    report = agent.generate_comprehensive_report(sample_project.id)
    print(report)
    
    # Analyse des risques
    print("\n=== ANALYSE DES RISQUES ===")
    risks = agent.predict_project_risks(sample_project.id)
    for key, value in risks.items():
        print(f"{key}: {value}")
    
    # Optimisation du planning
    print("\n=== OPTIMISATION DU PLANNING ===")
    await agent.optimize_project_schedule(sample_project.id)
    
    # Completion automatique d'une tâche
    print("\n=== COMPLETION AUTOMATIQUE ===")
    success = await agent.complete_task_automatically("task-002", sample_project.id)
    print(f"Completion automatique: {'✅ Réussie' if success else '❌ Échouée'}")
    
    print("\n🎉 Démonstration terminée!")


if __name__ == "__main__":
    # Lancement via CLI ou démonstration
    import sys
    
    if len(sys.argv) > 1:
        cli()
    else:
        # Mode démonstration
        asyncio.run(main())
