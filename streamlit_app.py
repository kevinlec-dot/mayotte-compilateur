import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import asyncio

st.set_page_config(
    page_title="IDEPHI Mayotte - Agent IA",
    page_icon="🌴",
    layout="wide"
)

st.title("🌴 IDEPHI Mayotte - Agent IA de Gestion de Projets")
st.subtitle("Application inspirée de France 2030 pour Mayotte")

# Sidebar pour navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisir une page", 
                           ["Accueil", "Créer un Projet", "Analyser un Projet", "Rapports"])

if page == "Accueil":
    st.header("Bienvenue dans IDEPHI Mayotte ! 🚀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objectifs")
        st.write("""
        - Gestion intelligente de projets 50k-200k€
        - Prédiction IA des retards de projet
        - Optimisation automatique des plannings
        - Conformité France 2030 Mayotte
        """)
        
    with col2:
        st.subheader("🌟 Fonctionnalités")
        st.write("""
        - Agent IA avec machine learning
        - Notifications automatiques
        - Rapports intelligents
        - Interface web intuitive
        """)
    
    st.success("✅ Application déployée avec succès sur Streamlit Cloud !")

elif page == "Créer un Projet":
    st.header("📝 Créer un Nouveau Projet")
    
    with st.form("project_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom du projet", placeholder="Ex: Énergie Solaire Mayotte")
            budget = st.number_input("Budget (€)", min_value=50000, max_value=200000, value=100000)
            manager = st.text_input("Gestionnaire", placeholder="Votre nom")
            
        with col2:
            duration = st.number_input("Durée (jours)", min_value=30, max_value=365, value=90)
            sector = st.selectbox("Secteur", 
                                ["Énergies Renouvelables", "Construction Durable", 
                                 "Numérique", "Urbanisme", "Formation"])
            
        submitted = st.form_submit_button("🚀 Créer le Projet")
        
        if submitted and name and manager:
            st.success(f"✅ Projet '{name}' créé avec succès !")
            st.info(f"💰 Budget: {budget:,}€ | 👤 Gestionnaire: {manager} | ⏱️ Durée: {duration} jours")
            
            # Simulation des données du projet
            st.subheader("📊 Aperçu du Projet")
            
            progress_data = pd.DataFrame({
                'Tâche': ['Étude de faisabilité', 'Développement', 'Tests', 'Déploiement'],
                'Progression': [100, 60, 20, 0],
                'Assigné à': ['Expert', 'Développeur', 'Testeur', 'Équipe']
            })
            
            st.dataframe(progress_data, use_container_width=True)
            st.bar_chart(progress_data.set_index('Tâche')['Progression'])

elif page == "Analyser un Projet":
    st.header("🔍 Analyse de Projet avec IA")
    
    # Simulation de l'analyse IA
    project_id = st.text_input("ID du Projet", placeholder="Ex: mayotte-solaire-2025")
    
    if st.button("🤖 Analyser avec IA"):
        if project_id:
            with st.spinner("Analyse en cours..."):
                import time
                time.sleep(2)  # Simulation
                
            st.success("✅ Analyse terminée !")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risque de Retard", "Faible", "-10%")
            with col2:
                st.metric("Progression", "65%", "+5%")
            with col3:
                st.metric("Budget Utilisé", "45%", "+2%")
            
            st.subheader("🎯 Recommandations IA")
            st.info("• Projet sur la bonne voie")
            st.info("• Surveiller la tâche 'Tests' (échéance proche)")
            st.info("• Optimiser l'allocation des ressources")
            
            # Graphique de prédiction
            dates = pd.date_range(start='2025-01-01', periods=10, freq='W')
            predictions = [65, 68, 72, 75, 78, 82, 85, 88, 92, 95]
            
            chart_data = pd.DataFrame({
                'Date': dates,
                'Progression Prédite (%)': predictions
            })
            
            st.line_chart(chart_data.set_index('Date'))

elif page == "Rapports":
    st.header("📈 Rapports et Métriques")
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Projets Actifs", "12", "+3")
    with col2:
        st.metric("Taux de Réussite", "87%", "+5%")
    with col3:
        st.metric("Budget Total", "1.2M€", "+200k€")
    with col4:
        st.metric("Équipes", "45", "+8")
    
    # Graphiques
    st.subheader("📊 Répartition par Secteur")
    
    sectors_data = pd.DataFrame({
        'Secteur': ['Énergies Renouvelables', 'Construction', 'Numérique', 'Formation'],
        'Nombre de Projets': [5, 3, 2, 2],
        'Budget (k€)': [600, 350, 180, 70]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(sectors_data.set_index('Secteur')['Nombre de Projets'])
    
    with col2:
        st.bar_chart(sectors_data.set_index('Secteur')['Budget (k€)'])

# Footer
st.markdown("---")
st.markdown("🌴 **IDEPHI Mayotte** - Agent IA pour projets France 2030 | Fait avec ❤️ à Mayotte")
