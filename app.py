import streamlit as st
import re
import pandas as pd

st.set_page_config(page_title="Agent Compilateur Projet", layout="centered")
st.title("🏝️ Mayotte Résiliente – Compilateur")

uploaded = st.file_uploader("Glisse ton PDF", type=["pdf"], accept_multiple_files=True)
if uploaded:
    data = []
    alerts = ["Pas de diagnostic structure", "Absence étude thermique", "Planning non défini"]
    for f in uploaded:
        txt = f.read().decode(errors="ignore")
        surface = re.search(r"(\d{1,5})\s*m", txt)
        budget  = re.search(r"(\d{3,7})", txt)
        found   = [a for a in alerts if a.lower() in txt.lower()]
        data.append({
            "Fichier": f.name,
            "Surface (m²)": surface.group(1) if surface else "–",
            "Budget (€)": budget.group(1) if budget else "–",
            "Alertes": ", ".join(found) if found else "Aucune"
        })
    df = pd.DataFrame(data)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Télécharger Excel", csv, "fiche_projet.csv")
