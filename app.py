import os
import time
import json

# Constants
PROJECT_ROOT = os.getcwd()  # Utilise le répertoire courant (adaptable si besoin)
DOCS_ENTRANTS_PATH = os.path.join(PROJECT_ROOT, 'docs_entrants')
RESULTATS_PATH = os.path.join(PROJECT_ROOT, 'resultats')
SNAPSHOT_FILE_PATH = os.path.join(RESULTATS_PATH, 'project_snapshot.json')

# Parsing functions (simulated; in real env, integrate libs like pdfplumber, python-docx, ezdxf, Pillow)
def parse_pdf(file_path):
    print(f"Analyzing PDF: {file_path}")
    # TODO: Real impl: import pdfplumber; with pdfplumber.open(file_path) as pdf: text = '\n'.join(page.extract_text() for page in pdf.pages)
    return {"type": "pdf", "content": "Simulated extracted PDF content (CCTP, plans)"}

def parse_docx(file_path):
    print(f"Analyzing DOCX: {file_path}")
    # TODO: Real impl: from docx import Document; doc = Document(file_path); text = '\n'.join(p.text for p in doc.paragraphs)
    return {"type": "docx", "content": "Simulated extracted DOCX content (cahiers des charges)"}

def parse_dwg(file_path):
    print(f"Analyzing DWG/DXF: {file_path}")
    # TODO: Real impl: import ezdxf; doc = ezdxf.readfile(file_path); entities = [e.dxf for e in doc.modelspace()]
    return {"type": "dwg", "content": "Simulated extracted DWG metadata (plans)"}

def parse_image(file_path):
    print(f"Analyzing image: {file_path}")
    # TODO: Real impl: from PIL import Image; img = Image.open(file_path); metadata = img.info
    return {"type": "image", "content": "Simulated extracted image metadata (photos projet)"}

# Main compilateur function
def run_compilateur(file_path):
    print("\n--- Agent 1: Compilateur Projet Tropical ---")
    print(f"New file detected: {os.path.basename(file_path)}")

    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.pdf':
            extracted_data = parse_pdf(file_path)
        elif file_extension == '.docx':
            extracted_data = parse_docx(file_path)
        elif file_extension in ['.dwg', '.dxf']:
            extracted_data = parse_dwg(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            extracted_data = parse_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Improved simulated extraction with more tropical-specific fields
        project_info = {
            "maitre_ouvrage": "Simulated Maître d'Ouvrage (e.g., Conseil Départemental)",
            "typologie": "Neuf/Rénovation (e.g., ERP, logement)",
            "surface": 1500,  # m²
            "budget": 750000,  # €
            "normes_visees": ["Résistance cyclonique Cat.5", "Bioclimatisme tropical"],
            "resistance_cyclone_kmh": 230,  # Vents max supportés
            "temperature_interieure_max_c": 27,  # Confort thermique
            "stock_tampon_semaines": 5,  # Logistique insulaire
            "source_files": [os.path.basename(file_path)]
        }
        print(f"Extracted data (simulation): {project_info}")

        # Improved 3C Tropical rules with detailed checks
        risks = []
        gaps = []
        if project_info["resistance_cyclone_kmh"] < 250:
            risks.append("Risque cyclonique: Résistance vents insuffisante (<250 km/h)")
        if project_info["temperature_interieure_max_c"] > 28:
            gaps.append("Gap bioclimatique: Confort thermique non optimal (>28°C)")
        if project_info["stock_tampon_semaines"] < 4:
            gaps.append("Gap logistique: Stock tampon trop faible pour insularité")

        # Generate JSON snapshot with timestamp
        snapshot = {
            "id": f"proj_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "risks": risks,
            "gaps": gaps,
            "data": project_info,
            "next": ["run_agent2"]  # Pour chaîne avec Agent 2
        }

        with open(SNAPSHOT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=4, ensure_ascii=False)

        print(f"Generated 'project_snapshot.json' in {RESULTATS_PATH}")
        print("--- End of Agent 1 execution ---")

    except ValueError as ve:
        print(f"Validation error for file {file_path}: {str(ve)}")
    except Exception as e:
        print(f"Unexpected error processing file {file_path}: {str(e)}")

# Polling-based file watcher (using standard libs, configurable interval)
def file_watcher(path, interval=5):
    if not os.path.exists(path):
        print(f"Warning: Watched path {path} does not exist. Creating it.")
        os.makedirs(path)
    
    known_files = set(os.listdir(path))
    print(f"Watching for new files in {path} every {interval} seconds...")
    
    while True:
        current_files = set(os.listdir(path))
        new_files = current_files - known_files
        for new_file in new_files:
            full_path = os.path.join(path, new_file)
            if os.path.isfile(full_path):
                run_compilateur(full_path)
        known_files = current_files
        time.sleep(interval)

if __name__ == "__main__":
    print("Agent 1 (Compilateur) starting...")
    
    if not os.path.exists(RESULTATS_PATH):
        os.makedirs(RESULTATS_PATH)
    
    file_watcher(DOCS_ENTRANTS_PATH, interval=10)  # Interval de 10s pour éviter surcharge
