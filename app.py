import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdMolTransforms
import requests
import re
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(page_title="ChemStudio 2D/3D", layout="wide")

# --- FONCTIONS D'INTERROGATION CHIMIQUE (PUBCHEM & CACTUS) ---
@st.cache_data(show_spinner=False)
def get_chemical_identifiers(smiles):
    """Récupère le nom IUPAC, le numéro CAS et le nom trivial à partir d'un SMILES."""
    result = {
        "iupac": "Non répertorié (structure nouvelle ou absente de PubChem)",
        "cas": "Non trouvé",
        "trivial": "Non trouvé",
        "cid": None
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return result
        inchikey = Chem.MolToInchiKey(mol)

        # 1. Récupération CID + Nom IUPAC via PubChem
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/IUPACName,Title/JSON"
        r_prop = requests.get(prop_url, timeout=5)
        if r_prop.status_code == 200:
            props = r_prop.json()["PropertyTable"]["Properties"][0]
            result["cid"] = props.get("CID")
            result["iupac"] = props.get("IUPACName", result["iupac"])
            result["trivial"] = props.get("Title", result["trivial"])

        # 2. Récupération des synonymes et du numéro CAS via PubChem
        if result["cid"]:
            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{result['cid']}/synonyms/JSON"
            r_syn = requests.get(syn_url, timeout=5)
            if r_syn.status_code == 200:
                synonyms = r_syn.json()["InformationList"]["Information"][0]["Synonym"]
                if result["trivial"] == "Non trouvé" and synonyms:
                    result["trivial"] = synonyms[0]
                
                # Filtrage par expression régulière du format officiel CAS (XX-XX-X)
                cas_pattern = re.compile(r"^\d{2,7}-\d{2}-\d$")
                for syn in synonyms:
                    if cas_pattern.match(syn):
                        result["cas"] = syn
                        break
    except Exception:
        pass
    return result

@st.cache_data(show_spinner=False)
def resolve_query_to_smiles(query):
    """Convertit un numéro CAS ou un nom trivial en SMILES via PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query.strip()}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
    except Exception:
        return None
    return None

# --- GÉNÉRATION ET MODIFICATION 3D ---
def generate_3d_mol(smiles, force_field="MMFF94", conformer_seed=42):
    """Génère un objet RDKit 3D optimisé."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = conformer_seed
    res = AllChem.EmbedMolecule(mol_3d, params)
    if res == 0:
        if force_field == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(mol_3d)
        else:
            AllChem.UFFOptimizeMolecule(mol_3d)
    return mol_3d

def render_3d_viewer(mol_block_3d, style="stick", show_hydrogens=True, spin=False):
    """Affiche le visualiseur 3D interactif avec étiquettes d'atomes."""
    view = py3Dmol.view(width=550, height=420)
    view.addModel(mol_block_3d, "mol")
    
    if style == "Bâtonnets & Sphères (Ball & Stick)":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    elif style == "Bâtonnets (Stick)":
        view.setStyle({"stick": {"radius": 0.2}})
    elif style == "Sphères de Van der Waals (Spacefill)":
        view.setStyle({"sphere": {}})
        
    if not show_hydrogens:
        view.setStyle({"elem": "H"}, {})

    view.zoomTo()
    view.spin(spin)
    components.html(view._make_html(), height=430)

# --- INITIALISATION DE L'ÉTAT ---
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirine par défaut
if "custom_3d_molblock" not in st.session_state:
    st.session_state.custom_3d_molblock = None

# --- BARRE LATÉRALE : RECHERCHE PAR CAS OU NOM TRIVIAL ---
with st.sidebar:
    st.header("🔍 Recherche par CAS / Nom")
    st.caption("Entrez un numéro CAS (ex: `50-78-2`, `58-08-2`) ou un nom trivial (`Paracetamol`, `Caffeine`) pour charger la structure.")
    search_input = st.text_input("Numéro CAS ou Nom :", placeholder="Ex: 50-78-2 ou Aspirin")
    if st.button("Charger la molécule", use_container_width=True):
        if search_input:
            found_smiles = resolve_query_to_smiles(search_input)
            if found_smiles:
                st.session_state.current_smiles = found_smiles
                st.session_state.custom_3d_molblock = None
                st.success("Molécule chargée !")
            else:
                st.error("Impossible de trouver ce CAS ou ce nom dans PubChem.")
                
    st.divider()
    st.header("⚙️ Paramètres 3D")
    style_3d = st.selectbox(
        "Style de rendu 3D",
        ["Bâtonnets & Sphères (Ball & Stick)", "Bâtonnets (Stick)", "Sphères de Van der Waals (Spacefill)"]
    )
    ff_choice = st.radio("Champ de force d'optimisation", ["MMFF94", "UFF"], horizontal=True)
    show_h = st.checkbox("Afficher les hydrogènes (H)", value=True)
    spin_3d = st.checkbox("Rotation automatique", value=False)
    conf_seed = st.slider("Explorer les conformères 3D", min_value=1, max_value=100, value=42)

# --- INTERFACE PRINCIPALE ---
st.title("🧪 Studio Moléculaire 2D ⇄ 3D & Nomenclature")

col_2d, col_3d = st.columns([1.3, 1.2])

with col_2d:
    st.subheader("1. Éditeur 2D (Style ChemDraw)")
    st.caption("Dessinez ou modifiez la molécule puis cliquez sur **Apply** (en bas à droite du cadre).")
    ketcher_smiles = st_ketcher(st.session_state.current_smiles, height=480)
    
    if ketcher_smiles and ketcher_smiles != st.session_state.current_smiles:
        st.session_state.current_smiles = ketcher_smiles
        st.session_state.custom_3d_molblock = None

active_smiles = st.session_state.current_smiles

with col_3d:
    st.subheader("2. Visualisation & Modification 3D")
    
    if active_smiles:
        mol_3d = generate_3d_mol(active_smiles, force_field=ff_choice, conformer_seed=conf_seed)
        
        if mol_3d:
            # --- OUTIL DE MODIFICATION 3D (Longueurs de liaison & Angles dièdres) ---
            with st.expander("🛠️ Modifier la géométrie 3D (Liaisons / Angles / Coordonnées)"):
                num_atoms = mol_3d.GetNumAtoms()
                st.caption(f"La molécule contient **{num_atoms} atomes** (indices de `0` à `{num_atoms - 1}`).")
                
                mod_col1, mod_col2, mod_col3 = st.columns(3)
                atom_i = mod_col1.number_input("Atome 1 (index)", 0, num_atoms - 1, 0)
                atom_j = mod_col2.number_input("Atome 2 (index)", 0, num_atoms - 1, min(1, num_atoms - 1))
                
                conf = mol_3d.GetConformer()
                current_dist = round(rdMolTransforms.GetBondLength(conf, int(atom_i), int(atom_j)), 2)
                new_dist = mod_col3.slider("Distance (Å)", 0.8, 3.5, float(min(max(current_dist, 0.8), 3.5)), 0.05)
                
                if new_dist != current_dist and atom_i != atom_j:
                    try:
                        rdMolTransforms.SetBondLength(conf, int(atom_i), int(atom_j), new_dist)
                    except Exception:
                        st.warning("Ces deux atomes font partie d'un cycle rigide et ne peuvent pas être étirés directement.")

            mol_block_3d = Chem.MolToMolBlock(mol_3d)
            render_3d_viewer(mol_block_3d, style=style_3d, show_hydrogens=show_h, spin=spin_3d)
            
            # Bouton d'export 3D (.sdf / .mol)
            st.download_button(
                "📥 Télécharger la géométrie 3D (.sdf)",
                data=mol_block_3d,
                file_name="molecule_3d.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True
            )
        else:
            st.error("Structure chimique invalide.")

# --- SECTION NOMENCLATURE : IUPAC, CAS & NOM TRIVIAL ---
st.divider()
st.subheader("3. Identification : Nom IUPAC, Numéro CAS & Nom Trivial")

if active_smiles:
    with st.spinner("Recherche de la nomenclature officielle (PubChem)..."):
        info = get_chemical_identifiers(active_smiles)
        mol_check = Chem.MolFromSmiles(active_smiles)
        
    if mol_check:
        id_col1, id_col2, id_col3, id_col4 = st.columns([1.2, 1, 1, 1])
        
        id_col1.metric("🏷️ Nom Trivial / Commun", info["trivial"])
        id_col2.metric("🔢 Numéro CAS", info["cas"])
        id_col3.metric("🧪 Formule Brute", rdMolDescriptors.CalcMolFormula(mol_check))
        id_col4.metric("⚖️ Masse Molaire", f"{round(Descriptors.MolWt(mol_check), 2)} g/mol")
        
        st.markdown("**📘 Nom IUPAC systématique :**")
        st.info(f"**{info['iupac']}**")
        
        with st.expander("Voir les identifiants structurels complets (SMILES, InChI, InChIKey)"):
            st.code(f"SMILES   : {Chem.MolToSmiles(mol_check, canonical=True)}\nInChIKey : {Chem.MolToInchiKey(mol_check)}", language="text")
