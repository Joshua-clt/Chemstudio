import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdMolTransforms, Draw, rdChemReactions, rdFMCS, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize
import requests
import re
import numpy as np
import pandas as pd
import py3Dmol
from collections import Counter

st.set_page_config(page_title="ChemStudio Ultra", layout="wide")

# ==========================================
# 1. FONCTIONS RÉSEAU & CHIMIE
# ==========================================
@st.cache_data(show_spinner=False)
def get_chemical_identifiers(smiles):
    res = {"iupac": "Non répertorié", "cas": "Non trouvé", "trivial": "Composé personnalisé", "cid": None}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return res
        inchikey = Chem.MolToInchiKey(mol)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/IUPACName,Title/JSON"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            res["cid"] = props.get("CID")
            res["iupac"] = props.get("IUPACName", res["iupac"])
            res["trivial"] = props.get("Title", res["trivial"])

        if res["cid"]:
            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{res['cid']}/synonyms/JSON"
            r_syn = requests.get(syn_url, timeout=5)
            if r_syn.status_code == 200:
                synonyms = r_syn.json()["InformationList"]["Information"][0]["Synonym"]
                cas_pattern = re.compile(r"^\d{2,7}-\d{2}-\d$")
                for syn in synonyms:
                    if cas_pattern.match(syn):
                        res["cas"] = syn
                        break
    except Exception:
        pass
    return res

@st.cache_data(show_spinner=False)
def resolve_query_to_smiles(query):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query.strip()}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def get_ghs_safety(cid):
    if not cid:
        return [], []
    pictos, hazards = [], []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            raw_text = r.text
            for code in sorted(set(re.findall(r"GHS0[1-9]", raw_text))):
                pictos.append(f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.svg")
            hazards = sorted(list(set(re.findall(r"H[234]\d{2}[^\"\\]*", raw_text))))[:6]
    except Exception:
        pass
    return pictos, hazards

def generate_3d_mol(smiles, force_field="MMFF94", seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_3d, params) == 0:
        if force_field == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(mol_3d)
        else:
            AllChem.UFFOptimizeMolecule(mol_3d)
    return mol_3d

def render_3d_viewer(mol_3d, style="Ball & Stick", show_h=True, color_charges=False, spin=False):
    mol_block = Chem.MolToMolBlock(mol_3d)
    view = py3Dmol.view(width=540, height=410)
    view.addModel(mol_block, "mol")
    if style == "Ball & Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.2}})
    else:
        view.setStyle({"sphere": {}})
    if color_charges:
        AllChem.ComputeGasteigerCharges(mol_3d)
        view.setStyle({"stick": {"radius": 0.18, "colorscheme": "RdYlBu"}})
    if not show_h:
        view.setStyle({"elem": "H"}, {})
    view.zoomTo()
    view.spin(spin)
    components.html(view._make_html(), height=420)

def estimate_chemprop(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    ar_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = ar_atoms / heavy if heavy > 0 else 0

    logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * ap
    sol_mg_ml = (10 ** logs) * mw
    tb_k = 198.2 + 2.04 * mw + 1.15 * tpsa + 18.0 * hbd - 4.5 * rot
    tm_k = 115.0 + 1.15 * mw + 0.85 * tpsa + 22.0 * hbd + 14.0 * rings - 5.5 * rot
    return {
        "mr": round(mr, 2),
        "logs": round(logs, 2),
        "sol_mg_ml": round(sol_mg_ml, 3),
        "tb_c": round(tb_k - 273.15, 1),
        "tm_c": round(tm_k - 273.15, 1)
    }

def calc_isotopic_spectrum(mol):
    iso_table = {
        "H": [(1.007825, 0.999885), (2.014102, 0.000115)],
        "C": [(12.000000, 0.9893), (13.003355, 0.0107)],
        "N": [(14.003074, 0.99636), (15.000109, 0.00364)],
        "O": [(15.994915, 0.99757), (16.999132, 0.00038), (17.999160, 0.00205)],
        "F": [(18.998403, 1.0)], "P": [(30.973762, 1.0)],
        "S": [(31.972071, 0.9499), (32.971459, 0.0075), (33.967867, 0.0425)],
        "Cl": [(34.968853, 0.7576), (36.965903, 0.2424)],
        "Br": [(78.918337, 0.5069), (80.916290, 0.4931)],
        "I": [(126.904473, 1.0)]
    }
    mol_h = Chem.AddHs(mol)
    pt = Chem.GetPeriodicTable()
    dist = {0: (0.0, 1.0)}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        isotopes = iso_table.get(sym, [(pt.GetMostCommonIsotopeMass(sym), 1.0)])
        base_m = isotopes[0][0]
        new_dist = {}
        for shift_old, (m_old, p_old) in dist.items():
            for m_iso, p_iso in isotopes:
                shift_add = int(round(m_iso - base_m))
                s_tot = shift_old + shift_add
                p_new = p_old * p_iso
                if p_new > 1e-6:
                    m_new = m_old + m_iso
                    if s_tot in new_dist:
                        m_prev, p_prev = new_dist[s_tot]
                        p_sum = p_prev + p_new
                        new_dist[s_tot] = ((m_prev * p_prev + m_new * p_new) / p_sum, p_sum)
                    else:
                        new_dist[s_tot] = (m_new, p_new)
        dist = new_dist
    max_p = max(p for _, p in dist.values()) if dist else 1.0
    rows = []
    for s in sorted(dist.keys()):
        m_val, p_val = dist[s]
        rel_int = (p_val / max_p) * 100.0
        if rel_int >= 0.1:
            rows.append({"Pic": f"M+{s}" if s > 0 else "M", "m/z exact": round(m_val, 5), "Intensité Relative (%)": round(rel_int, 2)})
    return pd.DataFrame(rows)

def simulate_nmr(mol):
    mol_h = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    c_groups = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "C":
            c_groups.setdefault(ranks[idx], []).append(atom)
    c_rows = []
    for r, atoms in c_groups.items():
        a = atoms[0]
        hyb = str(a.GetHybridization())
        arom = a.GetIsAromatic()
        o_nb = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "O")
        n_nb = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "N")
        h_nb = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
        has_c_o = any(nb.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0 for nb in a.GetNeighbors())
        if has_c_o:
            shift, env = (172.0 if (o_nb >= 2 or n_nb >= 1) else 202.0), "Carbonyle (C=O)"
        elif arom:
            shift, env = (155.0 if (o_nb or n_nb) else 128.5), "Aromatique (Ar-C)"
        elif hyb == "SP2":
            shift, env = 135.0, "Alcène (C=C)"
        else:
            shift, env = 18.0 + 28.0 * o_nb + 18.0 * n_nb + (3 - h_nb) * 9.5, f"Aliphatique (CH{h_nb})"
        c_rows.append({"Atomes #": ", ".join(str(x.GetIdx()) for x in atoms), "Type": env, "Nb C": len(atoms), "δ ¹³C estimé (ppm)": round(shift, 1)})

    h_groups = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "H":
            h_groups.setdefault(ranks[idx], []).append(atom)
    h_rows = []
    mult_names = {0: "s", 1: "d", 2: "t", 3: "q"}
    for r, h_atoms in h_groups.items():
        h0 = h_atoms[0]
        parent = h0.GetNeighbors()[0]
        p_sym = parent.GetSymbol()
        if p_sym == "O":
            shift, mult, env = 4.8, "s large", "Hydroxyle (-OH)"
        elif p_sym == "N":
            shift, mult, env = 6.2, "s large", "Amine / Amide (-NH)"
        else:
            vicinal_h = sum(sum(1 for n2 in nb.GetNeighbors() if n2.GetSymbol() == "H") for nb in parent.GetNeighbors() if nb.GetSymbol() == "C")
            mult = mult_names.get(vicinal_h, "m")
            shift, env = (7.35 if parent.GetIsAromatic() else 1.25), ("Aromatique (Ar-H)" if parent.GetIsAromatic() else "Aliphatique (C-H)")
        h_rows.append({"Intégration": f"{len(h_atoms)}H", "Type": env, "Multiplicité": mult, "δ ¹H estimé (ppm)": round(shift, 2)})
    return pd.DataFrame(h_rows), pd.DataFrame(c_rows)

def build_lorentzian_spectrum(df_peaks, col_shift, x_min, x_max, width=0.08):
    x = np.linspace(x_min, x_max, 800)
    y = np.zeros_like(x)
    for _, row in df_peaks.iterrows():
        x0 = row[col_shift]
        y += 1.0 / (1.0 + ((x - x0) / width) ** 2)
    return pd.DataFrame({"δ (ppm)": x, "Intensité": y}).set_index("δ (ppm)")

def render_publication_svg(mol, style_name="ACS Document 1996 (Noir & Blanc)", show_indices=False):
    mol_copy = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol_copy)
    Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
    drawer = Draw.MolDraw2DSVG(520, 420)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = show_indices
    if style_name.startswith("ACS"):
        Draw.SetACS1996Mode(opts, Draw.MeanBondLength(mol_copy))
        opts.useBWAtomPalette()
    elif style_name.startswith("RSC"):
        opts.bondLineWidth = 2.0
    elif style_name.startswith("Wiley"):
        opts.bondLineWidth = 2.6
        opts.useBWAtomPalette()
    elif style_name.startswith("Mode Sombre"):
        opts.setBackgroundColour((0.12, 0.14, 0.18))
        opts.setSymbolColour((0.95, 0.95, 0.95))
        opts.bondLineWidth = 2.2
    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# ==========================================
# 2. ÉTAT DE SESSION & BARRE LATÉRALE
# ==========================================
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
if "history" not in st.session_state:
    st.session_state.history = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"]
if "rxn_smiles" not in st.session_state:
    st.session_state.rxn_smiles = "CC(=O)O.OCC>>CC(=O)OCC"

with st.sidebar:
    st.title("🧪 ChemStudio Ultra")
    query = st.text_input("🔍 Recherche (CAS, Nom, SMILES) :", placeholder="Ex: 50-78-2, Ibuprofen...")
    if st.button("Charger", use_container_width=True) and query:
        mol_t = Chem.MolFromSmiles(query)
        s_res = Chem.MolToSmiles(mol_t) if mol_t else resolve_query_to_smiles(query)
        if s_res:
            st.session_state.current_smiles = s_res
            if s_res not in st.session_state.history:
                st.session_state.history.insert(0, s_res)
            st.success("Chargé !")
        else:
            st.error("Introuvable.")

    st.divider()
    exemples = {
        "Aspirine (50-78-2)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Paracétamol (103-90-2)": "CC(=O)NC1=CC=C(O)C=C1",
        "Ibuprofène (15687-27-1)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Caféine (58-08-2)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    ex_sel = st.selectbox("⚡ Exemples :", list(exemples.keys()))
    if st.button("Appliquer", use_container_width=True):
        st.session_state.current_smiles = exemples[ex_sel]

    st.divider()
    hist_sel = st.selectbox("🕘 Historique :", st.session_state.history)
    if st.button("Recharger", use_container_width=True):
        st.session_state.current_smiles = hist_sel

# Entête permanent
raw_in = st.session_state.current_smiles
single_smiles = raw_in.split(">")[0].split(".")[0] if (">" in raw_in or "." in raw_in) else raw_in
mol_main = Chem.MolFromSmiles(single_smiles)

if mol_main:
    ident = get_chemical_identifiers(single_smiles)
    b1, b2, b3, b4 = st.columns([1.4, 1, 1, 1])
    b1.metric("🏷️ Nom Trivial", ident["trivial"])
    b2.metric("🔢 Numéro CAS", ident["cas"])
    b3.metric("🧪 Formule", rdMolDescriptors.CalcMolFormula(mol_main))
    b4.metric("⚖️ Masse Molaire", f"{round(Descriptors.MolWt(mol_main), 2)} g/mol")
    st.caption(f"**Nom IUPAC :** `{ident['iupac']}`")

# ==========================================
# 3. ONGLETS PRINCIPAUX
# ==========================================
tab_2d3d, tab_rxn, tab_struct, tab_spec, tab_export = st.tabs([
    "🎨 1. Atelier 2D & 3D",
    "⚗️ 2. Réactions",
    "🔬 3. Structure & ChemProp",
    "📈 4. Spectrométrie & RMN",
    "📥 5. Sécurité & Publication"
])

with tab_2d3d:
    col_l, col_r = st.columns([1.2, 1.1])
    with col_l:
        st.markdown("##### Entrée SMILES (Éditeur textuel)")
        # Saisie SMILES directe (plus stable et sans bug d'installation de composants graphiques lourds)
        user_smiles = st.text_input("Modifier le SMILES actif :", value=single_smiles)
        if user_smiles != single_smiles:
            if Chem.MolFromSmiles(user_smiles):
                st.session_state.current_smiles = user_smiles
                st.rerun()
            else:
                st.error("SMILES invalide.")
        
        st.info("💡 Astuce : Vous pouvez coller n'importe quel code SMILES ou charger une molécule depuis la barre latérale.")
        
        if mol_main:
            img_2d = Draw.MolToImage(mol_main, size=(450, 350))
            st.image(img_2d, caption="Aperçu 2D de la molécule active")

    with col_r:
        st.markdown("##### Modèle 3D Interactif")
        if mol_main:
            c1, c2, c3 = st.columns(3)
            style_3d = c1.selectbox("Style", ["Ball & Stick", "Stick", "Spacefill"])
            ff_3d = c2.selectbox("Champ de force", ["MMFF94", "UFF"])
            seed_3d = c3.number_input("Conformère", 1, 100, 42)
            mol_3d = generate_3d_mol(single_smiles, force_field=ff_3d, seed=int(seed_3d))
            if mol_3d:
                render_3d_viewer(mol_3d, style=style_3d, show_h=True, color_charges=False, spin=False)

with tab_rxn:
    st.markdown("##### Grille de Stœchiométrie & Bilan")
    rxn_str = st.text_input("SMILES de réaction :", value=st.session_state.rxn_smiles)
    st.session_state.rxn_smiles = rxn_str
    if ">>" in rxn_str:
        parts = rxn_str.split(">>")
        r_mols = [Chem.MolFromSmiles(s) for s in parts[0].split(".") if Chem.MolFromSmiles(s)]
        p_mols = [Chem.MolFromSmiles(s) for s in parts[1].split(".") if Chem.MolFromSmiles(s)]
        if r_mols and p_mols:
            mw_r1 = Descriptors.MolWt(r_mols[0])
            ref_mg = st.number_input("Masse engagée Réactif 1 (mg) :", value=500.0)
            mmol = ref_mg / mw_r1
            st.metric("Quantité Réactif 1", f"{mmol:.3f} mmol")

with tab_struct:
    if mol_main:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Propriétés Physico-Chimiques")
            cprop = estimate_chemprop(mol_main)
            st.metric("Ébullition estimée (Tb)", f"{cprop['tb_c']} °C")
            st.metric("Fusion estimée (Tf)", f"{cprop['tm_c']} °C")
            st.metric("LogS (Solubilité)", f"{cprop['logs']} ({cprop['sol_mg_ml']} mg/mL)")
        with c2:
            st.markdown("##### Descripteurs Lipinski & TPSA")
            st.metric("LogP", round(Crippen.MolLogP(mol_main), 2))
            st.metric("TPSA", f"{round(Descriptors.TPSA(mol_main), 2)} Å²")

with tab_spec:
    if mol_main:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("##### Massif Isotopique")
            df_iso = calc_isotopic_spectrum(mol_main)
            st.dataframe(df_iso, use_container_width=True, hide_index=True)
        with s2:
            st.markdown("##### RMN Estimée")
            df_h, df_c = simulate_nmr(mol_main)
            st.dataframe(df_h, use_container_width=True, hide_index=True)

with tab_export:
    if mol_main:
        st.markdown("##### Studio de Publication & Sécurité SGH")
        pictos, hazards = get_ghs_safety(ident["cid"])
        if pictos:
            cols = st.columns(len(pictos))
            for i, url in enumerate(pictos):
                cols[i].image(url, width=80)
        svg_pub = render_publication_svg(mol_main)
        components.html(f"<div style='text-align:center;'>{svg_pub}</div>", height=420)
