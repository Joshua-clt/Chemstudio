import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors, rdMolTransforms,
    Draw, rdChemReactions, rdFMCS, Crippen, Fragments
)
from rdkit.Chem.MolStandardize import rdMolStandardize
import requests
import re
import numpy as np
import pandas as pd
import py3Dmol
import streamlit.components.v1 as components
from collections import Counter

st.set_page_config(page_title="ChemStudio Ultra", layout="wide")

# ==========================================
# 1. FONCTIONS RÉSEAU (PUBCHEM : CAS, IUPAC, SGH, EXP)
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

# ==========================================
# 2. FONCTIONS 3D, SUPERPOSITION & CHEMPROP
# ==========================================
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

def overlay_molecules_3d(smiles1, smiles2):
    """Superpose deux molécules en 3D sur leur sous-structure commune maximale (MCS)."""
    m1 = generate_3d_mol(smiles1)
    m2 = generate_3d_mol(smiles2)
    if not m1 or not m2:
        return None, None, None
    m1_noh = Chem.RemoveHs(m1)
    m2_noh = Chem.RemoveHs(m2)
    mcs = rdFMCS.FindMCS([m1_noh, m2_noh], timeout=3)
    rmsd = None
    if mcs.numAtoms >= 3:
        core = Chem.MolFromSmarts(mcs.smartsString)
        match1 = m1_noh.GetSubstructMatch(core)
        match2 = m2_noh.GetSubstructMatch(core)
        if match1 and match2:
            atom_map = list(zip(match2, match1))
            try:
                rmsd = AllChem.AlignMol(m2, m1, atomMap=atom_map)
            except Exception:
                rmsd = None
    return m1, m2, (mcs, rmsd)

def render_overlay_3d(m1, m2):
    view = py3Dmol.view(width=540, height=390)
    view.addModel(Chem.MolToMolBlock(m1), "mol")
    view.setStyle({"model": 0}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}})
    view.addModel(Chem.MolToMolBlock(m2), "mol")
    view.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.18}})
    view.zoomTo()
    components.html(view._make_html(), height=400)

def estimate_chemprop(mol):
    """Calcule les estimations ChemProp : LogS (ESOL), MR, Tb (Joback/corrélation), Tf."""
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

    # Équation ESOL (Delaney) pour la solubilité aqueuse LogS (mol/L)
    logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * ap
    sol_mg_ml = (10 ** logs) * mw

    # Estimations empiriques de Point d'Ébullition (Tb) et Point de Fusion (Tf) en K puis °C
    tb_k = 198.2 + 2.04 * mw + 1.15 * tpsa + 18.0 * hbd - 4.5 * rot
    tm_k = 115.0 + 1.15 * mw + 0.85 * tpsa + 22.0 * hbd + 14.0 * rings - 5.5 * rot
    return {
        "mr": round(mr, 2),
        "logs": round(logs, 2),
        "sol_mg_ml": round(sol_mg_ml, 3),
        "tb_c": round(tb_k - 273.15, 1),
        "tm_c": round(tm_k - 273.15, 1)
    }

# ==========================================
# 3. FONCTIONS SPECTROMÉTRIE (ISOTOPES, FRAGMENTS, RMN)
# ==========================================
def calc_isotopic_spectrum(mol):
    """Calcule le vrai massif isotopique par convolution polynomiale des abondances naturelles."""
    iso_table = {
        "H": [(1.007825, 0.999885), (2.014102, 0.000115)],
        "C": [(12.000000, 0.9893), (13.003355, 0.0107)],
        "N": [(14.003074, 0.99636), (15.000109, 0.00364)],
        "O": [(15.994915, 0.99757), (16.999132, 0.00038), (17.999160, 0.00205)],
        "F": [(18.998403, 1.0)],
        "P": [(30.973762, 1.0)],
        "S": [(31.972071, 0.9499), (32.971459, 0.0075), (33.967867, 0.0425)],
        "Cl": [(34.968853, 0.7576), (36.965903, 0.2424)],
        "Br": [(78.918337, 0.5069), (80.916290, 0.4931)],
        "I": [(126.904473, 1.0)]
    }
    mol_h = Chem.AddHs(mol)
    pt = Chem.GetPeriodicTable()
    dist = {0: (0.0, 1.0)}  # shift_nominal -> (exact_mass, prob)

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
            rows.append({
                "Pic": f"M+{s}" if s > 0 else "M",
                "m/z exact": round(m_val, 5),
                "Intensité Relative (%)": round(rel_int, 2)
            })
    return pd.DataFrame(rows)

def simulate_nmr(mol):
    """Estime les déplacements chimiques RMN 1H et 13C et génère les courbes spectrales."""
    mol_h = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    
    # --- 13C NMR ---
    c_groups = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "C":
            r = ranks[idx]
            c_groups.setdefault(r, []).append(atom)

    c_rows = []
    for r, atoms in c_groups.items():
        a = atoms[0]
        hyb = str(a.GetHybridization())
        arom = a.GetIsAromatic()
        o_nb = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "O")
        n_nb = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "N")
        h_nb = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
        has_c_o = any(
            nb.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nb in a.GetNeighbors()
        )
        if has_c_o:
            shift = 172.0 if (o_nb >= 2 or n_nb >= 1) else 202.0
            env = "Carbonyle (C=O)"
        elif arom:
            shift = 155.0 if (o_nb or n_nb) else 128.5
            env = "Aromatique (Ar-C)"
        elif hyb == "SP2":
            shift = 135.0
            env = "Alcène (C=C)"
        elif hyb == "SP":
            shift = 82.0
            env = "Alcyne / Nitrile"
        else:
            shift = 18.0 + 28.0 * o_nb + 18.0 * n_nb + (3 - h_nb) * 9.5
            env = f"Aliphatique (CH{h_nb})"
        c_rows.append({
            "Atomes #": ", ".join(str(x.GetIdx()) for x in atoms),
            "Type": env,
            "Nb C": len(atoms),
            "δ ¹³C estimé (ppm)": round(shift, 1)
        })

    # --- 1H NMR ---
    h_groups = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "H":
            r = ranks[idx]
            h_groups.setdefault(r, []).append(atom)

    h_rows = []
    mult_names = {0: "s (singulet)", 1: "d (doublet)", 2: "t (triplet)", 3: "q (quadruplet)"}
    for r, h_atoms in h_groups.items():
        h0 = h_atoms[0]
        parent = h0.GetNeighbors()[0]
        p_sym = parent.GetSymbol()
        if p_sym == "O":
            is_acid = any(
                nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" and n2.GetIdx() != parent.GetIdx() for n2 in nb.GetNeighbors())
                for nb in parent.GetNeighbors()
            )
            shift, mult, env = (11.5, "s large", "Acide (-COOH)") if is_acid else (4.8, "s large", "Hydroxyle (-OH)")
        elif p_sym == "N":
            shift, mult, env = 6.2, "s large", "Amine / Amide (-NH)"
        else:
            arom = parent.GetIsAromatic()
            hyb = str(parent.GetHybridization())
            o_nb = sum(1 for nb in parent.GetNeighbors() if nb.GetSymbol() in ["O", "F", "Cl", "Br"])
            # Calcul du couplage vicinal (règle n+1)
            vicinal_h = 0
            for nb in parent.GetNeighbors():
                if nb.GetSymbol() == "C":
                    vicinal_h += sum(1 for n2 in nb.GetNeighbors() if n2.GetSymbol() == "H")
            mult = mult_names.get(vicinal_h, "m (multiplet)")
            if arom:
                shift, env = 7.35 + 0.25 * o_nb, "Aromatique (Ar-H)"
            elif hyb == "SP2":
                shift, env = 5.85, "Vinylique (=CH)"
            else:
                shift = 1.25 + 2.3 * o_nb
                env = "Aliphatique (C-H)"
        h_rows.append({
            "Intégration": f"{len(h_atoms)}H",
            "Type": env,
            "Multiplicité": mult,
            "δ ¹H estimé (ppm)": round(shift, 2)
        })

    return pd.DataFrame(h_rows), pd.DataFrame(c_rows)

def build_lorentzian_spectrum(df_peaks, col_shift, x_min, x_max, width=0.08):
    x = np.linspace(x_min, x_max, 800)
    y = np.zeros_like(x)
    for _, row in df_peaks.iterrows():
        x0 = row[col_shift]
        y += 1.0 / (1.0 + ((x - x0) / width) ** 2)
    return pd.DataFrame({"δ (ppm)": x, "Intensité": y}).set_index("δ (ppm)")

# ==========================================
# 4. STUDIO DE PUBLICATION (ACS / RSC / WILEY)
# ==========================================
def render_publication_svg(mol, style_name="ACS Document 1996", show_indices=False):
    mol_copy = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol_copy)
    Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
    drawer = Draw.MolDraw2DSVG(520, 420)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = show_indices

    if style_name == "ACS Document 1996 (Noir & Blanc)":
        Draw.SetACS1996Mode(opts, Draw.MeanBondLength(mol_copy))
        opts.useBWAtomPalette()
    elif style_name == "RSC Standard (Couleurs sobres)":
        opts.bondLineWidth = 2.0
        opts.multipleBondOffset = 0.15
    elif style_name == "Wiley Clean (Trait épais N&B)":
        opts.bondLineWidth = 2.6
        opts.useBWAtomPalette()
    elif style_name == "Mode Sombre (Présentation)":
        opts.setBackgroundColour((0.12, 0.14, 0.18))
        opts.setSymbolColour((0.95, 0.95, 0.95))
        opts.bondLineWidth = 2.2

    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# ==========================================
# 5. INITIALISATION DE LA SESSION
# ==========================================
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirine
if "history" not in st.session_state:
    st.session_state.history = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    ]
if "rxn_smiles" not in st.session_state:
    st.session_state.rxn_smiles = "CC(=O)O.OCC>>CC(=O)OCC"

# ==========================================
# 6. BARRE LATÉRALE ÉPURÉE : CONTRÔLE CENTRAL
# ==========================================
with st.sidebar:
    st.title("🧪 ChemStudio Ultra")
    query = st.text_input("🔍 Recherche (CAS, Nom ou SMILES) :", placeholder="Ex: 50-78-2, Ibuprofen...")
    if st.button("Charger la structure", use_container_width=True) and query:
        mol_t = Chem.MolFromSmiles(query)
        s_res = Chem.MolToSmiles(mol_t) if mol_t else resolve_query_to_smiles(query)
        if s_res:
            st.session_state.current_smiles = s_res
            if s_res not in st.session_state.history:
                st.session_state.history.insert(0, s_res)
            st.success("Structure chargée !")
        else:
            st.error("Composé introuvable.")

    st.divider()
    exemples = {
        "Aspirine (50-78-2)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Paracétamol (103-90-2)": "CC(=O)NC1=CC=C(O)C=C1",
        "Ibuprofène (15687-27-1)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Caféine (58-08-2)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "4-Bromoacétophénone (Isotopes Br)": "CC(=O)c1ccc(Br)cc1",
        "L-DOPA (Chiralité R/S)": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O"
    }
    ex_sel = st.selectbox("⚡ Molécules de démonstration :", list(exemples.keys()))
    if st.button("Appliquer l'exemple", use_container_width=True):
        st.session_state.current_smiles = exemples[ex_sel]
        if exemples[ex_sel] not in st.session_state.history:
            st.session_state.history.insert(0, exemples[ex_sel])

    st.divider()
    st.markdown("**🕘 Historique de session**")
    hist_sel = st.selectbox("Recharger une molécule :", st.session_state.history)
    if st.button("Charger depuis l'historique", use_container_width=True):
        st.session_state.current_smiles = hist_sel

# ==========================================
# 7. BANDEAU D'IDENTITÉ PERMANENT
# ==========================================
raw_in = st.session_state.current_smiles
single_smiles = raw_in.split(">")[0].split(".")[0] if (">" in raw_in or "." in raw_in) else raw_in
mol_main = Chem.MolFromSmiles(single_smiles)

if mol_main:
    ident = get_chemical_identifiers(single_smiles)
    b1, b2, b3, b4 = st.columns([1.4, 1, 1, 1])
    b1.metric("🏷️ Nom Trivial", ident["trivial"])
    b2.metric("🔢 Numéro CAS", ident["cas"])
    b3.metric("🧪 Formule Brute", rdMolDescriptors.CalcMolFormula(mol_main))
    b4.metric("⚖️ Masse Molaire", f"{round(Descriptors.MolWt(mol_main), 2)} g/mol")
    st.caption(f"**Nom IUPAC :** `{ident['iupac']}`")
else:
    st.warning("Structure chimique invalide.")

# ==========================================
# 8. NAVIGATION EN 5 ONGLETS PRINCIPAUX
# ==========================================
tab_2d3d, tab_rxn, tab_struct, tab_spec, tab_export = st.tabs([
    "🎨 1. Atelier 2D, 3D & Overlay",
    "⚗️ 2. Laboratoire de Réactions",
    "🔬 3. Structure & ChemProp",
    "📈 4. Spectrométrie (MS, Fragments, RMN)",
    "📥 5. Sécurité & Publication (ACS/RSC)"
])

# ------------------------------------------
# ONGLET 1 : ATELIER 2D, 3D & SUPERPOSITION (OVERLAY)
# ------------------------------------------
with tab_2d3d:
    col_2d, col_3d = st.columns([1.25, 1.15])
    
    with col_2d:
        st.markdown("##### Éditeur Moléculaire 2D (Ketcher)")
        st.caption("Dessinez une molécule ou une réaction puis cliquez sur **Apply**.")
        ketcher_out = st_ketcher(st.session_state.current_smiles, height=470)
        if ketcher_out and ketcher_out != st.session_state.current_smiles:
            st.session_state.current_smiles = ketcher_out
            if ">>" in ketcher_out or ">" in ketcher_out:
                st.session_state.rxn_smiles = ketcher_out
            elif ketcher_out not in st.session_state.history:
                st.session_state.history.insert(0, ketcher_out)
            st.rerun()

    with col_3d:
        mode_3d = st.radio(
            "Outil 3D actif :",
            ["🧊 Visualisation & Géométrie 3D", "🔀 Superposition 2D/3D (Structure Overlay)"],
            horizontal=True
        )
        
        if mode_3d.startswith("🧊") and mol_main:
            c1, c2, c3 = st.columns(3)
            style_3d = c1.selectbox("Style", ["Ball & Stick", "Stick", "Spacefill"])
            ff_3d = c2.selectbox("Champ de force", ["MMFF94", "UFF"])
            seed_3d = c3.number_input("Conformère #", 1, 100, 42)
            
            k1, k2, k3 = st.columns(3)
            show_h = k1.checkbox("Hydrogènes", value=True)
            color_q = k2.checkbox("Charges δ+/δ-", value=False)
            spin_3d = k3.checkbox("Rotation", value=False)

            mol_3d = generate_3d_mol(single_smiles, force_field=ff_3d, seed=int(seed_3d))
            if mol_3d:
                with st.expander("🛠️ Modifier une distance interatomique en 3D"):
                    n_at = mol_3d.GetNumAtoms()
                    mc1, mc2, mc3 = st.columns(3)
                    ai = mc1.number_input("Atome A", 0, n_at - 1, 0)
                    aj = mc2.number_input("Atome B", 0, n_at - 1, min(1, n_at - 1))
                    conf = mol_3d.GetConformer()
                    cur_d = round(rdMolTransforms.GetBondLength(conf, int(ai), int(aj)), 2)
                    new_d = mc3.slider("Distance (Å)", 0.8, 3.5, float(min(max(cur_d, 0.8), 3.5)), 0.05)
                    if new_d != cur_d and ai != aj:
                        try:
                            rdMolTransforms.SetBondLength(conf, int(ai), int(aj), new_d)
                        except Exception:
                            st.caption("⚠️ Liaison verrouillée dans un cycle.")
                render_3d_viewer(mol_3d, style=style_3d, show_h=show_h, color_charges=color_q, spin=spin_3d)

        elif mode_3d.startswith("🔀") and mol_main:
            st.caption("Superpose la molécule active (**Vert**) avec une molécule de référence (**Cyan**) sur leur squelette commun (MCS).")
            ref_comp = st.selectbox("Choisir la 2e molécule (Historique ou SMILES) :", st.session_state.history, index=min(1, len(st.session_state.history)-1))
            custom_comp = st.text_input("Ou entrer un CAS / Nom / SMILES à superposer :", value="")
            target_s2 = resolve_query_to_smiles(custom_comp) if custom_comp.strip() else ref_comp

            if target_s2:
                m1_ov, m2_ov, (mcs_res, rmsd_val) = overlay_molecules_3d(single_smiles, target_s2)
                if m1_ov and m2_ov:
                    ov_c1, ov_c2 = st.columns(2)
                    ov_c1.metric("Atomes communs (MCS)", mcs_res.numAtoms if mcs_res else 0)
                    ov_c2.metric("Écart géométrique (RMSD)", f"{rmsd_val:.3f} Å" if rmsd_val is not None else "N/A")
                    render_overlay_3d(m1_ov, m2_ov)

# ------------------------------------------
# ONGLET 2 : LABORATOIRE DE RÉACTIONS
# ------------------------------------------
with tab_rxn:
    rxn_mode = st.radio(
        "Mode de création de la réaction :",
        [
            "📐 Depuis l'éditeur Ketcher / SMILES",
            "🔎 Par Numéros CAS / Noms triviaux",
            "🤖 Prédicteur automatique de synthèse"
        ],
        horizontal=True
    )

    if rxn_mode.startswith("📐"):
        st.session_state.rxn_smiles = st.text_input(
            "SMILES de réaction (Réactif1.Réactif2>>Produit1) :",
            value=st.session_state.rxn_smiles
        )
    elif rxn_mode.startswith("🔎"):
        rc1, rc2, rc3 = st.columns(3)
        r1_q = rc1.text_input("Réactif 1 (CAS ou Nom)", value="Salicylic acid")
        r2_q = rc2.text_input("Réactif 2 (Optionnel)", value="Acetic anhydride")
        p1_q = rc3.text_input("Produit (CAS ou Nom)", value="Aspirin")
        if st.button("Assembler la réaction"):
            s_r1 = resolve_query_to_smiles(r1_q)
            s_r2 = resolve_query_to_smiles(r2_q) if r2_q.strip() else ""
            s_p1 = resolve_query_to_smiles(p1_q)
            if s_r1 and s_p1:
                st.session_state.rxn_smiles = f"{s_r1}.{s_r2}>>{s_p1}" if s_r2 else f"{s_r1}>>{s_p1}"
                st.success("Schéma réactionnel mis à jour !")
    elif rxn_mode.startswith("🤖"):
        templates = {
            "Estérification de Fischer (Acide + Alcool -> Ester)": "[C:1](=[O:2])[OH].[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
            "Formation d'Amide (Acide + Amine -> Amide)": "[C:1](=[O:2])[OH].[N;H2,H1:3]>>[C:1](=[O:2])[N:3]",
            "Couplage de Suzuki-Miyaura (Ar-Br/Cl + Ar-B(OH)2)": "[c:1][Br,Cl,I].[c:2][B](O)O>>[c:1]-[c:2]",
            "Réduction de Cétone/Aldéhyde en Alcool": "[C:1]=[O:2]>>[C:1][OH:2]",
            "Click Chemistry CuAAC (Alcyne + Azoture -> Triazole)": "[C:1]#[CH:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[cH:2][n:3][n:4][n:5]1"
        }
        tc1, tc2, tc3 = st.columns([1.4, 1, 1])
        chosen_tpl = tc1.selectbox("Réaction type :", list(templates.keys()))
        sub1 = tc2.text_input("Réactif 1 (SMILES/Nom)", value="CC(=O)O")
        sub2 = tc3.text_input("Réactif 2 (si bimoléculaire)", value="OCC")
        if st.button("⚡ Prédire le produit"):
            m1 = Chem.MolFromSmiles(sub1) or Chem.MolFromSmiles(resolve_query_to_smiles(sub1) or "")
            m2 = Chem.MolFromSmiles(sub2) or Chem.MolFromSmiles(resolve_query_to_smiles(sub2) or "") if sub2.strip() else None
            rxn_eng = rdChemReactions.ReactionFromSmarts(templates[chosen_tpl])
            try:
                prods = rxn_eng.RunReactants((m1, m2) if m2 else (m1,))
                if prods and len(prods[0]) > 0:
                    prod_mol = prods[0][0]
                    Chem.SanitizeMol(prod_mol)
                    prod_smi = Chem.MolToSmiles(prod_mol)
                    r_str = f"{Chem.MolToSmiles(m1)}.{Chem.MolToSmiles(m2)}" if m2 else Chem.MolToSmiles(m1)
                    st.session_state.rxn_smiles = f"{r_str}>>{prod_smi}"
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.divider()
    rxn_str = st.session_state.rxn_smiles
    if ">>" in rxn_str or ">" in rxn_str:
        parts = rxn_str.split(">")
        r_mols = [Chem.MolFromSmiles(s) for s in parts[0].split(".") if s.strip() and Chem.MolFromSmiles(s)]
        p_mols = [Chem.MolFromSmiles(s) for s in parts[-1].split(".") if s.strip() and Chem.MolFromSmiles(s)]
        if r_mols and p_mols:
            rxn_obj = rdChemReactions.ReactionFromSmarts(
                f"{'.'.join(Chem.MolToSmiles(m) for m in r_mols)}>>{'.'.join(Chem.MolToSmiles(m) for m in p_mols)}",
                useSmiles=True
            )
            st.image(Draw.ReactionToImage(rxn_obj, subImgSize=(250, 170)))

            sc1, sc2 = st.columns(2)
            ref_mass_mg = sc1.number_input("Masse engagée du Réactif 1 (mg) :", min_value=1.0, value=500.0, step=50.0)
            exp_mass_mg = sc2.number_input("Masse de Produit 1 isolée au labo (mg) :", min_value=0.0, value=0.0, step=10.0)
            ref_mmol = ref_mass_mg / Descriptors.MolWt(r_mols[0])

            rows = []
            for i, m in enumerate(r_mols):
                inf = get_chemical_identifiers(Chem.MolToSmiles(m))
                rows.append({"Rôle": f"Réactif {i+1}", "Nom": inf["trivial"], "CAS": inf["cas"], "MW (g/mol)": round(Descriptors.MolWt(m), 2), "Équivalents": 1.0 if i == 0 else 1.2, "Densité (g/mL)": 1.0})
            for i, m in enumerate(p_mols):
                inf = get_chemical_identifiers(Chem.MolToSmiles(m))
                rows.append({"Rôle": f"Produit {i+1}", "Nom": inf["trivial"], "CAS": inf["cas"], "MW (g/mol)": round(Descriptors.MolWt(m), 2), "Équivalents": 1.0, "Densité (g/mL)": 1.0})

            ed_df = st.data_editor(pd.DataFrame(rows), disabled=["Rôle", "Nom", "CAS", "MW (g/mol)"], use_container_width=True, hide_index=True)
            ed_df["Quantité (mmol)"] = (ed_df["Équivalents"] * ref_mmol).round(3)
            ed_df["Masse (mg)"] = (ed_df["Quantité (mmol)"] * ed_df["MW (g/mol)"]).round(1)
            ed_df["Volume (µL)"] = (ed_df["Masse (mg)"] / ed_df["Densité (g/mL)"]).round(1)
            st.dataframe(ed_df, use_container_width=True, hide_index=True)

            mw_p1 = Descriptors.MolWt(p_mols[0])
            theo_mg = ref_mmol * mw_p1
            y_pct = (exp_mass_mg / theo_mg * 100) if (theo_mg > 0 and exp_mass_mg > 0) else 0.0
            atom_ec = (mw_p1 / sum(Descriptors.MolWt(m) for m in r_mols)) * 100
            dmz = Descriptors.ExactMolWt(p_mols[0]) - Descriptors.ExactMolWt(r_mols[0])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🎯 Masse théorique", f"{round(theo_mg, 1)} mg")
            k2.metric("🏆 Rendement", f"{round(y_pct, 1)} %" if exp_mass_mg > 0 else "Entrer masse isolée")
            k3.metric("♻️ Économie d'atomes", f"{round(atom_ec, 1)} %")
            k4.metric("📡 Δ m/z (R1→P1)", f"{dmz:+.4f} Da")

# ------------------------------------------
# ONGLET 3 : STRUCTURE, STÉRÉOCHIMIE & CHEMPROP
# ------------------------------------------
with tab_struct:
    if mol_main:
        st_left, st_right = st.columns([1.1, 1.2])
        
        with st_left:
            st.markdown("##### Carte 2D Annotée (Atomes, Liaisons & Stéréochimie CIP)")
            mol_annot = Chem.Mol(mol_main)
            Chem.rdDepictor.Compute2DCoords(mol_annot)
            Chem.AssignStereochemistry(mol_annot, cleanIt=True, force=True)
            d2d = Draw.MolDraw2DCairo(480, 360)
            opts = d2d.drawOptions()
            opts.addStereoAnnotation = True
            opts.addAtomIndices = True
            opts.addBondIndices = True
            d2d.DrawMolecule(mol_annot)
            d2d.FinishDrawing()
            st.image(d2d.GetDrawingText(), caption="Noir = Indices d'atomes | Bleu = Indices de liaisons | (R/S, E/Z)")

            chiral = Chem.FindMolChiralCenters(mol_annot, includeUnassigned=True)
            st.caption(f"**Centres chiraux détectés :** `{chiral if chiral else 'Aucun (Molécule achirale)'}`")

            with st.expander("🔄 Explorer les formes tautomères"):
                tauts = list(rdMolStandardize.TautomerEnumerator().Enumerate(mol_main))[:6]
                st.image(Draw.MolsToGridImage(tauts, molsPerRow=3, subImgSize=(190, 150)))

        with st_right:
            st.markdown("##### Panneau ChemProp (Propriétés Physiques & Lipinski)")
            cprop = estimate_chemprop(mol_main)
            mol_h = Chem.AddHs(mol_main)
            cnt = Counter(a.GetSymbol() for a in mol_h.GetAtoms())
            dbe = cnt["C"] + 1 - (cnt["H"] + cnt["F"] + cnt["Cl"] + cnt["Br"] + cnt["I"] - cnt["N"]) / 2

            p1, p2, p3 = st.columns(3)
            p1.metric("🌡️ Ébullition estimée (Tb)", f"{cprop['tb_c']} °C")
            p2.metric("❄️ Fusion estimée (Tf)", f"{cprop['tm_c']} °C")
            p3.metric("💧 Solubilité (LogS ESOL)", f"{cprop['logs']} ({cprop['sol_mg_ml']} mg/mL)")

            p4, p5, p6 = st.columns(3)
            p4.metric("LogP (Crippen)", round(Descriptors.MolLogP(mol_main), 2))
            p5.metric("Réfringence Molaire (MR)", f"{cprop['mr']} cm³/mol")
            p6.metric("Surface Polaire (TPSA)", f"{round(Descriptors.TPSA(mol_main), 2)} Å²")

            p7, p8, p9 = st.columns(3)
            p7.metric("Insaturation (DBE)", f"{dbe:.0f}")
            p8.metric("Donneurs / Accepteurs H", f"{rdMolDescriptors.CalcNumHBD(mol_main)} / {rdMolDescriptors.CalcNumHBA(mol_main)}")
            p9.metric("Liaisons rotatives", rdMolDescriptors.CalcNumRotatableBonds(mol_main))

# ------------------------------------------
# ONGLET 4 : SPECTROMÉTRIE (SOUS-NAVIGATION 3 INSTRUMENTS)
# ------------------------------------------
with tab_spec:
    if mol_main:
        spec_tool = st.radio(
            "Sélectionner l'instrument analytique :",
            [
                "📡 1. Spectrométrie de Masse (HRMS & Massif Isotopique)",
                "✂️ 2. Outil de Fragmentation MS par coupure de liaison",
                "🧲 3. Simulateur RMN ¹H & ¹³C (Spectres & Attributions)"
            ],
            horizontal=True
        )
        st.divider()

        # VUE A : HRMS & SPECTRE ISOTOPIQUE EN BÂTONS
        if spec_tool.startswith("📡"):
            ca, cb = st.columns([1.1, 1.2])
            with ca:
                exact_m = Descriptors.ExactMolWt(mol_main)
                st.metric("Masse Exacte Monoisotopique (M)", f"{exact_m:.5f} Da")
                adducts = pd.DataFrame([
                    {"Ion / Adduit": "[M + H]⁺", "m/z exact": round(exact_m + 1.00728, 5)},
                    {"Ion / Adduit": "[M + Na]⁺", "m/z exact": round(exact_m + 22.98922, 5)},
                    {"Ion / Adduit": "[M + K]⁺", "m/z exact": round(exact_m + 38.96316, 5)},
                    {"Ion / Adduit": "[M + NH₄]⁺", "m/z exact": round(exact_m + 18.03382, 5)},
                    {"Ion / Adduit": "[M - H]⁻", "m/z exact": round(exact_m - 1.00728, 5)},
                ])
                st.dataframe(adducts, use_container_width=True, hide_index=True)

            with cb:
                st.markdown("**Spectre du Massif Isotopique Théorique ($M, M+1, M+2...$)**")
                df_iso = calc_isotopic_spectrum(mol_main)
                st.bar_chart(df_iso.set_index("Pic")["Intensité Relative (%)"], height=220)
                st.dataframe(df_iso, use_container_width=True, hide_index=True)

        # VUE B : OUTIL DE FRAGMENTATION MS PAR COUPURE
        elif spec_tool.startswith("✂️"):
            f_left, f_right = st.columns([1.1, 1.3])
            with f_left:
                mol_b = Chem.Mol(mol_main)
                Chem.rdDepictor.Compute2DCoords(mol_b)
                d_b = Draw.MolDraw2DCairo(440, 340)
                d_b.drawOptions().addBondIndices = True
                d_b.drawOptions().addAtomIndices = True
                d_b.DrawMolecule(mol_b)
                d_b.FinishDrawing()
                st.image(d_b.GetDrawingText(), caption="Repérez l'indice de la liaison à couper")

            with f_right:
                non_ring_bonds = [
                    f"Liaison #{b.GetIdx()} : {b.GetBeginAtom().GetSymbol()}{b.GetBeginAtomIdx()} - {b.GetEndAtom().GetSymbol()}{b.GetEndAtomIdx()}"
                    for b in mol_main.GetBonds() if not b.IsInRing()
                ]
                if non_ring_bonds:
                    sel_b = st.selectbox("Choisir la liaison acyclique à couper :", non_ring_bonds)
                    b_idx = int(sel_b.split("#")[1].split(" ")[0])
                    frag_mol = Chem.FragmentOnBonds(mol_main, [b_idx], addDummies=False)
                    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
                    
                    frag_cols = st.columns(len(frags))
                    for i, fm in enumerate(frags):
                        fm.UpdatePropertyCache(strict=False)
                        Chem.rdDepictor.Compute2DCoords(fm)
                        f_form = rdMolDescriptors.CalcMolFormula(fm)
                        f_mass = Descriptors.ExactMolWt(fm)
                        with frag_cols[i]:
                            st.image(Draw.MolToImage(fm, size=(220, 180)))
                            st.metric(f"Fragment {i+1} ({f_form})", f"{f_mass:.4f} Da")
                            st.caption(f"Cation $[F_{i+1}]^+$ : `{f_mass - 0.00055:.4f} m/z`")
                else:
                    st.info("Toutes les liaisons de cette molécule appartiennent à un cycle (nécessite 2 coupures).")

        # VUE C : SIMULATEUR RMN 1H ET 13C
        elif spec_tool.startswith("🧲"):
            df_h, df_c = simulate_nmr(mol_main)
            nmr_n1, nmr_n2 = st.columns(2)
            with nmr_n1:
                st.markdown("##### Spectre RMN ¹H estimé (0 – 13 ppm)")
                if not df_h.empty:
                    spec_h = build_lorentzian_spectrum(df_h, "δ ¹H estimé (ppm)", 0.0, 13.0, width=0.06)
                    st.line_chart(spec_h, height=220)
                    st.dataframe(df_h, use_container_width=True, hide_index=True)
            with nmr_n2:
                st.markdown("##### Spectre RMN ¹³C estimé (0 – 220 ppm)")
                if not df_c.empty:
                    spec_c = build_lorentzian_spectrum(df_c, "δ ¹³C estimé (ppm)", 0.0, 220.0, width=0.8)
                    st.line_chart(spec_c, height=220)
                    st.dataframe(df_c, use_container_width=True, hide_index=True)

# ------------------------------------------
# ONGLET 5 : SÉCURITÉ SGH & STUDIO DE PUBLICATION (ACS/RSC)
# ------------------------------------------
with tab_export:
    if mol_main:
        ex_l, ex_r = st.columns([1, 1.3])
        
        with ex_l:
            st.markdown("##### ⚠️ Fiche de Sécurité SGH / GHS (PubChem)")
            pictos, hazards = get_ghs_safety(ident["cid"])
            if pictos:
                p_cols = st.columns(len(pictos))
                for i, url in enumerate(pictos):
                    p_cols[i].image(url, width=85)
                for h in hazards:
                    st.caption(f"• {h}")
            else:
                st.info("Aucun pictogramme SGH répertorié pour ce composé.")

            st.divider()
            st.markdown("**🔑 Identifiants copiables**")
            st.code(
                f"SMILES   : {Chem.MolToSmiles(mol_main, canonical=True)}\n"
                f"InChI    : {Chem.MolToInchi(mol_main)}\n"
                f"InChIKey : {Chem.MolToInchiKey(mol_main)}",
                language="text"
            )

        with ex_r:
            st.markdown("##### 🖋️ Studio de Publication Scientifique (ACS / RSC / Wiley)")
            st_col1, st_col2 = st.columns([1.5, 1])
            pub_style = st_col1.selectbox(
                "Feuille de style du journal :",
                [
                    "ACS Document 1996 (Noir & Blanc)",
                    "RSC Standard (Couleurs sobres)",
                    "Wiley Clean (Trait épais N&B)",
                    "Mode Sombre (Présentation)"
                ]
            )
            add_idx = st_col2.checkbox("Numéroter les atomes", value=False)

            svg_pub = render_publication_svg(mol_main, style_name=pub_style, show_indices=add_idx)
            components.html(f"<div style='text-align:center;'>{svg_pub}</div>", height=430)

            mol_2d_str = Chem.MolToMolBlock(mol_main)
            mol_3d_obj = generate_3d_mol(single_smiles)
            sdf_3d_str = Chem.MolToMolBlock(mol_3d_obj) if mol_3d_obj else ""

            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button("🖼️ Vectoriel (.svg)", svg_pub, file_name="publication.svg", mime="image/svg+xml", use_container_width=True)
            dl2.download_button("📄 Structure 2D (.mol)", mol_2d_str, file_name="structure_2d.mol", mime="chemical/x-mdl-molfile", use_container_width=True)
            dl3.download_button("🧊 Géométrie 3D (.sdf)", sdf_3d_str, file_name="structure_3d.sdf", mime="chemical/x-mdl-sdfile", use_container_width=True)
