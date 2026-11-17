import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, rdMolDescriptors, rdMolTransforms,
    Draw, rdChemReactions, rdFMCS, Crippen
)
from rdkit.Chem.MolStandardize import rdMolStandardize
import requests
import re
import numpy as np
import pandas as pd
import py3Dmol
import streamlit.components.v1 as components
from collections import Counter

st.set_page_config(page_title="ChemStudio Ultra — Quantum, 2D-NMR & Retrosynthesis", layout="wide")

# ==========================================
# 1. FONCTIONS RÉSEAU (PUBCHEM : CAS, IUPAC, SGH, MARKUSH)
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

@st.cache_data(show_spinner=False)
def search_pubchem_markush_analogues(smiles, mode="similarity", threshold=85, max_records=6):
    """Recherche d'analogues structuraux ou de sous-structures (Markush) via PubChem FastSearch."""
    results = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return results
        can_smi = Chem.MolToSmiles(mol, canonical=True)
        if mode == "similarity":
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{requests.utils.quote(can_smi)}/property/Title,CanonicalSMILES,MolecularWeight/JSON?Threshold={threshold}&MaxRecords={max_records}"
        else:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/smiles/{requests.utils.quote(can_smi)}/property/Title,CanonicalSMILES,MolecularWeight/JSON?MaxRecords={max_records}"
        r = requests.get(url, timeout=7)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            for p in props:
                smi_found = p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
                if smi_found:
                    results.append({
                        "CID": p.get("CID"),
                        "Nom": p.get("Title", f"CID {p.get('CID')}"),
                        "MW": p.get("MolecularWeight"),
                        "SMILES": smi_found
                    })
    except Exception:
        pass
    return results

# ==========================================
# 2. MOTEUR QUANTIQUE (HÜCKEL ÉTENDU : HOMO, LUMO, FUKUI, DIPÔLE)
# ==========================================
def compute_quantum_orbitals(mol):
    """
    Construit et diagonalise l'Hamiltonien de Hückel étendu aux hétéroatomes (STREITWIESER parameters).
    Retourne les énergies HOMO/LUMO (en eV équivalents), le gap ΔE, les coefficients orbitalaires
    et les indices de réactivité de Fukui (f-, f+, f0) par atome.
    """
    # Paramètres de Coulomb h_X (alpha_X = alpha + h_X * beta) et d'intégrale de résonance k_XY
    h_params = {"C": 0.0, "N": 0.8, "O": 1.5, "S": 0.6, "F": 2.2, "Cl": 1.4, "Br": 1.2, "I": 1.0, "B": -0.8, "P": 0.4}
    k_params = {
        ("C", "C"): 1.0, ("C", "N"): 0.9, ("C", "O"): 0.85, ("C", "S"): 0.7,
        ("C", "F"): 0.6, ("C", "Cl"): 0.4, ("C", "Br"): 0.3, ("N", "N"): 0.8, ("N", "O"): 0.7
    }
    n = mol.GetNumAtoms()
    if n == 0:
        return None

    H_mat = np.zeros((n, n))
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        # Ajustement selon l'hybridation et la charge formelle
        hyb_shift = 0.15 if str(atom.GetHybridization()) == "SP3" else 0.0
        H_mat[i, i] = h_params.get(sym, 0.0) + 0.3 * atom.GetFormalCharge() - hyb_shift

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        s1, s2 = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        bo = bond.GetBondTypeAsDouble()
        k_base = k_params.get((s1, s2), k_params.get((s2, s1), 0.75))
        # Renforcement pour les liaisons multiples / conjuguées
        k_val = k_base * (1.0 + 0.22 * (bo - 1.0))
        H_mat[i, j] = k_val
        H_mat[j, i] = k_val

    # En unités de beta (< 0), valeurs propres triées de la plus liante (plus grande valeur positive de x) à la plus antiliante
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    # Conversion en échelle énergétique physique approximative (eV) : E(eV) = -7.2 - 2.8 * x_i
    energies_ev = -7.2 - 2.8 * eigvals
    homo_idx = max(0, (n // 2) - 1)
    lumo_idx = min(n - 1, homo_idx + 1)

    e_homo = float(energies_ev[homo_idx])
    e_lumo = float(energies_ev[lumo_idx])
    gap_ev = max(0.1, e_lumo - e_homo)

    # Descripteurs DFT conceptuels (Koopmans) : Potentiel chimique mu, Dureté eta, Électrophilie globale omega
    mu = 0.5 * (e_homo + e_lumo)
    eta = 0.5 * gap_ev
    omega = (mu ** 2) / (2.0 * eta) if eta > 0 else 0.0
    lambda_max_nm = round(1240.0 / gap_ev, 1) if gap_ev > 0 else 200.0

    c_homo = eigvecs[:, homo_idx]
    c_lumo = eigvecs[:, lumo_idx]
    f_minus = c_homo ** 2  # Susceptibilité à l'attaque électrophile (nucléophilie locale)
    f_plus = c_lumo ** 2   # Susceptibilité à l'attaque nucléophile (électrophilie locale)

    fukui_rows = []
    for i, a in enumerate(mol.GetAtoms()):
        fukui_rows.append({
            "Atome": f"{a.GetSymbol()}{i}",
            "Coeff HOMO": round(float(c_homo[i]), 3),
            "Coeff LUMO": round(float(c_lumo[i]), 3),
            "Fukui f⁻ (Site Nucléophile)": round(float(f_minus[i]), 3),
            "Fukui f⁺ (Site Électrophile)": round(float(f_plus[i]), 3)
        })

    return {
        "e_homo": round(e_homo, 2),
        "e_lumo": round(e_lumo, 2),
        "gap": round(gap_ev, 2),
        "omega": round(omega, 2),
        "lambda_nm": lambda_max_nm,
        "c_homo": c_homo,
        "c_lumo": c_lumo,
        "fukui_df": pd.DataFrame(fukui_rows)
    }

def compute_dipole_moment_3d(mol_3d):
    """Calcule le moment dipolaire 3D (en Debye) à partir des charges de Gasteiger et des coordonnées 3D."""
    try:
        AllChem.ComputeGasteigerCharges(mol_3d)
        conf = mol_3d.GetConformer()
        dip_vec = np.zeros(3)
        for i, atom in enumerate(mol_3d.GetAtoms()):
            q = float(atom.GetProp("_GasteigerCharge"))
            if not np.isnan(q) and not np.isinf(q):
                pos = conf.GetAtomPosition(i)
                dip_vec += q * np.array([pos.x, pos.y, pos.z])
        # 1 e·Å = 4.8032 Debye
        return round(float(np.linalg.norm(dip_vec) * 4.8032), 2)
    except Exception:
        return 0.0

def render_quantum_orbital_3d(mol_3d, coeffs, mode="HOMO"):
    """Affiche en 3D les lobes d'orbitales quantiques (Phase + en Bleu, Phase - en Rouge) ou la surface ESP."""
    mol_block = Chem.MolToMolBlock(mol_3d)
    view = py3Dmol.view(width=540, height=400)
    view.addModel(mol_block, "mol")
    view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.22}})

    if mode in ["HOMO (Orbitale Haute Occupée)", "LUMO (Orbitale Basse Vacante)"]:
        conf = mol_3d.GetConformer()
        for i, c_val in enumerate(coeffs):
            amp = abs(float(c_val))
            if amp >= 0.12:
                pos = conf.GetAtomPosition(i)
                radius = min(1.35, max(0.32, amp * 1.65))
                color_pos = "#2563eb" if c_val > 0 else "#dc2626"
                color_neg = "#dc2626" if c_val > 0 else "#2563eb"
                # Lobes p_z au-dessus et en dessous du plan atomique
                view.addSphere({
                    "center": {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z + 0.42)},
                    "radius": radius, "color": color_pos, "opacity": 0.78
                })
                view.addSphere({
                    "center": {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z - 0.42)},
                    "radius": radius * 0.85, "color": color_neg, "opacity": 0.65
                })
    else:
        # Potentiel Électrostatique (ESP) sur surface de Van der Waals
        AllChem.ComputeGasteigerCharges(mol_3d)
        view.addSurface(py3Dmol.VDW, {"opacity": 0.78, "colorscheme": "RdYlBu"})

    view.zoomTo()
    components.html(view._make_html(), height=415)

# ==========================================
# 3. FONCTIONS 3D, OVERLAY & CHEMPROP
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
    view = py3Dmol.view(width=540, height=400)
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
    components.html(view._make_html(), height=415)

def overlay_molecules_3d(smiles1, smiles2):
    m1 = generate_3d_mol(smiles1)
    m2 = generate_3d_mol(smiles2)
    if not m1 or not m2:
        return None, None, None
    m1_noh, m2_noh = Chem.RemoveHs(m1), Chem.RemoveHs(m2)
    mcs = rdFMCS.FindMCS([m1_noh, m2_noh], timeout=3)
    rmsd = None
    if mcs.numAtoms >= 3:
        core = Chem.MolFromSmarts(mcs.smartsString)
        match1, match2 = m1_noh.GetSubstructMatch(core), m2_noh.GetSubstructMatch(core)
        if match1 and match2:
            try:
                rmsd = AllChem.AlignMol(m2, m1, atomMap=list(zip(match2, match1)))
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
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    ap = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy if heavy > 0 else 0
    logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * ap
    sol_mg_ml = (10 ** logs) * mw
    tb_k = 198.2 + 2.04 * mw + 1.15 * tpsa + 18.0 * hbd - 4.5 * rot
    tm_k = 115.0 + 1.15 * mw + 0.85 * tpsa + 22.0 * hbd + 14.0 * rings - 5.5 * rot
    return {"mr": round(mr, 2), "logs": round(logs, 2), "sol_mg_ml": round(sol_mg_ml, 3), "tb_c": round(tb_k - 273.15, 1), "tm_c": round(tm_k - 273.15, 1)}

# ==========================================
# 4. MOTEUR DE RÉTROSENTHÈSE IA (DÉCONNEXIONS STRATÉGIQUES)
# ==========================================
def run_retrosynthesis_engine(mol):
    """
    Analyse la molécule cible et applique des règles de déconnexion rétrosynthétiques (Synthons -> Précurseurs réels).
    """
    retro_rules = [
        ("Coupure d'Ester (Estérification de Fischer / Schotten-Baumann)", "[C:1](=[O:2])[O:3][C,c:4]>>[C:1](=[O:2])O.[O:3][C,c:4]"),
        ("Coupure d'Amide (Couplage Peptidique / Chlorure d'acyle)", "[C:1](=[O:2])[N;H1,H0:3][C,c:4]>>[C:1](=[O:2])O.[N:3][C,c:4]"),
        ("Couplage Croisé de Suzuki-Miyaura (Liaison Biaryle Pd⁰)", "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O"),
        ("Synthèse d'Éther de Williamson (Substitution Nucléophile SN2)", "[C;!$(C=O):1][O:2][C,c;!$(C=O):3]>>[C:1]Br.[O:2][C,c:3]"),
        ("Addition de Grignard sur Carbonyle (Coupure d'Alcool II/III)", "[C,c:1][C;H1,H0:2]([OH:3])[C,c:4]>>[C,c:1]MgBr.[C:2](=[O:3])[C,c:4]"),
        ("Amination Réductrice (Imine / Borohydrure)", "[C,c:1][CH2:2][N;H1,H2:3]>>[C,c:1][CH:2]=O.[N:3]"),
        ("Cycloaddition Click CuAAC (Déconnexion de 1,2,3-Triazole)", "[c:1]1[cH:2][n:3][n:4][n:5]1>>[C:1]#[CH:2].[N:3]=[N+:4]=[N-:5]"),
        ("Oléfination de Wittig (Coupure d'Alcène C=C)", "[C,c:1][CH:2]=[CH:3][C,c:4]>>[C,c:1][CH:2]=O.[C,c:4][CH2:3]Br")
    ]
    routes = []
    seen = set()
    for name, smarts in retro_rules:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        try:
            prod_sets = rxn.RunReactants((mol,))
            for p_tuple in prod_sets:
                precursors = []
                valid = True
                for pm in p_tuple:
                    try:
                        Chem.SanitizeMol(pm)
                        precursors.append(Chem.MolToSmiles(pm, canonical=True))
                    except Exception:
                        valid = False
                if valid and len(precursors) >= 2:
                    sig = ".".join(sorted(precursors))
                    if sig not in seen:
                        seen.add(sig)
                        routes.append({
                            "strategy": name,
                            "precursors": precursors,
                            "rxn_smiles": f"{'.'.join(precursors)}>>{Chem.MolToSmiles(mol, canonical=True)}"
                        })
        except Exception:
            continue
    return routes

# ==========================================
# 5. MOTEUR RMN HAUTE PRÉCISION (INCRÉMENTS α/β/γ) & RMN 2D (COSY, HSQC, HMBC)
# ==========================================
def simulate_advanced_nmr_1d_2d(mol):
    """
    Calcule les déplacements chimiques RMN 1H et 13C par incréments topologiques (alpha, beta, gamma)
    et génère les tables de corrélation 2D : 1H-1H COSY (3J_HH), 1H-13C HSQC (1J_CH), et 1H-13C HMBC (2,3J_CH).
    """
    mol_h = Chem.AddHs(mol)
    dist_mat = Chem.GetDistanceMatrix(mol_h)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))

    # Incréments d'électronégativité alpha et beta pour 13C (Grant-Paul / Pretsch)
    alpha_c = {"O": 48.0, "N": 29.0, "F": 68.0, "Cl": 31.0, "Br": 20.0, "S": 18.0, "C": 9.1}
    beta_c = {"O": 10.0, "N": 11.0, "F": 7.5, "Cl": 10.5, "Br": 10.0, "C": 9.4}
    # Incréments de Shoolery pour 1H
    alpha_h = {"O": 2.35, "N": 1.55, "F": 3.1, "Cl": 2.15, "Br": 1.95, "S": 1.35}

    c_shifts = {}  # c_idx -> float ppm
    c_rows = []
    c_groups = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "C":
            c_groups.setdefault(ranks[idx], []).append(atom)

    for r, atoms in c_groups.items():
        for a in atoms:
            idx = a.GetIdx()
            hyb = str(a.GetHybridization())
            arom = a.GetIsAromatic()
            nbs_1 = [mol_h.GetAtomWithIdx(j) for j in range(mol_h.GetNumAtoms()) if dist_mat[idx, j] == 1 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"]
            nbs_2 = [mol_h.GetAtomWithIdx(j) for j in range(mol_h.GetNumAtoms()) if dist_mat[idx, j] == 2 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"]
            nbs_3 = [mol_h.GetAtomWithIdx(j) for j in range(mol_h.GetNumAtoms()) if dist_mat[idx, j] == 3 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"]
            has_c_o = any(nb.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(idx, nb.GetIdx()).GetBondTypeAsDouble() == 2.0 for nb in nbs_1)

            if has_c_o:
                het_1 = [nb.GetSymbol() for nb in nbs_1 if nb.GetSymbol() in ["O", "N", "Cl"]]
                shift = 171.5 if len(het_1) >= 2 else 201.5
                env = "Carbonyle (C=O)"
            elif arom:
                shift = 128.5
                for nb in nbs_1:
                    if nb.GetSymbol() == "O":
                        shift += 27.5
                    elif nb.GetSymbol() == "N":
                        shift += 18.0
                    elif nb.GetSymbol() in ["Cl", "Br", "F"]:
                        shift += 12.0
                    elif not nb.GetIsAromatic():
                        shift += 9.5
                for nb in nbs_2:
                    if nb.GetSymbol() in ["O", "N"]:
                        shift -= 11.5
                env = "Aromatique (Ar-C)"
            elif hyb == "SP2":
                shift = 123.0 + sum(alpha_c.get(nb.GetSymbol(), 5.0) * 0.45 for nb in nbs_1)
                env = "Alcène (C=C)"
            elif hyb == "SP":
                shift = 78.0 + sum(alpha_c.get(nb.GetSymbol(), 4.0) * 0.4 for nb in nbs_1)
                env = "Alcyne / Nitrile"
            else:
                shift = -2.3
                shift += sum(alpha_c.get(nb.GetSymbol(), 5.0) for nb in nbs_1)
                shift += sum(beta_c.get(nb.GetSymbol(), 3.0) * 0.65 for nb in nbs_2)
                shift -= 2.5 * len(nbs_3)
                shift = max(8.0, min(95.0, shift))
                h_cnt = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
                env = f"Aliphatique (CH{h_cnt})"
            c_shifts[idx] = round(shift, 1)

        a0 = atoms[0]
        c_rows.append({
            "Carbone(s) #": ", ".join(str(x.GetIdx()) for x in atoms),
            "Environnement": env,
            "Nb C": len(atoms),
            "δ ¹³C (ppm)": c_shifts[a0.GetIdx()]
        })

    # Calcul 1H avec constantes de Shoolery et effets anisotropes (carbonyle/aromatique en beta)
    h_shifts = {}  # h_idx -> float ppm
    h_rows = []
    h_groups = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "H":
            h_groups.setdefault(ranks[idx], []).append(atom)

    mult_names = {0: "s (singulet)", 1: "d (doublet)", 2: "t (triplet)", 3: "q (quadruplet)"}
    for r, h_atoms in h_groups.items():
        h0 = h_atoms[0]
        parent = h0.GetNeighbors()[0]
        p_idx = parent.GetIdx()
        p_sym = parent.GetSymbol()

        if p_sym == "O":
            is_acid = any(
                nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" and n2.GetIdx() != p_idx for n2 in nb.GetNeighbors())
                for nb in parent.GetNeighbors()
            )
            shift, mult, env = (11.65, "s large", "Acide carboxylique (-COOH)") if is_acid else (4.85, "s large", "Alcool / Phénol (-OH)")
        elif p_sym == "N":
            has_co = any(
                nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" for n2 in nb.GetNeighbors())
                for nb in parent.GetNeighbors()
            )
            shift, mult, env = (7.85, "s large", "Amide (-CONH)") if has_co else (3.65, "s large", "Amine (-NH)")
        else:
            arom = parent.GetIsAromatic()
            hyb = str(parent.GetHybridization())
            vicinal_h = sum(sum(1 for n2 in nb.GetNeighbors() if n2.GetSymbol() == "H") for nb in parent.GetNeighbors() if nb.GetSymbol() == "C")
            mult = mult_names.get(vicinal_h, "m (multiplet)")

            if arom:
                shift = 7.26
                # Effets ortho/méta des hétéroatomes ou carbonyles sur le cycle aromatique
                for j in range(mol_h.GetNumAtoms()):
                    if dist_mat[p_idx, j] == 2:
                        sym_j = mol_h.GetAtomWithIdx(j).GetSymbol()
                        if sym_j == "O":
                            shift -= 0.38
                        elif sym_j == "N":
                            shift -= 0.45
                    elif dist_mat[p_idx, j] == 3 and mol_h.GetAtomWithIdx(j).GetSymbol() == "O":
                        shift += 0.42
                env = "Aromatique (Ar-H)"
            elif hyb == "SP2":
                shift, env = 5.85, "Alcène (=CH)"
            else:
                shift = 0.88
                for nb in parent.GetNeighbors():
                    if nb.GetSymbol() != "H":
                        shift += alpha_h.get(nb.GetSymbol(), 0.25)
                        if nb.GetIsAromatic():
                            shift += 1.35  # Position benzylique
                        elif any(n2.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(nb.GetIdx(), n2.GetIdx()).GetBondTypeAsDouble() == 2.0 for n2 in nb.GetNeighbors()):
                            shift += 1.15  # Position alpha d'un carbonyle
                shift = min(6.2, shift)
                env = "Aliphatique (C-H)"

        val_h = round(shift, 2)
        for ha in h_atoms:
            h_shifts[ha.GetIdx()] = val_h

        h_rows.append({
            "Porté par": f"{p_sym}{p_idx}",
            "Intégration": f"{len(h_atoms)}H",
            "Multiplicité": mult,
            "Environnement": env,
            "δ ¹H (ppm)": val_h
        })

    # --- CARTES DE CORRÉLATION RMN 2D (COSY, HSQC, HMBC) ---
    cosy_pts, hsqc_pts, hmbc_pts = [], [], []
    seen_cosy, seen_hsqc, seen_hmbc = set(), set(), set()

    # 1. COSY (1H - 1H : diagonale + couplage vicinal à 3 liaisons H-C-C-H)
    for h1, s1 in h_shifts.items():
        if (s1, s1) not in seen_cosy:
            seen_cosy.add((s1, s1))
            cosy_pts.append({"δ ¹H F2 (ppm)": s1, "δ ¹H F1 (ppm)": s1, "Type": "Diagonale"})
        for h2, s2 in h_shifts.items():
            if h1 != h2 and dist_mat[h1, h2] == 3 and abs(s1 - s2) > 0.03:
                pair = (round(s1, 2), round(s2, 2))
                if pair not in seen_cosy:
                    seen_cosy.add(pair)
                    cosy_pts.append({"δ ¹H F2 (ppm)": s1, "δ ¹H F1 (ppm)": s2, "Type": "Corrélation ³J_HH (COSY)"})

    # 2. HSQC (1J_CH : distance = 1 entre H et C) & HMBC (2,3J_CH : distance = 2 ou 3 entre H et C)
    for h_idx, sh in h_shifts.items():
        for c_idx, sc in c_shifts.items():
            d = dist_mat[h_idx, c_idx]
            if d == 1:
                key = (sh, sc)
                if key not in seen_hsqc:
                    seen_hsqc.add(key)
                    hsqc_pts.append({"Carbone": f"C{c_idx}", "δ ¹H (ppm)": sh, "δ ¹³C (ppm)": sc, "Corrélation": "¹J_CH (HSQC directe)"})
            elif d in [2, 3]:
                key = (sh, sc)
                if key not in seen_hmbc:
                    seen_hmbc.add(key)
                    hmbc_pts.append({"Carbone": f"C{c_idx}", "δ ¹H (ppm)": sh, "δ ¹³C (ppm)": sc, "Corrélation": f"^{int(d)}J_CH (HMBC longue distance)"})

    return (
        pd.DataFrame(h_rows),
        pd.DataFrame(c_rows),
        pd.DataFrame(cosy_pts),
        pd.DataFrame(hsqc_pts),
        pd.DataFrame(hmbc_pts)
    )

def build_lorentzian_spectrum(df_peaks, col_shift, x_min, x_max, width=0.08):
    x = np.linspace(x_min, x_max, 800)
    y = np.zeros_like(x)
    for _, row in df_peaks.iterrows():
        x0 = row[col_shift]
        y += 1.0 / (1.0 + ((x - x0) / width) ** 2)
    return pd.DataFrame({"δ (ppm)": x, "Intensité": y}).set_index("δ (ppm)")

# ==========================================
# 6. SPECTRE ISOTOPIQUE & STUDIO DE PUBLICATION
# ==========================================
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
# 7. INITIALISATION DE LA SESSION & BARRE LATÉRALE
# ==========================================
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
if "history" not in st.session_state:
    st.session_state.history = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    ]
if "rxn_smiles" not in st.session_state:
    st.session_state.rxn_smiles = "CC(=O)O.OCC>>CC(=O)OCC"

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
        "4-Bromoacétophénone (Massif Br)": "CC(=O)c1ccc(Br)cc1",
        "Biphényle-4-carboxamide (Suzuki/Amide)": "NC(=O)c1ccc(-c2ccccc2)cc1"
    }
    ex_sel = st.selectbox("⚡ Molécules de démonstration :", list(exemples.keys()))
    if st.button("Appliquer l'exemple", use_container_width=True):
        st.session_state.current_smiles = exemples[ex_sel]
        if exemples[ex_sel] not in st.session_state.history:
            st.session_state.history.insert(0, exemples[ex_sel])

    st.divider()
    hist_sel = st.selectbox("🕘 Historique de session :", st.session_state.history)
    if st.button("Charger depuis l'historique", use_container_width=True):
        st.session_state.current_smiles = hist_sel

# ==========================================
# 8. BANDEAU D'IDENTITÉ PERMANENT
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
# 9. ONGLETS PRINCIPAUX (NAVIGATION AÉRÉE)
# ==========================================
tab_2d3d, tab_rxn, tab_struct, tab_spec, tab_export = st.tabs([
    "🎨 1. Atelier 2D, 3D & Quantique",
    "⚗️ 2. Réactions, Rétrosynthèse & Markush",
    "🔬 3. Structure & ChemProp",
    "📈 4. Spectrométrie & RMN 2D",
    "📥 5. Sécurité & Publication (ACS/RSC)"
])

# ------------------------------------------
# ONGLET 1 : 2D, 3D, OVERLAY & CHIMIE QUANTIQUE (HOMO/LUMO/ESP)
# ------------------------------------------
with tab_2d3d:
    col_2d, col_3d = st.columns([1.2, 1.2])
    
    with col_2d:
        st.markdown("##### Éditeur Moléculaire 2D (Ketcher)")
        st.caption("Dessinez votre structure ou réaction puis cliquez sur **Apply**.")
        ketcher_out = st_ketcher(st.session_state.current_smiles, height=480)
        if ketcher_out and ketcher_out != st.session_state.current_smiles:
            st.session_state.current_smiles = ketcher_out
            if ">>" in ketcher_out or ">" in ketcher_out:
                st.session_state.rxn_smiles = ketcher_out
            elif ketcher_out not in st.session_state.history:
                st.session_state.history.insert(0, ketcher_out)
            st.rerun()

    with col_3d:
        mode_3d = st.radio(
            "Outil 3D & Électronique :",
            [
                "🧊 Géométrie 3D",
                "⚛️ Orbitales Quantiques (HOMO / LUMO / ESP)",
                "🔀 Superposition 3D (Overlay)"
            ],
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

        elif mode_3d.startswith("⚛️") and mol_main:
            q_res = compute_quantum_orbitals(mol_main)
            mol_3d_q = generate_3d_mol(single_smiles)
            dip_debye = compute_dipole_moment_3d(mol_3d_q) if mol_3d_q else 0.0

            if q_res and mol_3d_q:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Énergie HOMO", f"{q_res['e_homo']} eV")
                q2.metric("Énergie LUMO", f"{q_res['e_lumo']} eV")
                q3.metric("Gap ΔE (LUMO-HOMO)", f"{q_res['gap']} eV")
                q4.metric("Moment Dipolaire", f"{dip_debye} D")

                surf_choice = st.selectbox(
                    "Surface quantique à projeter en 3D :",
                    [
                        "HOMO (Orbitale Haute Occupée)",
                        "LUMO (Orbitale Basse Vacante)",
                        "Potentiel Électrostatique Moléculaire (Surface ESP)"
                    ]
                )
                st.caption("🔵 **Lobe Bleu** : Phase d'onde positive ($\psi > 0$) | 🔴 **Lobe Rouge** : Phase négative ($\psi < 0$)")
                coeffs_to_plot = q_res["c_homo"] if surf_choice.startswith("HOMO") else q_res["c_lumo"]
                render_quantum_orbital_3d(mol_3d_q, coeffs_to_plot, mode=surf_choice)

                with st.expander("📊 Indices de Réactivité Locale de Fukui (f⁻ / f⁺ par atome)"):
                    st.dataframe(q_res["fukui_df"], use_container_width=True, hide_index=True)

        elif mode_3d.startswith("🔀") and mol_main:
            ref_comp = st.selectbox("2e molécule (Historique) :", st.session_state.history, index=min(1, len(st.session_state.history)-1))
            custom_comp = st.text_input("Ou entrer un CAS / Nom / SMILES :", value="")
            target_s2 = resolve_query_to_smiles(custom_comp) if custom_comp.strip() else ref_comp
            if target_s2:
                m1_ov, m2_ov, (mcs_res, rmsd_val) = overlay_molecules_3d(single_smiles, target_s2)
                if m1_ov and m2_ov:
                    ov1, ov2 = st.columns(2)
                    ov1.metric("Atomes communs (MCS)", mcs_res.numAtoms if mcs_res else 0)
                    ov2.metric("RMSD 3D", f"{rmsd_val:.3f} Å" if rmsd_val is not None else "N/A")
                    render_overlay_3d(m1_ov, m2_ov)

# ------------------------------------------
# ONGLET 2 : RÉACTIONS, RÉTROSENTHÈSE IA & RECHERCHE MARKUSH
# ------------------------------------------
with tab_rxn:
    lab_tool = st.radio(
        "Outil de synthèse actif :",
        [
            "⚖️ 1. Équation & Grille de Stœchiométrie",
            "🌳 2. Rétrosynthèse IA (Arbre de Déconnexion Stratégique)",
            "🔎 3. Recherche Générique Markush & Analogues Brevets"
        ],
        horizontal=True
    )
    st.divider()

    # SOUS-VUE 1 : STŒCHIOMÉTRIE & SYNTHÈSE DIRECTE
    if lab_tool.startswith("⚖️"):
        rxn_mode = st.radio(
            "Entrée de la réaction :",
            ["📐 Éditeur Ketcher / SMILES", "🔎 Par CAS / Noms", "🤖 Prédicteur de produit"],
            horizontal=True
        )
        if rxn_mode.startswith("📐"):
            st.session_state.rxn_smiles = st.text_input("SMILES de réaction :", value=st.session_state.rxn_smiles)
        elif rxn_mode.startswith("🔎"):
            rc1, rc2, rc3 = st.columns(3)
            r1_q = rc1.text_input("Réactif 1", value="Salicylic acid")
            r2_q = rc2.text_input("Réactif 2", value="Acetic anhydride")
            p1_q = rc3.text_input("Produit", value="Aspirin")
            if st.button("Assembler la réaction"):
                s_r1, s_p1 = resolve_query_to_smiles(r1_q), resolve_query_to_smiles(p1_q)
                s_r2 = resolve_query_to_smiles(r2_q) if r2_q.strip() else ""
                if s_r1 and s_p1:
                    st.session_state.rxn_smiles = f"{s_r1}.{s_r2}>>{s_p1}" if s_r2 else f"{s_r1}>>{s_p1}"
        elif rxn_mode.startswith("🤖"):
            templates = {
                "Estérification de Fischer (Acide + Alcool -> Ester)": "[C:1](=[O:2])[OH].[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
                "Formation d'Amide (Acide + Amine -> Amide)": "[C:1](=[O:2])[OH].[N;H2,H1:3]>>[C:1](=[O:2])[N:3]",
                "Couplage de Suzuki-Miyaura (Ar-Br/Cl + Ar-B(OH)2)": "[c:1][Br,Cl,I].[c:2][B](O)O>>[c:1]-[c:2]",
                "Réduction de Carbonyle en Alcool": "[C:1]=[O:2]>>[C:1][OH:2]",
                "Click Chemistry CuAAC (Alcyne + Azoture -> Triazole)": "[C:1]#[CH:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[cH:2][n:3][n:4][n:5]1"
            }
            tc1, tc2, tc3 = st.columns([1.4, 1, 1])
            chosen_tpl = tc1.selectbox("Réaction :", list(templates.keys()))
            sub1 = tc2.text_input("Réactif 1", value="CC(=O)O")
            sub2 = tc3.text_input("Réactif 2", value="OCC")
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
                exp_mass_mg = sc2.number_input("Masse de Produit 1 isolée (mg) :", min_value=0.0, value=0.0, step=10.0)
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

    # SOUS-VUE 2 : RÉTROSENTHÈSE AUTOMATIQUE (ARBRE DE DÉCONNEXION)
    elif lab_tool.startswith("🌳") and mol_main:
        st.markdown("##### 🌳 Analyse Rétrosynthétique de la Molécule Active")
        st.caption("Identifie automatiquement les liaisons stratégiques sécables et génère les briques commerciales (précurseurs) avec leur numéro CAS.")
        routes = run_retrosynthesis_engine(mol_main)
        if routes:
            for idx, rt in enumerate(routes):
                with st.container():
                    st.markdown(f"**Voie #{idx+1} — {rt['strategy']}**")
                    rxn_retro = rdChemReactions.ReactionFromSmarts(rt["rxn_smiles"], useSmiles=True)
                    rc_img, rc_info = st.columns([1.4, 1.2])
                    with rc_img:
                        st.image(Draw.ReactionToImage(rxn_retro, subImgSize=(220, 150)), caption="Sens synthétique proposé (Précurseurs → Cible)")
                    with rc_info:
                        for p_i, p_smi in enumerate(rt["precursors"]):
                            p_id = get_chemical_identifiers(p_smi)
                            st.markdown(f"• **Brique {p_i+1} :** `{p_id['trivial']}` | **CAS :** `{p_id['cas']}` (`{p_smi}`)")
                        if st.button(f"📥 Envoyer la Voie #{idx+1} vers la Grille de Stœchiométrie", key=f"retro_btn_{idx}"):
                            st.session_state.rxn_smiles = rt["rxn_smiles"]
                            st.success("Voie chargée dans le tableau de stœchiométrie !")
                    st.divider()
        else:
            st.info("Aucune coupure stratégique standard (Ester, Amide, Suzuki, Williamson, Grignard, Wittig, Click) détectée sur cette molécule.")

    # SOUS-VUE 3 : RECHERCHE DE STRUCTURES DE MARKUSH & ANALOGUES PUBCHEM
    elif lab_tool.startswith("🔎") and mol_main:
        st.markdown("##### 🔎 Recherche de Structures Génériques (Markush / Sous-structures) & Analogues")
        m_col1, m_col2, m_col3 = st.columns([1.3, 1, 1])
        m_mode = m_col1.selectbox("Mode de recherche PubChem :", ["Analogues proches (Similarité 2D Tanimoto)", "Dérivés de Markush (Contenant ce squelette comme sous-structure)"])
        sim_thresh = m_col2.slider("Seuil de similarité (%)", 70, 98, 85)
        max_rec = m_col3.number_input("Nombre de composés à extraire", 3, 12, 6)

        if st.button("🚀 Lancer la recherche dans la base mondiale PubChem"):
            mode_key = "similarity" if m_mode.startswith("Analogues") else "substructure"
            with st.spinner("Recherche des dérivés et analogues dans PubChem..."):
                found_list = search_pubchem_markush_analogues(single_smiles, mode=mode_key, threshold=sim_thresh, max_records=int(max_rec))
            if found_list:
                mols_found = [Chem.MolFromSmiles(item["SMILES"]) for item in found_list if Chem.MolFromSmiles(item["SMILES"])]
                legends = [f"{item['Nom'][:22]} (MW: {item['MW']})" for item in found_list]
                st.image(Draw.MolsToGridImage(mols_found, molsPerRow=3, subImgSize=(260, 200), legends=legends))
                st.dataframe(pd.DataFrame(found_list), use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun analogue ou dérivé de sous-structure retourné par PubChem avec ces critères.")

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
            d2d.drawOptions().addStereoAnnotation = True
            d2d.drawOptions().addAtomIndices = True
            d2d.drawOptions().addBondIndices = True
            d2d.DrawMolecule(mol_annot)
            d2d.FinishDrawing()
            st.image(d2d.GetDrawingText(), caption="Noir = Atomes | Bleu = Liaisons | (R/S, E/Z)")

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
# ONGLET 4 : SPECTROMÉTRIE & RMN 2D (COSY, HSQC, HMBC)
# ------------------------------------------
with tab_spec:
    if mol_main:
        spec_tool = st.radio(
            "Instrument analytique :",
            [
                "📡 1. Masse Haute Résolution (HRMS & Massif Isotopique)",
                "✂️ 2. Fragmentation MS par coupure",
                "🧲 3. RMN 1D Haute Précision (¹H & ¹³C)",
                "🗺️ 4. RMN 2D Corrélations (COSY, HSQC & HMBC)"
            ],
            horizontal=True
        )
        st.divider()

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
                st.markdown("**Spectre du Massif Isotopique ($M, M+1, M+2...$)**")
                df_iso = calc_isotopic_spectrum(mol_main)
                st.bar_chart(df_iso.set_index("Pic")["Intensité Relative (%)"], height=220)
                st.dataframe(df_iso, use_container_width=True, hide_index=True)

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
                st.image(d_b.GetDrawingText(), caption="Indices des liaisons (en bleu)")
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
                else:
                    st.info("Toutes les liaisons appartiennent à un cycle.")

        elif spec_tool.startswith("🧲"):
            df_h, df_c, _, _, _ = simulate_advanced_nmr_1d_2d(mol_main)
            n1, n2 = st.columns(2)
            with n1:
                st.markdown("##### Spectre RMN ¹H Haute Précision (Incréments de Shoolery)")
                if not df_h.empty:
                    st.line_chart(build_lorentzian_spectrum(df_h, "δ ¹H (ppm)", 0.0, 13.0, width=0.05), height=220)
                    st.dataframe(df_h, use_container_width=True, hide_index=True)
            with n2:
                st.markdown("##### Spectre RMN ¹³C Haute Précision (Incréments α/β/γ)")
                if not df_c.empty:
                    st.line_chart(build_lorentzian_spectrum(df_c, "δ ¹³C (ppm)", 0.0, 220.0, width=0.7), height=220)
                    st.dataframe(df_c, use_container_width=True, hide_index=True)

        elif spec_tool.startswith("🗺️"):
            _, _, df_cosy, df_hsqc, df_hmbc = simulate_advanced_nmr_1d_2d(mol_main)
            c2d_1, c2d_2 = st.columns(2)
            with c2d_1:
                st.markdown("##### Carte 2D ¹H–¹H COSY (Couplages géminés & vicinaux ³J_HH)")
                if not df_cosy.empty:
                    st.scatter_chart(df_cosy, x="δ ¹H F2 (ppm)", y="δ ¹H F1 (ppm)", color="Type", height=320)
                    st.dataframe(df_cosy, use_container_width=True, hide_index=True)
            with c2d_2:
                st.markdown("##### Carte 2D Hétéronucléaire ¹H–¹³C : HSQC (¹J_CH) & HMBC (²˒³J_CH)")
                df_het = pd.concat([df_hsqc, df_hmbc], ignore_index=True)
                if not df_het.empty:
                    st.scatter_chart(df_het, x="δ ¹H (ppm)", y="δ ¹³C (ppm)", color="Corrélation", height=320)
                    st.dataframe(df_het, use_container_width=True, hide_index=True)

# ------------------------------------------
# ONGLET 5 : SÉCURITÉ SGH & STUDIO DE PUBLICATION
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
                "Feuille de style :",
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
