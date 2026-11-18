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

st.set_page_config(
    page_title="ChemStudio Ultra",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        hyb_shift = 0.15 if str(atom.GetHybridization()) == "SP3" else 0.0
        H_mat[i, i] = h_params.get(sym, 0.0) + 0.3 * atom.GetFormalCharge() - hyb_shift

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        s1, s2 = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        bo = bond.GetBondTypeAsDouble()
        k_base = k_params.get((s1, s2), k_params.get((s2, s1), 0.75))
        k_val = k_base * (1.0 + 0.22 * (bo - 1.0))
        H_mat[i, j] = k_val
        H_mat[j, i] = k_val

    eigvals, eigvecs = np.linalg.eigh(H_mat)
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    energies_ev = -7.2 - 2.8 * eigvals
    homo_idx = max(0, (n // 2) - 1)
    lumo_idx = min(n - 1, homo_idx + 1)

    e_homo = float(energies_ev[homo_idx])
    e_lumo = float(energies_ev[lumo_idx])
    gap_ev = max(0.1, e_lumo - e_homo)

    mu = 0.5 * (e_homo + e_lumo)
    eta = 0.5 * gap_ev
    omega = (mu ** 2) / (2.0 * eta) if eta > 0 else 0.0

    c_homo = eigvecs[:, homo_idx]
    c_lumo = eigvecs[:, lumo_idx]
    f_minus = c_homo ** 2
    f_plus = c_lumo ** 2

    fukui_rows = []
    for i, a in enumerate(mol.GetAtoms()):
        fukui_rows.append({
            "Atome": f"{a.GetSymbol()}{i}",
            "Coeff HOMO": round(float(c_homo[i]), 3),
            "Coeff LUMO": round(float(c_lumo[i]), 3),
            "Fukui f⁻ (Nucléophile)": round(float(f_minus[i]), 3),
            "Fukui f⁺ (Électrophile)": round(float(f_plus[i]), 3)
        })

    return {
        "e_homo": round(e_homo, 2),
        "e_lumo": round(e_lumo, 2),
        "gap": round(gap_ev, 2),
        "omega": round(omega, 2),
        "c_homo": c_homo,
        "c_lumo": c_lumo,
        "fukui_df": pd.DataFrame(fukui_rows)
    }

def compute_dipole_moment_3d(mol_3d):
    try:
        AllChem.ComputeGasteigerCharges(mol_3d)
        conf = mol_3d.GetConformer()
        dip_vec = np.zeros(3)
        for i, atom in enumerate(mol_3d.GetAtoms()):
            q = float(atom.GetProp("_GasteigerCharge"))
            if not np.isnan(q) and not np.isinf(q):
                pos = conf.GetAtomPosition(i)
                dip_vec += q * np.array([pos.x, pos.y, pos.z])
        return round(float(np.linalg.norm(dip_vec) * 4.8032), 2)
    except Exception:
        return 0.0

def render_quantum_orbital_3d(mol_3d, coeffs, mode="HOMO"):
    mol_block = Chem.MolToMolBlock(mol_3d)
    view = py3Dmol.view(width=540, height=420)
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
                view.addSphere({
                    "center": {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z + 0.42)},
                    "radius": radius, "color": color_pos, "opacity": 0.78
                })
                view.addSphere({
                    "center": {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z - 0.42)},
                    "radius": radius * 0.85, "color": color_neg, "opacity": 0.65
                })
    else:
        AllChem.ComputeGasteigerCharges(mol_3d)
        view.addSurface(py3Dmol.VDW, {"opacity": 0.78, "colorscheme": "RdYlBu"})

    view.zoomTo()
    components.html(view._make_html(), height=430)

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
    view = py3Dmol.view(width=540, height=450)
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
    components.html(view._make_html(), height=460)

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
    view = py3Dmol.view(width=540, height=420)
    view.addModel(Chem.MolToMolBlock(m1), "mol")
    view.setStyle({"model": 0}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}})
    view.addModel(Chem.MolToMolBlock(m2), "mol")
    view.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.18}})
    view.zoomTo()
    components.html(view._make_html(), height=430)

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
# 4. MOTEUR DE RÉTROSENTHÈSE IA SÉCURISÉ
# ==========================================
def run_retrosynthesis_engine(mol):
    retro_rules = [
        ("Coupure d'Ester (Estérification de Fischer / Schotten-Baumann)", "[C:1](=[O:2])-[O:3]-[C,c:4]>>[C:1](=[O:2])O.[O:3][C,c:4]"),
        ("Coupure d'Amide (Couplage Peptidique / Chlorure d'acyle)", "[C:1](=[O:2])-[N;H1,H0:3]-[C,c:4]>>[C:1](=[O:2])O.[N:3][C,c:4]"),
        ("Couplage Croisé de Suzuki-Miyaura (Liaison Biaryle Pd⁰)", "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O"),
        ("Synthèse d'Éther de Williamson (Substitution Nucléophile SN2)", "[CX4:1]-[OX2:2]-[CX4,c:3]>>[C:1]Br.[O:2][C,c:3]"),
        ("Addition de Grignard sur Carbonyle (Coupure d'Alcool II/III)", "[C,c:1]-[CX4;H1,H0:2](-[OX2H1:3])-[C,c:4]>>[C,c:1][Mg]Br.[C:2](=[O:3])[C,c:4]"),
        ("Amination Réductrice (Imine / Borohydrure)", "[C,c:1]-[CH2:2]-[NX3;H1,H2:3]>>[C,c:1][CH:2]=O.[N:3]"),
        ("Cycloaddition Click CuAAC (Déconnexion de 1,2,3-Triazole)", "[c:1]1[cH:2][n:3][n:4][n:5]1>>[C:1]#[CH:2].[N:3]=[N+:4]=[N-:5]"),
        ("Oléfination de Wittig (Coupure d'Alcène C=C)", "[C,c:1]-[CH:2]=[CH:3]-[C,c:4]>>[C,c:1][CH:2]=O.[C,c:4][CH2:3]Br")
    ]
    routes = []
    seen = set()
    for name, smarts in retro_rules:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if not rxn:
                continue
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
# 5. MOTEUR RMN 1D (MULTIPLETS PASCAL) & RMN 2D (CONTOURS VEGA-LITE)
# ==========================================
def simulate_advanced_nmr_1d_2d(mol):
    mol_h = Chem.AddHs(mol)
    dist_mat = Chem.GetDistanceMatrix(mol_h)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))

    alpha_c = {"O": 48.0, "N": 29.0, "F": 68.0, "Cl": 31.0, "Br": 20.0, "S": 18.0, "C": 9.1}
    beta_c = {"O": 10.0, "N": 11.0, "F": 7.5, "Cl": 10.5, "Br": 10.0, "C": 9.4}
    alpha_h = {"O": 2.35, "N": 1.55, "F": 3.1, "Cl": 2.15, "Br": 1.95, "S": 1.35}

    c_shifts, c_rows, c_groups = {}, [], {}
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
                shift, env = (171.5 if len(het_1) >= 2 else 201.5), "Carbonyle (C=O)"
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
                shift = -2.3 + sum(alpha_c.get(nb.GetSymbol(), 5.0) for nb in nbs_1) + sum(beta_c.get(nb.GetSymbol(), 3.0) * 0.65 for nb in nbs_2) - 2.5 * len(nbs_3)
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

    h_shifts, h_rows, h_groups = {}, [], {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "H":
            h_groups.setdefault(ranks[idx], []).append(atom)

    mult_names = {0: "s (singulet)", 1: "d (doublet)", 2: "t (triplet)", 3: "q (quadruplet)"}
    for r, h_atoms in h_groups.items():
        h0 = h_atoms[0]
        parent = h0.GetNeighbors()[0]
        p_idx, p_sym = parent.GetIdx(), parent.GetSymbol()

        if p_sym == "O":
            is_acid = any(nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" and n2.GetIdx() != p_idx for n2 in nb.GetNeighbors()) for nb in parent.GetNeighbors())
            shift, mult, env = (11.65, "s large", "Acide (-COOH)") if is_acid else (4.85, "s large", "Alcool / Phénol (-OH)")
        elif p_sym == "N":
            has_co = any(nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" for n2 in nb.GetNeighbors()) for nb in parent.GetNeighbors())
            shift, mult, env = (7.85, "s large", "Amide (-CONH)") if has_co else (3.65, "s large", "Amine (-NH)")
        else:
            arom = parent.GetIsAromatic()
            hyb = str(parent.GetHybridization())
            vicinal_h = sum(sum(1 for n2 in nb.GetNeighbors() if n2.GetSymbol() == "H") for nb in parent.GetNeighbors() if nb.GetSymbol() == "C")
            mult = mult_names.get(vicinal_h, "m (multiplet)")

            if arom:
                shift = 7.26
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
                            shift += 1.35
                        elif any(n2.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(nb.GetIdx(), n2.GetIdx()).GetBondTypeAsDouble() == 2.0 for n2 in nb.GetNeighbors()):
                            shift += 1.15
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

    cosy_pts, hsqc_pts, hmbc_pts = [], [], []
    seen_cosy, seen_hsqc, seen_hmbc = set(), set(), set()

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

    return pd.DataFrame(h_rows), pd.DataFrame(c_rows), pd.DataFrame(cosy_pts), pd.DataFrame(hsqc_pts), pd.DataFrame(hmbc_pts)

def build_lorentzian_spectrum(df_peaks, col_shift, x_min, x_max, width=0.003, freq_mhz=400.0):
    x = np.linspace(x_min, x_max, 4500)
    y = np.zeros_like(x)
    pascal_map = {
        "s": [1.0],
        "s large": [1.0],
        "d": [1.0, 1.0],
        "t": [1.0, 2.0, 1.0],
        "q": [1.0, 3.0, 3.0, 1.0],
        "m": [1.0, 4.0, 6.0, 4.0, 1.0]
    }
    delta_j_ppm = 7.5 / freq_mhz

    for _, row in df_peaks.iterrows():
        x0 = float(row[col_shift])
        n_nuclei = 1.0
        if "Intégration" in row:
            try:
                n_nuclei = float(str(row["Intégration"]).replace("H", "").strip())
            except Exception:
                n_nuclei = 1.0

        mult_key = "s"
        w_peak = width
        if "Multiplicité" in row:
            m_str = str(row["Multiplicité"]).strip()
            if "large" in m_str:
                mult_key = "s large"
                w_peak = width * 4.5
            else:
                mult_key = m_str.split(" ")[0]

        weights = pascal_map.get(mult_key, [1.0])
        tot_w = sum(weights)
        n_lines = len(weights)

        for k, w in enumerate(weights):
            offset_ppm = (k - (n_lines - 1) / 2.0) * delta_j_ppm
            xk = x0 + offset_ppm
            amp = n_nuclei * (w / tot_w)
            y += amp / (1.0 + ((x - xk) / w_peak) ** 2)

    return pd.DataFrame({"δ (ppm)": x, "Intensité": y}).set_index("δ (ppm)")

def render_2d_nmr_contour_plot(df_2d, x_col, y_col, color_col, x_domain, y_domain, is_cosy=False):
    layers = []
    if is_cosy:
        diag_df = pd.DataFrame({x_col: [x_domain[1], x_domain[0]], y_col: [y_domain[1], y_domain[0]]})
        layers.append({
            "data": {"values": diag_df.to_dict(orient="records")},
            "mark": {"type": "line", "strokeDash": [5, 5], "color": "#94a3b8", "opacity": 0.6},
            "encoding": {"x": {"field": x_col, "type": "quantitative"}, "y": {"field": y_col, "type": "quantitative"}}
        })

    for size_pt, opacity_val in [(480, 0.20), (200, 0.45), (55, 0.95)]:
        layers.append({
            "mark": {"type": "circle", "size": size_pt, "opacity": opacity_val, "strokeWidth": 1},
            "encoding": {
                "x": {"field": x_col, "type": "quantitative", "scale": {"domain": x_domain}, "title": x_col},
                "y": {"field": y_col, "type": "quantitative", "scale": {"domain": y_domain}, "title": y_col},
                "color": {"field": color_col, "type": "nominal", "legend": {"orient": "top"}},
                "tooltip": [
                    {"field": x_col, "type": "quantitative"},
                    {"field": y_col, "type": "quantitative"},
                    {"field": color_col, "type": "nominal"}
                ]
            }
        })

    spec = {"height": 380, "data": {"values": df_2d.to_dict(orient="records")}, "layer": layers}
    st.vega_lite_chart(spec, use_container_width=True)

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
    drawer = Draw.MolDraw2DSVG(520, 400)
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
# 7. INITIALISATION DE LA SESSION & PANIER RÉACTIONNEL
# ==========================================
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
if "history" not in st.session_state:
    st.session_state.history = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    ]
if "rxn_reactants" not in st.session_state:
    st.session_state.rxn_reactants = ["CC(=O)O", "OCC"]
if "rxn_products" not in st.session_state:
    st.session_state.rxn_products = ["CC(=O)OCC"]
if "rxn_smiles" not in st.session_state:
    st.session_state.rxn_smiles = "CC(=O)O.OCC>>CC(=O)OCC"

def sync_rxn_smiles_from_basket():
    r_part = ".".join(st.session_state.rxn_reactants) if st.session_state.rxn_reactants else "CC(=O)O"
    p_part = ".".join(st.session_state.rxn_products) if st.session_state.rxn_products else "CC(=O)OCC"
    st.session_state.rxn_smiles = f"{r_part}>>{p_part}"

# ==========================================
# 8. BARRE D'OUTILS SUPÉRIEURE (SANS SIDEBAR)
# ==========================================
top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([1.3, 2.2, 0.8, 1.1, 1.1])

with top_c1:
    st.markdown("### 🧪 ChemStudio Ultra")

with top_c2:
    query = st.text_input(
        "Recherche rapide",
        placeholder="🔍 Rechercher par Numéro CAS, Nom trivial ou SMILES (ex: 50-78-2, Ibuprofen...)",
        label_visibility="collapsed"
    )

with top_c3:
    if st.button("Charger", use_container_width=True) and query:
        mol_t = Chem.MolFromSmiles(query)
        s_res = Chem.MolToSmiles(mol_t) if mol_t else resolve_query_to_smiles(query)
        if s_res:
            st.session_state.current_smiles = s_res
            if s_res not in st.session_state.history:
                st.session_state.history.insert(0, s_res)
            st.rerun()
        else:
            st.toast("❌ Composé introuvable dans PubChem.")

with top_c4:
    with st.popover("⚡ Exemples & Historique", use_container_width=True):
        st.markdown("**Molécules de démonstration**")
        exemples = {
            "Aspirine (50-78-2)": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Paracétamol (103-90-2)": "CC(=O)NC1=CC=C(O)C=C1",
            "Ibuprofène (15687-27-1)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "Caféine (58-08-2)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "4-Bromoacétophénone (Massif Br)": "CC(=O)c1ccc(Br)cc1",
            "Biphényle-4-carboxamide (Suzuki)": "NC(=O)c1ccc(-c2ccccc2)cc1"
        }
        ex_sel = st.selectbox("Choisir un exemple :", list(exemples.keys()))
        if st.button("Appliquer l'exemple", use_container_width=True):
            st.session_state.current_smiles = exemples[ex_sel]
            if exemples[ex_sel] not in st.session_state.history:
                st.session_state.history.insert(0, exemples[ex_sel])
            st.rerun()

        st.divider()
        st.markdown("**Historique de session**")
        hist_sel = st.selectbox("Molécules récentes :", st.session_state.history)
        if st.button("Recharger depuis l'historique", use_container_width=True):
            st.session_state.current_smiles = hist_sel
            st.rerun()

# Extraction de la molécule active
raw_in = st.session_state.current_smiles
single_smiles = raw_in.split(">")[0].split(".")[0] if (">" in raw_in or "." in raw_in) else raw_in
mol_main = Chem.MolFromSmiles(single_smiles)
ident = get_chemical_identifiers(single_smiles) if mol_main else {"iupac": "-", "cas": "-", "trivial": "Invalide", "cid": None}

with top_c5:
    with st.popover(f"⚗️ Panier Réaction ({len(st.session_state.rxn_reactants)}R → {len(st.session_state.rxn_products)}P)", use_container_width=True):
        st.caption("Envoyez la molécule active directement dans votre réaction :")
        if st.button("➕ Ajouter comme Réactif", use_container_width=True) and mol_main:
            can_s = Chem.MolToSmiles(mol_main, canonical=True)
            st.session_state.rxn_reactants.append(can_s)
            sync_rxn_smiles_from_basket()
            st.toast(f"✅ {ident['trivial']} ajouté aux réactifs !")
            st.rerun()
        if st.button("🎯 Définir comme Produit", use_container_width=True) and mol_main:
            can_s = Chem.MolToSmiles(mol_main, canonical=True)
            st.session_state.rxn_products = [can_s]
            sync_rxn_smiles_from_basket()
            st.toast(f"🎯 {ident['trivial']} défini comme produit cible !")
            st.rerun()
        if st.button("➕ Ajouter comme Co-Produit", use_container_width=True) and mol_main:
            can_s = Chem.MolToSmiles(mol_main, canonical=True)
            st.session_state.rxn_products.append(can_s)
            sync_rxn_smiles_from_basket()
            st.rerun()
        st.divider()
        if st.button("🗑️ Vider les réactifs", use_container_width=True):
            st.session_state.rxn_reactants = []
            sync_rxn_smiles_from_basket()
            st.rerun()

# ==========================================
# 9. BARRE DES 5 ONGLETS PRINCIPAUX (EN HAUT)
# ==========================================
tab_2d3d, tab_rxn, tab_struct, tab_spec, tab_export = st.tabs([
    "🎨 1. Atelier 2D, 3D & Quantique",
    "⚗️ 2. Réactions, Rétrosynthèse & Markush",
    "🔬 3. Structure & ChemProp",
    "📈 4. Spectrométrie & RMN 2D",
    "📥 5. Sécurité & Publication (ACS/RSC)"
])

# Fonction d'affichage de la carte d'identité moléculaire compacte SOUS la barre des onglets
def render_compact_molecule_header(key_suffix=""):
    if not mol_main:
        st.warning("Structure chimique invalide.")
        return
    with st.container(border=True):
        h1, h2, h3, h4, h5, h6 = st.columns([1.6, 1.1, 1.0, 1.1, 1.1, 1.3])
        h1.markdown(f"**🏷️ Nom :** `{ident['trivial']}`")
        h2.markdown(f"**🔢 CAS :** `{ident['cas']}`")
        h3.markdown(f"**🧪 Formule :** `{rdMolDescriptors.CalcMolFormula(mol_main)}`")
        h4.markdown(f"**⚖️ MW :** `{round(Descriptors.MolWt(mol_main), 2)} g/mol`")
        with h5:
            with st.popover("📘 IUPAC & SMILES", use_container_width=True):
                st.markdown(f"**Nom IUPAC :**\n`{ident['iupac']}`")
                st.code(
                    f"SMILES   : {Chem.MolToSmiles(mol_main, canonical=True)}\n"
                    f"InChIKey : {Chem.MolToInchiKey(mol_main)}",
                    language="text"
                )
        with h6:
            b_r, b_p = st.columns(2)
            if b_r.button("➕ Réactif", key=f"add_r_{key_suffix}", help="Ajouter la molécule active aux réactifs", use_container_width=True):
                st.session_state.rxn_reactants.append(Chem.MolToSmiles(mol_main, canonical=True))
                sync_rxn_smiles_from_basket()
                st.toast(f"✅ Ajouté aux réactifs ({ident['trivial']})")
                st.rerun()
            if b_p.button("🎯 Produit", key=f"add_p_{key_suffix}", help="Définir la molécule active comme produit", use_container_width=True):
                st.session_state.rxn_products = [Chem.MolToSmiles(mol_main, canonical=True)]
                sync_rxn_smiles_from_basket()
                st.toast(f"🎯 Défini comme produit ({ident['trivial']})")
                st.rerun()

# ------------------------------------------
# ONGLET 1 : 2D, 3D, OVERLAY & CHIMIE QUANTIQUE
# ------------------------------------------
with tab_2d3d:
    render_compact_molecule_header("t1")
    col_2d, col_3d = st.columns([1.22, 1.18])
    
    with col_2d:
        st.caption("✏️ **Éditeur 2D (Ketcher)** — Dessinez puis cliquez sur **Apply** en bas à droite du cadre.")
        ketcher_out = st_ketcher(st.session_state.current_smiles, height=480)
        if ketcher_out and ketcher_out != st.session_state.current_smiles:
            st.session_state.current_smiles = ketcher_out
            if ">>" in ketcher_out or ">" in ketcher_out:
                st.session_state.rxn_smiles = ketcher_out
            elif ketcher_out not in st.session_state.history:
                st.session_state.history.insert(0, ketcher_out)
            st.rerun()

    with col_3d:
        hdr_3d_1, hdr_3d_2 = st.columns([2.4, 1])
        with hdr_3d_1:
            mode_3d = st.radio(
                "Mode 3D :",
                ["🧊 Modèle 3D", "⚛️ Quantique (HOMO/LUMO)", "🔀 Superposition"],
                horizontal=True,
                label_visibility="collapsed"
            )

        if mode_3d.startswith("🧊") and mol_main:
            mol_3d = generate_3d_mol(single_smiles, force_field="MMFF94", seed=42)
            with hdr_3d_2:
                with st.popover("⚙️ Réglages 3D", use_container_width=True):
                    style_3d = st.selectbox("Style visuel", ["Ball & Stick", "Stick", "Spacefill"])
                    ff_3d = st.selectbox("Champ de force", ["MMFF94", "UFF"])
                    seed_3d = st.number_input("Conformère #", 1, 100, 42)
                    show_h = st.checkbox("Afficher les hydrogènes", value=True)
                    color_q = st.checkbox("Carte des charges (δ+/δ-)", value=False)
                    spin_3d = st.checkbox("Rotation continue", value=False)
                    
                    mol_3d = generate_3d_mol(single_smiles, force_field=ff_3d, seed=int(seed_3d))
                    if mol_3d:
                        st.divider()
                        st.caption("🛠️ **Modifier une distance 3D**")
                        n_at = mol_3d.GetNumAtoms()
                        ai = st.number_input("Atome A", 0, n_at - 1, 0)
                        aj = st.number_input("Atome B", 0, n_at - 1, min(1, n_at - 1))
                        conf = mol_3d.GetConformer()
                        cur_d = round(rdMolTransforms.GetBondLength(conf, int(ai), int(aj)), 2)
                        new_d = st.slider("Distance (Å)", 0.8, 3.5, float(min(max(cur_d, 0.8), 3.5)), 0.05)
                        if new_d != cur_d and ai != aj:
                            try:
                                rdMolTransforms.SetBondLength(conf, int(ai), int(aj), new_d)
                            except Exception:
                                st.caption("⚠️ Liaison cyclique verrouillée.")
            if mol_3d:
                render_3d_viewer(mol_3d, style=style_3d, show_h=show_h, color_charges=color_q, spin=spin_3d)

        elif mode_3d.startswith("⚛️") and mol_main:
            q_res = compute_quantum_orbitals(mol_main)
            mol_3d_q = generate_3d_mol(single_smiles)
            dip_debye = compute_dipole_moment_3d(mol_3d_q) if mol_3d_q else 0.0

            if q_res and mol_3d_q:
                with hdr_3d_2:
                    with st.popover("📊 Table de Fukui", use_container_width=True):
                        st.caption("Réactivité locale par atome")
                        st.dataframe(q_res["fukui_df"], use_container_width=True, hide_index=True)

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("HOMO", f"{q_res['e_homo']} eV")
                q2.metric("LUMO", f"{q_res['e_lumo']} eV")
                q3.metric("Gap ΔE", f"{q_res['gap']} eV")
                q4.metric("Dipôle", f"{dip_debye} D")

                surf_choice = st.selectbox(
                    "Surface quantique :",
                    ["HOMO (Orbitale Haute Occupée)", "LUMO (Orbitale Basse Vacante)", "Potentiel Électrostatique (Surface ESP)"],
                    label_visibility="collapsed"
                )
                coeffs_to_plot = q_res["c_homo"] if surf_choice.startswith("HOMO") else q_res["c_lumo"]
                render_quantum_orbital_3d(mol_3d_q, coeffs_to_plot, mode=surf_choice)

        elif mode_3d.startswith("🔀") and mol_main:
            with hdr_3d_2:
                with st.popover("🔍 Choisir 2e Mol", use_container_width=True):
                    ref_comp = st.selectbox("Depuis l'historique :", st.session_state.history, index=min(1, len(st.session_state.history)-1))
                    custom_comp = st.text_input("Ou CAS / Nom / SMILES :", value="")
            target_s2 = resolve_query_to_smiles(custom_comp) if custom_comp.strip() else ref_comp
            if target_s2:
                m1_ov, m2_ov, (mcs_res, rmsd_val) = overlay_molecules_3d(single_smiles, target_s2)
                if m1_ov and m2_ov:
                    ov1, ov2 = st.columns(2)
                    ov1.metric("Squelette commun (MCS)", f"{mcs_res.numAtoms if mcs_res else 0} atomes")
                    ov2.metric("Écart RMSD 3D", f"{rmsd_val:.3f} Å" if rmsd_val is not None else "N/A")
                    render_overlay_3d(m1_ov, m2_ov)

# ------------------------------------------
# ONGLET 2 : RÉACTIONS, RÉTROSENTHÈSE IA & MARKUSH
# ------------------------------------------
with tab_rxn:
    render_compact_molecule_header("t2")
    lab_tool = st.radio(
        "Outil de synthèse :",
        [
            "⚖️ 1. Constructeur & Grille de Stœchiométrie",
            "🌳 2. Rétrosynthèse IA (Déconnexion de la molécule active)",
            "🔎 3. Recherche Markush & Analogues PubChem"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    if lab_tool.startswith("⚖️"):
        # Barre rapide de gestion du panier réactionnel + constructeurs
        with st.expander("🛠️ Modifier les composés de la réaction (Panier, Recherche CAS ou Prédicteur automatique)", expanded=False):
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("**Ajouter un composé par CAS, Nom ou SMILES**")
                add_q = st.text_input("Composé à ajouter :", placeholder="Ex: Acetic anhydride, 108-24-7, CCO...")
                ac1, ac2, ac3 = st.columns(3)
                if ac1.button("➕ Aux Réactifs", use_container_width=True) and add_q:
                    m_test = Chem.MolFromSmiles(add_q)
                    s_add = Chem.MolToSmiles(m_test) if m_test else resolve_query_to_smiles(add_q)
                    if s_add:
                        st.session_state.rxn_reactants.append(s_add)
                        sync_rxn_smiles_from_basket()
                        st.rerun()
                if ac2.button("🎯 Comme Produit", use_container_width=True) and add_q:
                    m_test = Chem.MolFromSmiles(add_q)
                    s_add = Chem.MolToSmiles(m_test) if m_test else resolve_query_to_smiles(add_q)
                    if s_add:
                        st.session_state.rxn_products = [s_add]
                        sync_rxn_smiles_from_basket()
                        st.rerun()
                if ac3.button("🗑️ Réinitialiser", use_container_width=True):
                    st.session_state.rxn_reactants = []
                    st.session_state.rxn_products = []
                    st.rerun()

            with b_col2:
                st.markdown("**🤖 Prédire automatiquement le produit à partir des réactifs du panier**")
                templates = {
                    "Estérification de Fischer (Acide + Alcool -> Ester)": "[C:1](=[O:2])[OH].[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
                    "Formation d'Amide (Acide + Amine -> Amide)": "[C:1](=[O:2])[OH].[N;H2,H1:3]>>[C:1](=[O:2])[N:3]",
                    "Couplage de Suzuki-Miyaura (Ar-Br/Cl + Ar-B(OH)2)": "[c:1][Br,Cl,I].[c:2][B](O)O>>[c:1]-[c:2]",
                    "Réduction de Carbonyle en Alcool": "[C:1]=[O:2]>>[C:1][OH:2]",
                    "Click Chemistry CuAAC (Alcyne + Azoture -> Triazole)": "[C:1]#[CH:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[cH:2][n:3][n:4][n:5]1"
                }
                chosen_tpl = st.selectbox("Transformation type :", list(templates.keys()))
                if st.button("⚡ Générer le produit depuis les réactifs actuels", use_container_width=True):
                    r_mols_curr = [Chem.MolFromSmiles(s) for s in st.session_state.rxn_reactants if Chem.MolFromSmiles(s)]
                    rxn_eng = rdChemReactions.ReactionFromSmarts(templates[chosen_tpl])
                    try:
                        prods = rxn_eng.RunReactants(tuple(r_mols_curr[:2]))
                        if prods and len(prods[0]) > 0:
                            prod_mol = prods[0][0]
                            Chem.SanitizeMol(prod_mol)
                            st.session_state.rxn_products = [Chem.MolToSmiles(prod_mol, canonical=True)]
                            sync_rxn_smiles_from_basket()
                            st.rerun()
                        else:
                            st.warning("Les réactifs du panier ne correspondent pas aux fonctions attendues par cette réaction.")
                    except Exception as e:
                        st.error(f"Erreur de prédiction : {e}")

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
                
                sch_col, metric_col = st.columns([1.8, 1.2])
                with sch_col:
                    st.image(Draw.ReactionToImage(rxn_obj, subImgSize=(240, 155)))
                with metric_col:
                    ref_mass_mg = st.number_input("⚖️ Masse engagée du Réactif 1 (mg) :", min_value=1.0, value=500.0, step=50.0)
                    exp_mass_mg = st.number_input("🏆 Masse de Produit 1 isolée au labo (mg) :", min_value=0.0, value=0.0, step=10.0)

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
                ed_df["Masse à peser (mg)"] = (ed_df["Quantité (mmol)"] * ed_df["MW (g/mol)"]).round(1)
                ed_df["Volume si liquide (µL)"] = (ed_df["Masse à peser (mg)"] / ed_df["Densité (g/mL)"]).round(1)
                st.dataframe(ed_df, use_container_width=True, hide_index=True)

                mw_p1 = Descriptors.MolWt(p_mols[0])
                theo_mg = ref_mmol * mw_p1
                y_pct = (exp_mass_mg / theo_mg * 100) if (theo_mg > 0 and exp_mass_mg > 0) else 0.0
                atom_ec = (mw_p1 / sum(Descriptors.MolWt(m) for m in r_mols)) * 100
                dmz = Descriptors.ExactMolWt(p_mols[0]) - Descriptors.ExactMolWt(r_mols[0])

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("🎯 Masse théorique (100%)", f"{round(theo_mg, 1)} mg")
                k2.metric("🏆 Rendement expérimental", f"{round(y_pct, 1)} %" if exp_mass_mg > 0 else "Renseigner masse")
                k3.metric("♻️ Économie d'atomes", f"{round(atom_ec, 1)} %")
                k4.metric("📡 Suivi MS (Δ m/z R1→P1)", f"{dmz:+.4f} Da")

    elif lab_tool.startswith("🌳") and mol_main:
        routes = run_retrosynthesis_engine(mol_main)
        if routes:
            for idx, rt in enumerate(routes):
                with st.container(border=True):
                    st.markdown(f"**Voie #{idx+1} — {rt['strategy']}**")
                    rxn_retro = rdChemReactions.ReactionFromSmarts(rt["rxn_smiles"], useSmiles=True)
                    rc_img, rc_info = st.columns([1.4, 1.2])
                    with rc_img:
                        st.image(Draw.ReactionToImage(rxn_retro, subImgSize=(220, 145)))
                    with rc_info:
                        for p_i, p_smi in enumerate(rt["precursors"]):
                            p_id = get_chemical_identifiers(p_smi)
                            b_col_txt, b_col_act = st.columns([2.2, 1])
                            b_col_txt.markdown(f"• **Brique {p_i+1} :** `{p_id['trivial']}` (CAS: `{p_id['cas']}`)")
                            if b_col_act.button("✏️ Éditer 2D", key=f"load_prec_{idx}_{p_i}", use_container_width=True):
                                st.session_state.current_smiles = p_smi
                                st.rerun()
                        if st.button(f"⚗️ Charger toute la Voie #{idx+1} dans la Grille de Stœchiométrie", key=f"retro_btn_{idx}", use_container_width=True):
                            st.session_state.rxn_reactants = list(rt["precursors"])
                            st.session_state.rxn_products = [Chem.MolToSmiles(mol_main, canonical=True)]
                            sync_rxn_smiles_from_basket()
                            st.toast("✅ Voie chargée dans la stœchiométrie !")
                            st.rerun()
        else:
            st.info("Aucune coupure stratégique standard détectée sur cette molécule.")

    elif lab_tool.startswith("🔎") and mol_main:
        m_col1, m_col2, m_col3, m_col4 = st.columns([1.6, 1, 0.8, 1])
        m_mode = m_col1.selectbox("Type de recherche :", ["Analogues proches (Similarité 2D)", "Dérivés de Markush (Sous-structures)"])
        sim_thresh = m_col2.slider("Similarité (%)", 70, 98, 85)
        max_rec = m_col3.number_input("Max résultats", 3, 12, 6)
        with m_col4:
            st.write("")
            run_mk = st.button("🚀 Rechercher sur PubChem", use_container_width=True)

        if run_mk:
            mode_key = "similarity" if m_mode.startswith("Analogues") else "substructure"
            with st.spinner("Interrogation de PubChem..."):
                found_list = search_pubchem_markush_analogues(single_smiles, mode=mode_key, threshold=sim_thresh, max_records=int(max_rec))
            if found_list:
                cols_mk = st.columns(3)
                for i, item in enumerate(found_list):
                    m_f = Chem.MolFromSmiles(item["SMILES"])
                    if m_f:
                        with cols_mk[i % 3]:
                            with st.container(border=True):
                                st.image(Draw.MolToImage(m_f, size=(230, 160)))
                                st.markdown(f"**{item['Nom'][:26]}** (`{item['MW']} g/mol`)")
                                c_a1, c_a2 = st.columns(2)
                                if c_a1.button("✏️ Charger", key=f"mk_ld_{i}", use_container_width=True):
                                    st.session_state.current_smiles = item["SMILES"]
                                    st.rerun()
                                if c_a2.button("➕ Réactif", key=f"mk_rx_{i}", use_container_width=True):
                                    st.session_state.rxn_reactants.append(item["SMILES"])
                                    sync_rxn_smiles_from_basket()
                                    st.toast("✅ Ajouté aux réactifs !")
            else:
                st.warning("Aucun analogue retourné par PubChem.")

# ------------------------------------------
# ONGLET 3 : STRUCTURE, STÉRÉOCHIMIE & CHEMPROP
# ------------------------------------------
with tab_struct:
    render_compact_molecule_header("t3")
    if mol_main:
        st_left, st_right = st.columns([1.05, 1.25])
        with st_left:
            st.markdown("##### Carte 2D Annotée (Atomes, Liaisons & Stéréochimie CIP)")
            mol_annot = Chem.Mol(mol_main)
            Chem.rdDepictor.Compute2DCoords(mol_annot)
            Chem.AssignStereochemistry(mol_annot, cleanIt=True, force=True)
            d2d = Draw.MolDraw2DCairo(460, 340)
            d2d.drawOptions().addStereoAnnotation = True
            d2d.drawOptions().addAtomIndices = True
            d2d.drawOptions().addBondIndices = True
            d2d.DrawMolecule(mol_annot)
            d2d.FinishDrawing()
            st.image(d2d.GetDrawingText(), caption="Noir = Indices d'atomes | Bleu = Indices de liaisons | (R/S, E/Z)")

            with st.expander("🔄 Explorer les formes tautomères"):
                tauts = list(rdMolStandardize.TautomerEnumerator().Enumerate(mol_main))[:6]
                st.image(Draw.MolsToGridImage(tauts, molsPerRow=3, subImgSize=(180, 140)))

        with st_right:
            st.markdown("##### Panneau ChemProp (Propriétés Physiques & Lipinski)")
            cprop = estimate_chemprop(mol_main)
            mol_h = Chem.AddHs(mol_main)
            cnt = Counter(a.GetSymbol() for a in mol_h.GetAtoms())
            dbe = cnt["C"] + 1 - (cnt["H"] + cnt["F"] + cnt["Cl"] + cnt["Br"] + cnt["I"] - cnt["N"]) / 2

            p1, p2, p3 = st.columns(3)
            p1.metric("🌡️ Ébullition estimée (Tb)", f"{cprop['tb_c']} °C")
            p2.metric("❄️ Fusion estimée (Tf)", f"{cprop['tm_c']} °C")
            p3.metric("💧 Solubilité (LogS)", f"{cprop['logs']} ({cprop['sol_mg_ml']} mg/mL)")

            p4, p5, p6 = st.columns(3)
            p4.metric("LogP (Crippen)", round(Descriptors.MolLogP(mol_main), 2))
            p5.metric("Réfringence Molaire", f"{cprop['mr']} cm³/mol")
            p6.metric("Surface Polaire (TPSA)", f"{round(Descriptors.TPSA(mol_main), 2)} Å²")

            p7, p8, p9 = st.columns(3)
            p7.metric("Insaturation (DBE)", f"{dbe:.0f}")
            p8.metric("Donneurs / Accepteurs H", f"{rdMolDescriptors.CalcNumHBD(mol_main)} / {rdMolDescriptors.CalcNumHBA(mol_main)}")
            p9.metric("Liaisons rotatives", rdMolDescriptors.CalcNumRotatableBonds(mol_main))

# ------------------------------------------
# ONGLET 4 : SPECTROMÉTRIE & RMN 2D
# ------------------------------------------
with tab_spec:
    render_compact_molecule_header("t4")
    if mol_main:
        spec_tool = st.radio(
            "Instrument analytique :",
            [
                "📡 1. Masse Haute Résolution (HRMS & Isotopes)",
                "✂️ 2. Fragmentation MS par coupure",
                "🧲 3. RMN 1D Haute Précision (¹H & ¹³C)",
                "🗺️ 4. RMN 2D Corrélations (COSY, HSQC & HMBC)"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )

        if spec_tool.startswith("📡"):
            ca, cb = st.columns([1.05, 1.25])
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
                st.bar_chart(df_iso.set_index("Pic")["Intensité Relative (%)"], height=230)
                with st.expander("📋 Voir la table des pics isotopiques"):
                    st.dataframe(df_iso, use_container_width=True, hide_index=True)

        elif spec_tool.startswith("✂️"):
            f_left, f_right = st.columns([1.05, 1.3])
            with f_left:
                mol_b = Chem.Mol(mol_main)
                Chem.rdDepictor.Compute2DCoords(mol_b)
                d_b = Draw.MolDraw2DCairo(440, 330)
                d_b.drawOptions().addBondIndices = True
                d_b.drawOptions().addAtomIndices = True
                d_b.DrawMolecule(mol_b)
                d_b.FinishDrawing()
                st.image(d_b.GetDrawingText(), caption="Indices des liaisons en bleu")
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
                            st.image(Draw.MolToImage(fm, size=(220, 175)))
                            st.metric(f"Fragment {i+1} ({f_form})", f"{f_mass:.4f} Da")
                else:
                    st.info("Toutes les liaisons appartiennent à un cycle.")

        elif spec_tool.startswith("🧲"):
            df_h, df_c, _, _, _ = simulate_advanced_nmr_1d_2d(mol_main)
            f_col1, f_col2, f_col3 = st.columns([1.1, 1.4, 1.4])
            freq_mhz = f_col1.selectbox("Fréquence :", [300.0, 400.0, 500.0], index=1, format_func=lambda v: f"{int(v)} MHz")
            zoom_h = f_col2.slider("Zoom RMN ¹H (ppm) :", 0.0, 13.0, (0.0, 10.5), 0.5)
            zoom_c = f_col3.slider("Zoom RMN ¹³C (ppm) :", 0.0, 220.0, (0.0, 210.0), 5.0)

            n1, n2 = st.columns(2)
            with n1:
                st.markdown(f"##### Spectre RMN ¹H ({int(freq_mhz)} MHz — Multiplets résolus)")
                if not df_h.empty:
                    spec_h = build_lorentzian_spectrum(df_h, "δ ¹H (ppm)", zoom_h[0], zoom_h[1], width=0.0025, freq_mhz=freq_mhz)
                    st.line_chart(spec_h, height=250)
                    with st.expander("📋 Table des attributions ¹H", expanded=True):
                        st.dataframe(df_h, use_container_width=True, hide_index=True)
            with n2:
                st.markdown("##### Spectre RMN ¹³C {¹H}")
                if not df_c.empty:
                    spec_c = build_lorentzian_spectrum(df_c, "δ ¹³C (ppm)", zoom_c[0], zoom_c[1], width=0.35, freq_mhz=freq_mhz)
                    st.line_chart(spec_c, height=250)
                    with st.expander("📋 Table des attributions ¹³C", expanded=True):
                        st.dataframe(df_c, use_container_width=True, hide_index=True)

        elif spec_tool.startswith("🗺️"):
            _, _, df_cosy, df_hsqc, df_hmbc = simulate_advanced_nmr_1d_2d(mol_main)
            c2d_1, c2d_2 = st.columns(2)
            with c2d_1:
                st.markdown("##### Carte 2D ¹H–¹H COSY (Contours & Diagonale)")
                if not df_cosy.empty:
                    render_2d_nmr_contour_plot(df_cosy, "δ ¹H F2 (ppm)", "δ ¹H F1 (ppm)", "Type", [11.0, 0.0], [11.0, 0.0], is_cosy=True)
                    with st.expander("📋 Voir les taches de corrélation COSY"):
                        st.dataframe(df_cosy, use_container_width=True, hide_index=True)
            with c2d_2:
                st.markdown("##### Carte 2D ¹H–¹³C HSQC (¹J_CH) & HMBC (²˒³J_CH)")
                df_het = pd.concat([df_hsqc, df_hmbc], ignore_index=True)
                if not df_het.empty:
                    render_2d_nmr_contour_plot(df_het, "δ ¹H (ppm)", "δ ¹³C (ppm)", "Corrélation", [11.0, 0.0], [215.0, 0.0], is_cosy=False)
                    with st.expander("📋 Voir les taches de corrélation HSQC / HMBC"):
                        st.dataframe(df_het, use_container_width=True, hide_index=True)

# ------------------------------------------
# ONGLET 5 : SÉCURITÉ SGH & STUDIO DE PUBLICATION
# ------------------------------------------
with tab_export:
    render_compact_molecule_header("t5")
    if mol_main:
        ex_l, ex_r = st.columns([1, 1.3])
        with ex_l:
            st.markdown("##### ⚠️ Sécurité SGH / GHS (PubChem)")
            pictos, hazards = get_ghs_safety(ident["cid"])
            if pictos:
                p_cols = st.columns(len(pictos))
                for i, url in enumerate(pictos):
                    p_cols[i].image(url, width=85)
                for h in hazards:
                    st.caption(f"• {h}")
            else:
                st.info("Aucun pictogramme SGH répertorié pour ce composé.")

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
                ],
                label_visibility="collapsed"
            )
            add_idx = st_col2.checkbox("Numéroter les atomes", value=False)
            svg_pub = render_publication_svg(mol_main, style_name=pub_style, show_indices=add_idx)
            components.html(f"<div style='text-align:center;'>{svg_pub}</div>", height=415)

            mol_2d_str = Chem.MolToMolBlock(mol_main)
            mol_3d_obj = generate_3d_mol(single_smiles)
            sdf_3d_str = Chem.MolToMolBlock(mol_3d_obj) if mol_3d_obj else ""

            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button("🖼️ Vectoriel (.svg)", svg_pub, file_name="publication.svg", mime="image/svg+xml", use_container_width=True)
            dl2.download_button("📄 Structure 2D (.mol)", mol_2d_str, file_name="structure_2d.mol", mime="chemical/x-mdl-molfile", use_container_width=True)
            dl3.download_button("🧊 Géométrie 3D (.sdf)", sdf_3d_str, file_name="structure_3d.sdf", mime="chemical/x-mdl-sdfile", use_container_width=True)
