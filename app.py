import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdMolTransforms, Draw, rdChemReactions
from rdkit.Chem.MolStandardize import rdMolStandardize
import requests
import re
import io
import pandas as pd
import py3Dmol
import streamlit.components.v1 as components
from collections import Counter

st.set_page_config(page_title="ChemStudio Pro — 2D/3D & Réactions", layout="wide")

# ==========================================
# 1. FONCTIONS RÉSEAU (PUBCHEM : CAS, IUPAC, SGH)
# ==========================================
@st.cache_data(show_spinner=False)
def get_chemical_identifiers(smiles):
    """Récupère Nom IUPAC, CAS et Nom Trivial depuis PubChem via InChIKey."""
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
    """Convertit un numéro CAS ou un nom en SMILES canonique."""
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
    """Extrait les pictogrammes SGH et mentions de danger H depuis PubChem."""
    if not cid:
        return [], []
    pictos, hazards = [], []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            raw_text = r.text
            found_codes = set(re.findall(r"GHS0[1-9]", raw_text))
            for code in sorted(found_codes):
                pictos.append(f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.svg")
            found_h = set(re.findall(r"H[234]\d{2}[^\"\\]*", raw_text))
            hazards = sorted(list(found_h))[:6]
    except Exception:
        pass
    return pictos, hazards

# ==========================================
# 2. FONCTIONS CHIMIE, 3D & ANALYTIQUE (RDKIT)
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
    view = py3Dmol.view(width=540, height=420)
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
    components.html(view._make_html(), height=430)

def detect_functional_groups(mol):
    patterns = {
        "Acide carboxylique": "[CX3](=O)[OX2H1]",
        "Ester": "[#6][CX3](=O)[OX2H0][#6]",
        "Amide": "[NX3][CX3](=[OX1])[#6]",
        "Cétone": "[#6][CX3](=O)[#6]",
        "Aldéhyde": "[CX3H1](=O)[#6]",
        "Alcool": "[#6][OX2H]",
        "Phénol": "c[OX2H]",
        "Amine primaire/secondaire": "[NX3;H2,H1;!$(NC=O)]",
        "Amine tertiaire": "[NX3;H0;!$(NC=O)]",
        "Halogénure": "[#6][F,Cl,Br,I]",
        "Nitrile": "[NX1]#[CX2]",
        "Cycle aromatique": "a1aaaa1|a1aaaaa1"
    }
    found = []
    for name, smarts in patterns.items():
        if "|" in smarts:
            if any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in smarts.split("|")):
                found.append(name)
        else:
            p = Chem.MolFromSmarts(smarts)
            if p and mol.HasSubstructMatch(p):
                found.append(name)
    return found

def elemental_analysis(mol):
    mol_h = Chem.AddHs(mol)
    total_mw = Descriptors.MolWt(mol_h)
    counts = Counter(atom.GetSymbol() for atom in mol_h.GetAtoms())
    pt = Chem.GetPeriodicTable()
    rows = []
    for sym, count in sorted(counts.items()):
        atomic_w = pt.GetAtomicWeight(sym)
        pct = (count * atomic_w / total_mw) * 100
        rows.append({"Élément": sym, "Atomes": count, "% Massique": round(pct, 2)})
    return pd.DataFrame(rows)

def check_atom_balance(reactant_mols, product_mols):
    r_counts, p_counts = Counter(), Counter()
    for m in reactant_mols:
        for a in Chem.AddHs(m).GetAtoms():
            r_counts[a.GetSymbol()] += 1
    for m in product_mols:
        for a in Chem.AddHs(m).GetAtoms():
            p_counts[a.GetSymbol()] += 1
    diff = r_counts - p_counts
    missing = p_counts - r_counts
    return r_counts == p_counts, diff, missing

# ==========================================
# 3. INITIALISATION DE LA SESSION
# ==========================================
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirine
if "history" not in st.session_state:
    st.session_state.history = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
if "rxn_smiles" not in st.session_state:
    st.session_state.rxn_smiles = "CC(=O)O.OCC>>CC(=O)OCC"  # Estérification par défaut

# ==========================================
# 4. BARRE LATÉRALE : RECHERCHE & HISTORIQUE
# ==========================================
with st.sidebar:
    st.title("🧪 ChemStudio Pro")
    st.subheader("🔍 Charger par CAS / Nom")
    query = st.text_input("CAS, Nom trivial ou SMILES :", placeholder="Ex: 50-78-2, Ibuprofen...")
    if st.button("Charger la molécule", use_container_width=True) and query:
        mol_test = Chem.MolFromSmiles(query)
        smiles_res = Chem.MolToSmiles(mol_test) if mol_test else resolve_query_to_smiles(query)
        if smiles_res:
            st.session_state.current_smiles = smiles_res
            if smiles_res not in st.session_state.history:
                st.session_state.history.insert(0, smiles_res)
            st.success("Molécule chargée !")
        else:
            st.error("Composé introuvable.")

    st.divider()
    exemples = {
        "Aspirine (50-78-2)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caféine (58-08-2)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofène (15687-27-1)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "L-DOPA (59-92-7)": "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
        "Paracétamol (103-90-2)": "CC(=O)NC1=CC=C(O)C=C1"
    }
    ex_sel = st.selectbox("⚡ Molécules de référence :", list(exemples.keys()))
    if st.button("Appliquer l'exemple", use_container_width=True):
        st.session_state.current_smiles = exemples[ex_sel]

    st.divider()
    st.subheader("🕘 Historique de session")
    hist_choice = st.selectbox("Molécules récentes :", st.session_state.history)
    if st.button("Recharger depuis l'historique", use_container_width=True):
        st.session_state.current_smiles = hist_choice

# ==========================================
# 5. BANDEAU D'IDENTITÉ PERMANENT (EN-TÊTE)
# ==========================================
# Si l'utilisateur dessine une réaction dans Ketcher, on garde le 1er composant pour l'analyse individuelle
raw_input = st.session_state.current_smiles
single_smiles = raw_input.split(">")[0].split(".")[0] if (">" in raw_input or "." in raw_input) else raw_input
mol_main = Chem.MolFromSmiles(single_smiles)

if mol_main:
    ident = get_chemical_identifiers(single_smiles)
    b1, b2, b3, b4 = st.columns([1.4, 1, 1, 1])
    b1.metric("🏷️ Nom Trivial", ident["trivial"])
    b2.metric("🔢 Numéro CAS", ident["cas"])
    b3.metric("🧪 Formule Brute", rdMolDescriptors.CalcMolFormula(mol_main))
    b4.metric("⚖️ Masse Molaire", f"{round(Descriptors.MolWt(mol_main), 2)} g/mol")
    st.caption(f"**Nom IUPAC officiel :** `{ident['iupac']}`")
else:
    st.warning("Veuillez dessiner ou charger une structure chimique valide.")

# ==========================================
# 6. ESPACE DE TRAVAIL À 5 ONGLETS ÉPURÉS
# ==========================================
tab_2d3d, tab_rxn, tab_struct, tab_spec, tab_export = st.tabs([
    "🎨 1. Atelier 2D & 3D",
    "⚗️ 2. Laboratoire de Réactions",
    "🔬 3. Structure & Stéréochimie",
    "📈 4. Spectrométrie & Analyse",
    "📥 5. Sécurité & Exports"
])

# ------------------------------------------
# ONGLET 1 : ATELIER 2D & MODÉLISATION 3D
# ------------------------------------------
with tab_2d3d:
    col_left, col_right = st.columns([1.3, 1.1])
    
    with col_left:
        st.markdown("##### Éditeur 2D (Molécule ou Réaction)")
        st.caption("Dessinez une molécule ou une réaction complète (avec flèche `->`), puis cliquez sur **Apply**.")
        ketcher_out = st_ketcher(st.session_state.current_smiles, height=470)
        if ketcher_out and ketcher_out != st.session_state.current_smiles:
            st.session_state.current_smiles = ketcher_out
            if ">>" in ketcher_out or ">" in ketcher_out:
                st.session_state.rxn_smiles = ketcher_out
            elif ketcher_out not in st.session_state.history:
                st.session_state.history.insert(0, ketcher_out)
            st.rerun()

    with col_right:
        st.markdown("##### Modèle 3D Interactif")
        if mol_main:
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            style_3d = c_opt1.selectbox("Style", ["Ball & Stick", "Stick", "Spacefill"])
            ff_3d = c_opt2.selectbox("Champ de force", ["MMFF94", "UFF"])
            seed_3d = c_opt3.number_input("Conformère #", 1, 100, 42)
            
            chk1, chk2, chk3 = st.columns(3)
            show_h = chk1.checkbox("Hydrogènes", value=True)
            color_q = chk2.checkbox("Charges δ+/δ-", value=False)
            spin_3d = chk3.checkbox("Rotation", value=False)

            mol_3d = generate_3d_mol(single_smiles, force_field=ff_3d, seed=int(seed_3d))
            if mol_3d:
                with st.expander("🛠️ Modifier une distance interatomique en 3D"):
                    n_at = mol_3d.GetNumAtoms()
                    mc1, mc2, mc3 = st.columns(3)
                    ai = mc1.number_input("Atome A", 0, n_at - 1, 0)
                    aj = mc2.number_input("Atome B", 0, n_at - 1, min(1, n_at - 1))
                    conf = mol_3d.GetConformer()
                    cur_d = round(rdMolTransforms.GetBondLength(conf, int(ai), int(aj)), 2)
                    new_d = mc3.slider("Longueur (Å)", 0.8, 3.5, float(min(max(cur_d, 0.8), 3.5)), 0.05)
                    if new_d != cur_d and ai != aj:
                        try:
                            rdMolTransforms.SetBondLength(conf, int(ai), int(aj), new_d)
                        except Exception:
                            st.caption("⚠️ Liaison verrouillée dans un cycle.")

                render_3d_viewer(mol_3d, style=style_3d, show_h=show_h, color_charges=color_q, spin=spin_3d)

# ------------------------------------------
# ONGLET 2 : LABORATOIRE DE RÉACTIONS (LES 6 OPTIONS)
# ------------------------------------------
with tab_rxn:
    st.markdown("##### 1. Définition ou Prédiction de la Réaction")
    rxn_mode = st.radio(
        "Source de la réaction :",
        [
            "📐 Utiliser la réaction dessinée dans Ketcher (ou SMILES de réaction)",
            "🔎 Construire par Numéros CAS / Noms triviaux",
            "🤖 Simulateur automatique (Prédire le produit à partir des réactifs)"
        ],
        horizontal=True
    )

    if rxn_mode.startswith("📐"):
        rxn_in = st.text_input(
            "SMILES de la réaction (Format : Réactif1.Réactif2>>Produit1) :",
            value=st.session_state.rxn_smiles
        )
        st.session_state.rxn_smiles = rxn_in

    elif rxn_mode.startswith("🔎"):
        rc1, rc2, rc3 = st.columns(3)
        r1_q = rc1.text_input("Réactif 1 (CAS ou Nom)", value="Salicylic acid")
        r2_q = rc2.text_input("Réactif 2 (Optionnel)", value="Acetic anhydride")
        p1_q = rc3.text_input("Produit attendu (CAS ou Nom)", value="Aspirin")
        if st.button("Générer l'équation réactionnelle"):
            s_r1 = resolve_query_to_smiles(r1_q)
            s_r2 = resolve_query_to_smiles(r2_q) if r2_q.strip() else ""
            s_p1 = resolve_query_to_smiles(p1_q)
            if s_r1 and s_p1:
                reacts = f"{s_r1}.{s_r2}" if s_r2 else s_r1
                st.session_state.rxn_smiles = f"{reacts}>>{s_p1}"
                st.success("Réaction assemblée avec succès !")
            else:
                st.error("Un des composés n'a pas pu être résolu via PubChem.")

    elif rxn_mode.startswith("🤖"):
        templates = {
            "Estérification de Fischer (Acide + Alcool -> Ester)": "[C:1](=[O:2])[OH].[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
            "Formation d'Amide (Acide + Amine -> Amide)": "[C:1](=[O:2])[OH].[N;H2,H1:3]>>[C:1](=[O:2])[N:3]",
            "Réduction de Cétone/Aldéhyde en Alcool": "[C:1]=[O:2]>>[C:1][OH:2]",
            "Saponification d'Ester en Acide Carboxylique": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH]",
            "Click Chemistry CuAAC (Alcyne + Azoture -> Triazole)": "[C:1]#[CH:2].[N:3]=[N+:4]=[N-:5]>>[c:1]1[cH:2][n:3][n:4][n:5]1"
        }
        tc1, tc2, tc3 = st.columns([1.4, 1, 1])
        chosen_tpl = tc1.selectbox("Type de transformation :", list(templates.keys()))
        sub1 = tc2.text_input("SMILES ou Nom Réactif 1", value="CC(=O)O")
        sub2 = tc3.text_input("SMILES ou Nom Réactif 2 (si bimoléculaire)", value="NCC")
        if st.button("⚡ Prédire et construire le produit"):
            m1 = Chem.MolFromSmiles(sub1) or Chem.MolFromSmiles(resolve_query_to_smiles(sub1) or "")
            m2 = Chem.MolFromSmiles(sub2) or Chem.MolFromSmiles(resolve_query_to_smiles(sub2) or "") if sub2.strip() else None
            rxn_engine = rdChemReactions.ReactionFromSmarts(templates[chosen_tpl])
            try:
                prods = rxn_engine.RunReactants((m1, m2) if m2 else (m1,))
                if prods and len(prods[0]) > 0:
                    prod_mol = prods[0][0]
                    Chem.SanitizeMol(prod_mol)
                    prod_smi = Chem.MolToSmiles(prod_mol)
                    r_str = f"{Chem.MolToSmiles(m1)}.{Chem.MolToSmiles(m2)}" if m2 else Chem.MolToSmiles(m1)
                    st.session_state.rxn_smiles = f"{r_str}>>{prod_smi}"
                    st.success(f"Produit prédit : `{prod_smi}`")
                else:
                    st.warning("Les groupes fonctionnels des réactifs ne correspondent pas à cette réaction.")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

    # --- ANALYSE & TABLEAU DE STŒCHIOMÉTRIE DE LA RÉACTION ACTIVE ---
    st.divider()
    rxn_str = st.session_state.rxn_smiles
    if ">>" in rxn_str or ">" in rxn_str:
        parts = rxn_str.split(">")
        r_smiles_list = [s for s in parts[0].split(".") if s.strip()]
        p_smiles_list = [s for s in parts[-1].split(".") if s.strip()]
        r_mols = [Chem.MolFromSmiles(s) for s in r_smiles_list if Chem.MolFromSmiles(s)]
        p_mols = [Chem.MolFromSmiles(s) for s in p_smiles_list if Chem.MolFromSmiles(s)]

        if r_mols and p_mols:
            # Affichage graphique de l'équation bilan
            rxn_obj = rdChemReactions.ReactionFromSmarts(f"{'.'.join(r_smiles_list)}>>{'.'.join(p_smiles_list)}", useSmiles=True)
            rxn_img = Draw.ReactionToImage(rxn_obj, subImgSize=(260, 180))
            st.image(rxn_img, caption="Schéma réactionnel validé")

            st.markdown("##### 2. Grille de Stœchiométrie Interactive & Calcul de Rendement")
            sc1, sc2, sc3 = st.columns(3)
            ref_mass_mg = sc1.number_input("Masse engagée du Réactif 1 de référence (mg) :", min_value=1.0, value=500.0, step=50.0)
            exp_mass_mg = sc2.number_input("Masse de Produit 1 réellement isolée au labo (mg) :", min_value=0.0, value=0.0, step=10.0)
            
            mw_r1 = Descriptors.MolWt(r_mols[0])
            ref_mmol = ref_mass_mg / mw_r1  # mmol du réactif 1

            # Construction de la table des réactifs et produits
            stoich_rows = []
            for idx, m in enumerate(r_mols):
                smi = Chem.MolToSmiles(m)
                info = get_chemical_identifiers(smi)
                mw = round(Descriptors.MolWt(m), 2)
                stoich_rows.append({
                    "Rôle": f"Réactif {idx+1}",
                    "Nom": info["trivial"],
                    "CAS": info["cas"],
                    "MW (g/mol)": mw,
                    "Équivalents (eq)": 1.0 if idx == 0 else 1.2,
                    "Densité (g/mL)": 1.0
                })
            for idx, m in enumerate(p_mols):
                smi = Chem.MolToSmiles(m)
                info = get_chemical_identifiers(smi)
                mw = round(Descriptors.MolWt(m), 2)
                stoich_rows.append({
                    "Rôle": f"Produit {idx+1}",
                    "Nom": info["trivial"],
                    "CAS": info["cas"],
                    "MW (g/mol)": mw,
                    "Équivalents (eq)": 1.0,
                    "Densité (g/mL)": 1.0
                })

            df_in = pd.DataFrame(stoich_rows)
            st.caption("💡 *Modifiez directement les colonnes **Équivalents (eq)** ou **Densité (g/mL)** dans le tableau ci-dessous :*")
            edited_df = st.data_editor(
                df_in,
                disabled=["Rôle", "Nom", "CAS", "MW (g/mol)"],
                use_container_width=True,
                hide_index=True
            )

            # Calcul automatique des quantités (mmol, mg, mL)
            edited_df["Quantité (mmol)"] = (edited_df["Équivalents (eq)"] * ref_mmol).round(3)
            edited_df["Masse à peser / attendue (mg)"] = (edited_df["Quantité (mmol)"] * edited_df["MW (g/mol)"]).round(1)
            edited_df["Volume si liquide (µL)"] = (edited_df["Masse à peser / attendue (mg)"] / edited_df["Densité (g/mL)"]).round(1)
            st.dataframe(
                edited_df[["Rôle", "Nom", "CAS", "MW (g/mol)", "Équivalents (eq)", "Quantité (mmol)", "Masse à peser / attendue (mg)", "Volume si liquide (µL)"]],
                use_container_width=True,
                hide_index=True
            )

            # Indicateurs de synthèse : Rendement, Atom Economy, Delta m/z, Bilan atomique
            mw_p1 = Descriptors.MolWt(p_mols[0])
            theo_yield_mg = ref_mmol * mw_p1
            actual_yield_pct = (exp_mass_mg / theo_yield_mg * 100) if (theo_yield_mg > 0 and exp_mass_mg > 0) else 0.0
            sum_mw_reactants = sum(Descriptors.MolWt(m) for m in r_mols)
            atom_economy = (mw_p1 / sum_mw_reactants) * 100
            delta_mz = Descriptors.ExactMolWt(p_mols[0]) - Descriptors.ExactMolWt(r_mols[0])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎯 Masse théorique (100%)", f"{round(theo_yield_mg, 1)} mg")
            m2.metric("🏆 Rendement expérimental", f"{round(actual_yield_pct, 1)} %" if exp_mass_mg > 0 else "Renseigner masse isolée")
            m3.metric("♻️ Économie d'atomes", f"{round(atom_economy, 1)} %")
            m4.metric("📡 Suivi MS (Δ m/z R1→P1)", f"{delta_mz:+.4f} Da")

            is_bal, lost_atoms, gained_atoms = check_atom_balance(r_mols, p_mols)
            if is_bal:
                st.success("✅ **Bilan atomique équilibré :** Tous les atomes des réactifs se retrouvent dans les produits dessinés.")
            else:
                lost_str = "".join(f"{k}{v if v>1 else ''} " for k, v in lost_atoms.items())
                st.info(f"ℹ️ **Sous-produit non dessiné (groupe partant / condensation) :** Atomes perdus = `{lost_str.strip() or 'Aucun'}`")

            # Zone C : Conditions opératoires & Export Cahier de Labo
            with st.expander("📋 3. Conditions Opératoires & Export Cahier de Laboratoire (ELN)"):
                cc1, cc2, cc3, cc4 = st.columns(4)
                solv = cc1.selectbox("Solvant", ["DCM", "THF", "DMF", "MeOH", "EtOH", "AcOEt", "Toluène", "Eau", "Acétonitrile", "Sans solvant"])
                temp = cc2.text_input("Température", value="TA (20 °C)")
                time_rxn = cc3.text_input("Durée", value="2 h")
                atm = cc4.selectbox("Atmosphère", ["Air ambiant", "Sous Argon (Ar)", "Sous Azote (N₂)", "Tube scellé"])
                notes = st.text_area("Protocole expérimental / Observations :", value=f"Réaction menée dans {solv} à {temp} pendant {time_rxn} ({atm}).")
                
                csv_export = edited_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Télécharger la Fiche Stœchiométrique (.csv / Excel)",
                    data=csv_export,
                    file_name="cahier_de_labo_stoechiometrie.csv",
                    mime="text/csv"
                )

# ------------------------------------------
# ONGLET 3 : STRUCTURE, STÉRÉOCHIMIE & PROPRIÉTÉS
# ------------------------------------------
with tab_struct:
    if mol_main:
        sc_left, sc_right = st.columns([1.1, 1.2])
        
        with sc_left:
            st.markdown("##### Carte 2D Annotée (Labels R/S, E/Z & Numéros d'atomes)")
            mol_annot = Chem.Mol(mol_main)
            Chem.rdDepictor.Compute2DCoords(mol_annot)
            Chem.AssignStereochemistry(mol_annot, cleanIt=True, force=True)
            d2d = Draw.MolDraw2DCairo(480, 380)
            opts = d2d.drawOptions()
            opts.addStereoAnnotation = True
            opts.addAtomIndices = True
            d2d.DrawMolecule(mol_annot)
            d2d.FinishDrawing()
            st.image(d2d.GetDrawingText(), caption="Indices d'atomes et descripteurs Cahn-Ingold-Prelog (R/S, E/Z)")

            chiral_centers = Chem.FindMolChiralCenters(mol_annot, includeUnassigned=True)
            if chiral_centers:
                st.success(f"**Centres stéréogènes détectés :** {chiral_centers}")
            else:
                st.caption("Molécule achirale (aucun carbone asymétrique détecté).")

        with sc_right:
            st.markdown("##### Profil Physico-Chimique & Groupes Fonctionnels")
            # Calcul du degré d'insaturation (DBE / IHD)
            mol_h = Chem.AddHs(mol_main)
            cnt = Counter(a.GetSymbol() for a in mol_h.GetAtoms())
            dbe = cnt["C"] + 1 - (cnt["H"] + cnt["F"] + cnt["Cl"] + cnt["Br"] + cnt["I"] - cnt["N"]) / 2

            p1, p2, p3 = st.columns(3)
            logp = round(Descriptors.MolLogP(mol_main), 2)
            tpsa = round(Descriptors.TPSA(mol_main), 2)
            p1.metric("Degré d'insaturation (DBE)", f"{dbe:.0f}")
            p2.metric("LogP (Lipophilie)", logp)
            p3.metric("Surface Polaire (TPSA)", f"{tpsa} Å²")

            p4, p5, p6 = st.columns(3)
            hbd = rdMolDescriptors.CalcNumHBD(mol_main)
            hba = rdMolDescriptors.CalcNumHBA(mol_main)
            rot = rdMolDescriptors.CalcNumRotatableBonds(mol_main)
            p4.metric("Donneurs H (HBD)", hbd)
            p5.metric("Accepteurs H (HBA)", hba)
            p6.metric("Liaisons rotatives", rot)

            fg_list = detect_functional_groups(mol_main)
            st.markdown("**🧩 Fonctions chimiques identifiées :**")
            st.info(", ".join(fg_list) if fg_list else "Aucune fonction standard répertoriée.")

            with st.expander("🔄 Explorer les formes tautomères possibles"):
                enumerator = rdMolStandardize.TautomerEnumerator()
                tautomers = list(enumerator.Enumerate(mol_main))[:6]
                st.caption(f"{len(tautomers)} forme(s) tautomère(s) générée(s) par RDKit :")
                taut_img = Draw.MolsToGridImage(tautomers, molsPerRow=3, subImgSize=(200, 160))
                st.image(taut_img)

# ------------------------------------------
# ONGLET 4 : SPECTROMÉTRIE & ANALYSE (HRMS, ÉLÉMENTAIRE, RMN)
# ------------------------------------------
with tab_spec:
    if mol_main:
        sp_col1, sp_col2 = st.columns(2)
        
        with sp_col1:
            st.markdown("##### 📡 Spectrométrie de Masse Haute Résolution (HRMS)")
            exact_m = Descriptors.ExactMolWt(mol_main)
            st.metric("Masse Exacte Monoisotopique (M)", f"{exact_m:.5f} Da")
            
            adducts = pd.DataFrame([
                {"Espèce ionique": "[M + H]⁺", "m/z théorique": round(exact_m + 1.007276, 5)},
                {"Espèce ionique": "[M + Na]⁺", "m/z théorique": round(exact_m + 22.989218, 5)},
                {"Espèce ionique": "[M + K]⁺", "m/z théorique": round(exact_m + 38.963158, 5)},
                {"Espèce ionique": "[M + NH₄]⁺", "m/z théorique": round(exact_m + 18.033823, 5)},
                {"Espèce ionique": "[M - H]⁻", "m/z théorique": round(exact_m - 1.007276, 5)},
            ])
            st.dataframe(adducts, use_container_width=True, hide_index=True)

            st.markdown("##### 🧲 Symétrie RMN (Signaux attendus)")
            mol_with_h = Chem.AddHs(mol_main)
            ranks = list(Chem.CanonicalRankAtoms(mol_with_h, breakTies=False))
            c_sym = len(set(ranks[i] for i, a in enumerate(mol_with_h.GetAtoms()) if a.GetSymbol() == "C"))
            h_sym = len(set(ranks[i] for i, a in enumerate(mol_with_h.GetAtoms()) if a.GetSymbol() == "H"))
            n1, n2 = st.columns(2)
            n1.metric("Signaux RMN ¹³C théoriques", c_sym)
            n2.metric("Environnements ¹H distincts", h_sym)

        with sp_col2:
            st.markdown("##### 🧪 Analyse Centésimale Élémentaire")
            df_elem = elemental_analysis(mol_main)
            st.dataframe(df_elem, use_container_width=True, hide_index=True)
            st.bar_chart(df_elem.set_index("Élément")["% Massique"])

# ------------------------------------------
# ONGLET 5 : SÉCURITÉ SGH & EXPORTS DE PUBLICATION
# ------------------------------------------
with tab_export:
    if mol_main:
        ex_left, ex_right = st.columns(2)
        
        with ex_left:
            st.markdown("##### ⚠️ Sécurité SGH / GHS (Données PubChem)")
            pictos, hazards = get_ghs_safety(ident["cid"])
            if pictos:
                p_cols = st.columns(len(pictos))
                for i, url in enumerate(pictos):
                    p_cols[i].image(url, width=90)
                for h in hazards:
                    st.caption(f"• {h}")
            else:
                st.info("Aucun pictogramme de danger SGH répertorié sur PubChem pour cette structure.")

        with ex_right:
            st.markdown("##### 📥 Formats d'Exportation & Publication")
            # Dessin SVG vectoriel haute qualité
            Chem.rdDepictor.Compute2DCoords(mol_main)
            svg_drawer = Draw.MolDraw2DSVG(500, 500)
            svg_drawer.DrawMolecule(mol_main)
            svg_drawer.FinishDrawing()
            svg_data = svg_drawer.GetDrawingText()

            mol_2d_block = Chem.MolToMolBlock(mol_main)
            mol_3d_exp = generate_3d_mol(single_smiles)
            sdf_3d_block = Chem.MolToMolBlock(mol_3d_exp) if mol_3d_exp else ""

            e1, e2, e3 = st.columns(3)
            e1.download_button("🖼️ Vectoriel (.svg)", svg_data, file_name="structure.svg", mime="image/svg+xml", use_container_width=True)
            e2.download_button("📄 Fichier 2D (.mol)", mol_2d_block, file_name="structure_2d.mol", mime="chemical/x-mdl-molfile", use_container_width=True)
            e3.download_button("🧊 Géométrie 3D (.sdf)", sdf_3d_block, file_name="structure_3d.sdf", mime="chemical/x-mdl-sdfile", use_container_width=True)

            st.markdown("**Identifiants copiables :**")
            st.code(
                f"SMILES   : {Chem.MolToSmiles(mol_main, canonical=True)}\n"
                f"InChI    : {Chem.MolToInchi(mol_main)}\n"
                f"InChIKey : {Chem.MolToInchiKey(mol_main)}",
                language="text"
            )
