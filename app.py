import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw
import io

st.set_page_config(page_title="ChemStudio — Éditeur Moléculaire", layout="wide")

st.title("🧪 ChemStudio — Éditeur & Analyseur Moléculaire")

# Barre latérale pour charger des exemples rapides ou un SMILES personnalisé
with st.sidebar:
    st.header("⚡ Chargement rapide")
    exemples = {
        "Vide": "",
        "Aspirine": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caféine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracétamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Ibuprofène": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Pénicilline G": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
    }
    choix = st.selectbox("Molécules exemples :", list(exemples.keys()), index=1)
    custom_smiles = st.text_input("Ou coller un SMILES :", value="")
    
    mol_initiale = custom_smiles if custom_smiles else exemples[choix]

# Mise en page : Éditeur à gauche, Résultats à droite
col_editor, col_analysis = st.columns([1.6, 1])

with col_editor:
    st.subheader("Éditeur 2D")
    st.caption("Dessinez votre structure puis cliquez sur **Apply** en bas à droite de l'éditeur.")
    # Éditeur Ketcher
    smiles_dessine = st_ketcher(mol_initiale, height=580)

with col_analysis:
    st.subheader("Analyse & Export")
    
    if smiles_dessine:
        mol = Chem.MolFromSmiles(smiles_dessine)
        
        if mol:
            # --- 1. IDENTIFIANTS ---
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            inchi = Chem.MolToInchi(mol)
            inchikey = Chem.MolToInchiKey(mol)
            
            with st.expander("🔑 Identifiants chimiques", expanded=True):
                st.markdown("**SMILES Canonique :**")
                st.code(canonical_smiles, language="text")
                st.markdown(f"**InChIKey :** `{inchikey}`")
            
            # --- 2. PROPRIÉTÉS PHYSICO-CHIMIQUES ---
            formula = rdMolDescriptors.CalcMolFormula(mol)
            mw = round(Descriptors.MolWt(mol), 2)
            exact_mw = round(Descriptors.ExactMolWt(mol), 4)
            logp = round(Descriptors.MolLogP(mol), 2)
            tpsa = round(Descriptors.TPSA(mol), 2)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            
            st.markdown("**📊 Propriétés physico-chimiques**")
            c1, c2 = st.columns(2)
            c1.metric("Formule brute", formula)
            c2.metric("Masse molaire", f"{mw} g/mol")
            
            c3, c4, c5 = st.columns(3)
            c3.metric("LogP (Lipophilie)", logp)
            c4.metric("TPSA (Surface polaire)", f"{tpsa} Å²")
            c5.metric("Liaisons rotatives", rot_bonds)
            
            # --- 3. RÈGLE DES 5 DE LIPINSKI ---
            lipinski_ok = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)
            st.markdown("**💊 Règle de Lipinski (Drug-Likeness)**")
            st.caption(f"Donneurs H (HBD) : **{hbd}** (≤ 5) | Accepteurs H (HBA) : **{hba}** (≤ 10)")
            if lipinski_ok:
                st.success("✅ Conforme aux règles de Lipinski")
            else:
                st.warning("⚠️ Non conforme à au moins un critère de Lipinski")
            
            st.divider()
            
            # --- 4. EXPORTATIONS (Image PNG & Fichier MOL) ---
            st.markdown("**📥 Exporter la structure**")
            
            # Génération du MolBlock (coordonnées 2D)
            Chem.rdDepictor.Compute2DCoords(mol)
            mol_block = Chem.MolToMolBlock(mol)
            
            # Génération de l'image PNG via RDKit
            img = Draw.MolToImage(mol, size=(500, 500))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label="📄 Télécharger (.mol)",
                    data=mol_block,
                    file_name=f"{formula}.mol",
                    mime="chemical/x-mdl-molfile",
                    use_container_width=True
                )
            with btn_col2:
                st.download_button(
                    label="🖼️ Image 2D (.png)",
                    data=buf.getvalue(),
                    file_name=f"{formula}.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.error("Structure chimique invalide ou valence incorrecte.")
    else:
        st.info("Dessinez une molécule dans l'éditeur et cliquez sur 'Apply' pour voir ses propriétés.")
