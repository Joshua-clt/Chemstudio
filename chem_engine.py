from collections import Counter
from functools import lru_cache
from itertools import permutations
import html
import json
import os
import re
from typing import Any
import urllib.parse

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Draw,
    rdChemReactions,
    rdDepictor,
    rdFMCS,
    rdMolDescriptors,
    rdMolTransforms,
)
from rdkit.Chem.Draw import rdMolDraw2D # type: ignore
from rdkit.Chem.MolStandardize import rdMolStandardize

EXEMPLES_MOLECULES: dict[str, str] = {
    "Aspirine (50-78-2)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Paracétamol (103-90-2)": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofène (15687-27-1)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Caféine (58-08-2)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "4-Bromoacétophénone (Massif Br)": "CC(=O)c1ccc(Br)cc1",
    "Biphényle-4-carboxamide (Suzuki)": "NC(=O)c1ccc(-c2ccccc2)cc1",
}

from reactions_db import REACTION_CATALOG, REACTION_TEMPLATES, RETRO_RULES


def clean_val(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def extract_single_smiles(raw_smiles: str) -> str:
    raw = clean_val(raw_smiles)
    if ">" in raw or "." in raw:
        return raw.split(">")[0].split(".")[0].strip()
    return raw


def parse_smiles(smiles: str):
    s = clean_val(smiles)
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


# ==================================================================
# 1. CLIENT API CHEMBASE v2.0 (X-API-Key & PRÉ-REMPLISSAGE)
# ==================================================================
def get_chembase_config(override_url: str = "", override_key: str = "") -> tuple[str, str]:
    url = clean_val(override_url or os.environ.get("CHEMBASE_URL", "")).rstrip("/")
    key = clean_val(override_key or os.environ.get("CHEMBASE_API_KEY", ""))
    return url, key


def fetch_chembase_catalog(query: str = "", override_url: str = "", override_key: str = "") -> tuple[bool, list[dict[str, Any]], str]:
    base_url, api_key = get_chembase_config(override_url, override_key)
    if not base_url or not api_key:
        return False, [], "CHEMBASE_URL / CHEMBASE_API_KEY non configurés."
    try:
        r = requests.get(
            f"{base_url}/api/v1/catalog",
            params={"q": query} if clean_val(query) else None,
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            return True, list(data.get("items", [])), f"{data.get('count', 0)} fiche(s) récupérée(s) depuis ChemBase."
        if r.status_code == 401:
            return False, [], "Clé X-API-Key refusée par ChemBase (401)."
        return False, [], f"Réponse HTTP {r.status_code} de ChemBase."
    except Exception as ex:
        return False, [], f"ChemBase injoignable : {ex}"


def lookup_chembase_by_cas(cas: str, override_url: str = "", override_key: str = "") -> dict[str, Any] | None:
    cas_c = clean_val(cas)
    base_url, api_key = get_chembase_config(override_url, override_key)
    if not cas_c or cas_c in ("Non trouvé", "-", "N/A") or not base_url or not api_key:
        return None
    try:
        r = requests.get(
            f"{base_url}/api/v1/molecule/{urllib.parse.quote(cas_c)}",
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=5,
        )
        if r.status_code == 200:
            return dict(r.json())
    except Exception:
        pass
    return None


def build_chembase_send_url(cas: str, name: str, smiles: str, override_url: str = "") -> str:
    base_url, _k = get_chembase_config(override_url, "")
    if not base_url:
        return ""
    params: dict[str, str] = {}
    if clean_val(cas) and clean_val(cas) not in ("Non trouvé", "-", "N/A"):
        params["new_cas"] = clean_val(cas)
    if clean_val(name) and clean_val(name) not in ("Composé personnalisé", "Invalide"):
        params["new_name"] = clean_val(name)
    if clean_val(smiles):
        params["new_smiles"] = clean_val(smiles)
    return f"{base_url}/?{urllib.parse.urlencode(params)}" if params else f"{base_url}/"


# ==================================================================
# 2. FONCTIONS RÉSEAU PUBCHEM (CAS, IUPAC, SGH, MARKUSH)
# ==================================================================
@lru_cache(maxsize=256)
def get_chemical_identifiers(smiles: str) -> dict[str, Any]:
    res: dict[str, Any] = {
        "iupac": "Non répertorié",
        "cas": "Non trouvé",
        "trivial": "Composé personnalisé",
        "cid": None,
    }
    try:
        mol = parse_smiles(smiles)
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
                    if cas_pattern.match(str(syn)):
                        res["cas"] = str(syn)
                        break
    except Exception:
        pass
    return res


@lru_cache(maxsize=256)
def resolve_query_to_smiles(query: str) -> str | None:
    q = clean_val(query)
    if not q:
        return None
    mol_t = parse_smiles(q)
    if mol_t:
        return Chem.MolToSmiles(mol_t, canonical=True)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(q)}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
    except Exception:
        return None
    return None


@lru_cache(maxsize=256)
def get_ghs_safety(cid: int | None) -> tuple[list[dict[str, str]], list[str]]:
    if not cid:
        return [], []
    pictos: list[dict[str, str]] = []
    hazards: list[str] = []
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS+Classification"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            raw_text = r.text
            for code in sorted(set(re.findall(r"GHS0[1-9]", raw_text))):
                pictos.append({"code": code, "url": f"https://pubchem.ncbi.nlm.nih.gov/images/ghs/{code}.svg"})
            hazards = sorted(set(re.findall(r"H[234]\d{2}[^\"\\]*", raw_text)))[:8]
    except Exception:
        pass
    return pictos, hazards


@lru_cache(maxsize=64)
def search_pubchem_markush_analogues(smiles: str, mode: str = "similarity", threshold: int = 85, max_records: int = 6) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    try:
        mol = parse_smiles(smiles)
        if not mol:
            return results
        can_smi = Chem.MolToSmiles(mol, canonical=True)
        quoted = urllib.parse.quote(can_smi, safe="")
        if mode == "similarity":
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{quoted}"
                f"/property/Title,CanonicalSMILES,MolecularWeight/JSON?Threshold={int(threshold)}&MaxRecords={int(max_records)}"
            )
        else:
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/smiles/{quoted}"
                f"/property/Title,CanonicalSMILES,MolecularWeight/JSON?MaxRecords={int(max_records)}"
            )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            for p in props:
                smi_found = p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
                if smi_found:
                    results.append({
                        "cid": p.get("CID"),
                        "nom": str(p.get("Title") or f"CID {p.get('CID')}"),
                        "mw": p.get("MolecularWeight"),
                        "smiles": str(smi_found),
                    })
    except Exception:
        pass
    return results


# ==================================================================
# 3. MOTEUR QUANTIQUE (HÜCKEL ÉTENDU : HOMO, LUMO, FUKUI, DIPÔLE)
# ==================================================================
def compute_quantum_orbitals(mol) -> dict[str, Any] | None:
    if mol is None:
        return None
    h_params = {"C": 0.0, "N": 0.8, "O": 1.5, "S": 0.6, "F": 2.2, "Cl": 1.4, "Br": 1.2, "I": 1.0, "B": -0.8, "P": 0.4}
    k_params = {
        ("C", "C"): 1.0, ("C", "N"): 0.9, ("C", "O"): 0.85, ("C", "S"): 0.7,
        ("C", "F"): 0.6, ("C", "Cl"): 0.4, ("C", "Br"): 0.3, ("N", "N"): 0.8, ("N", "O"): 0.7,
    }
    n = mol.GetNumAtoms()
    if n == 0:
        return None

    h_mat = np.zeros((n, n))
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        hyb_shift = 0.15 if str(atom.GetHybridization()) == "SP3" else 0.0
        h_mat[i, i] = h_params.get(sym, 0.0) + 0.3 * atom.GetFormalCharge() - hyb_shift

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        s1, s2 = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        bo = bond.GetBondTypeAsDouble()
        k_base = k_params.get((s1, s2), k_params.get((s2, s1), 0.75))
        k_val = k_base * (1.0 + 0.22 * (bo - 1.0))
        h_mat[i, j] = k_val
        h_mat[j, i] = k_val

    eigvals, eigvecs = np.linalg.eigh(h_mat)
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

    fukui_rows: list[dict[str, Any]] = []
    for i, a in enumerate(mol.GetAtoms()):
        fukui_rows.append({
            "atome": f"{a.GetSymbol()}{i}",
            "coeff_homo": round(float(c_homo[i]), 3),
            "coeff_lumo": round(float(c_lumo[i]), 3),
            "f_minus": round(float(f_minus[i]), 3),
            "f_plus": round(float(f_plus[i]), 3),
        })

    return {
        "e_homo": round(e_homo, 2),
        "e_lumo": round(e_lumo, 2),
        "gap": round(gap_ev, 2),
        "omega": round(omega, 2),
        "c_homo": [float(v) for v in c_homo],
        "c_lumo": [float(v) for v in c_lumo],
        "fukui_rows": fukui_rows,
    }


def compute_gasteiger_charges(mol_3d) -> list[float]:
    charges: list[float] = []
    try:
        AllChem.ComputeGasteigerCharges(mol_3d)
        for atom in mol_3d.GetAtoms():
            q = float(atom.GetProp("_GasteigerCharge"))
            charges.append(0.0 if (np.isnan(q) or np.isinf(q)) else q)
    except Exception:
        charges = [0.0] * mol_3d.GetNumAtoms()
    return charges


def compute_dipole_moment_3d(mol_3d) -> float:
    try:
        charges = compute_gasteiger_charges(mol_3d)
        conf = mol_3d.GetConformer()
        dip_vec = np.zeros(3)
        for i, q in enumerate(charges):
            pos = conf.GetAtomPosition(i)
            dip_vec += q * np.array([pos.x, pos.y, pos.z])
        return round(float(np.linalg.norm(dip_vec) * 4.8032), 2)
    except Exception:
        return 0.0


# ==================================================================
# 4. MOTEUR 3D, OVERLAY MCS/RMSD, VUES 3DMOL.JS & ÉDITEUR 2D JSME
# ==================================================================
def generate_3d_mol(smiles: str, force_field: str = "MMFF94", seed: int = 42):
    mol = parse_smiles(smiles)
    if not mol:
        return None
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    res = AllChem.EmbedMolecule(mol_3d, params)
    if res != 0:
        res = AllChem.EmbedMolecule(mol_3d, randomSeed=int(seed))
    if res == 0:
        if force_field == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(mol_3d)
        else:
            AllChem.UFFOptimizeMolecule(mol_3d)
    return mol_3d


def get_bond_length_3d(mol_3d, atom_i: int, atom_j: int) -> float:
    try:
        conf = mol_3d.GetConformer()
        return round(float(rdMolTransforms.GetBondLength(conf, int(atom_i), int(atom_j))), 2)
    except Exception:
        return 1.50


def set_bond_length_3d(mol_3d, atom_i: int, atom_j: int, new_dist: float) -> tuple[bool, str]:
    try:
        if atom_i == atom_j:
            return False, "Choisissez deux atomes distincts."
        conf = mol_3d.GetConformer()
        rdMolTransforms.SetBondLength(conf, int(atom_i), int(atom_j), float(new_dist))
        return True, f"Distance A{atom_i}–A{atom_j} ajustée à {new_dist:.2f} Å."
    except Exception:
        return False, "Liaison cyclique verrouillée ou non connectée directement."


def overlay_molecules_3d(smiles1: str, smiles2: str):
    m1 = generate_3d_mol(smiles1)
    m2 = generate_3d_mol(smiles2)
    if not m1 or not m2:
        return None, None, 0, None
    m1_noh, m2_noh = Chem.RemoveHs(m1), Chem.RemoveHs(m2)
    mcs = rdFMCS.FindMCS([m1_noh, m2_noh], timeout=3)
    rmsd = None
    mcs_atoms = int(mcs.numAtoms) if mcs else 0
    if mcs_atoms >= 3:
        core = Chem.MolFromSmarts(mcs.smartsString)
        match1 = m1_noh.GetSubstructMatch(core) if core else ()
        match2 = m2_noh.GetSubstructMatch(core) if core else ()
        if match1 and match2:
            try:
                rmsd = float(AllChem.AlignMol(m2, m1, atomMap=list(zip(match2, match1))))
            except Exception:
                rmsd = None
    return m1, m2, mcs_atoms, rmsd


def _charge_to_hex(q: float) -> str:
    """Palette bleu (δ+) -> blanc (0) -> rouge (δ-) pour charges Gasteiger."""
    clamped = max(-0.45, min(0.45, q)) / 0.45
    if clamped >= 0:
        r = int(245 - 195 * clamped)
        g = int(245 - 125 * clamped)
        b = 250
    else:
        t = -clamped
        r = 245
        g = int(245 - 185 * t)
        b = int(245 - 185 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_3dmol_standard_html(
    mol_3d,
    style: str = "Ball & Stick",
    show_h: bool = True,
    color_charges: bool = False,
    spin: bool = False,
    dark_mode: bool = True,
) -> str:
    mol_block = Chem.MolToMolBlock(mol_3d)
    charges = compute_gasteiger_charges(mol_3d) if color_charges else []
    hex_map = [_charge_to_hex(q) for q in charges]
    bg = "#0b1416" if dark_mode else "#f8fafc"
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"/>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body,#v3d{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:{bg};}}</style>
</head><body><div id="v3d"></div>
<script>
const molBlock = {json.dumps(mol_block)};
const hexMap = {json.dumps(hex_map)};
const styleName = {json.dumps(style)};
const showH = {json.dumps(bool(show_h))};
const colorCharges = {json.dumps(bool(color_charges))};
const doSpin = {json.dumps(bool(spin))};
const viewer = $3Dmol.createViewer("v3d", {{backgroundColor: {json.dumps(bg)}}});
viewer.addModel(molBlock, "mol");
if (colorCharges && hexMap.length > 0) {{
  const cFunc = (atom) => hexMap[atom.serial - 1] || "#94a3b8";
  if (styleName === "Spacefill") {{
    viewer.setStyle({{}}, {{sphere: {{colorfunc: cFunc}}}});
  }} else if (styleName === "Stick") {{
    viewer.setStyle({{}}, {{stick: {{radius: 0.2, colorfunc: cFunc}}}});
  }} else {{
    viewer.setStyle({{}}, {{stick: {{radius: 0.16, colorfunc: cFunc}}, sphere: {{scale: 0.24, colorfunc: cFunc}}}});
  }}
}} else if (styleName === "Stick") {{
  viewer.setStyle({{}}, {{stick: {{radius: 0.20}}}});
}} else if (styleName === "Spacefill") {{
  viewer.setStyle({{}}, {{sphere: {{}}}});
}} else {{
  viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.25}}}});
}}
if (!showH) viewer.setStyle({{elem: "H"}}, {{}});
viewer.zoomTo();
if (doSpin) viewer.spin("y", 0.7);
viewer.render();
</script></body></html>"""


def build_3dmol_quantum_html(
    mol_3d,
    coeffs: list[float],
    mode: str = "HOMO",
    dark_mode: bool = True,
) -> str:
    mol_block = Chem.MolToMolBlock(mol_3d)
    bg = "#0b1416" if dark_mode else "#f8fafc"
    spheres: list[dict[str, Any]] = []
    hex_map: list[str] = []

    if mode in ("HOMO", "LUMO"):
        conf = mol_3d.GetConformer()
        for i, c_val in enumerate(coeffs):
            amp = abs(float(c_val))
            if amp >= 0.12 and i < mol_3d.GetNumAtoms():
                pos = conf.GetAtomPosition(i)
                radius = min(1.35, max(0.32, amp * 1.65))
                color_pos = "#38bdf8" if c_val > 0 else "#f43f5e"
                color_neg = "#f43f5e" if c_val > 0 else "#38bdf8"
                spheres.append({"x": float(pos.x), "y": float(pos.y), "z": float(pos.z + 0.42), "r": radius, "c": color_pos, "o": 0.76})
                spheres.append({"x": float(pos.x), "y": float(pos.y), "z": float(pos.z - 0.42), "r": radius * 0.85, "c": color_neg, "o": 0.62})
    else:
        charges = compute_gasteiger_charges(mol_3d)
        hex_map = [_charge_to_hex(q) for q in charges]

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"/>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body,#v3d{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:{bg};}}</style>
</head><body><div id="v3d"></div>
<script>
const molBlock = {json.dumps(mol_block)};
const spheres = {json.dumps(spheres)};
const hexMap = {json.dumps(hex_map)};
const mode = {json.dumps(mode)};
const viewer = $3Dmol.createViewer("v3d", {{backgroundColor: {json.dumps(bg)}}});
viewer.addModel(molBlock, "mol");
viewer.setStyle({{}}, {{stick: {{radius: 0.14}}, sphere: {{scale: 0.22}}}});
if (mode === "HOMO" || mode === "LUMO") {{
  spheres.forEach(s => {{
    viewer.addSphere({{center: {{x: s.x, y: s.y, z: s.z}}, radius: s.r, color: s.c, opacity: s.o}});
  }});
}} else {{
  viewer.addSurface($3Dmol.SurfaceType.VDW, {{
    opacity: 0.78,
    colorfunc: (atom) => hexMap[atom.serial - 1] || "#94a3b8"
  }});
}}
viewer.zoomTo();
viewer.render();
</script></body></html>"""


def build_3dmol_overlay_html(mol_1, mol_2, dark_mode: bool = True) -> str:
    b1 = Chem.MolToMolBlock(mol_1)
    b2 = Chem.MolToMolBlock(mol_2)
    bg = "#0b1416" if dark_mode else "#f8fafc"
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"/>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>html,body,#v3d{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:{bg};}}</style>
</head><body><div id="v3d"></div>
<script>
const viewer = $3Dmol.createViewer("v3d", {{backgroundColor: {json.dumps(bg)}}});
viewer.addModel({json.dumps(b1)}, "mol");
viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: "greenCarbon", radius: 0.18}}}});
viewer.addModel({json.dumps(b2)}, "mol");
viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: "cyanCarbon", radius: 0.18}}}});
viewer.zoomTo();
viewer.render();
</script></body></html>"""


def build_jsme_editor_html(initial_smiles: str = "", dark_mode: bool = True) -> str:
    bar_bg = "#0b1416" if dark_mode else "#f1f5f9"
    bar_txt = "#e6edf3" if dark_mode else "#0f172a"
    bar_border = "rgba(148,163,184,0.28)" if dark_mode else "rgba(15,23,42,0.14)"
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"/>
<script src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
<style>
  html, body {{
    margin: 0; padding: 0; width: 100%; height: 100%;
    overflow: hidden; display: flex; flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #ffffff;
  }}
  #jsme_container {{ flex: 1; width: 100%; min-height: 0; }}
  .jsme-bar {{
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
    padding: 8px 12px; background: {bar_bg}; color: {bar_txt};
    border-top: 1px solid {bar_border}; font-size: 12px;
  }}
  .smi-preview {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    opacity: 0.85; flex: 1;
  }}
  .apply-btn {{
    padding: 7px 14px; border-radius: 9px; border: 1px solid rgba(16,185,129,0.65);
    background: linear-gradient(135deg, #10b981, #0284c7); color: #ffffff;
    font-weight: 700; font-size: 12px; cursor: pointer; white-space: nowrap;
  }}
  .apply-btn:hover {{ filter: brightness(1.08); }}
</style>
</head><body>
  <div id="jsme_container"></div>
  <div class="jsme-bar">
    <span id="smi-preview" class="smi-preview">{html.escape(initial_smiles)}</span>
    <button class="apply-btn" onclick="applyToStudio()">Appliquer au Studio ➔</button>
  </div>
<script>
  let jsmeApplet = null;
  const initSmi = {json.dumps(initial_smiles)};

  function jsmeOnLoad() {{
    jsmeApplet = new JSApplet.JSME("jsme_container", "100%", "100%", {{
      options: "newlook,star,hydrogens,query"
    }});
    if (initSmi) {{
      try {{ jsmeApplet.readGenericMolecularInput(initSmi); }} catch (e) {{}}
    }}
    jsmeApplet.setCallBack("AfterStructureModified", function() {{
      const s = jsmeApplet.smiles() || "";
      document.getElementById("smi-preview").textContent = s || "(vide)";
    }});
  }}

  function applyToStudio() {{
    if (!jsmeApplet) return;
    const smi = (jsmeApplet.smiles() || "").trim();
    if (!smi) return;
    window.parent.postMessage({{ type: "jsme_smiles", smiles: smi }}, "*");
  }}
</script>
</body></html>"""


# ==================================================================
# 5. PROPRIÉTÉS CHEMPROP, RENDUS 2D SVG & FRAGMENTATION
# ==================================================================
def estimate_chemprop(mol) -> dict[str, Any]:
    mw = float(Descriptors.MolWt(mol))
    exact_mw = float(Descriptors.ExactMolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    mr = float(Crippen.MolMR(mol))
    rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    heavy = mol.GetNumHeavyAtoms()
    ap = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy if heavy > 0 else 0.0
    logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * ap
    sol_mg_ml = (10 ** logs) * mw
    tb_k = 198.2 + 2.04 * mw + 1.15 * tpsa + 18.0 * hbd - 4.5 * rot
    tm_k = 115.0 + 1.15 * mw + 0.85 * tpsa + 22.0 * hbd + 14.0 * rings - 5.5 * rot

    mol_h = Chem.AddHs(mol)
    cnt = Counter(a.GetSymbol() for a in mol_h.GetAtoms())
    dbe = cnt["C"] + 1 - (cnt["H"] + cnt["F"] + cnt["Cl"] + cnt["Br"] + cnt["I"] - cnt["N"]) / 2.0

    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = "N/A"

    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(mw, 2),
        "exact_mw": round(exact_mw, 5),
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "inchikey": inchikey,
        "mr": round(mr, 2),
        "logp": round(logp, 2),
        "logs": round(logs, 2),
        "sol_mg_ml": round(sol_mg_ml, 3),
        "tb_c": round(tb_k - 273.15, 1),
        "tm_c": round(tm_k - 273.15, 1),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rot": rot,
        "rings": rings,
        "dbe": round(dbe, 1),
    }


def render_2d_svg(
    mol,
    width: int = 440,
    height: int = 310,
    atom_indices: bool = False,
    bond_indices: bool = False,
    stereo: bool = True,
    style_name: str = "default",
) -> str:
    if mol is None:
        return ""
    mol_copy = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol_copy)
    if stereo:
        try:
            Chem.AssignStereochemistry(mol_copy, cleanIt=True, force=True)
        except Exception:
            pass
    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = bool(stereo)
    opts.addAtomIndices = bool(atom_indices)
    opts.addBondIndices = bool(bond_indices)

    if style_name == "ACS Document 1996 (Noir & Blanc)":
        try:
            Draw.SetACS1996Mode(opts, Draw.MeanBondLength(mol_copy))
        except Exception:
            opts.bondLineWidth = 1.6
        opts.useBWAtomPalette()
    elif style_name == "RSC Standard (Couleurs sobres)":
        opts.bondLineWidth = 2.0
        opts.multipleBondOffset = 0.15
    elif style_name == "Wiley Clean (Trait épais N&B)":
        opts.bondLineWidth = 2.6
        opts.useBWAtomPalette()
    elif style_name == "Mode Sombre (Présentation)":
        opts.setBackgroundColour((0.07, 0.10, 0.12, 1.0))
        opts.setSymbolColour((0.95, 0.96, 0.98, 1.0))
        opts.bondLineWidth = 2.2
    else:
        opts.bondLineWidth = 2.0

    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_reaction_svg(rxn_smiles: str, width: int = 680, height: int = 210) -> str:
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        if not rxn:
            return ""
        drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
        drawer.drawOptions().bondLineWidth = 2.0
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def enumerate_tautomers_svgs(mol, max_count: int = 6) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetMaxTautomers(int(max_count))
        tauts = list(enumerator.Enumerate(mol))[:max_count]
        for t_mol in tauts:
            out.append({
                "smiles": Chem.MolToSmiles(t_mol, canonical=True),
                "svg": render_2d_svg(t_mol, width=220, height=160, stereo=True),
            })
    except Exception:
        pass
    return out


def get_acyclic_bonds(mol) -> list[dict[str, Any]]:
    bonds: list[dict[str, Any]] = []
    if not mol:
        return bonds
    for b in mol.GetBonds():
        if not b.IsInRing():
            idx = b.GetIdx()
            a1 = f"{b.GetBeginAtom().GetSymbol()}{b.GetBeginAtomIdx()}"
            a2 = f"{b.GetEndAtom().GetSymbol()}{b.GetEndAtomIdx()}"
            bonds.append({"idx": idx, "label": f"Liaison #{idx} : {a1} — {a2}"})
    return bonds


def fragment_on_bond(mol, bond_idx: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        frag_mol = Chem.FragmentOnBonds(mol, [int(bond_idx)], addDummies=False)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
        for idx, fm in enumerate(frags):
            fm.UpdatePropertyCache(strict=False)
            out.append({
                "index": idx + 1,
                "formula": rdMolDescriptors.CalcMolFormula(fm),
                "exact_mass": round(float(Descriptors.ExactMolWt(fm)), 4),
                "svg": render_2d_svg(fm, width=250, height=185, stereo=False),
            })
    except Exception:
        pass
    return out


# ==================================================================
# 6. RÉACTIONS, STŒCHIOMÉTRIE & RÉTROSENTHÈSE IA
# ==================================================================
def predict_reaction_product(reactant_smiles_list: list[str], template_smarts: str) -> tuple[str | None, str]:
    r_mols = [m for s in reactant_smiles_list if (m := parse_smiles(s)) is not None]
    if len(r_mols) < 1:
        return None, "Ajoutez au moins un réactif dans le panier."
    try:
        rxn_eng = rdChemReactions.ReactionFromSmarts(template_smarts)
        needed = rxn_eng.GetNumReactantTemplates()
        if len(r_mols) < needed:
            return None, f"Cette transformation nécessite {needed} réactif(s) dans le panier."
        prods = rxn_eng.RunReactants(tuple(r_mols[:needed]))
        if prods and len(prods[0]) > 0:
            prod_mol = prods[0][0]
            Chem.SanitizeMol(prod_mol)
            return Chem.MolToSmiles(prod_mol, canonical=True), "Produit prédit avec succès."
        return None, "Les réactifs du panier ne correspondent pas aux fonctions attendues par ce modèle."
    except Exception as ex:
        return None, f"Erreur de prédiction : {ex}"


def predict_reaction_candidates(
    reactant_smiles_list: list[str],
    template_name: str = "__auto__",
) -> list[dict[str, Any]]:
    """Teste le catalogue de réactions et renvoie les produits candidats dédupliqués."""
    r_mols = [m for s in reactant_smiles_list if (m := parse_smiles(s)) is not None]
    if not r_mols:
        return []

    input_cans = {Chem.MolToSmiles(m, canonical=True) for m in r_mols}

    def _run_one_item(item: dict[str, Any]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        seen_local: set[str] = set()
        try:
            rxn_eng = rdChemReactions.ReactionFromSmarts(item["smarts"])
            needed = rxn_eng.GetNumReactantTemplates()
        except Exception:
            return found

        if len(r_mols) < needed:
            return found

        if needed == 1:
            tuples_to_try = [(m,) for m in r_mols]
        elif needed == 2:
            tuples_to_try = list(permutations(r_mols[:5], 2))
        else:
            tuples_to_try = [tuple(r_mols[:needed])]

        for r_tup in tuples_to_try:
            try:
                prod_sets = rxn_eng.RunReactants(r_tup)
            except Exception:
                continue
            for p_set in prod_sets:
                if not p_set:
                    continue
                pm = Chem.Mol(p_set[0])
                try:
                    Chem.SanitizeMol(pm)
                    can = Chem.MolToSmiles(pm, canonical=True)
                except Exception:
                    continue
                if not can or can in seen_local or can in input_cans:
                    continue
                seen_local.add(can)
                found.append({
                    "smiles": can,
                    "name": item["name"],
                    "category": item["category"],
                    "reagents": item.get("reagents", ""),
                    "n_reactants": needed,
                })
                # Limite à 2 régioisomères distincts max par modèle (évite de noyer l'écran)
                if len(found) >= 2:
                    return found
        return found

    # Mode modèle précis choisi dans le catalogue
    if template_name != "__auto__":
        target = next((it for it in REACTION_CATALOG if it["name"] == template_name), None)
        return _run_one_item(target) if target else []

    # Mode automatique : priorité aux réactions utilisant 2 réactifs s'il y en a >= 2
    candidates: list[dict[str, Any]] = []
    if len(r_mols) >= 2:
        for item in REACTION_CATALOG:
            if int(item.get("n_reactants", 1)) == 2:
                candidates.extend(_run_one_item(item))

    # Si 1 seul réactif (ou aucune réaction à 2 réactifs compatible), teste les réactions à 1 réactif
    if not candidates:
        for item in REACTION_CATALOG:
            if int(item.get("n_reactants", 1)) == 1:
                candidates.extend(_run_one_item(item))

    # Déduplication globale sur (smiles, nom)
    unique: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for c in candidates:
        key = (c["smiles"], c["name"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique.append(c)
    return unique


def run_retrosynthesis_engine(mol) -> list[dict[str, Any]]:
    if mol is None:
        return []
    routes: list[dict[str, Any]] = []
    seen: set[str] = set()
    target_can = Chem.MolToSmiles(mol, canonical=True)

    for name, smarts in RETRO_RULES:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if not rxn:
                continue
            prod_sets = rxn.RunReactants((mol,))
            for p_tuple in prod_sets:
                precursors: list[str] = []
                valid = True
                for pm in p_tuple:
                    try:
                        pm_copy = Chem.Mol(pm)
                        Chem.SanitizeMol(pm_copy)
                        p_can = Chem.MolToSmiles(pm_copy, canonical=True)
                        if not p_can or p_can == target_can:
                            valid = False
                        else:
                            precursors.append(p_can)
                    except Exception:
                        valid = False
                if valid and len(precursors) >= 2:
                    sig = ".".join(sorted(precursors))
                    if sig not in seen:
                        seen.add(sig)
                        routes.append({
                            "strategy": name,
                            "precursors": precursors,
                            "rxn_smiles": f"{'.'.join(precursors)}>>{target_can}",
                        })
        except Exception:
            continue
    return routes


# ==================================================================
# 7. MOTEUR RMN 1D (PASCAL) & RMN 2D (COSY, HSQC, HMBC)
# ==================================================================
def simulate_advanced_nmr_1d_2d(mol) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mol_h = Chem.AddHs(mol)
    dist_mat = Chem.GetDistanceMatrix(mol_h)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))

    alpha_c = {"O": 48.0, "N": 29.0, "F": 68.0, "Cl": 31.0, "Br": 20.0, "S": 18.0, "C": 9.1}
    beta_c = {"O": 10.0, "N": 11.0, "F": 7.5, "Cl": 10.5, "Br": 10.0, "C": 9.4}
    alpha_h = {"O": 2.35, "N": 1.55, "F": 3.1, "Cl": 2.15, "Br": 1.95, "S": 1.35}

    c_shifts: dict[int, float] = {}
    c_rows: list[dict[str, Any]] = []
    c_groups: dict[int, list[Any]] = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "C":
            c_groups.setdefault(ranks[idx], []).append(atom)

    for atoms in c_groups.values():
        env = "Aliphatique"
        for a in atoms:
            idx = a.GetIdx()
            hyb = str(a.GetHybridization())
            arom = a.GetIsAromatic()
            nbs_1 = [mol_h.GetAtomWithIdx(j) for j in range(mol_h.GetNumAtoms()) if dist_mat[idx, j] == 1 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"]
            nbs_2 = [mol_h.GetAtomWithIdx(j) for j in range(mol_h.GetNumAtoms()) if dist_mat[idx, j] == 2 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"]
            nbs_3 = [mol_h.GetAtomWithIdx(j) for j in range(mol_h.GetNumAtoms()) if dist_mat[idx, j] == 3 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"]
            has_c_o = any(
                nb.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(idx, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                for nb in nbs_1
            )

            if has_c_o:
                het_1 = [nb.GetSymbol() for nb in nbs_1 if nb.GetSymbol() in ("O", "N", "Cl")]
                shift, env = (171.5 if len(het_1) >= 2 else 201.5), "Carbonyle (C=O)"
            elif arom:
                shift = 128.5
                for nb in nbs_1:
                    if nb.GetSymbol() == "O":
                        shift += 27.5
                    elif nb.GetSymbol() == "N":
                        shift += 18.0
                    elif nb.GetSymbol() in ("Cl", "Br", "F"):
                        shift += 12.0
                    elif not nb.GetIsAromatic():
                        shift += 9.5
                for nb in nbs_2:
                    if nb.GetSymbol() in ("O", "N"):
                        shift -= 11.5
                env = "Aromatique (Ar-C)"
            elif hyb == "SP2":
                shift = 123.0 + sum(alpha_c.get(nb.GetSymbol(), 5.0) * 0.45 for nb in nbs_1)
                env = "Alcène (C=C)"
            elif hyb == "SP":
                shift = 78.0 + sum(alpha_c.get(nb.GetSymbol(), 4.0) * 0.4 for nb in nbs_1)
                env = "Alcyne / Nitrile"
            else:
                shift = (
                    -2.3
                    + sum(alpha_c.get(nb.GetSymbol(), 5.0) for nb in nbs_1)
                    + sum(beta_c.get(nb.GetSymbol(), 3.0) * 0.65 for nb in nbs_2)
                    - 2.5 * len(nbs_3)
                )
                shift = max(8.0, min(95.0, shift))
                h_cnt = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
                env = f"Aliphatique (CH{h_cnt})"
            c_shifts[idx] = round(shift, 1)

        a0 = atoms[0]
        c_rows.append({
            "carbones": ", ".join(str(x.GetIdx()) for x in atoms),
            "environnement": env,
            "nb_c": len(atoms),
            "shift": c_shifts[a0.GetIdx()],
        })

    h_shifts: dict[int, float] = {}
    h_rows: list[dict[str, Any]] = []
    h_groups: dict[int, list[Any]] = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "H":
            h_groups.setdefault(ranks[idx], []).append(atom)

    mult_names = {0: "s (singulet)", 1: "d (doublet)", 2: "t (triplet)", 3: "q (quadruplet)"}
    for h_atoms in h_groups.values():
        h0 = h_atoms[0]
        parent = h0.GetNeighbors()[0]
        p_idx, p_sym = parent.GetIdx(), parent.GetSymbol()

        if p_sym == "O":
            is_acid = any(
                nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" and n2.GetIdx() != p_idx for n2 in nb.GetNeighbors())
                for nb in parent.GetNeighbors()
            )
            shift, mult, env = (11.65, "s large", "Acide (-COOH)") if is_acid else (4.85, "s large", "Alcool / Phénol (-OH)")
        elif p_sym == "N":
            has_co = any(
                nb.GetSymbol() == "C" and any(n2.GetSymbol() == "O" for n2 in nb.GetNeighbors())
                for nb in parent.GetNeighbors()
            )
            shift, mult, env = (7.85, "s large", "Amide (-CONH)") if has_co else (3.65, "s large", "Amine (-NH)")
        else:
            arom = parent.GetIsAromatic()
            hyb = str(parent.GetHybridization())
            vicinal_h = sum(
                sum(1 for n2 in nb.GetNeighbors() if n2.GetSymbol() == "H")
                for nb in parent.GetNeighbors()
                if nb.GetSymbol() == "C"
            )
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
                        elif any(
                            n2.GetSymbol() == "O" and mol_h.GetBondBetweenAtoms(nb.GetIdx(), n2.GetIdx()).GetBondTypeAsDouble() == 2.0
                            for n2 in nb.GetNeighbors()
                        ):
                            shift += 1.15
                shift = min(6.2, shift)
                env = "Aliphatique (C-H)"

        val_h = round(shift, 2)
        for ha in h_atoms:
            h_shifts[ha.GetIdx()] = val_h

        h_rows.append({
            "porte_par": f"{p_sym}{p_idx}",
            "integration": f"{len(h_atoms)}H",
            "n_h": len(h_atoms),
            "multiplicite": mult,
            "environnement": env,
            "shift": val_h,
        })

    h_rows.sort(key=lambda x: -float(x["shift"]))
    c_rows.sort(key=lambda x: -float(x["shift"]))

    cosy_pts: list[dict[str, Any]] = []
    hsqc_pts: list[dict[str, Any]] = []
    hmbc_pts: list[dict[str, Any]] = []
    seen_cosy: set[tuple[float, float]] = set()
    seen_hsqc: set[tuple[float, float]] = set()
    seen_hmbc: set[tuple[float, float]] = set()

    for h1, s1 in h_shifts.items():
        if (s1, s1) not in seen_cosy:
            seen_cosy.add((s1, s1))
            cosy_pts.append({"f2": s1, "f1": s1, "type": "Diagonale"})
        for h2, s2 in h_shifts.items():
            if h1 != h2 and dist_mat[h1, h2] == 3 and abs(s1 - s2) > 0.03:
                pair = (round(s1, 2), round(s2, 2))
                if pair not in seen_cosy:
                    seen_cosy.add(pair)
                    cosy_pts.append({"f2": s1, "f1": s2, "type": "Corrélation ³J_HH (COSY)"})

    for h_idx, sh in h_shifts.items():
        for c_idx, sc in c_shifts.items():
            d = dist_mat[h_idx, c_idx]
            if d == 1:
                key = (sh, sc)
                if key not in seen_hsqc:
                    seen_hsqc.add(key)
                    hsqc_pts.append({"carbone": f"C{c_idx}", "h": sh, "c": sc, "type": "¹J_CH (HSQC)"})
            elif d in (2, 3):
                key = (sh, sc)
                if key not in seen_hmbc:
                    seen_hmbc.add(key)
                    hmbc_pts.append({"carbone": f"C{c_idx}", "h": sh, "c": sc, "type": f"^{int(d)}J_CH (HMBC)"})

    return h_rows, c_rows, cosy_pts, hsqc_pts, hmbc_pts


def build_lorentzian_points(
    rows: list[dict[str, Any]],
    x_min: float,
    x_max: float,
    width: float = 0.003,
    freq_mhz: float = 400.0,
    num_points: int = 1600,
) -> list[list[float]]:
    x = np.linspace(float(x_min), float(x_max), int(num_points))
    y = np.zeros_like(x)
    pascal_map = {
        "s": [1.0],
        "s large": [1.0],
        "d": [1.0, 1.0],
        "t": [1.0, 2.0, 1.0],
        "q": [1.0, 3.0, 3.0, 1.0],
        "m": [1.0, 4.0, 6.0, 4.0, 1.0],
    }
    delta_j_ppm = 7.5 / float(freq_mhz)

    for row in rows:
        x0 = float(row["shift"])
        n_nuclei = float(row.get("n_h", row.get("nb_c", 1.0)))
        mult_key = "s"
        w_peak = float(width)
        m_str = clean_val(row.get("multiplicite"))
        if m_str:
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

    return [[round(float(xv), 4), round(float(yv), 4)] for xv, yv in zip(x, y)]


# ==================================================================
# 8. SPECTRE ISOTOPIQUE HRMS
# ==================================================================
def calc_isotopic_spectrum(mol) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_m = float(Descriptors.ExactMolWt(mol))
    adducts = [
        {"ion": "[M + H]⁺", "mz": round(exact_m + 1.00728, 5)},
        {"ion": "[M + Na]⁺", "mz": round(exact_m + 22.98922, 5)},
        {"ion": "[M + K]⁺", "mz": round(exact_m + 38.96316, 5)},
        {"ion": "[M + NH₄]⁺", "mz": round(exact_m + 18.03382, 5)},
        {"ion": "[M - H]⁻", "mz": round(exact_m - 1.00728, 5)},
    ]

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
        "I": [(126.904473, 1.0)],
    }
    mol_h = Chem.AddHs(mol)
    pt = Chem.GetPeriodicTable()
    dist: dict[int, tuple[float, float]] = {0: (0.0, 1.0)}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        isotopes = iso_table.get(sym, [(pt.GetMostCommonIsotopeMass(sym), 1.0)])
        base_m = isotopes[0][0]
        new_dist: dict[int, tuple[float, float]] = {}
        for shift_old, (m_old, p_old) in dist.items():
            for m_iso, p_iso in isotopes:
                shift_add = round(m_iso - base_m)
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

    max_p = max((p for _, p in dist.values()), default=1.0)
    iso_rows: list[dict[str, Any]] = []
    for s in sorted(dist.keys()):
        m_val, p_val = dist[s]
        rel_int = (p_val / max_p) * 100.0
        if rel_int >= 0.1:
            iso_rows.append({
                "pic": f"M+{s}" if s > 0 else "M",
                "mz": round(m_val, 5),
                "rel": round(rel_int, 2),
            })
    return adducts, iso_rows