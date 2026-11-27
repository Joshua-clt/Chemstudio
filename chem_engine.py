import html
from collections import Counter
from functools import lru_cache
from itertools import permutations
import ipaddress
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
try:
    rdDepictor.SetPreferCoordGen(True)
except Exception:
    pass


def compute_clean_2d_coords(mol) -> None:
    """Calcule des coordonnées 2D propres, horizontales et alignées sur des angles de 30°."""
    if mol is None:
        return
    rdDepictor.Compute2DCoords(mol)
    try:
        rdDepictor.NormalizeDepiction(mol)
    except Exception:
        pass
    try:
        rdDepictor.StraightenDepiction(mol, minimizeRotation=True)
    except Exception:
        pass

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
    if not s or len(s) > 1500:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def is_mol_too_large(mol, max_heavy_atoms: int = 120) -> bool:
    return mol is None or int(mol.GetNumHeavyAtoms()) > int(max_heavy_atoms)


def is_safe_public_url(url: str) -> bool:
    raw = clean_val(url)
    if not raw:
        return False
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return False
    host = (parsed.hostname or "").lower().strip()
    if not host:
        return False
    if os.environ.get("RENDER") is None and host in ("localhost", "127.0.0.1") and parsed.scheme in ("http", "https"):
        return True
    if parsed.scheme != "https":
        return False
    if host in ("localhost", "0.0.0.0") or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    except ValueError:
        pass
    return True


# ==================================================================
# 1. CLIENT API CHEMBASE v2.0 (X-API-Key & PRÉ-REMPLISSAGE)
# ==================================================================
def get_chembase_config(override_url: str = "", override_key: str = "") -> tuple[str, str]:
    url = clean_val(override_url or os.environ.get("CHEMBASE_URL", "")).rstrip("/")
    if url and not is_safe_public_url(url):
        url = ""
    env_key = "" if os.environ.get("RENDER") is not None else os.environ.get("CHEMBASE_API_KEY", "")
    key = clean_val(override_key or env_key)
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
_IDENT_CACHE: dict[str, dict[str, Any]] = {}
_RESOLVE_CACHE: dict[str, str] = {}


def get_chemical_identifiers(smiles: str) -> dict[str, Any]:
    key = clean_val(smiles)
    if key in _IDENT_CACHE:
        return dict(_IDENT_CACHE[key])
    res: dict[str, Any] = {
        "iupac": "Non répertorié",
        "cas": "Non trouvé",
        "trivial": "Composé personnalisé",
        "cid": None,
    }
    try:
        mol = parse_smiles(key)
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
            if len(_IDENT_CACHE) >= 256:
                _IDENT_CACHE.pop(next(iter(_IDENT_CACHE)))
            _IDENT_CACHE[key] = dict(res)
    except Exception:
        pass
    return res


def resolve_query_to_smiles(query: str) -> str | None:
    q = clean_val(query)
    if not q:
        return None
    if q in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[q]
    mol_t = parse_smiles(q)
    if mol_t:
        return Chem.MolToSmiles(mol_t, canonical=True)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(q)}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            found = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
            if found:
                if len(_RESOLVE_CACHE) >= 256:
                    _RESOLVE_CACHE.pop(next(iter(_RESOLVE_CACHE)))
                _RESOLVE_CACHE[q] = str(found)
                return str(found)
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
    if is_mol_too_large(mol, 100):
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
_MOL3D_CACHE: dict[tuple[str, str, int], Any] = {}
_OVERLAY_CACHE: dict[tuple[str, str], tuple[Any, Any, int, float | None]] = {}


def generate_3d_mol(smiles: str, force_field: str = "MMFF94", seed: int = 42):
    mol = parse_smiles(smiles)
    if is_mol_too_large(mol, 120):
        return None
    can_smi = Chem.MolToSmiles(mol, canonical=True)
    cache_key = (can_smi, str(force_field), int(seed))
    cached = _MOL3D_CACHE.get(cache_key)
    if cached is not None:
        return Chem.Mol(cached)

    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    res = AllChem.EmbedMolecule(mol_3d, params)
    if res != 0:
        res = AllChem.EmbedMolecule(mol_3d, randomSeed=int(seed))
    if res != 0 or mol_3d.GetNumConformers() == 0:
        return None

    if force_field == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mol_3d):
        AllChem.MMFFOptimizeMolecule(mol_3d)
    else:
        AllChem.UFFOptimizeMolecule(mol_3d)

    if len(_MOL3D_CACHE) >= 64:
        _MOL3D_CACHE.pop(next(iter(_MOL3D_CACHE)))
    _MOL3D_CACHE[cache_key] = Chem.Mol(mol_3d)
    return mol_3d


def get_bond_length_3d(mol_3d, atom_i: int, atom_j: int) -> float:
    try:
        conf = mol_3d.GetConformer()
        return round(float(rdMolTransforms.GetBondLength(conf, int(atom_i), int(atom_j))), 2)
    except Exception:
        return 1.50


def get_adjustable_bonds_3d(mol_3d) -> list[dict[str, Any]]:
    """Liste les liaisons acycliques entre atomes lourds dont la longueur 3D peut être ajustée."""
    bonds: list[dict[str, Any]] = []
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return bonds
    for b in mol_3d.GetBonds():
        if b.IsInRing():
            continue
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        if a1.GetSymbol() == "H" or a2.GetSymbol() == "H":
            continue
        i, j = int(a1.GetIdx()), int(a2.GetIdx())
        length = get_bond_length_3d(mol_3d, i, j)
        bonds.append({
            "key": f"{i}_{j}",
            "ai": i,
            "aj": j,
            "length": length,
            "label": f"{a1.GetSymbol()}{i} — {a2.GetSymbol()}{j} ({length:.2f} Å)",
        })
    return bonds


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
    mol1 = parse_smiles(smiles1)
    mol2 = parse_smiles(smiles2)
    if is_mol_too_large(mol1, 120) or is_mol_too_large(mol2, 120):
        return None, None, 0, None

    c1 = Chem.MolToSmiles(mol1, canonical=True)
    c2 = Chem.MolToSmiles(mol2, canonical=True)
    cache_key = (c1, c2)
    cached = _OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        m1_c, m2_c, mcs_c, rmsd_c = cached
        return Chem.Mol(m1_c), Chem.Mol(m2_c), mcs_c, rmsd_c

    m1 = generate_3d_mol(c1)
    m2 = generate_3d_mol(c2)
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

    if len(_OVERLAY_CACHE) >= 32:
        _OVERLAY_CACHE.pop(next(iter(_OVERLAY_CACHE)))
    _OVERLAY_CACHE[cache_key] = (Chem.Mol(m1), Chem.Mol(m2), mcs_atoms, rmsd)
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


_THREED_THEME_SYNC_JS = """
const _apply3dBg = (isDark) => {
  const nb = isDark ? "#0b1416" : "#f8fafc";
  document.documentElement.style.background = nb;
  document.body.style.background = nb;
  const el = document.getElementById("v3d");
  if (el) el.style.background = nb;
  viewer.setBackgroundColor(nb);
  viewer.render();
};
window.addEventListener("message", (ev) => {
  if (ev.data && ev.data.type === "cs_set_theme") _apply3dBg(!!ev.data.dark);
});
"""


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
let initBg = {json.dumps(bg)};
try {{
  const pt = window.parent && window.parent.document.documentElement.getAttribute("data-theme");
  if (pt === "light") initBg = "#f8fafc";
  else if (pt === "dark") initBg = "#0b1416";
}} catch (_e) {{}}
document.documentElement.style.background = initBg;
document.body.style.background = initBg;
const molBlock = {json.dumps(mol_block)};
const hexMap = {json.dumps(hex_map)};
const styleName = {json.dumps(style)};
const showH = {json.dumps(bool(show_h))};
const colorCharges = {json.dumps(bool(color_charges))};
const doSpin = {json.dumps(bool(spin))};
const viewer = $3Dmol.createViewer("v3d", {{backgroundColor: initBg}});
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
{_THREED_THEME_SYNC_JS}
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
let initBg = {json.dumps(bg)};
try {{
  const pt = window.parent && window.parent.document.documentElement.getAttribute("data-theme");
  if (pt === "light") initBg = "#f8fafc";
  else if (pt === "dark") initBg = "#0b1416";
}} catch (_e) {{}}
document.documentElement.style.background = initBg;
document.body.style.background = initBg;
const molBlock = {json.dumps(mol_block)};
const spheres = {json.dumps(spheres)};
const hexMap = {json.dumps(hex_map)};
const mode = {json.dumps(mode)};
const viewer = $3Dmol.createViewer("v3d", {{backgroundColor: initBg}});
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
{_THREED_THEME_SYNC_JS}
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
let initBg = {json.dumps(bg)};
try {{
  const pt = window.parent && window.parent.document.documentElement.getAttribute("data-theme");
  if (pt === "light") initBg = "#f8fafc";
  else if (pt === "dark") initBg = "#0b1416";
}} catch (_e) {{}}
document.documentElement.style.background = initBg;
document.body.style.background = initBg;
const viewer = $3Dmol.createViewer("v3d", {{backgroundColor: initBg}});
viewer.addModel({json.dumps(b1)}, "mol");
viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: "greenCarbon", radius: 0.18}}}});
viewer.addModel({json.dumps(b2)}, "mol");
viewer.setStyle({{model: 1}}, {{stick: {{colorscheme: "cyanCarbon", radius: 0.18}}}});
viewer.zoomTo();
viewer.render();
{_THREED_THEME_SYNC_JS}
</script></body></html>"""

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
    transparent: bool = False,
) -> str:
    if mol is None:
        return ""
    mol_copy = Chem.Mol(mol)
    compute_clean_2d_coords(mol_copy)
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

    if transparent:
        try:
            opts.clearBackground = False
        except Exception:
            pass

    drawer.DrawMolecule(mol_copy)
    drawer.FinishDrawing()
    svg_txt = drawer.GetDrawingText()
    if transparent:
        svg_txt = re.sub(r"<rect\b[^>]*x='0\.0'\s+y='0\.0'[^>]*>\s*</rect>", "", svg_txt, count=1)
    return svg_txt


def render_reaction_svg(rxn_smiles: str, width: int = 680, height: int = 210) -> str:
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        if not rxn:
            return ""
        for i in range(rxn.GetNumReactantTemplates()):
            compute_clean_2d_coords(rxn.GetReactantTemplate(i))
        for i in range(rxn.GetNumProductTemplates()):
            compute_clean_2d_coords(rxn.GetProductTemplate(i))
        drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
        drawer.drawOptions().bondLineWidth = 2.0
        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def render_annotated_reaction_svg(
    reactant_smiles: list[str],
    product_smiles: list[str],
    solvent: str = "",
    temp: str = "",
    duration: str = "",
    style_name: str = "default",
    transparent: bool = True,
) -> str:
    """Compose un schéma de réaction vectoriel complet avec flèche annotée (solvant, T°, durée)."""
    r_mols = [m for s in reactant_smiles[:6] if (m := parse_smiles(s)) is not None]
    p_mols = [m for s in product_smiles[:6] if (m := parse_smiles(s)) is not None]
    if not r_mols and not p_mols:
        return ""

    mol_w, mol_h = 210, 155
    plus_w = 28
    pad_x, pad_y = 22, 24

    top_txt = clean_val(solvent)
    bot_parts = [p for p in (clean_val(temp), clean_val(duration)) if p]
    bot_txt = " · ".join(bot_parts)
    max_lbl_len = max(len(top_txt), len(bot_txt), 0)
    arrow_w = max(130, min(260, int(max_lbl_len * 7.5 + 42)))

    r_block_w = len(r_mols) * mol_w + max(0, len(r_mols) - 1) * plus_w if r_mols else 40
    p_block_w = len(p_mols) * mol_w + max(0, len(p_mols) - 1) * plus_w if p_mols else 40
    total_w = pad_x * 2 + r_block_w + arrow_w + p_block_w
    total_h = mol_h + pad_y * 2
    cy = total_h / 2.0

    is_dark_style = "Sombre" in str(style_name)
    stroke_col = "#f1f5f9" if is_dark_style else "#0f172a"
    cond_col = "#cbd5e1" if is_dark_style else "#334155"

    def _embed_mol_svg(mol_obj, x_pos: float, y_pos: float) -> str:
        raw = render_2d_svg(
            mol_obj,
            width=mol_w,
            height=mol_h,
            stereo=True,
            style_name=style_name,
            transparent=True,
        )
        cleaned = re.sub(r"<\?xml[^>]*\?>", "", raw).strip()
        return re.sub(r"<svg\b", f'<svg x="{x_pos:.1f}" y="{y_pos:.1f}"', cleaned, count=1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}">'
    ]
    if is_dark_style:
        parts.append(f'<rect width="{total_w}" height="{total_h}" rx="10" fill="#121a1e"/>')
    elif not transparent:
        parts.append(f'<rect width="{total_w}" height="{total_h}" fill="#ffffff"/>')

    cur_x = float(pad_x)
    # 1) Réactifs
    if r_mols:
        for idx, rm in enumerate(r_mols):
            parts.append(_embed_mol_svg(rm, cur_x, float(pad_y)))
            cur_x += mol_w
            if idx < len(r_mols) - 1:
                parts.append(
                    f'<text x="{cur_x + plus_w / 2:.1f}" y="{cy:.1f}" fill="{stroke_col}" '
                    f'font-family="Segoe UI, Arial, sans-serif" font-size="22" font-weight="600" '
                    f'text-anchor="middle" dominant-baseline="central">+</text>'
                )
                cur_x += plus_w
    else:
        cur_x += 40

    # 2) Flèche centrale + conditions
    ax1 = cur_x + 14
    ax2 = cur_x + arrow_w - 16
    amid = (ax1 + ax2) / 2.0
    parts.append(
        f'<line x1="{ax1:.1f}" y1="{cy:.1f}" x2="{ax2 - 6:.1f}" y2="{cy:.1f}" '
        f'stroke="{stroke_col}" stroke-width="2.1" stroke-linecap="round"/>'
    )
    parts.append(
        f'<polygon points="{ax2:.1f},{cy:.1f} {ax2 - 11:.1f},{cy - 5.5:.1f} {ax2 - 11:.1f},{cy + 5.5:.1f}" '
        f'fill="{stroke_col}"/>'
    )
    if top_txt:
        parts.append(
            f'<text x="{amid:.1f}" y="{cy - 12:.1f}" fill="{cond_col}" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12.5" font-weight="600" '
            f'text-anchor="middle">{html.escape(top_txt)}</text>'
        )
    if bot_txt:
        parts.append(
            f'<text x="{amid:.1f}" y="{cy + 21:.1f}" fill="{cond_col}" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="500" '
            f'text-anchor="middle">{html.escape(bot_txt)}</text>'
        )
    cur_x += arrow_w

    # 3) Produits
    for idx, pm in enumerate(p_mols):
        parts.append(_embed_mol_svg(pm, cur_x, float(pad_y)))
        cur_x += mol_w
        if idx < len(p_mols) - 1:
            parts.append(
                f'<text x="{cur_x + plus_w / 2:.1f}" y="{cy:.1f}" fill="{stroke_col}" '
                f'font-family="Segoe UI, Arial, sans-serif" font-size="22" font-weight="600" '
                f'text-anchor="middle" dominant-baseline="central">+</text>'
            )
            cur_x += plus_w

    parts.append("</svg>")
    return "".join(parts)


def enumerate_tautomers_svgs(mol, max_count: int = 6) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if is_mol_too_large(mol, 80):
        return out
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
# 7. MOTEUR RMN 1D (INCRÉMENTS EMPIRIQUES & PASCAL) & RMN 2D
# ==================================================================
NMR_SOLVENTS: dict[str, dict[str, Any]] = {
    "CDCl3": {
        "label": "CDCl₃ (Chloroforme-d)",
        "h_peaks": [{"ppm": 7.26, "label": "CHCl₃ (7.26)"}, {"ppm": 1.56, "label": "H₂O (1.56)"}],
        "c_peaks": [{"ppm": 77.16, "label": "CDCl₃ (77.16)"}],
    },
    "DMSO-d6": {
        "label": "DMSO-d₆",
        "h_peaks": [{"ppm": 2.50, "label": "DMSO (2.50)"}, {"ppm": 3.33, "label": "H₂O (3.33)"}],
        "c_peaks": [{"ppm": 39.52, "label": "DMSO-d₆ (39.52)"}],
    },
    "MeOD": {
        "label": "CD₃OD (Méthanol-d₄)",
        "h_peaks": [{"ppm": 3.31, "label": "CHD₂OD (3.31)"}, {"ppm": 4.87, "label": "HDO (4.87)"}],
        "c_peaks": [{"ppm": 49.00, "label": "CD₃OD (49.00)"}],
    },
    "Acetone-d6": {
        "label": "Acétone-d₆",
        "h_peaks": [{"ppm": 2.05, "label": "Acétone (2.05)"}, {"ppm": 2.84, "label": "H₂O (2.84)"}],
        "c_peaks": [{"ppm": 29.84, "label": "Acétone (29.84)"}, {"ppm": 206.26, "label": "Acétone C=O (206.3)"}],
    },
    "D2O": {
        "label": "D₂O (Eau lourde)",
        "h_peaks": [{"ppm": 4.79, "label": "HDO (4.79)"}],
        "c_peaks": [],
    },
    "none": {
        "label": "Aucun solvant (spectre seul)",
        "h_peaks": [],
        "c_peaks": [],
    },
}

# Tables d'incréments aromatiques (ipso, ortho, meta, para) — type Pretsch / Silverstein
_AROM_C_INC: dict[str, tuple[float, float, float, float]] = {
    "OH": (26.9, -12.7, 1.4, -7.3),
    "OR": (31.4, -14.4, 1.0, -7.7),
    "OCOR": (23.0, -6.4, 1.3, -2.3),
    "NR2": (18.0, -13.3, 0.9, -9.8),
    "NHCOR": (9.7, -8.1, 0.2, -4.4),
    "NO2": (20.0, -4.8, 0.9, 5.8),
    "COOH": (2.1, 1.5, 0.0, 5.1),
    "COOR": (2.0, 1.2, -0.1, 4.3),
    "COR": (9.1, 0.1, 0.0, 4.2),
    "CHO": (8.6, 1.3, 0.6, 5.5),
    "CONR2": (5.4, -0.6, 0.1, 3.0),
    "CN": (-15.4, 3.6, 0.6, 3.9),
    "F": (34.8, -12.9, 1.4, -4.5),
    "Cl": (6.2, 0.4, 1.3, -1.9),
    "Br": (-5.5, 3.4, 1.7, -1.6),
    "I": (-34.1, 8.7, 1.4, -1.4),
    "CF3": (2.6, -3.3, 0.3, 3.2),
    "S": (10.2, -1.5, 0.3, -3.2),
    "Ar": (13.1, -1.1, 0.4, -1.2),
    "Alkene": (9.5, -2.0, 0.2, -0.5),
    "Alkyne": (-6.1, 3.8, 0.4, -0.2),
    "Alkyl": (8.9, 0.7, -0.1, -2.9),
}

# Tables d'incréments aromatiques ¹H (ortho, meta, para) par rapport au benzène (7.26 ppm)
_AROM_H_INC: dict[str, tuple[float, float, float]] = {
    "OH": (-0.56, -0.12, -0.45),
    "OR": (-0.48, -0.09, -0.44),
    "OCOR": (-0.25, 0.03, -0.13),
    "NR2": (-0.75, -0.25, -0.65),
    "NHCOR": (0.12, -0.07, -0.28),
    "NO2": (0.95, 0.26, 0.38),
    "COOH": (0.85, 0.18, 0.27),
    "COOR": (0.71, 0.11, 0.21),
    "COR": (0.62, 0.14, 0.21),
    "CHO": (0.56, 0.22, 0.29),
    "CONR2": (0.61, 0.10, 0.17),
    "CN": (0.36, 0.18, 0.28),
    "F": (-0.29, -0.02, -0.23),
    "Cl": (0.03, -0.02, -0.09),
    "Br": (0.18, -0.08, -0.04),
    "I": (0.39, -0.21, 0.00),
    "CF3": (0.32, 0.14, 0.20),
    "S": (-0.08, -0.16, -0.22),
    "Ar": (0.37, 0.20, 0.10),
    "Alkene": (0.06, -0.03, -0.09),
    "Alkyne": (0.15, -0.02, -0.01),
    "Alkyl": (-0.14, -0.06, -0.17),
}


def _bond_order_nmr(mol_h, idx1: int, idx2: int) -> float:
    bd = mol_h.GetBondBetweenAtoms(int(idx1), int(idx2))
    return float(bd.GetBondTypeAsDouble()) if bd is not None else 1.0


def _has_carbonyl_o(mol_h, c_idx: int) -> bool:
    atom = mol_h.GetAtomWithIdx(int(c_idx))
    if atom.GetSymbol() != "C":
        return False
    return any(
        nb.GetSymbol() == "O" and _bond_order_nmr(mol_h, c_idx, nb.GetIdx()) == 2.0
        for nb in atom.GetNeighbors()
    )


def _classify_substituent_nmr(mol_h, parent_idx: int, sub_atom) -> str:
    """Identifie la classe chimique d'un substituant lié à un atome donné."""
    sym = sub_atom.GetSymbol()
    s_idx = int(sub_atom.GetIdx())
    nbs = list(sub_atom.GetNeighbors())
    other_heavy = [nb for nb in nbs if int(nb.GetIdx()) != int(parent_idx) and nb.GetSymbol() != "H"]
    h_cnt = sum(1 for nb in nbs if nb.GetSymbol() == "H")

    if sym == "O":
        if h_cnt >= 1:
            return "OH"
        if any(_has_carbonyl_o(mol_h, int(nb.GetIdx())) for nb in other_heavy):
            return "OCOR"
        return "OR"

    if sym == "N":
        o_nbs = [nb for nb in other_heavy if nb.GetSymbol() == "O"]
        if len(o_nbs) >= 2:
            return "NO2"
        if any(_has_carbonyl_o(mol_h, int(nb.GetIdx())) for nb in other_heavy):
            return "NHCOR"
        return "NR2"

    if sym == "C":
        if any(nb.GetSymbol() == "N" and _bond_order_nmr(mol_h, s_idx, nb.GetIdx()) == 3.0 for nb in other_heavy):
            return "CN"
        if _has_carbonyl_o(mol_h, s_idx):
            if h_cnt >= 1:
                return "CHO"
            for nb in other_heavy:
                if nb.GetSymbol() == "O" and _bond_order_nmr(mol_h, s_idx, nb.GetIdx()) == 1.0:
                    has_oh = any(n2.GetSymbol() == "H" for n2 in nb.GetNeighbors())
                    return "COOH" if has_oh else "COOR"
            if any(nb.GetSymbol() == "N" for nb in other_heavy):
                return "CONR2"
            return "COR"
        if sub_atom.GetIsAromatic():
            return "Ar"
        hyb = str(sub_atom.GetHybridization())
        if hyb == "SP2":
            return "Alkene"
        if hyb == "SP":
            return "Alkyne"
        if sum(1 for nb in other_heavy if nb.GetSymbol() == "F") >= 3:
            return "CF3"
        return "Alkyl"

    if sym in ("F", "Cl", "Br", "I", "S"):
        return sym
    return "Alkyl"


def simulate_advanced_nmr_1d_2d(
    mol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if is_mol_too_large(mol, 120):
        return [], [], [], [], []

    mol_h = Chem.AddHs(mol)
    dist_mat = Chem.GetDistanceMatrix(mol_h)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    atom_rings = list(mol_h.GetRingInfo().AtomRings())

    arom_rings_6 = [
        list(r) for r in atom_rings
        if len(r) == 6 and all(mol_h.GetAtomWithIdx(int(i)).GetIsAromatic() for i in r)
    ]
    arom_rings_5 = [
        list(r) for r in atom_rings
        if len(r) == 5 and all(mol_h.GetAtomWithIdx(int(i)).GetIsAromatic() for i in r)
    ]

    # Incréments aliphatiques ¹³C (Grant-Paul étendu)
    alpha_c_cls = {
        "Alkyl": 9.1, "Ar": 23.0, "Alkene": 20.5, "Alkyne": 12.0,
        "OH": 48.5, "OR": 54.0, "OCOR": 52.0, "NR2": 29.0, "NHCOR": 31.0,
        "NO2": 62.0, "COOH": 21.5, "COOR": 22.0, "COR": 28.0, "CHO": 30.0,
        "CONR2": 23.0, "CN": 14.5, "F": 68.0, "Cl": 31.2, "Br": 20.0,
        "I": -6.0, "S": 16.0, "CF3": 20.0,
    }
    beta_c_sym = {"C": 9.4, "O": 10.1, "N": 11.3, "F": 7.8, "Cl": 10.5, "Br": 10.6, "S": 6.5}

    # Incréments aliphatiques ¹H (Shoolery étendu)
    alpha_h_cls = {
        "Alkyl": 0.0, "Ar": 1.32, "Alkene": 0.85, "Alkyne": 0.90,
        "OH": 2.40, "OR": 2.38, "OCOR": 2.90, "NR2": 1.55, "NHCOR": 2.08,
        "NO2": 3.40, "COOH": 1.18, "COOR": 1.20, "COR": 1.30, "CHO": 1.32,
        "CONR2": 1.15, "CN": 1.42, "F": 3.50, "Cl": 2.53, "Br": 2.33,
        "I": 2.15, "S": 1.45, "CF3": 1.15,
    }

    # --------------------------------------------------------------
    # 1) CALCUL DES DÉPLACEMENTS ¹³C
    # --------------------------------------------------------------
    c_shifts: dict[int, float] = {}
    c_rows: list[dict[str, Any]] = []
    c_groups: dict[int, list[Any]] = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "C":
            c_groups.setdefault(ranks[idx], []).append(atom)

    for atoms in c_groups.values():
        env = "Aliphatique"
        for a in atoms:
            idx = int(a.GetIdx())
            hyb = str(a.GetHybridization())
            arom = a.GetIsAromatic()
            nbs_all = list(a.GetNeighbors())
            nbs_1 = [nb for nb in nbs_all if nb.GetSymbol() != "H"]
            h_cnt = sum(1 for nb in nbs_all if nb.GetSymbol() == "H")
            nbs_2 = [
                mol_h.GetAtomWithIdx(j)
                for j in range(mol_h.GetNumAtoms())
                if dist_mat[idx, j] == 2 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"
            ]
            nbs_3 = [
                mol_h.GetAtomWithIdx(j)
                for j in range(mol_h.GetNumAtoms())
                if dist_mat[idx, j] == 3 and mol_h.GetAtomWithIdx(j).GetSymbol() != "H"
            ]

            has_co = _has_carbonyl_o(mol_h, idx)
            has_cn = any(
                nb.GetSymbol() == "N" and _bond_order_nmr(mol_h, idx, nb.GetIdx()) == 3.0
                for nb in nbs_1
            )

            if has_co:
                conj = any(nb.GetIsAromatic() or str(nb.GetHybridization()) == "SP2" for nb in nbs_1 if nb.GetSymbol() == "C")
                single_o = [nb for nb in nbs_1 if nb.GetSymbol() == "O" and _bond_order_nmr(mol_h, idx, nb.GetIdx()) == 1.0]
                n_nbs = [nb for nb in nbs_1 if nb.GetSymbol() == "N"]
                cl_nbs = [nb for nb in nbs_1 if nb.GetSymbol() == "Cl"]

                if h_cnt >= 1:
                    shift, env = (192.0 if conj else 201.5), "Aldéhyde (C=O)"
                elif len(single_o) + len(n_nbs) >= 2:
                    shift, env = 156.5, "Urée / Carbamate (C=O)"
                elif single_o:
                    is_acid = any(n2.GetSymbol() == "H" for n2 in single_o[0].GetNeighbors())
                    if is_acid:
                        shift, env = (171.0 if conj else 178.0), "Acide carboxylique (C=O)"
                    else:
                        shift, env = (166.5 if conj else 171.0), "Ester (C=O)"
                elif n_nbs:
                    shift, env = (168.5 if conj else 172.0), "Amide (C=O)"
                elif cl_nbs:
                    shift, env = (168.0 if conj else 170.5), "Chlorure d'acyle (C=O)"
                else:
                    shift, env = (197.5 if conj else 208.0), "Cétone (C=O)"

            elif has_cn:
                shift, env = 118.2, "Nitrile (C≡N)"

            elif arom:
                ring6 = next((r for r in arom_rings_6 if idx in r), None)
                ring5 = next((r for r in arom_rings_5 if idx in r), None)
                if ring6 is not None:
                    shift = 128.5
                    pos_i = ring6.index(idx)
                    ring_set = set(ring6)
                    for pos_r, r_idx in enumerate(ring6):
                        d_ring = min((pos_r - pos_i) % 6, (pos_i - pos_r) % 6)
                        r_atom = mol_h.GetAtomWithIdx(int(r_idx))
                        if d_ring > 0 and r_atom.GetSymbol() == "N":
                            shift += {1: 21.0, 2: -4.5, 3: 7.5}.get(d_ring, 0.0)
                        for nb in r_atom.GetNeighbors():
                            nb_i = int(nb.GetIdx())
                            if nb_i in ring_set or nb.GetSymbol() == "H":
                                continue
                            cls = _classify_substituent_nmr(mol_h, int(r_idx), nb)
                            inc_tup = _AROM_C_INC.get(cls, (0.0, 0.0, 0.0, 0.0))
                            shift += inc_tup[d_ring]
                    env = "Aromatique (Ar-CH)" if h_cnt > 0 else "Aromatique quaternaire (Ar-C)"
                elif ring5 is not None:
                    pos_i = ring5.index(idx)
                    het_list = [
                        (min((pr - pos_i) % 5, (pos_i - pr) % 5), mol_h.GetAtomWithIdx(int(ri)).GetSymbol())
                        for pr, ri in enumerate(ring5)
                        if mol_h.GetAtomWithIdx(int(ri)).GetSymbol() != "C"
                    ]
                    shift = 126.0
                    if het_list:
                        d_het, h_sym = min(het_list, key=lambda x: x[0])
                        if h_sym == "O":
                            shift = 142.5 if d_het == 1 else 110.2
                        elif h_sym == "S":
                            shift = 125.2 if d_het == 1 else 127.0
                        elif h_sym == "N":
                            shift = 136.0 if len(het_list) >= 2 else (118.5 if d_het == 1 else 108.5)
                    for nb in nbs_1:
                        if int(nb.GetIdx()) not in set(ring5):
                            cls = _classify_substituent_nmr(mol_h, idx, nb)
                            shift += _AROM_C_INC.get(cls, (0.0, 0.0, 0.0, 0.0))[0] * 0.5
                    env = "Hétéroaromatique (5 chaînons)"
                else:
                    shift, env = 130.0, "Aromatique (Ar-C)"
                shift = max(95.0, min(170.0, shift))

            elif hyb == "SP2":
                shift = 123.3
                for nb in nbs_1:
                    if _bond_order_nmr(mol_h, idx, nb.GetIdx()) == 2.0:
                        for nb2 in nb.GetNeighbors():
                            if int(nb2.GetIdx()) != idx and nb2.GetSymbol() != "H":
                                cls2 = _classify_substituent_nmr(mol_h, int(nb.GetIdx()), nb2)
                                shift += {"Alkyl": -7.9, "Ar": -6.0, "OR": -34.0, "COOH": 10.0, "COR": 14.0}.get(cls2, -3.0)
                    else:
                        cls1 = _classify_substituent_nmr(mol_h, idx, nb)
                        shift += {"Alkyl": 10.6, "Ar": 13.0, "OR": 29.0, "COOH": 4.0, "COR": 15.0, "Cl": 3.0}.get(cls1, 7.0)
                shift = max(85.0, min(168.0, shift))
                env = f"Alcène (=CH{h_cnt})" if h_cnt > 0 else "Alcène (=C)"

            elif hyb == "SP":
                shift = 71.5 if h_cnt >= 1 else 81.5
                env = "Alcyne (C≡C)"

            else:
                shift = -2.3
                for nb in nbs_1:
                    cls = _classify_substituent_nmr(mol_h, idx, nb)
                    shift += alpha_c_cls.get(cls, 8.0)
                for nb2 in nbs_2:
                    shift += beta_c_sym.get(nb2.GetSymbol(), 4.0) * 0.55
                shift -= 2.2 * min(4, len(nbs_3))
                shift = max(5.0, min(105.0, shift))
                env = f"Aliphatique (CH{h_cnt})" if h_cnt > 0 else "Aliphatique (Cq)"

            c_shifts[idx] = round(shift, 1)

        a0 = atoms[0]
        c_rows.append({
            "carbones": ", ".join(str(x.GetIdx()) for x in atoms),
            "environnement": env,
            "nb_c": len(atoms),
            "shift": c_shifts[int(a0.GetIdx())],
        })

    # --------------------------------------------------------------
    # 2) CALCUL DES DÉPLACEMENTS ¹H, COUPS J ET MULTIPLETS
    # --------------------------------------------------------------
    h_shifts: dict[int, float] = {}
    h_rows: list[dict[str, Any]] = []
    h_groups: dict[int, list[Any]] = {}
    for idx, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() == "H":
            h_groups.setdefault(ranks[idx], []).append(atom)

    single_mult = {
        0: "s (singulet)",
        1: "d (doublet)",
        2: "t (triplet)",
        3: "q (quadruplet)",
        4: "quint (quintuplet)",
        5: "sext (sextuplet)",
        6: "sept (septuplet)",
    }

    for h_atoms in h_groups.values():
        h0 = h_atoms[0]
        parent = h0.GetNeighbors()[0]
        p_idx = int(parent.GetIdx())
        p_sym = parent.GetSymbol()
        j_hz = 0.0
        j2_hz = 0.0

        if p_sym == "O":
            o_nbs = [nb for nb in parent.GetNeighbors() if nb.GetSymbol() != "H"]
            is_acid = any(_has_carbonyl_o(mol_h, int(nb.GetIdx())) for nb in o_nbs)
            is_phenol = any(nb.GetIsAromatic() for nb in o_nbs)
            if is_acid:
                shift, mult, env = 11.50, "s large", "Acide carboxylique (-COOH)"
            elif is_phenol:
                shift, mult, env = 5.85, "s large", "Phénol (Ar-OH)"
            else:
                shift, mult, env = 2.85, "s large", "Alcool (-OH)"

        elif p_sym == "N":
            n_nbs = [nb for nb in parent.GetNeighbors() if nb.GetSymbol() != "H"]
            has_co = any(_has_carbonyl_o(mol_h, int(nb.GetIdx())) for nb in n_nbs)
            is_anil = any(nb.GetIsAromatic() for nb in n_nbs)
            if has_co:
                shift, mult, env = 7.75, "s large", "Amide (-CONH)"
            elif is_anil:
                shift, mult, env = 4.10, "s large", "Aniline (Ar-NH)"
            else:
                shift, mult, env = 1.85, "s large", "Amine (-NH)"

        elif p_sym == "S":
            shift, mult, env = 2.15, "s large", "Thiol (-SH)"

        else:
            arom = parent.GetIsAromatic()
            hyb = str(parent.GetHybridization())
            p_heavy_nbs = [nb for nb in parent.GetNeighbors() if nb.GetSymbol() != "H"]
            p_h_cnt = sum(1 for nb in parent.GetNeighbors() if nb.GetSymbol() == "H")
            is_aldehyde = _has_carbonyl_o(mol_h, p_idx)

            # Regroupement des protons vicinaux par rang d'équivalence chimique
            vic_by_rank: dict[int, int] = {}
            for nb in p_heavy_nbs:
                if nb.GetSymbol() != "C":
                    continue
                for n2 in nb.GetNeighbors():
                    if n2.GetSymbol() == "H":
                        rk = int(ranks[int(n2.GetIdx())])
                        vic_by_rank[rk] = vic_by_rank.get(rk, 0) + 1

            vic_counts = sorted(vic_by_rank.values(), reverse=True)
            total_vic = sum(vic_counts)

            if is_aldehyde:
                conj = any(nb.GetIsAromatic() or str(nb.GetHybridization()) == "SP2" for nb in p_heavy_nbs if nb.GetSymbol() == "C")
                shift = 9.95 if conj else 9.72
                env = "Aldéhyde (-CHO)"
                j_hz = 2.0 if total_vic > 0 else 0.0
                mult = single_mult.get(total_vic, "m (multiplet)")

            elif arom:
                ring6 = next((r for r in arom_rings_6 if p_idx in r), None)
                ring5 = next((r for r in arom_rings_5 if p_idx in r), None)
                if ring6 is not None:
                    shift = 7.26
                    pos_i = ring6.index(p_idx)
                    ring_set = set(ring6)
                    for pos_r, r_idx in enumerate(ring6):
                        d_ring = min((pos_r - pos_i) % 6, (pos_i - pos_r) % 6)
                        if d_ring == 0:
                            continue
                        r_atom = mol_h.GetAtomWithIdx(int(r_idx))
                        if r_atom.GetSymbol() == "N":
                            shift += {1: 1.35, 2: 0.02, 3: 0.42}.get(d_ring, 0.0)
                        for nb in r_atom.GetNeighbors():
                            nb_i = int(nb.GetIdx())
                            if nb_i in ring_set or nb.GetSymbol() == "H":
                                continue
                            cls = _classify_substituent_nmr(mol_h, int(r_idx), nb)
                            inc_h = _AROM_H_INC.get(cls, (0.0, 0.0, 0.0))
                            shift += inc_h[d_ring - 1]
                    env = "Aromatique (Ar-H)"
                elif ring5 is not None:
                    pos_i = ring5.index(p_idx)
                    het_list = [
                        (min((pr - pos_i) % 5, (pos_i - pr) % 5), mol_h.GetAtomWithIdx(int(ri)).GetSymbol())
                        for pr, ri in enumerate(ring5)
                        if mol_h.GetAtomWithIdx(int(ri)).GetSymbol() != "C"
                    ]
                    shift = 7.15
                    if het_list:
                        d_het, h_sym = min(het_list, key=lambda x: x[0])
                        if h_sym == "O":
                            shift = 7.42 if d_het == 1 else 6.38
                        elif h_sym == "S":
                            shift = 7.32 if d_het == 1 else 7.10
                        elif h_sym == "N":
                            shift = 7.62 if len(het_list) >= 2 else (6.68 if d_het == 1 else 6.22)
                    env = "Hétéroaromatique (Ar-H)"
                else:
                    shift, env = 7.30, "Aromatique (Ar-H)"

                shift = max(5.8, min(9.6, shift))
                j_hz = 8.2 if total_vic > 0 else 0.0
                if len(vic_counts) == 0:
                    mult = "s (singulet)"
                elif len(vic_counts) == 1:
                    mult = single_mult.get(vic_counts[0], "m (multiplet)")
                elif vic_counts == [1, 1]:
                    mult = "dd (doublet de doublets)"
                    j_hz, j2_hz = 8.2, 7.4
                else:
                    mult = "m (multiplet)"

            elif hyb == "SP2":
                shift = 5.25
                for nb in p_heavy_nbs:
                    if _bond_order_nmr(mol_h, p_idx, nb.GetIdx()) == 2.0:
                        for nb2 in nb.GetNeighbors():
                            if int(nb2.GetIdx()) != p_idx and nb2.GetSymbol() != "H":
                                cls2 = _classify_substituent_nmr(mol_h, int(nb.GetIdx()), nb2)
                                shift += {"Ar": 0.45, "COOH": 0.80, "COOR": 0.75, "COR": 0.90, "OR": -0.95, "Alkyl": -0.22}.get(cls2, 0.0)
                    else:
                        cls1 = _classify_substituent_nmr(mol_h, p_idx, nb)
                        shift += {"Ar": 1.38, "COOH": 0.90, "COOR": 0.80, "COR": 1.10, "OR": 1.22, "Alkyl": 0.45}.get(cls1, 0.25)
                shift = max(4.2, min(8.2, shift))
                env = "Alcène (=CH)"
                j_hz = 11.2 if total_vic > 0 else 0.0
                if len(vic_counts) <= 1:
                    mult = single_mult.get(total_vic, "m (multiplet)")
                elif vic_counts == [1, 1]:
                    mult = "dd (doublet de doublets)"
                    j_hz, j2_hz = 15.5, 10.2
                else:
                    mult = "m (multiplet)"

            elif hyb == "SP":
                conj = any(
                    n2.GetIsAromatic()
                    for nb in p_heavy_nbs
                    for n2 in nb.GetNeighbors()
                    if int(n2.GetIdx()) != p_idx
                )
                shift = 3.05 if conj else 2.45
                env = "Alcyne terminal (≡CH)"
                j_hz = 2.4 if total_vic > 0 else 0.0
                mult = single_mult.get(total_vic, "m (multiplet)")

            else:
                shift = 0.86 if p_h_cnt >= 3 else (1.25 if p_h_cnt == 2 else 1.50)
                for nb in p_heavy_nbs:
                    cls = _classify_substituent_nmr(mol_h, p_idx, nb)
                    shift += alpha_h_cls.get(cls, 0.25)
                    # Effet bêta subtil (ex. méthyle en bêta d'un carbonyle/aromatique/oxygène)
                    if cls in ("Alkyl", "COR", "COOR", "CONR2"):
                        for n2 in nb.GetNeighbors():
                            if int(n2.GetIdx()) == p_idx or n2.GetSymbol() == "H":
                                continue
                            if cls == "Alkyl":
                                c2 = _classify_substituent_nmr(mol_h, int(nb.GetIdx()), n2)
                                shift += {"OH": 0.26, "OR": 0.24, "OCOR": 0.30, "COOH": 0.28, "COOR": 0.25, "COR": 0.28, "Ar": 0.22, "Cl": 0.35, "Br": 0.35}.get(c2, 0.0)
                            elif n2.GetIsAromatic():
                                shift += 0.30
                shift = max(0.65, min(6.6, shift))
                env = f"Aliphatique (CH{p_h_cnt})" if p_h_cnt > 1 else "Aliphatique (CH)"
                j_hz = 7.2 if total_vic > 0 else 0.0
                if len(vic_counts) == 0:
                    mult = "s (singulet)"
                elif len(vic_counts) == 1:
                    mult = single_mult.get(vic_counts[0], "m (multiplet)")
                elif vic_counts == [1, 1]:
                    mult = "dd (doublet de doublets)"
                    j_hz, j2_hz = 7.5, 4.5
                elif vic_counts == [2, 1]:
                    mult = "dt (doublet de triplets)"
                    j_hz, j2_hz = 7.2, 6.8
                else:
                    mult = "m (multiplet)"

        val_h = round(shift, 2)
        for ha in h_atoms:
            h_shifts[int(ha.GetIdx())] = val_h

        h_rows.append({
            "porte_par": f"{p_sym}{p_idx}",
            "integration": f"{len(h_atoms)}H",
            "n_h": len(h_atoms),
            "multiplicite": mult,
            "j_hz": round(j_hz, 1),
            "j2_hz": round(j2_hz, 1),
            "environnement": env,
            "shift": val_h,
        })

    h_rows.sort(key=lambda x: -float(x["shift"]))
    c_rows.sort(key=lambda x: -float(x["shift"]))

    # --------------------------------------------------------------
    # 3) CORRÉLATIONS 2D (COSY, HSQC, HMBC)
    # --------------------------------------------------------------
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
    width: float = 0.0035,
    freq_mhz: float = 400.0,
    num_points: int = 2200,
) -> list[list[float]]:
    x = np.linspace(float(x_min), float(x_max), int(num_points))
    y = np.zeros_like(x)
    pascal_map: dict[str, list[float]] = {
        "s": [1.0],
        "s large": [1.0],
        "d": [1.0, 1.0],
        "t": [1.0, 2.0, 1.0],
        "q": [1.0, 3.0, 3.0, 1.0],
        "quint": [1.0, 4.0, 6.0, 4.0, 1.0],
        "sext": [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        "sept": [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
        "m": [1.0, 3.0, 6.0, 7.0, 6.0, 3.0, 1.0],
    }

    for row in rows:
        x0 = float(row["shift"])
        n_nuclei = float(row.get("n_h", row.get("nb_c", 1.0)))
        mult_key = "s"
        w_peak = float(width)
        m_str = clean_val(row.get("multiplicite"))
        if m_str:
            if "large" in m_str:
                mult_key = "s large"
                w_peak = width * 6.0
            else:
                mult_key = m_str.split(" ")[0]

        j1_hz = float(row.get("j_hz") or (7.2 if mult_key not in ("s", "s large") else 0.0))
        j2_hz = float(row.get("j2_hz") or 0.0)
        d1_ppm = j1_hz / float(freq_mhz)
        d2_ppm = (j2_hz if j2_hz > 0 else j1_hz * 0.6) / float(freq_mhz)

        if mult_key == "dd":
            lines = [
                (x0 - 0.5 * d1_ppm - 0.5 * d2_ppm, 0.25),
                (x0 - 0.5 * d1_ppm + 0.5 * d2_ppm, 0.25),
                (x0 + 0.5 * d1_ppm - 0.5 * d2_ppm, 0.25),
                (x0 + 0.5 * d1_ppm + 0.5 * d2_ppm, 0.25),
            ]
        elif mult_key == "dt":
            lines = [
                (x0 - 0.5 * d1_ppm - d2_ppm, 0.125),
                (x0 - 0.5 * d1_ppm, 0.25),
                (x0 - 0.5 * d1_ppm + d2_ppm, 0.125),
                (x0 + 0.5 * d1_ppm - d2_ppm, 0.125),
                (x0 + 0.5 * d1_ppm, 0.25),
                (x0 + 0.5 * d1_ppm + d2_ppm, 0.125),
            ]
        else:
            weights = pascal_map.get(mult_key, [1.0])
            tot_w = sum(weights)
            n_lines = len(weights)
            lines = [
                (x0 + (k - (n_lines - 1) / 2.0) * d1_ppm, w / tot_w)
                for k, w in enumerate(weights)
            ]

        for xk, frac in lines:
            amp = n_nuclei * frac
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
    if is_mol_too_large(mol, 120):
        return adducts, []

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