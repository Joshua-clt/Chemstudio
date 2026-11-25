import json
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D


def mol_to_graph(smiles: str) -> dict[str, Any]:
    """Convertit un SMILES en graphe 2D JSON (coordonnées RDKit + kekulisation + stéréo)."""
    s = (smiles or "").strip()
    if not s:
        return {"atoms": [], "bonds": []}
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return {"atoms": [], "bonds": []}

    mol_copy = Chem.Mol(mol)
    try:
        Chem.Kekulize(mol_copy, clearAromaticFlags=True)
    except Exception:
        pass

    rdDepictor.Compute2DCoords(mol_copy)
    conf = mol_copy.GetConformer()
    try:
        Chem.WedgeMolBonds(mol_copy, conf)
    except Exception:
        pass

    atoms: list[dict[str, Any]] = []
    for i, a in enumerate(mol_copy.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append({
            "id": i,
            "el": a.GetSymbol(),
            "x": round(float(pos.x), 4),
            "y": round(float(pos.y), 4),
            "charge": int(a.GetFormalCharge()),
        })

    bonds: list[dict[str, Any]] = []
    for b in mol_copy.GetBonds():
        btype = b.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            order = 2
        elif btype == Chem.BondType.TRIPLE:
            order = 3
        else:
            order = 1

        bdir = b.GetBondDir()
        if bdir == Chem.BondDir.BEGINWEDGE:
            stereo = 1
        elif bdir == Chem.BondDir.BEGINDASH:
            stereo = 2
        else:
            stereo = 0

        bonds.append({
            "a": int(b.GetBeginAtomIdx()),
            "b": int(b.GetEndAtomIdx()),
            "order": order,
            "stereo": stereo,
        })

    return {"atoms": atoms, "bonds": bonds}


def graph_to_mol(graph: dict[str, Any]) -> tuple[Any, str, list[int]]:
    """Reconstruit une molécule RDKit sanitisée et identifie précisément les atomes en erreur."""
    raw_atoms = graph.get("atoms") or []
    raw_bonds = graph.get("bonds") or []
    if not raw_atoms:
        return None, "Toile vide", []

    rw = Chem.RWMol()
    id_to_idx: dict[int, int] = {}
    idx_to_id: dict[int, int] = {}
    coords: list[tuple[float, float]] = []

    for a in raw_atoms:
        aid = int(a["id"])
        el = str(a.get("el") or "C").strip()
        try:
            atom = Chem.Atom(el)
        except Exception:
            return None, f"Élément inconnu : {el}", [aid]
        atom.SetFormalCharge(int(a.get("charge") or 0))
        idx = rw.AddAtom(atom)
        id_to_idx[aid] = idx
        idx_to_id[idx] = aid
        coords.append((float(a.get("x") or 0.0), float(a.get("y") or 0.0)))

    seen_pairs: set[tuple[int, int]] = set()
    for b in raw_bonds:
        a_id = int(b.get("a", -1))
        b_id = int(b.get("b", -1))
        if a_id not in id_to_idx or b_id not in id_to_idx or a_id == b_id:
            continue
        ia, ib = id_to_idx[a_id], id_to_idx[b_id]
        pair = (min(ia, ib), max(ia, ib))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        order = int(b.get("order") or 1)
        if order == 2:
            btype = Chem.BondType.DOUBLE
        elif order == 3:
            btype = Chem.BondType.TRIPLE
        else:
            btype = Chem.BondType.SINGLE

        rw.AddBond(ia, ib, btype)
        stereo = int(b.get("stereo") or 0)
        if order == 1 and stereo in (1, 2):
            bond_obj = rw.GetBondBetweenAtoms(ia, ib)
            if bond_obj is not None:
                bond_obj.SetBondDir(
                    Chem.BondDir.BEGINWEDGE if stereo == 1 else Chem.BondDir.BEGINDASH
                )

    conf = Chem.Conformer(rw.GetNumAtoms())
    for idx, (x, y) in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(x, y, 0.0))
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()

    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        bad_ids: list[int] = []
        details: list[str] = []
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass

        for prob in problems:
            ptype = prob.GetType()
            if hasattr(prob, "GetAtomIdx"):
                idx = int(prob.GetAtomIdx())
                if idx in idx_to_id:
                    aid = idx_to_id[idx]
                    if aid not in bad_ids:
                        bad_ids.append(aid)
                    atom_obj = mol.GetAtomWithIdx(idx)
                    sym = atom_obj.GetSymbol()
                    bond_sum = int(round(sum(bd.GetBondTypeAsDouble() for bd in atom_obj.GetBonds())))
                    if "Valence" in ptype:
                        details.append(f"{sym} ({bond_sum} liaisons)")
                    else:
                        details.append(f"{sym} (valence/charge)")
            elif hasattr(prob, "GetAtomIndices"):
                for idx in prob.GetAtomIndices():
                    if int(idx) in idx_to_id:
                        aid = idx_to_id[int(idx)]
                        if aid not in bad_ids:
                            bad_ids.append(aid)
                details.append("cycle aromatique")

        short_msg = "Valence : " + ", ".join(details[:2]) if details else "Valence invalide"
        if len(details) > 2:
            short_msg += f" (+{len(details) - 2})"
        return None, short_msg, bad_ids

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "Structure invalide", []

    try:
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass

    return mol, "Valence OK", []


def graph_to_smiles(graph: dict[str, Any]) -> tuple[str | None, str, list[int]]:
    mol, msg, bad_ids = graph_to_mol(graph)
    if mol is None:
        return None, msg, bad_ids
    try:
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi, "Valence OK", []
    except Exception:
        return None, "Structure invalide", []


def clean_graph_2d(graph: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str, list[int]]:
    """Redresse proprement la géométrie 2D via RDKit et renvoie le graphe nettoyé."""
    smi, msg, bad_ids = graph_to_smiles(graph)
    if not smi:
        return None, None, msg, bad_ids
    cleaned = mol_to_graph(smi)
    return cleaned, smi, "Géométrie 2D nettoyée", []


_SKETCHER_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr" data-theme="__INIT_THEME__">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: rgba(9, 17, 19, 0.94);
  --bar-bg: rgba(14, 24, 27, 0.90);
  --pop-bg: rgba(12, 22, 26, 0.96);
  --canvas-bg: #081113;
  --grid-dot: rgba(148, 163, 184, 0.14);
  --txt: #e6edf3;
  --sub: #94a3b8;
  --border: rgba(148, 163, 184, 0.22);
  --accent: #10b981;
  --cyan: #38bdf8;
  --danger: #f43f5e;
  --warn: #f59e0b;
  --btn-bg: rgba(255, 255, 255, 0.04);
  --btn-hover: rgba(56, 189, 248, 0.14);
  --btn-active-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.28), rgba(56, 189, 248, 0.26));
  --btn-active-bd: rgba(56, 189, 248, 0.65);
  --bond-color: #e2e8f0;
  --ghost-color: rgba(56, 189, 248, 0.78);
  --ghost-fill: rgba(16, 185, 129, 0.10);
  --hover-ring: rgba(56, 189, 248, 0.34);
  --sel-fill: rgba(56, 189, 248, 0.14);
  --sel-stroke: rgba(56, 189, 248, 0.75);
}

html[data-theme="light"] {
  --bg: rgba(248, 250, 252, 0.96);
  --bar-bg: rgba(255, 255, 255, 0.94);
  --pop-bg: rgba(255, 255, 255, 0.97);
  --canvas-bg: #f8fafc;
  --grid-dot: rgba(15, 23, 42, 0.10);
  --txt: #0f172a;
  --sub: #475569;
  --border: rgba(15, 23, 42, 0.13);
  --accent: #059669;
  --cyan: #0284c7;
  --danger: #dc2626;
  --warn: #d97706;
  --btn-bg: rgba(15, 23, 42, 0.03);
  --btn-hover: rgba(2, 132, 199, 0.11);
  --btn-active-bg: linear-gradient(135deg, rgba(5, 150, 105, 0.18), rgba(2, 132, 199, 0.18));
  --btn-active-bd: rgba(2, 132, 199, 0.55);
  --bond-color: #0f172a;
  --ghost-color: rgba(2, 132, 199, 0.78);
  --ghost-fill: rgba(5, 150, 105, 0.08);
  --hover-ring: rgba(2, 132, 199, 0.28);
  --sel-fill: rgba(2, 132, 199, 0.12);
  --sel-stroke: rgba(2, 132, 199, 0.70);
}

* { box-sizing: border-box; user-select: none; -webkit-user-select: none; }
html, body {
  margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
  background: var(--canvas-bg); color: var(--txt);
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

.shell {
  display: flex; flex-direction: column; width: 100%; height: 100%; position: relative;
}

.top-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 5px;
  padding: 6px 8px; background: var(--bar-bg);
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.tool-group {
  display: flex; align-items: center; gap: 3px; flex-wrap: wrap;
}
.sep {
  width: 1px; height: 20px; background: var(--border); margin: 0 2px;
}

.tbtn {
  display: inline-flex; align-items: center; justify-content: center; gap: 4px;
  min-width: 28px; height: 28px; padding: 0 6px;
  border-radius: 7px; border: 1px solid var(--border);
  background: var(--btn-bg); color: var(--txt);
  font-family: inherit; font-size: 11.5px; font-weight: 600;
  cursor: pointer; transition: all 0.14s ease;
}
.tbtn:hover { background: var(--btn-hover); border-color: var(--cyan); }
.tbtn.active {
  background: var(--btn-active-bg); border-color: var(--btn-active-bd);
  color: var(--txt); box-shadow: 0 0 10px rgba(56, 189, 248, 0.16);
}
.tbtn.danger:hover { border-color: var(--danger); color: var(--danger); background: rgba(244, 63, 94, 0.12); }
.tbtn svg { width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

.mid-area {
  flex: 1; min-height: 0; display: flex; position: relative;
}

.el-bar {
  width: 44px; padding: 5px 4px; background: var(--bar-bg);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; align-items: center; gap: 3px;
  overflow-y: auto;
}
.el-btn {
  width: 33px; height: 27px; padding: 0; border-radius: 7px;
  border: 1px solid var(--border); background: var(--btn-bg);
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 700;
  cursor: pointer; transition: all 0.14s ease; color: var(--txt); flex-shrink: 0;
}
.el-btn:hover { background: var(--btn-hover); border-color: var(--cyan); }
.el-btn.active { background: var(--btn-active-bg); border-color: var(--btn-active-bd); }
.el-btn.pt-btn {
  margin-top: auto; border-style: dashed; color: var(--cyan); font-size: 10.5px;
}

.canvas-wrap {
  flex: 1; position: relative; overflow: hidden;
  background-color: var(--canvas-bg);
  background-image: radial-gradient(var(--grid-dot) 1.2px, transparent 1.2px);
  background-size: 22px 22px;
}
#draw-canvas {
  width: 100%; height: 100%; display: block; cursor: crosshair;
}

.status-pill {
  position: absolute; right: 10px; bottom: 10px;
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px;
  background: var(--bar-bg); border: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--sub);
  pointer-events: none; backdrop-filter: blur(10px);
}
.status-pill.has-err {
  border-color: rgba(244, 63, 94, 0.55); color: var(--danger);
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
.dot.warn { background: var(--warn); }
.dot.err { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

.hint-pill {
  position: absolute; left: 10px; bottom: 10px;
  padding: 4px 9px; border-radius: 8px;
  background: var(--bar-bg); border: 1px solid var(--border);
  font-size: 10.5px; color: var(--sub); pointer-events: none; opacity: 0.88;
  max-width: calc(100% - 175px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

.bottom-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 5px;
  padding: 6px 8px; background: var(--bar-bg);
  border-top: 1px solid var(--border); flex-wrap: wrap; position: relative;
}
.ring-lbl {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--cyan); font-weight: 700;
  margin-right: 2px;
}

.popover-menu {
  position: absolute; bottom: 44px; left: 10px; z-index: 40;
  display: none; flex-direction: column; gap: 6px;
  padding: 10px; border-radius: 12px; min-width: 240px; max-width: 340px;
  background: var(--pop-bg); border: 1px solid var(--border);
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.42); backdrop-filter: blur(16px);
}
.popover-menu.open { display: flex; }
.pop-title {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--cyan); font-weight: 700;
}
.pop-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px;
}
.pop-item {
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px;
  padding: 6px 5px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--btn-bg); color: var(--txt); cursor: pointer;
  font-size: 11px; font-weight: 600; transition: all 0.14s ease;
}
.pop-item:hover { background: var(--btn-hover); border-color: var(--cyan); }
.pop-item.active { background: var(--btn-active-bg); border-color: var(--btn-active-bd); }
.pop-sub { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--sub); }
</style>
</head>
<body>
<div class="shell">
  <!-- BARRE SUPÉRIEURE -->
  <div class="top-bar">
    <div class="tool-group">
      <button class="tbtn" data-tool="select" title="Sélection / Lasso / Déplacer (V) · Ctrl+C/V pour dupliquer">
        <svg viewBox="0 0 24 24"><path d="M5 3l14 9-6 2-3 6z"/></svg>
      </button>
      <button class="tbtn active" data-tool="bond1" title="Liaison simple · Clic liaison = 1→2→3 (B)">
        <svg viewBox="0 0 24 24"><line x1="5" y1="19" x2="19" y2="5"/></svg>
      </button>
      <button class="tbtn" data-tool="bond2" title="Liaison double (2)">
        <svg viewBox="0 0 24 24"><line x1="4" y1="16" x2="16" y2="4"/><line x1="8" y1="20" x2="20" y2="8"/></svg>
      </button>
      <button class="tbtn" data-tool="bond3" title="Liaison triple (3)">
        <svg viewBox="0 0 24 24"><line x1="3" y1="15" x2="15" y2="3"/><line x1="6" y1="18" x2="18" y2="6"/><line x1="9" y1="21" x2="21" y2="9"/></svg>
      </button>
      <button class="tbtn" data-tool="chain" title="Chaîne carbonée zig-zag au clic-glisser (L)">
        <svg viewBox="0 0 24 24"><polyline points="2,15 7,9 12,15 17,9 22,15"/></svg>
      </button>
      <button class="tbtn" data-tool="wedge" title="Liaison avant · Coin plein (W)">
        <svg viewBox="0 0 24 24"><polygon points="5,18 16,4 20,8" fill="currentColor" stroke="none"/></svg>
      </button>
      <button class="tbtn" data-tool="dash" title="Liaison arrière · Hachurée (D)">
        <svg viewBox="0 0 24 24">
          <line x1="5" y1="18" x2="7" y2="19"/><line x1="8" y1="14" x2="11" y2="16"/>
          <line x1="11" y1="10" x2="15" y2="13"/><line x1="14" y1="6" x2="19" y2="10"/>
        </svg>
      </button>
      <div class="sep"></div>
      <button class="tbtn" data-tool="charge_plus" title="Ajouter charge (+)">+</button>
      <button class="tbtn" data-tool="charge_minus" title="Ajouter charge (−)">−</button>
      <button class="tbtn danger" data-tool="erase" title="Gomme (E ou Suppr)">
        <svg viewBox="0 0 24 24"><path d="M20 20H7L3 16l9-9 8 8-4 5z"/><line x1="12" y1="7" x2="18" y2="13"/></svg>
      </button>
    </div>

    <div class="tool-group">
      <button class="tbtn" id="btn-flip-h" title="Miroir horizontal (sélection ou molécule)">⇄</button>
      <button class="tbtn" id="btn-flip-v" title="Miroir vertical (sélection ou molécule)">⇅</button>
      <button class="tbtn" id="btn-rot" title="Pivoter de 30° (sélection ou molécule)">↻</button>
      <div class="sep"></div>
      <button class="tbtn" id="btn-undo" title="Annuler (Ctrl+Z)">↶</button>
      <button class="tbtn" id="btn-redo" title="Rétablir (Ctrl+Y)">↷</button>
      <button class="tbtn" id="btn-clean" title="Redresser proprement la géométrie 2D via RDKit">✦ 2D</button>
      <button class="tbtn" id="btn-center" title="Recentrer la vue">⌖</button>
      <button class="tbtn danger" id="btn-clear" title="Tout effacer">✕</button>
    </div>
  </div>

  <!-- ZONE CENTRALE -->
  <div class="mid-area">
    <div class="el-bar" id="el-bar"></div>
    <div class="canvas-wrap" id="canvas-wrap">
      <canvas id="draw-canvas"></canvas>
      <div class="hint-pill" id="hint-pill">Clic atome : chaîne 120° · Glisser : aimanté 30° · Survol + O/N/S/F/Suppr</div>
      <div class="status-pill" id="status-pill"><span class="dot" id="status-dot"></span><span id="status-txt">Synchro directe</span></div>
    </div>
  </div>

  <!-- POPOVER HÉTÉROCYCLES -->
  <div class="popover-menu" id="pop-hetero">
    <div class="pop-title">Hétérocycles &amp; Bicycles</div>
    <div class="pop-grid">
      <button class="pop-item" data-tool="ring_pyridine"><span>Pyridine</span><span class="pop-sub">C₅H₅N</span></button>
      <button class="pop-item" data-tool="ring_furan"><span>Furane</span><span class="pop-sub">C₄H₄O</span></button>
      <button class="pop-item" data-tool="ring_thiophene"><span>Thiophène</span><span class="pop-sub">C₄H₄S</span></button>
      <button class="pop-item" data-tool="ring_pyrrole"><span>Pyrrole</span><span class="pop-sub">C₄H₅N</span></button>
      <button class="pop-item" data-tool="ring_piperidine"><span>Pipéridine</span><span class="pop-sub">C₅H₁₁N</span></button>
      <button class="pop-item" data-tool="ring_naphthalene"><span>Naphtalène</span><span class="pop-sub">C₁₀H₈</span></button>
    </div>
  </div>

  <!-- POPOVER GROUPEMENTS FONCTIONNELS -->
  <div class="popover-menu" id="pop-groups">
    <div class="pop-title">Groupements fonctionnels rapides</div>
    <div class="pop-grid">
      <button class="pop-item" data-tool="grp_cooh"><span>−COOH</span><span class="pop-sub">Acide</span></button>
      <button class="pop-item" data-tool="grp_cho"><span>−CHO</span><span class="pop-sub">Aldéhyde</span></button>
      <button class="pop-item" data-tool="grp_no2"><span>−NO₂</span><span class="pop-sub">Nitro</span></button>
      <button class="pop-item" data-tool="grp_cn"><span>−CN</span><span class="pop-sub">Nitrile</span></button>
      <button class="pop-item" data-tool="grp_cf3"><span>−CF₃</span><span class="pop-sub">Trifluoro</span></button>
      <button class="pop-item" data-tool="grp_ome"><span>−OMe</span><span class="pop-sub">Méthoxy</span></button>
      <button class="pop-item" data-tool="grp_boc"><span>−Boc</span><span class="pop-sub">Protecteur</span></button>
    </div>
  </div>

  <!-- BARRE INFÉRIEURE -->
  <div class="bottom-bar">
    <div class="tool-group">
      <span class="ring-lbl">Cycles</span>
      <button class="tbtn" data-tool="ring_benzene" title="Benzène (clic liaison = fusion · clic atome = greffe)">
        <svg viewBox="0 0 24 24"><polygon points="12,3 20,7.5 20,16.5 12,21 4,16.5 4,7.5"/><circle cx="12" cy="12" r="4.5"/></svg>
        <span>Ph</span>
      </button>
      <button class="tbtn" data-tool="ring_6" title="Cyclohexane"><span>C6</span></button>
      <button class="tbtn" data-tool="ring_5" title="Cyclopentane"><span>C5</span></button>
      <button class="tbtn" data-tool="ring_4" title="Cyclobutane"><span>C4</span></button>
      <button class="tbtn" data-tool="ring_3" title="Cyclopropane"><span>C3</span></button>
      <button class="tbtn" data-tool="ring_7" title="Cycloheptane"><span>C7</span></button>
      <button class="tbtn" id="btn-pop-hetero" title="Hétérocycles (Pyridine, Furane, Thiophène, Pyrrole, Pipéridine, Naphtalène)">+ Cycles ▾</button>
      <button class="tbtn" id="btn-pop-groups" title="Groupements fonctionnels rapides (−COOH, −NO₂, −CF₃, −Boc...)">Groupes ▾</button>
    </div>
    <div class="tool-group">
      <button class="tbtn" id="btn-copy-img" title="Copier l'image transparente HD (prête pour Ctrl+V dans Word / PowerPoint)">Img</button>
      <button class="tbtn" id="btn-dl-svg" title="Télécharger le fichier vectoriel .svg">↓ SVG</button>
      <button class="tbtn" id="btn-copy-smi" title="Copier le SMILES canonique">SMILES</button>
    </div>
  </div>
</div>

<script>
(() => {
  const INIT_GRAPH = __INIT_GRAPH__;
  const BASE_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"];
  const BOND_LEN = 1.5;

  const RING_META = {
    ring_benzene:    { n: 6, aromatic: true,  heteroEl: null },
    ring_6:          { n: 6, aromatic: false, heteroEl: null },
    ring_5:          { n: 5, aromatic: false, heteroEl: null },
    ring_4:          { n: 4, aromatic: false, heteroEl: null },
    ring_3:          { n: 3, aromatic: false, heteroEl: null },
    ring_7:          { n: 7, aromatic: false, heteroEl: null },
    ring_pyridine:   { n: 6, aromatic: true,  heteroEl: "N", heteroPos: 3 },
    ring_piperidine: { n: 6, aromatic: false, heteroEl: "N", heteroPos: 3 },
    ring_furan:      { n: 5, aromatic: false, hetero5: "O" },
    ring_thiophene:  { n: 5, aromatic: false, hetero5: "S" },
    ring_pyrrole:    { n: 5, aromatic: false, hetero5: "N" },
  };

  const L = BOND_LEN;
  const H3 = L * Math.sqrt(3) / 2;
  const H1 = L / 2;
  const TEMPLATES = {
    grp_cooh: {
      atoms: [{ el:"C", x:0, y:0, ch:0 }, { el:"O", x:H1, y:H3, ch:0 }, { el:"O", x:H1, y:-H3, ch:0 }],
      bonds: [[0,1,2],[0,2,1]],
    },
    grp_cho: {
      atoms: [{ el:"C", x:0, y:0, ch:0 }, { el:"O", x:H1, y:H3, ch:0 }],
      bonds: [[0,1,2]],
    },
    grp_no2: {
      atoms: [{ el:"N", x:0, y:0, ch:1 }, { el:"O", x:H1, y:H3, ch:0 }, { el:"O", x:H1, y:-H3, ch:-1 }],
      bonds: [[0,1,2],[0,2,1]],
    },
    grp_cn: {
      atoms: [{ el:"C", x:0, y:0, ch:0 }, { el:"N", x:L, y:0, ch:0 }],
      bonds: [[0,1,3]],
    },
    grp_cf3: {
      atoms: [{ el:"C", x:0, y:0, ch:0 }, { el:"F", x:H1, y:H3, ch:0 }, { el:"F", x:L, y:0, ch:0 }, { el:"F", x:H1, y:-H3, ch:0 }],
      bonds: [[0,1,1],[0,2,1],[0,3,1]],
    },
    grp_ome: {
      atoms: [{ el:"O", x:0, y:0, ch:0 }, { el:"C", x:H1, y:H3, ch:0 }],
      bonds: [[0,1,1]],
    },
    grp_boc: {
      atoms: [
        { el:"C", x:0, y:0, ch:0 },
        { el:"O", x:H1, y:H3, ch:0 },
        { el:"O", x:H1, y:-H3, ch:0 },
        { el:"C", x:H1+L, y:-H3, ch:0 },
        { el:"C", x:H1+L+H1, y:0, ch:0 },
        { el:"C", x:H1+2*L, y:-H3, ch:0 },
        { el:"C", x:H1+L+H1, y:-2*H3, ch:0 },
      ],
      bonds: [[0,1,2],[0,2,1],[2,3,1],[3,4,1],[3,5,1],[3,6,1]],
    },
    ring_naphthalene: {
      atoms: [
        { el:"C", x:0,      y:0,     ch:0 },
        { el:"C", x:H3,     y:H1,    ch:0 },
        { el:"C", x:2*H3,   y:0,     ch:0 },
        { el:"C", x:2*H3,   y:-L,    ch:0 },
        { el:"C", x:H3,     y:-L-H1, ch:0 },
        { el:"C", x:0,      y:-L,    ch:0 },
        { el:"C", x:3*H3,   y:H1,    ch:0 },
        { el:"C", x:4*H3,   y:0,     ch:0 },
        { el:"C", x:4*H3,   y:-L,    ch:0 },
        { el:"C", x:3*H3,   y:-L-H1, ch:0 },
      ],
      bonds: [
        [2,3,2],
        [0,1,2],[1,2,1],[3,4,1],[4,5,2],[5,0,1],
        [2,6,1],[6,7,2],[7,8,1],[8,9,2],[9,3,1]
      ],
    },
  };

  const canvas = document.getElementById("draw-canvas");
  const wrap = document.getElementById("canvas-wrap");
  const ctx = canvas.getContext("2d");
  const statusPill = document.getElementById("status-pill");
  const statusDot = document.getElementById("status-dot");
  const statusTxt = document.getElementById("status-txt");
  const hintPill = document.getElementById("hint-pill");
  const popHetero = document.getElementById("pop-hetero");
  const popGroups = document.getElementById("pop-groups");

  let atoms = [];
  let bonds = [];
  let nextId = 1;
  let lastValidSmiles = __INIT_SMILES__;
  let invalidAtomIds = new Set();

  let scale = 42;
  let ox = 260;
  let oy = 180;

  let tool = "bond1";
  let activeEl = "C";
  let recentCustomEl = null;

  let selectedIds = new Set();
  let clipboardFrag = null;
  let marqueeRect = null;
  let movingOrigin = null;
  let mergeMap = new Map();

  let mouseScreen = { x: 0, y: 0 };
  let mouseWorld = { x: 0, y: 0 };
  let hoverAtom = null;
  let hoverBond = null;
  let isPointerDown = false;
  let pointerButton = 0;
  let dragStartWorld = null;
  let dragStartScreen = null;
  let dragMoved = false;
  let dragFromAtom = null;
  let dragTargetAtom = null;
  let dragEndWorld = null;
  let chainPreview = null;
  let panStartOx = 0;
  let panStartOy = 0;

  let undoStack = [];
  let redoStack = [];
  let emitTimer = null;

  function cloneState() {
    return JSON.stringify({ atoms, bonds, nextId });
  }
  function pushHistory() {
    undoStack.push(cloneState());
    if (undoStack.length > 70) undoStack.shift();
    redoStack = [];
  }
  function restoreState(snapStr) {
    const s = JSON.parse(snapStr);
    atoms = s.atoms || [];
    bonds = s.bonds || [];
    nextId = s.nextId || 1;
    selectedIds.clear();
    draw();
    scheduleEmit();
  }

  function closePopovers() {
    popHetero.classList.remove("open");
    popGroups.classList.remove("open");
  }

  const elBar = document.getElementById("el-bar");
  function rebuildElBar() {
    elBar.innerHTML = "";
    const list = [...BASE_ELEMENTS];
    if (recentCustomEl && !list.includes(recentCustomEl)) list.push(recentCustomEl);
    list.forEach(el => {
      const b = document.createElement("button");
      b.className = "el-btn";
      b.dataset.el = el;
      b.textContent = el;
      b.title = "Atome " + el + " (clic atome pour remplacer, clic vide pour poser)";
      b.addEventListener("click", () => {
        activeEl = el;
        setTool("atom");
      });
      elBar.appendChild(b);
    });
    const ptBtn = document.createElement("button");
    ptBtn.className = "el-btn pt-btn";
    ptBtn.textContent = "PT";
    ptBtn.title = "Ouvrir le Tableau Périodique complet en plein écran (118 éléments)";
    ptBtn.addEventListener("click", () => {
      closePopovers();
      window.parent.postMessage({ type: "cs_open_periodic_table", activeEl }, "*");
    });
    elBar.appendChild(ptBtn);
    styleElButtons();
  }

  function isDark() {
    return document.documentElement.getAttribute("data-theme") !== "light";
  }

  function elColor(el) {
    const d = isDark();
    if (el === "O") return d ? "#f87171" : "#dc2626";
    if (el === "N") return d ? "#38bdf8" : "#0284c7";
    if (el === "S") return d ? "#fbbf24" : "#d97706";
    if (el === "P") return d ? "#fb923c" : "#ea580c";
    if (el === "F" || el === "Cl") return d ? "#34d399" : "#059669";
    if (el === "Br") return d ? "#f97316" : "#c2410c";
    if (el === "I") return d ? "#c084fc" : "#9333ea";
    if (el === "B" || el === "Si") return d ? "#2dd4bf" : "#0d9488";
    if (el === "C") return d ? "#e2e8f0" : "#0f172a";
    return d ? "#cbd5e1" : "#334155";
  }

  function styleElButtons() {
    document.querySelectorAll(".el-btn[data-el]").forEach(b => {
      const el = b.dataset.el;
      b.style.color = el === "C" ? "var(--txt)" : elColor(el);
      b.classList.toggle("active", tool === "atom" && activeEl === el);
    });
  }

  function setTool(newTool) {
    tool = newTool;
    closePopovers();
    if (tool !== "select") selectedIds.clear();
    document.querySelectorAll("[data-tool]").forEach(b => {
      b.classList.toggle("active", b.dataset.tool === tool);
    });
    const inHetero = ["ring_pyridine","ring_furan","ring_thiophene","ring_pyrrole","ring_piperidine","ring_naphthalene"].includes(tool);
    const inGroups = tool.startsWith("grp_");
    document.getElementById("btn-pop-hetero").classList.toggle("active", inHetero);
    document.getElementById("btn-pop-groups").classList.toggle("active", inGroups);
    styleElButtons();

    if (tool === "chain") {
      hintPill.textContent = "Chaîne zig-zag : cliquez-glissez depuis le vide ou un atome pour déployer C2…C20";
    } else if (tool in RING_META || tool in TEMPLATES) {
      hintPill.textContent = "Survolez une liaison (fusion), un atome (greffe) ou le vide (pose libre)";
    } else if (tool === "select") {
      hintPill.textContent = "Glissez dans le vide = sélection · Déplacez sur un atome = fusion · Ctrl+C/V · Clic droit = pan";
    } else if (tool === "erase") {
      hintPill.textContent = "Cliquez sur un atome ou une liaison pour l'effacer";
    } else if (tool === "atom") {
      hintPill.textContent = `Atome ${activeEl} : cliquez un atome pour remplacer ou le vide pour poser`;
    } else {
      hintPill.textContent = "Clic atome : chaîne 120° · Glisser : aimanté 30° · Survol + O/N/S/F/Suppr · Clic droit = pan";
    }
    canvas.style.cursor = tool === "select" ? "default" : "crosshair";
    draw();
  }

  document.querySelectorAll("[data-tool]").forEach(b => {
    b.addEventListener("click", () => setTool(b.dataset.tool));
  });

  document.getElementById("btn-pop-hetero").addEventListener("click", e => {
    e.stopPropagation();
    const open = !popHetero.classList.contains("open");
    closePopovers();
    if (open) {
      popHetero.style.left = Math.min(e.currentTarget.offsetLeft, wrap.clientWidth - 250) + "px";
      popHetero.classList.add("open");
    }
  });
  document.getElementById("btn-pop-groups").addEventListener("click", e => {
    e.stopPropagation();
    const open = !popGroups.classList.contains("open");
    closePopovers();
    if (open) {
      popGroups.style.left = Math.max(8, Math.min(e.currentTarget.offsetLeft - 60, wrap.clientWidth - 260)) + "px";
      popGroups.classList.add("open");
    }
  });
  document.addEventListener("pointerdown", e => {
    if (!popHetero.contains(e.target) && !popGroups.contains(e.target) && e.target.id !== "btn-pop-hetero" && e.target.id !== "btn-pop-groups") {
      closePopovers();
    }
  });

  function w2s(x, y) { return { x: ox + x * scale, y: oy - y * scale }; }
  function s2w(sx, sy) { return { x: (sx - ox) / scale, y: (oy - sy) / scale }; }
  function dist2(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

  function atomById(id) { return atoms.find(a => a.id === id) || null; }
  function neighborsOf(id) {
    const out = [];
    for (const b of bonds) {
      if (b.a === id) out.push({ id: b.b, bond: b });
      else if (b.b === id) out.push({ id: b.a, bond: b });
    }
    return out;
  }
  function findBond(id1, id2) {
    return bonds.find(b => (b.a === id1 && b.b === id2) || (b.a === id2 && b.b === id1)) || null;
  }

  function findAtomAtScreen(sx, sy, radiusPx = 15) {
    let best = null, bestD = radiusPx;
    for (const a of atoms) {
      const p = w2s(a.x, a.y);
      const d = dist2(sx, sy, p.x, p.y);
      if (d < bestD) { bestD = d; best = a; }
    }
    return best;
  }

  function distPointToSeg(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay;
    const len2 = vx * vx + vy * vy;
    if (len2 < 1e-6) return dist2(px, py, ax, ay);
    let t = ((px - ax) * vx + (py - ay) * vy) / len2;
    t = Math.max(0.12, Math.min(0.88, t));
    return dist2(px, py, ax + t * vx, ay + t * vy);
  }

  function findBondAtScreen(sx, sy, tolPx = 11) {
    let best = null, bestD = tolPx;
    for (const b of bonds) {
      const a1 = atomById(b.a), a2 = atomById(b.b);
      if (!a1 || !a2) continue;
      const p1 = w2s(a1.x, a1.y), p2 = w2s(a2.x, a2.y);
      const d = distPointToSeg(sx, sy, p1.x, p1.y, p2.x, p2.y);
      if (d < bestD) { bestD = d; best = b; }
    }
    return best;
  }

  function findExistingAtomNearWorld(wx, wy, tolWorld = 0.36, excludeSet = null) {
    let best = null, bestD = tolWorld;
    for (const a of atoms) {
      if (excludeSet && excludeSet.has(a.id)) continue;
      const d = dist2(wx, wy, a.x, a.y);
      if (d < bestD) { bestD = d; best = a; }
    }
    return best;
  }

  function bestOutgoingAngle(atom) {
    const nbs = neighborsOf(atom.id).map(n => atomById(n.id)).filter(Boolean);
    if (nbs.length === 0) return -Math.PI / 6;
    if (nbs.length === 1) {
      const base = Math.atan2(nbs[0].y - atom.y, nbs[0].x - atom.x);
      const c1 = base + (2 * Math.PI) / 3;
      const c2 = base - (2 * Math.PI) / 3;
      return Math.abs(Math.sin(c1)) <= Math.abs(Math.sin(c2)) ? c1 : c2;
    }
    const angles = nbs.map(nb => Math.atan2(nb.y - atom.y, nb.x - atom.x)).sort((u, v) => u - v);
    let maxGap = -1, bestAng = 0;
    for (let i = 0; i < angles.length; i++) {
      const a1 = angles[i];
      const a2 = i + 1 < angles.length ? angles[i + 1] : angles[0] + 2 * Math.PI;
      const gap = a2 - a1;
      if (gap > maxGap) { maxGap = gap; bestAng = a1 + gap / 2; }
    }
    return bestAng;
  }

  function computeRingPreview(ringKey) {
    const meta = RING_META[ringKey];
    if (!meta) return null;
    const n = meta.n;
    const aromatic = meta.aromatic;

    if (hoverBond) {
      const a1 = atomById(hoverBond.a), a2 = atomById(hoverBond.b);
      if (a1 && a2) {
        const dx = a2.x - a1.x, dy = a2.y - a1.y;
        const edgeLen = Math.hypot(dx, dy) || BOND_LEN;
        const mx = (a1.x + a2.x) / 2, my = (a1.y + a2.y) / 2;
        const nx = -dy / edgeLen, ny = dx / edgeLen;
        const side = (mouseWorld.x - mx) * nx + (mouseWorld.y - my) * ny >= 0 ? 1 : -1;
        const apothem = edgeLen / (2 * Math.tan(Math.PI / n));
        const cx = mx + side * nx * apothem;
        const cy = my + side * ny * apothem;
        const r = edgeLen / (2 * Math.sin(Math.PI / n));
        const ang1 = Math.atan2(a1.y - cy, a1.x - cx);
        const ang2 = Math.atan2(a2.y - cy, a2.x - cx);
        let step = ang2 - ang1;
        while (step > Math.PI) step -= 2 * Math.PI;
        while (step < -Math.PI) step += 2 * Math.PI;
        const dir = step >= 0 ? -1 : 1;
        const pts = [
          { x: a1.x, y: a1.y, existingId: a1.id, el: a1.el },
          { x: a2.x, y: a2.y, existingId: a2.id, el: a2.el }
        ];
        const angStep = (2 * Math.PI) / n;
        for (let k = 1; k <= n - 2; k++) {
          const ang = ang2 - dir * k * angStep;
          const wx = cx + r * Math.cos(ang), wy = cy + r * Math.sin(ang);
          const ex = findExistingAtomNearWorld(wx, wy, 0.35);
          let el = "C";
          if (meta.heteroEl && (k + 1) === meta.heteroPos) el = meta.heteroEl;
          if (meta.hetero5 && k === 1) el = meta.hetero5;
          pts.push({ x: ex ? ex.x : wx, y: ex ? ex.y : wy, existingId: ex ? ex.id : null, el: ex ? ex.el : el });
        }
        const sharedOrder = hoverBond.order || 1;
        const edgeOrders = [];
        for (let i = 0; i < n; i++) {
          if (i === 0) edgeOrders.push(sharedOrder);
          else if (meta.hetero5) edgeOrders.push(i === 3 ? 2 : 1);
          else if (aromatic) edgeOrders.push(sharedOrder === 2 ? (i % 2 === 0 ? 2 : 1) : (i % 2 === 1 ? 2 : 1));
          else edgeOrders.push(1);
        }
        return { mode: "fuse", pts, edgeOrders };
      }
    }

    if (hoverAtom) {
      const ang = bestOutgoingAngle(hoverAtom);
      const r = BOND_LEN / (2 * Math.sin(Math.PI / n));
      const p0x = hoverAtom.x + BOND_LEN * Math.cos(ang);
      const p0y = hoverAtom.y + BOND_LEN * Math.sin(ang);
      const cx = p0x + r * Math.cos(ang);
      const cy = p0y + r * Math.sin(ang);
      const startAng = ang + Math.PI;
      const pts = [];
      for (let k = 0; k < n; k++) {
        const a = startAng + (k * 2 * Math.PI) / n;
        const wx = cx + r * Math.cos(a), wy = cy + r * Math.sin(a);
        const ex = findExistingAtomNearWorld(wx, wy, 0.32);
        let el = "C";
        if (meta.heteroEl && k === meta.heteroPos) el = meta.heteroEl;
        if (meta.hetero5 && k === 2) el = meta.hetero5;
        pts.push({ x: ex ? ex.x : wx, y: ex ? ex.y : wy, existingId: ex ? ex.id : null, el: ex ? ex.el : el });
      }
      let edgeOrders;
      if (meta.hetero5) edgeOrders = [2, 1, 1, 2, 1];
      else edgeOrders = Array.from({ length: n }, (_, i) => (aromatic && i % 2 === 1 ? 2 : 1));
      return { mode: "attach", pts, edgeOrders, attachFrom: hoverAtom };
    }

    const r = BOND_LEN / (2 * Math.sin(Math.PI / n));
    const pts = [];
    for (let k = 0; k < n; k++) {
      const a = Math.PI / 2 + (k * 2 * Math.PI) / n;
      let el = "C";
      if (meta.heteroEl && k === meta.heteroPos) el = meta.heteroEl;
      if (meta.hetero5 && k === 0) el = meta.hetero5;
      pts.push({ x: mouseWorld.x + r * Math.cos(a), y: mouseWorld.y + r * Math.sin(a), existingId: null, el });
    }
    let edgeOrders;
    if (meta.hetero5) edgeOrders = [1, 2, 1, 2, 1];
    else edgeOrders = Array.from({ length: n }, (_, i) => (aromatic && i % 2 === 0 ? 2 : 1));
    return { mode: "free", pts, edgeOrders };
  }

  function applyRingPreview(prev) {
    if (!prev) return;
    pushHistory();
    const ids = prev.pts.map(pt => {
      if (pt.existingId !== null && atomById(pt.existingId)) return pt.existingId;
      const id = nextId++;
      atoms.push({ id, el: pt.el || "C", x: pt.x, y: pt.y, charge: 0 });
      return id;
    });
    if (prev.mode === "attach" && prev.attachFrom) {
      if (!findBond(prev.attachFrom.id, ids[0])) {
        bonds.push({ a: prev.attachFrom.id, b: ids[0], order: 1, stereo: 0 });
      }
    }
    const n = ids.length;
    for (let i = 0; i < n; i++) {
      if (prev.mode === "fuse" && i === 0) continue;
      const u = ids[i], v = ids[(i + 1) % n];
      if (!findBond(u, v)) {
        bonds.push({ a: u, b: v, order: prev.edgeOrders[i], stereo: 0 });
      }
    }
    draw();
    scheduleEmit();
  }

  function computeTemplatePreview(tplKey) {
    const tpl = TEMPLATES[tplKey];
    if (!tpl) return null;
    let originX = mouseWorld.x, originY = mouseWorld.y, ang = 0, attachFrom = null;
    if (hoverAtom) {
      ang = bestOutgoingAngle(hoverAtom);
      originX = hoverAtom.x + BOND_LEN * Math.cos(ang);
      originY = hoverAtom.y + BOND_LEN * Math.sin(ang);
      attachFrom = hoverAtom;
    } else if (tplKey === "ring_naphthalene") {
      originX -= 2 * H3;
      originY += L / 2;
    }
    const ca = Math.cos(ang), sa = Math.sin(ang);
    const pts = tpl.atoms.map(ta => {
      const wx = originX + ta.x * ca - ta.y * sa;
      const wy = originY + ta.x * sa + ta.y * ca;
      const ex = attachFrom ? findExistingAtomNearWorld(wx, wy, 0.28) : null;
      return { x: ex ? ex.x : wx, y: ex ? ex.y : wy, el: ta.el, ch: ta.ch || 0, existingId: ex ? ex.id : null };
    });
    return { pts, bonds: tpl.bonds, attachFrom };
  }

  function applyTemplatePreview(prev) {
    if (!prev) return;
    pushHistory();
    const ids = prev.pts.map(pt => {
      if (pt.existingId !== null && atomById(pt.existingId)) return pt.existingId;
      const id = nextId++;
      atoms.push({ id, el: pt.el, x: pt.x, y: pt.y, charge: pt.ch || 0 });
      return id;
    });
    if (prev.attachFrom && !findBond(prev.attachFrom.id, ids[0])) {
      bonds.push({ a: prev.attachFrom.id, b: ids[0], order: 1, stereo: 0 });
    }
    for (const [i, j, ord] of prev.bonds) {
      const u = ids[i], v = ids[j];
      if (!findBond(u, v)) bonds.push({ a: u, b: v, order: ord, stereo: 0 });
    }
    draw();
    scheduleEmit();
  }

  function computeChainPreview(startW, endW, startAtom) {
    const dx = endW.x - startW.x, dy = endW.y - startW.y;
    const dist = Math.hypot(dx, dy);
    const stepAxis = BOND_LEN * Math.cos(Math.PI / 6);
    const nBonds = Math.max(1, Math.min(24, Math.round(dist / stepAxis)));
    const rawAng = Math.atan2(dy, dx);
    const axisAng = Math.round(rawAng / (Math.PI / 6)) * (Math.PI / 6);
    const pts = [{ x: startW.x, y: startW.y, existingId: startAtom ? startAtom.id : null }];
    let cx = startW.x, cy = startW.y;
    for (let k = 0; k < nBonds; k++) {
      const bAng = axisAng + (k % 2 === 0 ? Math.PI / 6 : -Math.PI / 6);
      cx += BOND_LEN * Math.cos(bAng);
      cy += BOND_LEN * Math.sin(bAng);
      const ex = findExistingAtomNearWorld(cx, cy, 0.30);
      pts.push({ x: ex ? ex.x : cx, y: ex ? ex.y : cy, existingId: ex ? ex.id : null });
    }
    return { pts, count: pts.length };
  }

  function applyChainPreview(ch) {
    if (!ch || ch.pts.length < 2) return;
    pushHistory();
    const ids = ch.pts.map(pt => {
      if (pt.existingId !== null && atomById(pt.existingId)) return pt.existingId;
      const id = nextId++;
      atoms.push({ id, el: "C", x: pt.x, y: pt.y, charge: 0 });
      return id;
    });
    for (let i = 0; i < ids.length - 1; i++) {
      if (ids[i] !== ids[i + 1] && !findBond(ids[i], ids[i + 1])) {
        bonds.push({ a: ids[i], b: ids[i + 1], order: 1, stereo: 0 });
      }
    }
    draw();
    scheduleEmit();
  }

  function transformSelection(fn) {
    const target = selectedIds.size > 0
      ? atoms.filter(a => selectedIds.has(a.id))
      : atoms;
    if (!target.length) return;
    pushHistory();
    let cx = 0, cy = 0;
    for (const a of target) { cx += a.x; cy += a.y; }
    cx /= target.length; cy /= target.length;
    for (const a of target) fn(a, cx, cy);
    draw();
    scheduleEmit();
  }

  function implicitHCount(atom) {
    const baseVal = { C: 4, N: 3, O: 2, S: 2, P: 3, F: 1, Cl: 1, Br: 1, I: 1, B: 3, Si: 4, Se: 2, Te: 2, As: 3 };
    let v = baseVal[atom.el];
    if (v === undefined) return 0;
    const ch = atom.charge || 0;
    if (atom.el === "C" || atom.el === "Si") v = Math.max(0, 4 - Math.abs(ch));
    else if (atom.el === "N" || atom.el === "P" || atom.el === "As") v = Math.max(0, 3 + ch);
    else if (atom.el === "O" || atom.el === "S" || atom.el === "Se" || atom.el === "Te") v = Math.max(0, 2 + ch);
    else if (atom.el === "B") v = Math.max(0, 3 - ch);
    else v = Math.max(0, v + ch);
    const bondSum = neighborsOf(atom.id).reduce((acc, n) => acc + (n.bond.order || 1), 0);
    return Math.max(0, v - bondSum);
  }

  function formatAtomText(a) {
    const hCnt = implicitHCount(a);
    let txt = a.el;
    if (hCnt === 1) txt += "H";
    else if (hCnt > 1) txt += "H" + (hCnt === 2 ? "₂" : hCnt === 3 ? "₃" : hCnt === 4 ? "₄" : String(hCnt));
    return txt;
  }

  function bondSideHint(b) {
    const a1 = atomById(b.a), a2 = atomById(b.b);
    if (!a1 || !a2) return 0;
    const dx = a2.x - a1.x, dy = a2.y - a1.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    let sum = 0, cnt = 0;
    for (const nb of neighborsOf(b.a)) {
      if (nb.id === b.b) continue;
      const p = atomById(nb.id);
      if (p) { sum += (p.x - a1.x) * nx + (p.y - a1.y) * ny; cnt++; }
    }
    for (const nb of neighborsOf(b.b)) {
      if (nb.id === b.a) continue;
      const p = atomById(nb.id);
      if (p) { sum += (p.x - a2.x) * nx + (p.y - a2.y) * ny; cnt++; }
    }
    if (cnt === 0 || Math.abs(sum) < 0.08) return 0;
    return sum > 0 ? 1 : -1;
  }

  function drawWedge(p1, p2, isDash, color) {
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    const halfW = 5.5;
    if (!isDash) {
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x + nx * halfW, p2.y + ny * halfW);
      ctx.lineTo(p2.x - nx * halfW, p2.y - ny * halfW);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    } else {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.9;
      const steps = 6;
      for (let i = 1; i <= steps; i++) {
        const t = i / (steps + 0.5);
        const cx = p1.x + dx * t, cy = p1.y + dy * t;
        const w = halfW * t;
        ctx.beginPath();
        ctx.moveTo(cx + nx * w, cy + ny * w);
        ctx.lineTo(cx - nx * w, cy - ny * w);
        ctx.stroke();
      }
    }
  }

  function drawBondObj(b, isHovered) {
    const a1 = atomById(b.a), a2 = atomById(b.b);
    if (!a1 || !a2) return;
    const p1 = w2s(a1.x, a1.y), p2 = w2s(a2.x, a2.y);

    if (isHovered) {
      ctx.save();
      ctx.strokeStyle = tool === "erase" ? "rgba(244, 63, 94, 0.35)" : getComputedStyle(document.documentElement).getPropertyValue("--hover-ring").trim();
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      ctx.restore();
    }

    const bondCol = getComputedStyle(document.documentElement).getPropertyValue("--bond-color").trim();
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const nx = -uy, ny = ux;

    ctx.lineCap = "round";
    if (b.order === 1 && (b.stereo === 1 || b.stereo === 2)) {
      drawWedge(p1, p2, b.stereo === 2, bondCol);
      return;
    }

    ctx.strokeStyle = bondCol;
    ctx.lineWidth = 2.1;

    if (b.order === 1) {
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    } else if (b.order === 2) {
      const side = bondSideHint(b);
      if (side === 0) {
        const off = 3.0;
        ctx.beginPath();
        ctx.moveTo(p1.x + nx * off, p1.y + ny * off); ctx.lineTo(p2.x + nx * off, p2.y + ny * off);
        ctx.moveTo(p1.x - nx * off, p1.y - ny * off); ctx.lineTo(p2.x - nx * off, p2.y - ny * off);
        ctx.stroke();
      } else {
        const sOff = -side * 5.2;
        const inset = Math.min(9, len * 0.16);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
        ctx.moveTo(p1.x + ux * inset + nx * sOff, p1.y + uy * inset + ny * sOff);
        ctx.lineTo(p2.x - ux * inset + nx * sOff, p2.y - uy * inset + ny * sOff);
        ctx.stroke();
      }
    } else if (b.order === 3) {
      const off = 4.6;
      const inset = Math.min(6, len * 0.10);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
      ctx.moveTo(p1.x + ux * inset + nx * off, p1.y + uy * inset + ny * off);
      ctx.lineTo(p2.x - ux * inset + nx * off, p2.y - uy * inset + ny * off);
      ctx.moveTo(p1.x + ux * inset - nx * off, p1.y + uy * inset - ny * off);
      ctx.lineTo(p2.x - ux * inset - nx * off, p2.y - uy * inset - ny * off);
      ctx.stroke();
    }
  }

  function drawAtomLabel(a, isHovered, isTarget, isMergeTarget) {
    const p = w2s(a.x, a.y);
    const nbs = neighborsOf(a.id);
    const isInvalid = invalidAtomIds.has(a.id);
    const isSelected = selectedIds.has(a.id);
    const showLabel = a.el !== "C" || a.charge !== 0 || nbs.length === 0 || isInvalid;

    if (isInvalid) {
      ctx.save();
      ctx.fillStyle = "rgba(244, 63, 94, 0.22)";
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(p.x - 15, p.y - 13, 30, 26, 7);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    if (isSelected) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 13, 0, Math.PI * 2);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--sel-fill").trim();
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--sel-stroke").trim();
      ctx.lineWidth = 1.6;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    if (isMergeTarget) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 15, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(16, 185, 129, 0.30)";
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    } else if (isHovered || isTarget) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(p.x, p.y, isTarget ? 12 : 11, 0, Math.PI * 2);
      ctx.fillStyle = tool === "erase"
        ? "rgba(244, 63, 94, 0.28)"
        : (isTarget ? "rgba(16, 185, 129, 0.32)" : getComputedStyle(document.documentElement).getPropertyValue("--hover-ring").trim());
      ctx.fill();
      ctx.restore();
    }

    if (!showLabel) return;

    const txt = (a.el === "C" && isInvalid && nbs.length > 0 && a.charge === 0) ? "C" : formatAtomText(a);
    const ch = a.charge || 0;
    const chStr = ch === 1 ? "+" : ch === -1 ? "−" : ch > 1 ? ch + "+" : ch < -1 ? Math.abs(ch) + "−" : "";

    ctx.save();
    ctx.font = "700 14px 'Plus Jakarta Sans', sans-serif";
    const mW = ctx.measureText(txt).width;
    const bgR = Math.max(9.5, mW / 2 + 3.5);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--canvas-bg").trim();
    ctx.beginPath();
    ctx.arc(p.x, p.y, bgR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = isInvalid ? "#f43f5e" : elColor(a.el);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, p.x, p.y + 0.5);

    if (chStr) {
      ctx.font = "700 10px 'JetBrains Mono', monospace";
      ctx.fillText(chStr, p.x + mW / 2 + 5, p.y - 6);
    }
    ctx.restore();
  }

  function draw() {
    const rect = wrap.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    for (const b of bonds) {
      drawBondObj(b, hoverBond === b);
    }

    const gCol = getComputedStyle(document.documentElement).getPropertyValue("--ghost-color").trim();
    const gFill = getComputedStyle(document.documentElement).getPropertyValue("--ghost-fill").trim();

    if (isPointerDown && pointerButton === 0 && dragMoved && dragEndWorld && (tool === "bond1" || tool === "bond2" || tool === "bond3" || tool === "wedge" || tool === "dash")) {
      const startW = dragFromAtom ? { x: dragFromAtom.x, y: dragFromAtom.y } : dragStartWorld;
      const endW = dragTargetAtom ? { x: dragTargetAtom.x, y: dragTargetAtom.y } : dragEndWorld;
      const p1 = w2s(startW.x, startW.y), p2 = w2s(endW.x, endW.y);
      ctx.save();
      ctx.strokeStyle = gCol; ctx.lineWidth = 2.6; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      ctx.restore();
    }

    if (isPointerDown && pointerButton === 0 && tool === "chain" && chainPreview && chainPreview.pts.length >= 2) {
      const sPts = chainPreview.pts.map(pt => w2s(pt.x, pt.y));
      ctx.save();
      ctx.strokeStyle = gCol; ctx.lineWidth = 2.6; ctx.setLineDash([5, 3]);
      ctx.beginPath();
      sPts.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.stroke();
      ctx.setLineDash([]);
      const lastP = sPts[sPts.length - 1];
      const badgeTxt = "C" + chainPreview.count;
      ctx.font = "700 11px 'JetBrains Mono', monospace";
      const bw = ctx.measureText(badgeTxt).width + 14;
      ctx.fillStyle = isDark() ? "rgba(12, 22, 26, 0.92)" : "rgba(255, 255, 255, 0.95)";
      ctx.strokeStyle = gCol;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.roundRect(lastP.x + 10, lastP.y - 24, bw, 20, 99);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = gCol;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(badgeTxt, lastP.x + 10 + bw / 2, lastP.y - 14);
      ctx.restore();
    }

    if (!isPointerDown && (tool in RING_META) && mouseScreen.x > 0 && mouseScreen.y > 0) {
      const prev = computeRingPreview(tool);
      if (prev) {
        const sPts = prev.pts.map(pt => w2s(pt.x, pt.y));
        ctx.save();
        ctx.beginPath();
        sPts.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.closePath();
        ctx.fillStyle = gFill; ctx.fill();
        ctx.strokeStyle = gCol; ctx.lineWidth = 2.1; ctx.setLineDash([5, 3]);
        ctx.stroke();
        if (prev.mode === "attach" && prev.attachFrom) {
          const pA = w2s(prev.attachFrom.x, prev.attachFrom.y);
          ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(sPts[0].x, sPts[0].y); ctx.stroke();
        }
        if (tool === "ring_benzene" || tool === "ring_pyridine") {
          const cx = sPts.reduce((s, p) => s + p.x, 0) / sPts.length;
          const cy = sPts.reduce((s, p) => s + p.y, 0) / sPts.length;
          ctx.beginPath(); ctx.arc(cx, cy, scale * 0.85, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        prev.pts.forEach((pt, idx) => {
          if (pt.el && pt.el !== "C") {
            ctx.fillStyle = gCol;
            ctx.fillText(pt.el, sPts[idx].x, sPts[idx].y);
          }
        });
        ctx.restore();
      }
    }

    if (!isPointerDown && (tool in TEMPLATES) && mouseScreen.x > 0 && mouseScreen.y > 0) {
      const prev = computeTemplatePreview(tool);
      if (prev) {
        const sPts = prev.pts.map(pt => w2s(pt.x, pt.y));
        ctx.save();
        ctx.strokeStyle = gCol; ctx.lineWidth = 2.1; ctx.setLineDash([5, 3]);
        if (prev.attachFrom) {
          const pA = w2s(prev.attachFrom.x, prev.attachFrom.y);
          ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(sPts[0].x, sPts[0].y); ctx.stroke();
        }
        for (const [i, j, ord] of prev.bonds) {
          ctx.beginPath(); ctx.moveTo(sPts[i].x, sPts[i].y); ctx.lineTo(sPts[j].x, sPts[j].y); ctx.stroke();
          if (ord >= 2) {
            const dx = sPts[j].x - sPts[i].x, dy = sPts[j].y - sPts[i].y;
            const ln = Math.hypot(dx, dy) || 1;
            const nx = -dy / ln * 4, ny = dx / ln * 4;
            ctx.beginPath(); ctx.moveTo(sPts[i].x + nx, sPts[i].y + ny); ctx.lineTo(sPts[j].x + nx, sPts[j].y + ny); ctx.stroke();
          }
        }
        ctx.setLineDash([]);
        ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = gCol;
        prev.pts.forEach((pt, idx) => {
          if (pt.el !== "C") ctx.fillText(pt.el, sPts[idx].x, sPts[idx].y);
        });
        ctx.restore();
      }
    }

    if (marqueeRect) {
      ctx.save();
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--sel-fill").trim();
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--sel-stroke").trim();
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 3]);
      ctx.fillRect(marqueeRect.x, marqueeRect.y, marqueeRect.w, marqueeRect.h);
      ctx.strokeRect(marqueeRect.x, marqueeRect.y, marqueeRect.w, marqueeRect.h);
      ctx.restore();
    }

    const mergeTargets = new Set(Array.from(mergeMap.values()).map(t => t.id));
    for (const a of atoms) {
      drawAtomLabel(a, hoverAtom === a, dragTargetAtom === a, mergeTargets.has(a.id));
    }
  }

  // Export SVG transparent propre (avec doubles liaisons alignées dans les cycles)
  function buildTransparentSvg() {
    if (!atoms.length) return "";
    const sc = 42, pad = 36;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const a of atoms) {
      if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
      if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
    }
    const w = Math.max(80, Math.round((maxX - minX) * sc + pad * 2));
    const h = Math.max(80, Math.round((maxY - minY) * sc + pad * 2));
    const tx = x => ((x - minX) * sc + pad);
    const ty = y => ((maxY - y) * sc + pad);
    const hasLbl = a => a.el !== "C" || (a.charge || 0) !== 0 || neighborsOf(a.id).length === 0;
    const strokeCol = "#0f172a";
    const parts = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">`];

    for (const b of bonds) {
      const a1 = atomById(b.a), a2 = atomById(b.b);
      if (!a1 || !a2) continue;
      let x1 = tx(a1.x), y1 = ty(a1.y);
      let x2 = tx(a2.x), y2 = ty(a2.y);
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len, nx = -uy, ny = ux;
      if (hasLbl(a1)) { x1 += ux * 11; y1 += uy * 11; }
      if (hasLbl(a2)) { x2 -= ux * 11; y2 -= uy * 11; }

      if (b.order === 1 && b.stereo === 1) {
        parts.push(`<polygon points="${x1.toFixed(1)},${y1.toFixed(1)} ${(x2+nx*5).toFixed(1)},${(y2+ny*5).toFixed(1)} ${(x2-nx*5).toFixed(1)},${(y2-ny*5).toFixed(1)}" fill="${strokeCol}"/>`);
      } else if (b.order === 1 && b.stereo === 2) {
        for (let i = 1; i <= 6; i++) {
          const t = i / 6.5, cx = x1 + (x2 - x1) * t, cy = y1 + (y2 - y1) * t, hw = 5 * t;
          parts.push(`<line x1="${(cx+nx*hw).toFixed(1)}" y1="${(cy+ny*hw).toFixed(1)}" x2="${(cx-nx*hw).toFixed(1)}" y2="${(cy-ny*hw).toFixed(1)}" stroke="${strokeCol}" stroke-width="1.9" stroke-linecap="round"/>`);
        }
      } else if (b.order === 1) {
        parts.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${strokeCol}" stroke-width="2.1" stroke-linecap="round"/>`);
      } else if (b.order === 2) {
        const side = bondSideHint(b);
        if (side === 0) {
          const off = 2.9;
          parts.push(`<line x1="${(x1+nx*off).toFixed(1)}" y1="${(y1+ny*off).toFixed(1)}" x2="${(x2+nx*off).toFixed(1)}" y2="${(y2+ny*off).toFixed(1)}" stroke="${strokeCol}" stroke-width="2" stroke-linecap="round"/>`);
          parts.push(`<line x1="${(x1-nx*off).toFixed(1)}" y1="${(y1-ny*off).toFixed(1)}" x2="${(x2-nx*off).toFixed(1)}" y2="${(y2-ny*off).toFixed(1)}" stroke="${strokeCol}" stroke-width="2" stroke-linecap="round"/>`);
        } else {
          const sOff = -side * 5.0;
          const inset = Math.min(8, len * 0.14);
          parts.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${strokeCol}" stroke-width="2.1" stroke-linecap="round"/>`);
          parts.push(`<line x1="${(x1+ux*inset+nx*sOff).toFixed(1)}" y1="${(y1+uy*inset+ny*sOff).toFixed(1)}" x2="${(x2-ux*inset+nx*sOff).toFixed(1)}" y2="${(y2-uy*inset+ny*sOff).toFixed(1)}" stroke="${strokeCol}" stroke-width="2" stroke-linecap="round"/>`);
        }
      } else if (b.order === 3) {
        const off = 4.4;
        const inset = Math.min(5, len * 0.08);
        parts.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${strokeCol}" stroke-width="1.9" stroke-linecap="round"/>`);
        parts.push(`<line x1="${(x1+ux*inset+nx*off).toFixed(1)}" y1="${(y1+uy*inset+ny*off).toFixed(1)}" x2="${(x2-ux*inset+nx*off).toFixed(1)}" y2="${(y2-uy*inset+ny*off).toFixed(1)}" stroke="${strokeCol}" stroke-width="1.9" stroke-linecap="round"/>`);
        parts.push(`<line x1="${(x1+ux*inset-nx*off).toFixed(1)}" y1="${(y1+uy*inset-ny*off).toFixed(1)}" x2="${(x2-ux*inset-nx*off).toFixed(1)}" y2="${(y2-uy*inset-ny*off).toFixed(1)}" stroke="${strokeCol}" stroke-width="1.9" stroke-linecap="round"/>`);
      }
    }
    const lightElCol = el => ({ O:"#dc2626", N:"#0284c7", S:"#d97706", P:"#ea580c", F:"#059669", Cl:"#059669", Br:"#c2410c", I:"#9333ea", B:"#0d9488", Si:"#0d9488" }[el] || "#0f172a");
    for (const a of atoms) {
      if (!hasLbl(a)) continue;
      const txt = formatAtomText(a);
      const ch = a.charge || 0;
      const chStr = ch === 1 ? "+" : ch === -1 ? "−" : ch > 1 ? ch + "+" : ch < -1 ? Math.abs(ch) + "−" : "";
      parts.push(`<text x="${tx(a.x).toFixed(2)}" y="${ty(a.y).toFixed(2)}" fill="${lightElCol(a.el)}" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" text-anchor="middle" dominant-baseline="central">${txt}${chStr ? `<tspan dy="-5" font-size="10">${chStr}</tspan>` : ""}</text>`);
    }
    parts.push("</svg>");
    return parts.join("");
  }

  function svgToPngBlob(svgText, dpr = 3) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const w = img.naturalWidth || img.width || 300;
        const h = img.naturalHeight || img.height || 240;
        const c = document.createElement("canvas");
        c.width = Math.round(w * dpr);
        c.height = Math.round(h * dpr);
        const cctx = c.getContext("2d");
        cctx.scale(dpr, dpr);
        cctx.drawImage(img, 0, 0, w, h);
        c.toBlob(b => (b ? resolve(b) : reject(new Error("PNG blob null"))), "image/png");
      };
      img.onerror = () => reject(new Error("SVG load error"));
      img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgText);
    });
  }

  function centerView() {
    const rect = wrap.getBoundingClientRect();
    const w = rect.width || 520, h = rect.height || 380;
    if (!atoms.length) {
      scale = 42; ox = w / 2; oy = h / 2; draw(); return;
    }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const a of atoms) {
      if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
      if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
    }
    const spanX = Math.max(2.6, maxX - minX), spanY = Math.max(2.6, maxY - minY);
    scale = Math.min(54, Math.max(20, Math.min((w - 110) / spanX, (h - 100) / spanY)));
    ox = w / 2 - ((minX + maxX) / 2) * scale;
    oy = h / 2 + ((minY + maxY) / 2) * scale;
    draw();
  }

  function loadGraph(g, keepHistory = false) {
    if (!g || !Array.isArray(g.atoms)) return;
    if (!keepHistory) { undoStack = []; redoStack = []; }
    atoms = g.atoms.map(a => ({
      id: Number(a.id), el: String(a.el || "C"),
      x: Number(a.x || 0), y: Number(a.y || 0), charge: Number(a.charge || 0),
    }));
    bonds = (g.bonds || []).map(b => ({
      a: Number(b.a), b: Number(b.b), order: Number(b.order || 1), stereo: Number(b.stereo || 0),
    }));
    nextId = atoms.reduce((m, a) => Math.max(m, a.id), 0) + 1;
    selectedIds.clear();
    invalidAtomIds.clear();
    centerView();
  }

  function setStatus(ok, text, isPending = false, badIds = null) {
    statusDot.className = isPending ? "dot warn" : (ok ? "dot" : "dot err");
    statusPill.classList.toggle("has-err", !isPending && !ok);
    statusTxt.textContent = text;
    if (Array.isArray(badIds)) {
      invalidAtomIds = new Set(badIds.map(Number));
      draw();
    } else if (ok && !isPending && invalidAtomIds.size > 0) {
      invalidAtomIds.clear();
      draw();
    }
  }

  function scheduleEmit() {
    setStatus(true, "Synchro...", true);
    if (emitTimer) clearTimeout(emitTimer);
    emitTimer = setTimeout(() => {
      window.parent.postMessage({ type: "cs_sketcher_change", graph: { atoms, bonds } }, "*");
    }, 240);
  }

  function eraseAtom(a) {
    pushHistory();
    atoms = atoms.filter(x => x.id !== a.id);
    bonds = bonds.filter(b => b.a !== a.id && b.b !== a.id);
    selectedIds.delete(a.id);
    hoverAtom = null;
    draw(); scheduleEmit();
  }

  function eraseSelection() {
    if (!selectedIds.size) return;
    pushHistory();
    atoms = atoms.filter(a => !selectedIds.has(a.id));
    bonds = bonds.filter(b => !selectedIds.has(b.a) && !selectedIds.has(b.b));
    selectedIds.clear();
    draw(); scheduleEmit();
  }

  function eraseBond(b) {
    pushHistory();
    bonds = bonds.filter(x => x !== b);
    hoverBond = null;
    draw(); scheduleEmit();
  }

  function performMerges() {
    if (!mergeMap.size) return;
    const remapped = new Map();
    for (const [movedId, targetAtom] of mergeMap.entries()) {
      remapped.set(movedId, targetAtom.id);
    }
    const removedIds = new Set(remapped.keys());
    atoms = atoms.filter(a => !removedIds.has(a.id));
    for (const id of removedIds) selectedIds.delete(id);

    const newBonds = [];
    const seen = new Map();
    for (const b of bonds) {
      const na = remapped.get(b.a) ?? b.a;
      const nb = remapped.get(b.b) ?? b.b;
      if (na === nb) continue;
      const key = na < nb ? `${na}_${nb}` : `${nb}_${na}`;
      if (seen.has(key)) {
        const ex = seen.get(key);
        ex.order = Math.max(ex.order, b.order);
      } else {
        const nbObj = { a: na, b: nb, order: b.order, stereo: b.stereo };
        seen.set(key, nbObj);
        newBonds.push(nbObj);
      }
    }
    bonds = newBonds;
    mergeMap.clear();
  }

  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  canvas.addEventListener("contextmenu", e => e.preventDefault());

  canvas.addEventListener("pointerdown", e => {
    closePopovers();
    canvas.setPointerCapture(e.pointerId);
    const p = getPos(e);
    mouseScreen = p; mouseWorld = s2w(p.x, p.y);
    isPointerDown = true; pointerButton = e.button;
    dragStartScreen = p; dragStartWorld = { ...mouseWorld };
    dragMoved = false; dragTargetAtom = null; dragEndWorld = null;
    chainPreview = null; marqueeRect = null; movingOrigin = null; mergeMap.clear();

    if (e.button === 2 || e.button === 1) {
      panStartOx = ox; panStartOy = oy;
      canvas.style.cursor = "grabbing";
      return;
    }

    hoverAtom = findAtomAtScreen(p.x, p.y);
    hoverBond = hoverAtom ? null : findBondAtScreen(p.x, p.y);

    if (tool === "select") {
      if (hoverAtom) {
        if (e.shiftKey) {
          if (selectedIds.has(hoverAtom.id)) selectedIds.delete(hoverAtom.id);
          else selectedIds.add(hoverAtom.id);
        } else if (!selectedIds.has(hoverAtom.id)) {
          selectedIds.clear();
          selectedIds.add(hoverAtom.id);
        }
        movingOrigin = new Map();
        for (const id of selectedIds) {
          const a = atomById(id);
          if (a) movingOrigin.set(id, { x: a.x, y: a.y });
        }
        pushHistory();
        canvas.style.cursor = "grabbing";
      } else {
        if (!e.shiftKey) selectedIds.clear();
        marqueeRect = { x: p.x, y: p.y, w: 0, h: 0 };
      }
      draw();
      return;
    }

    if (tool === "chain") {
      dragFromAtom = hoverAtom;
      return;
    }

    if (tool === "bond1" || tool === "bond2" || tool === "bond3" || tool === "wedge" || tool === "dash") {
      dragFromAtom = hoverAtom;
    }
  });

  canvas.addEventListener("pointermove", e => {
    const p = getPos(e);
    mouseScreen = p; mouseWorld = s2w(p.x, p.y);

    if (isPointerDown) {
      if (pointerButton === 2 || pointerButton === 1) {
        ox = panStartOx + (p.x - dragStartScreen.x);
        oy = panStartOy + (p.y - dragStartScreen.y);
        draw(); return;
      }

      if (tool === "select") {
        if (movingOrigin) {
          const dwx = mouseWorld.x - dragStartWorld.x;
          const dwy = mouseWorld.y - dragStartWorld.y;
          if (dist2(p.x, p.y, dragStartScreen.x, dragStartScreen.y) > 4) dragMoved = true;
          mergeMap.clear();
          for (const [id, orig] of movingOrigin.entries()) {
            const a = atomById(id);
            if (!a) continue;
            a.x = orig.x + dwx;
            a.y = orig.y + dwy;
            const target = findExistingAtomNearWorld(a.x, a.y, 0.38, selectedIds);
            if (target) mergeMap.set(id, target);
          }
          draw(); return;
        }
        if (marqueeRect) {
          marqueeRect = {
            x: Math.min(dragStartScreen.x, p.x),
            y: Math.min(dragStartScreen.y, p.y),
            w: Math.abs(p.x - dragStartScreen.x),
            h: Math.abs(p.y - dragStartScreen.y),
          };
          draw(); return;
        }
      }

      if (tool === "chain") {
        if (dist2(p.x, p.y, dragStartScreen.x, dragStartScreen.y) > 8) {
          dragMoved = true;
          const startW = dragFromAtom ? { x: dragFromAtom.x, y: dragFromAtom.y } : dragStartWorld;
          chainPreview = computeChainPreview(startW, mouseWorld, dragFromAtom);
          draw();
        }
        return;
      }

      if (dist2(p.x, p.y, dragStartScreen.x, dragStartScreen.y) > 9) {
        dragMoved = true;
        const orig = dragFromAtom ? { x: dragFromAtom.x, y: dragFromAtom.y } : dragStartWorld;
        const cand = findAtomAtScreen(p.x, p.y, 18);
        if (cand && cand !== dragFromAtom) {
          dragTargetAtom = cand;
          dragEndWorld = { x: cand.x, y: cand.y };
        } else {
          dragTargetAtom = null;
          const rawAng = Math.atan2(mouseWorld.y - orig.y, mouseWorld.x - orig.x);
          const snapAng = Math.round(rawAng / (Math.PI / 6)) * (Math.PI / 6);
          dragEndWorld = { x: orig.x + BOND_LEN * Math.cos(snapAng), y: orig.y + BOND_LEN * Math.sin(snapAng) };
        }
        draw();
      }
      return;
    }

    hoverAtom = findAtomAtScreen(p.x, p.y);
    hoverBond = hoverAtom ? null : findBondAtScreen(p.x, p.y);
    draw();
  });

  canvas.addEventListener("pointerup", e => {
    if (!isPointerDown) return;
    isPointerDown = false;
    const p = getPos(e);
    mouseScreen = p; mouseWorld = s2w(p.x, p.y);
    canvas.style.cursor = tool === "select" ? "default" : "crosshair";

    if (pointerButton === 2 || pointerButton === 1) return;

    if (tool === "select") {
      if (movingOrigin) {
        if (dragMoved) {
          performMerges();
          draw();
          scheduleEmit();
        } else {
          undoStack.pop();
        }
        movingOrigin = null; mergeMap.clear();
      } else if (marqueeRect) {
        if (marqueeRect.w > 5 || marqueeRect.h > 5) {
          for (const a of atoms) {
            const sp = w2s(a.x, a.y);
            if (sp.x >= marqueeRect.x && sp.x <= marqueeRect.x + marqueeRect.w && sp.y >= marqueeRect.y && sp.y <= marqueeRect.y + marqueeRect.h) {
              selectedIds.add(a.id);
            }
          }
        }
        marqueeRect = null;
        draw();
      }
      return;
    }

    if (tool === "chain") {
      if (dragMoved && chainPreview) applyChainPreview(chainPreview);
      chainPreview = null; dragFromAtom = null;
      return;
    }

    if (tool in RING_META) {
      hoverAtom = findAtomAtScreen(p.x, p.y);
      hoverBond = hoverAtom ? null : findBondAtScreen(p.x, p.y);
      applyRingPreview(computeRingPreview(tool));
      return;
    }

    if (tool in TEMPLATES) {
      hoverAtom = findAtomAtScreen(p.x, p.y);
      applyTemplatePreview(computeTemplatePreview(tool));
      return;
    }

    if (tool === "erase") {
      if (hoverAtom) eraseAtom(hoverAtom);
      else if (hoverBond) eraseBond(hoverBond);
      return;
    }

    if (tool === "charge_plus" || tool === "charge_minus") {
      if (hoverAtom) {
        pushHistory();
        hoverAtom.charge = Math.max(-3, Math.min(3, (hoverAtom.charge || 0) + (tool === "charge_plus" ? 1 : -1)));
        draw(); scheduleEmit();
      }
      return;
    }

    if (tool === "atom") {
      pushHistory();
      if (hoverAtom) hoverAtom.el = activeEl;
      else atoms.push({ id: nextId++, el: activeEl, x: mouseWorld.x, y: mouseWorld.y, charge: 0 });
      draw(); scheduleEmit();
      return;
    }

    const desiredOrder = tool === "bond2" ? 2 : tool === "bond3" ? 3 : 1;
    const desiredStereo = tool === "wedge" ? 1 : tool === "dash" ? 2 : 0;

    if (dragMoved && dragEndWorld) {
      pushHistory();
      let aStart = dragFromAtom;
      if (!aStart) {
        aStart = { id: nextId++, el: "C", x: dragStartWorld.x, y: dragStartWorld.y, charge: 0 };
        atoms.push(aStart);
      }
      let aEnd = dragTargetAtom || findExistingAtomNearWorld(dragEndWorld.x, dragEndWorld.y, 0.32);
      if (!aEnd) {
        aEnd = { id: nextId++, el: "C", x: dragEndWorld.x, y: dragEndWorld.y, charge: 0 };
        atoms.push(aEnd);
      }
      if (aStart.id !== aEnd.id) {
        const ex = findBond(aStart.id, aEnd.id);
        if (ex) { ex.order = desiredOrder; ex.stereo = desiredStereo; ex.a = aStart.id; ex.b = aEnd.id; }
        else bonds.push({ a: aStart.id, b: aEnd.id, order: desiredOrder, stereo: desiredStereo });
      }
      dragFromAtom = null; dragTargetAtom = null; dragEndWorld = null;
      draw(); scheduleEmit();
      return;
    }

    if (hoverBond) {
      pushHistory();
      if (tool === "bond1") { hoverBond.stereo = 0; hoverBond.order = (hoverBond.order % 3) + 1; }
      else if (tool === "bond2" || tool === "bond3") { hoverBond.stereo = 0; hoverBond.order = hoverBond.order === desiredOrder ? 1 : desiredOrder; }
      else if (tool === "wedge" || tool === "dash") {
        hoverBond.order = 1;
        if (hoverBond.stereo === desiredStereo) { const t = hoverBond.a; hoverBond.a = hoverBond.b; hoverBond.b = t; }
        else hoverBond.stereo = desiredStereo;
      }
      draw(); scheduleEmit();
    } else if (hoverAtom) {
      pushHistory();
      const ang = bestOutgoingAngle(hoverAtom);
      const nx = hoverAtom.x + BOND_LEN * Math.cos(ang), ny = hoverAtom.y + BOND_LEN * Math.sin(ang);
      let target = findExistingAtomNearWorld(nx, ny, 0.32);
      if (!target) { target = { id: nextId++, el: "C", x: nx, y: ny, charge: 0 }; atoms.push(target); }
      if (!findBond(hoverAtom.id, target.id)) bonds.push({ a: hoverAtom.id, b: target.id, order: desiredOrder, stereo: desiredStereo });
      draw(); scheduleEmit();
    } else {
      pushHistory();
      const id1 = nextId++, id2 = nextId++, ang = -Math.PI / 6;
      atoms.push({ id: id1, el: "C", x: mouseWorld.x, y: mouseWorld.y, charge: 0 });
      atoms.push({ id: id2, el: "C", x: mouseWorld.x + BOND_LEN * Math.cos(ang), y: mouseWorld.y + BOND_LEN * Math.sin(ang), charge: 0 });
      bonds.push({ a: id1, b: id2, order: desiredOrder, stereo: desiredStereo });
      draw(); scheduleEmit();
    }
  });

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const p = getPos(e), before = s2w(p.x, p.y);
    scale = Math.max(16, Math.min(95, scale * (e.deltaY < 0 ? 1.12 : 0.89)));
    ox = p.x - before.x * scale; oy = p.y + before.y * scale;
    draw();
  }, { passive: false });

  document.getElementById("btn-flip-h").addEventListener("click", () => transformSelection((a, cx) => { a.x = 2 * cx - a.x; }));
  document.getElementById("btn-flip-v").addEventListener("click", () => transformSelection((a, _cx, cy) => { a.y = 2 * cy - a.y; }));
  document.getElementById("btn-rot").addEventListener("click", () => {
    const rad = -Math.PI / 6, ca = Math.cos(rad), sa = Math.sin(rad);
    transformSelection((a, cx, cy) => {
      const dx = a.x - cx, dy = a.y - cy;
      a.x = cx + dx * ca - dy * sa;
      a.y = cy + dx * sa + dy * ca;
    });
  });

  document.getElementById("btn-undo").addEventListener("click", () => {
    if (!undoStack.length) return;
    redoStack.push(cloneState());
    restoreState(undoStack.pop());
  });
  document.getElementById("btn-redo").addEventListener("click", () => {
    if (!redoStack.length) return;
    undoStack.push(cloneState());
    restoreState(redoStack.pop());
  });
  document.getElementById("btn-center").addEventListener("click", centerView);
  document.getElementById("btn-clear").addEventListener("click", () => {
    if (!atoms.length) return;
    pushHistory(); atoms = []; bonds = []; selectedIds.clear(); invalidAtomIds.clear();
    draw(); scheduleEmit();
  });
  document.getElementById("btn-clean").addEventListener("click", async () => {
    if (!atoms.length) return;
    setStatus(true, "Nettoyage RDKit...", true);
    try {
      const r = await fetch("/api/studio/clean2d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ atoms, bonds }),
      });
      const data = await r.json();
      if (data.ok && data.graph) {
        pushHistory();
        loadGraph(data.graph, true);
        if (data.smiles) lastValidSmiles = data.smiles;
        setStatus(true, "Géométrie 2D nettoyée", false, []);
        scheduleEmit();
      } else {
        setStatus(false, data.message || "Valence invalide", false, data.invalid_ids || []);
      }
    } catch (_err) {
      setStatus(false, "Erreur réseau");
    }
  });

  // Copie directe d'une vraie image PNG transparente HD (prête pour Ctrl+V dans Word)
  document.getElementById("btn-copy-img").addEventListener("click", async () => {
    const svg = buildTransparentSvg();
    if (!svg) return;
    try {
      if (navigator.clipboard && typeof window.ClipboardItem === "function") {
        const pngPromise = svgToPngBlob(svg, 3);
        await navigator.clipboard.write([new ClipboardItem({ "image/png": pngPromise })]);
        setStatus(true, "Image copiée ✓");
        window.parent.postMessage({ type: "cs_toast", text: "Image transparente HD copiée (prête pour Word)" }, "*");
        return;
      }
    } catch (_err) {
      // Secours via la fenêtre parente
    }
    window.parent.postMessage({ type: "cs_copy_png", svg }, "*");
  });

  // Téléchargement du fichier vectoriel .svg
  document.getElementById("btn-dl-svg").addEventListener("click", () => {
    const svg = buildTransparentSvg();
    if (!svg) return;
    window.parent.postMessage({ type: "cs_download_svg", svg }, "*");
    setStatus(true, "Fichier .svg exporté");
  });

  document.getElementById("btn-copy-smi").addEventListener("click", () => {
    if (!lastValidSmiles) return;
    window.parent.postMessage({ type: "cs_copy_text", text: lastValidSmiles, label: "SMILES" }, "*");
  });

  window.addEventListener("keydown", e => {
    if (e.key === "Escape") {
      closePopovers();
      selectedIds.clear(); draw(); return;
    }
    const mod = e.ctrlKey || e.metaKey;
    const k = e.key.toLowerCase();
    if (mod && k === "z") { e.preventDefault(); document.getElementById("btn-undo").click(); return; }
    if (mod && k === "y") { e.preventDefault(); document.getElementById("btn-redo").click(); return; }
    if (mod && k === "a") {
      e.preventDefault();
      selectedIds = new Set(atoms.map(a => a.id));
      setTool("select");
      return;
    }
    if (mod && k === "c") {
      const srcIds = selectedIds.size > 0 ? selectedIds : new Set(atoms.map(a => a.id));
      if (!srcIds.size) return;
      e.preventDefault();
      clipboardFrag = {
        atoms: atoms.filter(a => srcIds.has(a.id)).map(a => ({ ...a })),
        bonds: bonds.filter(b => srcIds.has(b.a) && srcIds.has(b.b)).map(b => ({ ...b })),
      };
      setStatus(true, `Copié (${clipboardFrag.atoms.length} at.)`);
      return;
    }
    if (mod && k === "v" && clipboardFrag && clipboardFrag.atoms.length) {
      e.preventDefault();
      pushHistory();
      const mapId = new Map();
      selectedIds.clear();
      for (const a of clipboardFrag.atoms) {
        const nid = nextId++;
        mapId.set(a.id, nid);
        atoms.push({ id: nid, el: a.el, x: a.x + 0.9, y: a.y - 0.7, charge: a.charge });
        selectedIds.add(nid);
      }
      for (const b of clipboardFrag.bonds) {
        bonds.push({ a: mapId.get(b.a), b: mapId.get(b.b), order: b.order, stereo: b.stereo });
      }
      setTool("select");
      selectedIds = new Set(mapId.values());
      draw(); scheduleEmit();
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (selectedIds.size > 0) { e.preventDefault(); eraseSelection(); }
      else if (hoverAtom) { e.preventDefault(); eraseAtom(hoverAtom); }
      else if (hoverBond) { e.preventDefault(); eraseBond(hoverBond); }
      return;
    }

    const keyElMap = { c:"C", n:"N", o:"O", s:"S", p:"P", f:"F", l:"Cl", r:"Br", i:"I" };
    if (hoverAtom && keyElMap[k] && !mod) {
      pushHistory(); hoverAtom.el = keyElMap[k]; draw(); scheduleEmit(); return;
    }
    if (hoverBond && (k === "1" || k === "2" || k === "3")) {
      pushHistory(); hoverBond.order = Number(k); hoverBond.stereo = 0; draw(); scheduleEmit(); return;
    }
    if (!mod) {
      if (k === "v") setTool("select");
      else if (k === "b") setTool("bond1");
      else if (k === "2") setTool("bond2");
      else if (k === "3") setTool("bond3");
      else if (k === "l") setTool("chain");
      else if (k === "w") setTool("wedge");
      else if (k === "d") setTool("dash");
      else if (k === "e") setTool("erase");
    }
  });

  window.addEventListener("message", ev => {
    const d = ev.data;
    if (!d || typeof d !== "object") return;
    if (d.type === "cs_load_graph" && d.graph) {
      loadGraph(d.graph, false);
      if (d.smiles) lastValidSmiles = d.smiles;
      setStatus(true, "Structure chargée", false, []);
    } else if (d.type === "cs_set_theme") {
      document.documentElement.setAttribute("data-theme", d.dark ? "dark" : "light");
      styleElButtons();
      draw();
    } else if (d.type === "cs_sketcher_status") {
      if (d.smiles) lastValidSmiles = d.smiles;
      setStatus(!!d.ok, String(d.text || (d.ok ? "Valence OK" : "Valence ?")), false, d.invalidIds || []);
    } else if (d.type === "cs_select_element" && d.el) {
      activeEl = String(d.el);
      if (!BASE_ELEMENTS.includes(activeEl)) recentCustomEl = activeEl;
      rebuildElBar();
      setTool("atom");
    }
  });

  window.addEventListener("resize", draw);
  rebuildElBar();
  loadGraph(INIT_GRAPH, false);
})();
</script>
</body>
</html>
"""


def build_custom_sketcher_html(initial_smiles: str = "", dark_mode: bool = True) -> str:
    graph = mol_to_graph(initial_smiles)
    theme_str = "dark" if dark_mode else "light"
    return (
        _SKETCHER_HTML_TEMPLATE
        .replace("__INIT_THEME__", theme_str)
        .replace("__INIT_GRAPH__", json.dumps(graph))
        .replace("__INIT_SMILES__", json.dumps(initial_smiles or ""))
    )