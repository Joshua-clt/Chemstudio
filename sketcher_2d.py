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


def graph_to_mol(graph: dict[str, Any]):
    """Reconstruit une molécule RDKit sanitisée depuis le graphe 2D JSON."""
    raw_atoms = graph.get("atoms") or []
    raw_bonds = graph.get("bonds") or []
    if not raw_atoms:
        return None, "Toile vide"

    rw = Chem.RWMol()
    id_to_idx: dict[int, int] = {}
    coords: list[tuple[float, float]] = []

    for a in raw_atoms:
        el = str(a.get("el") or "C").strip()
        try:
            atom = Chem.Atom(el)
        except Exception:
            return None, f"Élément inconnu : {el}"
        atom.SetFormalCharge(int(a.get("charge") or 0))
        idx = rw.AddAtom(atom)
        id_to_idx[int(a["id"])] = idx
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
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "Valence invalide"

    try:
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass

    return mol, "OK"


def graph_to_smiles(graph: dict[str, Any]) -> tuple[str | None, str]:
    mol, msg = graph_to_mol(graph)
    if mol is None:
        return None, msg
    try:
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return smi, "Valence OK"
    except Exception:
        return None, "Structure invalide"


def clean_graph_2d(graph: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str]:
    """Redresse proprement la géométrie 2D via RDKit et renvoie le graphe nettoyé."""
    smi, msg = graph_to_smiles(graph)
    if not smi:
        return None, None, msg
    cleaned = mol_to_graph(smi)
    return cleaned, smi, "Géométrie 2D nettoyée"


_SKETCHER_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr" data-theme="__INIT_THEME__">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: rgba(9, 17, 19, 0.92);
  --bar-bg: rgba(14, 24, 27, 0.88);
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
  --ghost-color: rgba(56, 189, 248, 0.72);
  --ghost-fill: rgba(16, 185, 129, 0.10);
  --hover-ring: rgba(56, 189, 248, 0.35);
}

html[data-theme="light"] {
  --bg: rgba(248, 250, 252, 0.96);
  --bar-bg: rgba(255, 255, 255, 0.92);
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
  --ghost-color: rgba(2, 132, 199, 0.75);
  --ghost-fill: rgba(5, 150, 105, 0.08);
  --hover-ring: rgba(2, 132, 199, 0.28);
}

* { box-sizing: border-box; user-select: none; -webkit-user-select: none; }
html, body {
  margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
  background: var(--canvas-bg); color: var(--txt);
  font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

.shell {
  display: flex; flex-direction: column; width: 100%; height: 100%;
}

.top-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 6px;
  padding: 7px 10px; background: var(--bar-bg);
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.tool-group {
  display: flex; align-items: center; gap: 4px; flex-wrap: wrap;
}
.sep {
  width: 1px; height: 22px; background: var(--border); margin: 0 3px;
}

.tbtn {
  display: inline-flex; align-items: center; justify-content: center; gap: 5px;
  min-width: 30px; height: 30px; padding: 0 8px;
  border-radius: 8px; border: 1px solid var(--border);
  background: var(--btn-bg); color: var(--txt);
  font-family: inherit; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: all 0.15s ease;
}
.tbtn:hover { background: var(--btn-hover); border-color: var(--cyan); }
.tbtn.active {
  background: var(--btn-active-bg); border-color: var(--btn-active-bd);
  color: var(--txt); box-shadow: 0 0 12px rgba(56, 189, 248, 0.16);
}
.tbtn.danger:hover { border-color: var(--danger); color: var(--danger); background: rgba(244, 63, 94, 0.12); }
.tbtn svg { width: 16px; height: 16px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

.mid-area {
  flex: 1; min-height: 0; display: flex; position: relative;
}

.el-bar {
  width: 44px; padding: 6px 5px; background: var(--bar-bg);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; align-items: center; gap: 4px;
  overflow-y: auto;
}
.el-btn {
  width: 32px; height: 29px; padding: 0; border-radius: 8px;
  border: 1px solid var(--border); background: var(--btn-bg);
  font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
  cursor: pointer; transition: all 0.15s ease; color: var(--txt);
}
.el-btn:hover { background: var(--btn-hover); border-color: var(--cyan); }
.el-btn.active { background: var(--btn-active-bg); border-color: var(--btn-active-bd); }

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
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); }
.dot.warn { background: var(--warn); }
.dot.err { background: var(--danger); }

.hint-pill {
  position: absolute; left: 10px; bottom: 10px;
  padding: 4px 9px; border-radius: 8px;
  background: var(--bar-bg); border: 1px solid var(--border);
  font-size: 11px; color: var(--sub); pointer-events: none; opacity: 0.85;
}

.bottom-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 6px;
  padding: 7px 10px; background: var(--bar-bg);
  border-top: 1px solid var(--border); flex-wrap: wrap;
}
.ring-lbl {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--cyan); font-weight: 700;
  margin-right: 4px;
}
</style>
</head>
<body>
<div class="shell">
  <!-- BARRE SUPÉRIEURE : OUTILS DE LIAISON, STÉRÉO & ACTIONS -->
  <div class="top-bar">
    <div class="tool-group">
      <button class="tbtn" data-tool="select" title="Sélectionner / Déplacer (V)">
        <svg viewBox="0 0 24 24"><path d="M5 3l14 9-6 2-3 6z"/></svg>
      </button>
      <button class="tbtn active" data-tool="bond1" title="Liaison simple / Clic liaison = 1→2→3 (B)">
        <svg viewBox="0 0 24 24"><line x1="5" y1="19" x2="19" y2="5"/></svg>
      </button>
      <button class="tbtn" data-tool="bond2" title="Liaison double (2)">
        <svg viewBox="0 0 24 24"><line x1="4" y1="16" x2="16" y2="4"/><line x1="8" y1="20" x2="20" y2="8"/></svg>
      </button>
      <button class="tbtn" data-tool="bond3" title="Liaison triple (3)">
        <svg viewBox="0 0 24 24"><line x1="3" y1="15" x2="15" y2="3"/><line x1="6" y1="18" x2="18" y2="6"/><line x1="9" y1="21" x2="21" y2="9"/></svg>
      </button>
      <button class="tbtn" data-tool="wedge" title="Liaison avant · Coin plein (W)">
        <svg viewBox="0 0 24 24"><polygon points="5,18 19,5 15,12" fill="currentColor" stroke="none"/><polygon points="5,18 16,4 20,8" fill="currentColor"/></svg>
      </button>
      <button class="tbtn" data-tool="dash" title="Liaison arrière · Hachurée (D)">
        <svg viewBox="0 0 24 24">
          <line x1="5" y1="18" x2="7" y2="19"/>
          <line x1="8" y1="14" x2="11" y2="16"/>
          <line x1="11" y1="10" x2="15" y2="13"/>
          <line x1="14" y1="6" x2="19" y2="10"/>
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
      <button class="tbtn" id="btn-undo" title="Annuler (Ctrl+Z)">↶</button>
      <button class="tbtn" id="btn-redo" title="Rétablir (Ctrl+Y)">↷</button>
      <button class="tbtn" id="btn-clean" title="Redresser la géométrie 2D via RDKit">✦ Nettoyer 2D</button>
      <button class="tbtn" id="btn-center" title="Recentrer la vue">⌖</button>
      <button class="tbtn danger" id="btn-clear" title="Effacer toute la toile">✕</button>
    </div>
  </div>

  <!-- ZONE CENTRALE : ÉLÉMENTS À GAUCHE + TOILE CANVAS -->
  <div class="mid-area">
    <div class="el-bar" id="el-bar"></div>
    <div class="canvas-wrap" id="canvas-wrap">
      <canvas id="draw-canvas"></canvas>
      <div class="hint-pill" id="hint-pill">Clic atome : chaîne 120° · Glisser : aimanté 30° · Survol + O/N/S/F/Suppr</div>
      <div class="status-pill" id="status-pill"><span class="dot" id="status-dot"></span><span id="status-txt">Synchro directe</span></div>
    </div>
  </div>

  <!-- BARRE INFÉRIEURE : 6 CYCLES PRÉ-FAITS AVEC APERÇU FANTÔME -->
  <div class="bottom-bar">
    <div class="tool-group">
      <span class="ring-lbl">Cycles</span>
      <button class="tbtn" data-tool="ring_benzene" title="Benzène (clic liaison = fusion · clic atome = greffe)">
        <svg viewBox="0 0 24 24"><polygon points="12,3 20,7.5 20,16.5 12,21 4,16.5 4,7.5"/><circle cx="12" cy="12" r="4.5"/></svg>
        <span>Ph</span>
      </button>
      <button class="tbtn" data-tool="ring_6" title="Cyclohexane">
        <svg viewBox="0 0 24 24"><polygon points="12,3 20,7.5 20,16.5 12,21 4,16.5 4,7.5"/></svg>
        <span>C6</span>
      </button>
      <button class="tbtn" data-tool="ring_5" title="Cyclopentane">
        <svg viewBox="0 0 24 24"><polygon points="12,3 21,9.5 17.5,20 6.5,20 3,9.5"/></svg>
        <span>C5</span>
      </button>
      <button class="tbtn" data-tool="ring_4" title="Cyclobutane">
        <svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14"/></svg>
        <span>C4</span>
      </button>
      <button class="tbtn" data-tool="ring_3" title="Cyclopropane">
        <svg viewBox="0 0 24 24"><polygon points="12,4 20.5,19 3.5,19"/></svg>
        <span>C3</span>
      </button>
      <button class="tbtn" data-tool="ring_7" title="Cycloheptane">
        <svg viewBox="0 0 24 24"><polygon points="12,3 19,6.5 21,14 16,20.5 8,20.5 3,14 5,6.5"/></svg>
        <span>C7</span>
      </button>
    </div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--sub);">Molette : Zoom · Clic droit : Déplacer</span>
  </div>
</div>

<script>
(() => {
  const INIT_GRAPH = __INIT_GRAPH__;
  const ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"];
  const RING_META = {
    ring_benzene: { n: 6, aromatic: true },
    ring_6: { n: 6, aromatic: false },
    ring_5: { n: 5, aromatic: false },
    ring_4: { n: 4, aromatic: false },
    ring_3: { n: 3, aromatic: false },
    ring_7: { n: 7, aromatic: false },
  };
  const BOND_LEN = 1.5; // unités monde RDKit

  const canvas = document.getElementById("draw-canvas");
  const wrap = document.getElementById("canvas-wrap");
  const ctx = canvas.getContext("2d");
  const statusDot = document.getElementById("status-dot");
  const statusTxt = document.getElementById("status-txt");
  const hintPill = document.getElementById("hint-pill");

  let atoms = [];
  let bonds = [];
  let nextId = 1;

  // Caméra (monde -> écran : sx = ox + x * scale, sy = oy - y * scale)
  let scale = 42;
  let ox = 260;
  let oy = 180;

  let tool = "bond1";
  let activeEl = "C";

  // Interaction
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
  let panStartOx = 0;
  let panStartOy = 0;
  let movingAtom = null;

  // Historique Annuler / Rétablir
  let undoStack = [];
  let redoStack = [];
  let emitTimer = null;

  function cloneState() {
    return JSON.stringify({ atoms, bonds, nextId });
  }
  function pushHistory() {
    undoStack.push(cloneState());
    if (undoStack.length > 60) undoStack.shift();
    redoStack = [];
  }
  function restoreState(snapStr) {
    const s = JSON.parse(snapStr);
    atoms = s.atoms || [];
    bonds = s.bonds || [];
    nextId = s.nextId || 1;
    draw();
    scheduleEmit();
  }

  // Construction barre d'éléments
  const elBar = document.getElementById("el-bar");
  ELEMENTS.forEach(el => {
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
    return d ? "#e2e8f0" : "#0f172a";
  }

  function styleElButtons() {
    document.querySelectorAll(".el-btn").forEach(b => {
      const el = b.dataset.el;
      b.style.color = el === "C" ? "var(--txt)" : elColor(el);
      b.classList.toggle("active", tool === "atom" && activeEl === el);
    });
  }

  function setTool(newTool) {
    tool = newTool;
    document.querySelectorAll("[data-tool]").forEach(b => {
      b.classList.toggle("active", b.dataset.tool === tool);
    });
    styleElButtons();
    if (tool in RING_META) {
      hintPill.textContent = "Cycle : survolez une liaison (fusion), un atome (greffe) ou le vide (pose libre)";
    } else if (tool === "select") {
      hintPill.textContent = "Glissez un atome pour le déplacer · Glissez le vide pour parcourir";
    } else if (tool === "erase") {
      hintPill.textContent = "Cliquez sur un atome ou une liaison pour l'effacer";
    } else {
      hintPill.textContent = "Clic atome : chaîne 120° · Glisser : aimanté 30° · Survol + O/N/S/F/Suppr";
    }
    canvas.style.cursor = tool === "select" ? "grab" : "crosshair";
    draw();
  }

  document.querySelectorAll("[data-tool]").forEach(b => {
    b.addEventListener("click", () => setTool(b.dataset.tool));
  });

  // Coordonnées
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

  function findExistingAtomNearWorld(wx, wy, tolWorld = 0.36) {
    let best = null, bestD = tolWorld;
    for (const a of atoms) {
      const d = dist2(wx, wy, a.x, a.y);
      if (d < bestD) { bestD = d; best = a; }
    }
    return best;
  }

  function bestOutgoingAngle(atom) {
    const nbs = neighborsOf(atom.id).map(n => atomById(n.id)).filter(Boolean);
    if (nbs.length === 0) return -Math.PI / 6; // -30 deg
    if (nbs.length === 1) {
      const base = Math.atan2(nbs[0].y - atom.y, nbs[0].x - atom.x);
      // Choisit +120 ou -120 pour favoriser un zig-zag horizontal naturel
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
      if (gap > maxGap) {
        maxGap = gap;
        bestAng = a1 + gap / 2;
      }
    }
    return bestAng;
  }

  function computeRingPreview(ringKey) {
    const meta = RING_META[ringKey];
    if (!meta) return null;
    const n = meta.n;
    const aromatic = meta.aromatic;

    // 1. Fusion sur une liaison survolée
    if (hoverBond) {
      const a1 = atomById(hoverBond.a);
      const a2 = atomById(hoverBond.b);
      if (a1 && a2) {
        const dx = a2.x - a1.x, dy = a2.y - a1.y;
        const edgeLen = Math.hypot(dx, dy) || BOND_LEN;
        const mx = (a1.x + a2.x) / 2, my = (a1.y + a2.y) / 2;
        const nx = -dy / edgeLen, ny = dx / edgeLen;
        // Choisir le côté où pointe la souris (ou loin des autres voisins)
        let side = (mouseWorld.x - mx) * nx + (mouseWorld.y - my) * ny >= 0 ? 1 : -1;
        const apothem = edgeLen / (2 * Math.tan(Math.PI / n));
        const cx = mx + side * nx * apothem;
        const cy = my + side * ny * apothem;
        const r = edgeLen / (2 * Math.sin(Math.PI / n));
        const ang1 = Math.atan2(a1.y - cy, a1.x - cx);
        const ang2 = Math.atan2(a2.y - cy, a2.x - cx);
        let step = ang2 - ang1;
        while (step > Math.PI) step -= 2 * Math.PI;
        while (step < -Math.PI) step += 2 * Math.PI;
        const dir = step >= 0 ? -1 : 1; // sens opposé pour parcourir les nouveaux sommets
        const pts = [{ x: a1.x, y: a1.y, existingId: a1.id }, { x: a2.x, y: a2.y, existingId: a2.id }];
        const angStep = (2 * Math.PI) / n;
        for (let k = 1; k <= n - 2; k++) {
          const ang = ang2 - dir * k * angStep;
          const wx = cx + r * Math.cos(ang);
          const wy = cy + r * Math.sin(ang);
          const ex = findExistingAtomNearWorld(wx, wy, 0.35);
          pts.push({ x: ex ? ex.x : wx, y: ex ? ex.y : wy, existingId: ex ? ex.id : null });
        }
        // Ordre des liaisons (évite carbone pentavalent lors de la fusion benzène + benzène)
        const sharedOrder = hoverBond.order || 1;
        const edgeOrders = [];
        for (let i = 0; i < n; i++) {
          if (i === 0) {
            edgeOrders.push(sharedOrder); // arête partagée déjà existante
          } else if (aromatic) {
            // Si arête partagée = 2 -> i=1:1, i=2:2, i=3:1, i=4:2, i=5:1
            // Si arête partagée = 1 -> i=1:2, i=2:1, i=3:2, i=4:1, i=5:2
            edgeOrders.push(sharedOrder === 2 ? (i % 2 === 0 ? 2 : 1) : (i % 2 === 1 ? 2 : 1));
          } else {
            edgeOrders.push(1);
          }
        }
        return { mode: "fuse", pts, edgeOrders, anchorBond: hoverBond };
      }
    }

    // 2. Greffe sur un atome survolé (ex : ajout phényle / cyclohexyle)
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
        const wx = cx + r * Math.cos(a);
        const wy = cy + r * Math.sin(a);
        const ex = findExistingAtomNearWorld(wx, wy, 0.32);
        pts.push({ x: ex ? ex.x : wx, y: ex ? ex.y : wy, existingId: ex ? ex.id : null });
      }
      const edgeOrders = Array.from({ length: n }, (_, i) => (aromatic && i % 2 === 1 ? 2 : 1));
      return { mode: "attach", pts, edgeOrders, attachFrom: hoverAtom };
    }

    // 3. Pose libre dans le vide
    const r = BOND_LEN / (2 * Math.sin(Math.PI / n));
    const offsetAng = n % 2 === 0 ? Math.PI / 2 : Math.PI / 2;
    const pts = [];
    for (let k = 0; k < n; k++) {
      const a = offsetAng + (k * 2 * Math.PI) / n;
      pts.push({ x: mouseWorld.x + r * Math.cos(a), y: mouseWorld.y + r * Math.sin(a), existingId: null });
    }
    const edgeOrders = Array.from({ length: n }, (_, i) => (aromatic && i % 2 === 0 ? 2 : 1));
    return { mode: "free", pts, edgeOrders };
  }

  function applyRingPreview(prev) {
    if (!prev) return;
    pushHistory();
    const ids = prev.pts.map(pt => {
      if (pt.existingId !== null && atomById(pt.existingId)) return pt.existingId;
      const id = nextId++;
      atoms.push({ id, el: "C", x: pt.x, y: pt.y, charge: 0 });
      return id;
    });
    if (prev.mode === "attach" && prev.attachFrom) {
      if (!findBond(prev.attachFrom.id, ids[0])) {
        bonds.push({ a: prev.attachFrom.id, b: ids[0], order: 1, stereo: 0 });
      }
    }
    const n = ids.length;
    for (let i = 0; i < n; i++) {
      if (prev.mode === "fuse" && i === 0) continue; // arête déjà présente
      const u = ids[i];
      const v = ids[(i + 1) % n];
      const existing = findBond(u, v);
      if (!existing) {
        bonds.push({ a: u, b: v, order: prev.edgeOrders[i], stereo: 0 });
      }
    }
    draw();
    scheduleEmit();
  }

  function implicitHCount(atom) {
    const baseVal = { C: 4, N: 3, O: 2, S: 2, P: 3, F: 1, Cl: 1, Br: 1, I: 1, B: 3 };
    let v = baseVal[atom.el];
    if (v === undefined) return 0;
    const ch = atom.charge || 0;
    if (atom.el === "C") v = Math.max(0, 4 - Math.abs(ch));
    else if (atom.el === "N" || atom.el === "P") v = Math.max(0, 3 + ch);
    else if (atom.el === "O" || atom.el === "S") v = Math.max(0, 2 + ch);
    else v = Math.max(0, v + ch);
    const bondSum = neighborsOf(atom.id).reduce((acc, n) => acc + (n.bond.order || 1), 0);
    return Math.max(0, v - bondSum);
  }

  function bondSideHint(b) {
    // Indique vers quel côté décaler la 2e ligne d'une liaison double (vers l'intérieur du cycle/voisins)
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
        const cx = p1.x + dx * t;
        const cy = p1.y + dy * t;
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
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
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
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    } else if (b.order === 2) {
      const side = bondSideHint(b);
      if (side === 0) {
        const off = 3.0;
        ctx.beginPath();
        ctx.moveTo(p1.x + nx * off, p1.y + ny * off);
        ctx.lineTo(p2.x + nx * off, p2.y + ny * off);
        ctx.moveTo(p1.x - nx * off, p1.y - ny * off);
        ctx.lineTo(p2.x - nx * off, p2.y - ny * off);
        ctx.stroke();
      } else {
        // Note : Y écran est inversé -> inverser side pour l'orientation écran
        const sOff = -side * 5.2;
        const inset = Math.min(9, len * 0.16);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.moveTo(p1.x + ux * inset + nx * sOff, p1.y + uy * inset + ny * sOff);
        ctx.lineTo(p2.x - ux * inset + nx * sOff, p2.y - uy * inset + ny * sOff);
        ctx.stroke();
      }
    } else if (b.order === 3) {
      const off = 4.6;
      const inset = Math.min(6, len * 0.10);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.moveTo(p1.x + ux * inset + nx * off, p1.y + uy * inset + ny * off);
      ctx.lineTo(p2.x - ux * inset + nx * off, p2.y - uy * inset + ny * off);
      ctx.moveTo(p1.x + ux * inset - nx * off, p1.y + uy * inset - ny * off);
      ctx.lineTo(p2.x - ux * inset - nx * off, p2.y - uy * inset - ny * off);
      ctx.stroke();
    }
  }

  function drawAtomLabel(a, isHovered, isTarget) {
    const p = w2s(a.x, a.y);
    const nbs = neighborsOf(a.id);
    const showLabel = a.el !== "C" || a.charge !== 0 || nbs.length === 0;

    if (isHovered || isTarget) {
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

    const hCnt = implicitHCount(a);
    let txt = a.el;
    if (hCnt === 1) txt += "H";
    else if (hCnt > 1) txt += "H" + (hCnt === 2 ? "₂" : hCnt === 3 ? "₃" : hCnt === 4 ? "₄" : String(hCnt));

    const ch = a.charge || 0;
    const chStr = ch === 1 ? "+" : ch === -1 ? "−" : ch > 1 ? ch + "+" : ch < -1 ? Math.abs(ch) + "−" : "";

    ctx.save();
    ctx.font = "700 14px 'Plus Jakarta Sans', sans-serif";
    const mW = ctx.measureText(txt).width;
    const bgR = Math.max(10, mW / 2 + 4);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--canvas-bg").trim();
    ctx.beginPath();
    ctx.arc(p.x, p.y, bgR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = elColor(a.el);
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

    // Liaisons
    for (const b of bonds) {
      drawBondObj(b, hoverBond === b);
    }

    // Aperçu fantôme pendant clic-glisser liaison
    if (isPointerDown && pointerButton === 0 && dragMoved && dragEndWorld && (tool === "bond1" || tool === "bond2" || tool === "bond3" || tool === "wedge" || tool === "dash")) {
      const startW = dragFromAtom ? { x: dragFromAtom.x, y: dragFromAtom.y } : dragStartWorld;
      const endW = dragTargetAtom ? { x: dragTargetAtom.x, y: dragTargetAtom.y } : dragEndWorld;
      const p1 = w2s(startW.x, startW.y), p2 = w2s(endW.x, endW.y);
      ctx.save();
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--ghost-color").trim();
      ctx.lineWidth = 2.6;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      ctx.restore();
    }

    // Aperçu fantôme pour les cycles pré-faits
    if (!isPointerDown && (tool in RING_META) && mouseScreen.x > 0 && mouseScreen.y > 0) {
      const prev = computeRingPreview(tool);
      if (prev) {
        const gCol = getComputedStyle(document.documentElement).getPropertyValue("--ghost-color").trim();
        const gFill = getComputedStyle(document.documentElement).getPropertyValue("--ghost-fill").trim();
        const sPts = prev.pts.map(pt => w2s(pt.x, pt.y));
        ctx.save();
        ctx.beginPath();
        sPts.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.closePath();
        ctx.fillStyle = gFill;
        ctx.fill();
        ctx.strokeStyle = gCol;
        ctx.lineWidth = 2.1;
        ctx.setLineDash([5, 3]);
        ctx.stroke();

        if (prev.mode === "attach" && prev.attachFrom) {
          const pA = w2s(prev.attachFrom.x, prev.attachFrom.y);
          ctx.beginPath();
          ctx.moveTo(pA.x, pA.y);
          ctx.lineTo(sPts[0].x, sPts[0].y);
          ctx.stroke();
        }
        // Cercle interne pour le benzène fantôme
        if (tool === "ring_benzene") {
          const cx = sPts.reduce((s, p) => s + p.x, 0) / sPts.length;
          const cy = sPts.reduce((s, p) => s + p.y, 0) / sPts.length;
          ctx.beginPath();
          ctx.arc(cx, cy, scale * 0.85, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // Atomes
    for (const a of atoms) {
      drawAtomLabel(a, hoverAtom === a, dragTargetAtom === a);
    }
  }

  function centerView() {
    const rect = wrap.getBoundingClientRect();
    const w = rect.width || 520;
    const h = rect.height || 380;
    if (atoms.length === 0) {
      scale = 42; ox = w / 2; oy = h / 2;
      draw();
      return;
    }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const a of atoms) {
      if (a.x < minX) minX = a.x;
      if (a.x > maxX) maxX = a.x;
      if (a.y < minY) minY = a.y;
      if (a.y > maxY) maxY = a.y;
    }
    const spanX = Math.max(2.6, maxX - minX);
    const spanY = Math.max(2.6, maxY - minY);
    scale = Math.min(54, Math.max(22, Math.min((w - 110) / spanX, (h - 100) / spanY)));
    ox = w / 2 - ((minX + maxX) / 2) * scale;
    oy = h / 2 + ((minY + maxY) / 2) * scale;
    draw();
  }

  function loadGraph(g, keepHistory = false) {
    if (!g || !Array.isArray(g.atoms)) return;
    if (!keepHistory) {
      undoStack = [];
      redoStack = [];
    }
    atoms = g.atoms.map(a => ({
      id: Number(a.id),
      el: String(a.el || "C"),
      x: Number(a.x || 0),
      y: Number(a.y || 0),
      charge: Number(a.charge || 0),
    }));
    bonds = (g.bonds || []).map(b => ({
      a: Number(b.a),
      b: Number(b.b),
      order: Number(b.order || 1),
      stereo: Number(b.stereo || 0),
    }));
    nextId = atoms.reduce((m, a) => Math.max(m, a.id), 0) + 1;
    centerView();
  }

  function setStatus(ok, text, isPending = false) {
    statusDot.className = isPending ? "dot warn" : (ok ? "dot" : "dot err");
    statusTxt.textContent = text;
  }

  function scheduleEmit() {
    setStatus(true, "Synchro...", true);
    if (emitTimer) clearTimeout(emitTimer);
    emitTimer = setTimeout(() => {
      window.parent.postMessage({
        type: "cs_sketcher_change",
        graph: { atoms, bonds },
      }, "*");
    }, 240);
  }

  function eraseAtom(a) {
    pushHistory();
    atoms = atoms.filter(x => x.id !== a.id);
    bonds = bonds.filter(b => b.a !== a.id && b.b !== a.id);
    hoverAtom = null;
    draw();
    scheduleEmit();
  }

  function eraseBond(b) {
    pushHistory();
    bonds = bonds.filter(x => x !== b);
    hoverBond = null;
    draw();
    scheduleEmit();
  }

  // Événements souris
  function getPos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  canvas.addEventListener("contextmenu", e => e.preventDefault());

  canvas.addEventListener("pointerdown", e => {
    canvas.setPointerCapture(e.pointerId);
    const p = getPos(e);
    mouseScreen = p;
    mouseWorld = s2w(p.x, p.y);
    isPointerDown = true;
    pointerButton = e.button;
    dragStartScreen = p;
    dragStartWorld = { ...mouseWorld };
    dragMoved = false;
    dragTargetAtom = null;
    dragEndWorld = null;

    if (e.button === 2 || e.button === 1) {
      panStartOx = ox; panStartOy = oy;
      canvas.style.cursor = "grabbing";
      return;
    }

    hoverAtom = findAtomAtScreen(p.x, p.y);
    hoverBond = hoverAtom ? null : findBondAtScreen(p.x, p.y);

    if (tool === "select") {
      if (hoverAtom) {
        movingAtom = hoverAtom;
        pushHistory();
      } else {
        panStartOx = ox; panStartOy = oy;
      }
      canvas.style.cursor = "grabbing";
      return;
    }

    if (tool === "bond1" || tool === "bond2" || tool === "bond3" || tool === "wedge" || tool === "dash") {
      dragFromAtom = hoverAtom;
    }
  });

  canvas.addEventListener("pointermove", e => {
    const p = getPos(e);
    mouseScreen = p;
    mouseWorld = s2w(p.x, p.y);

    if (isPointerDown) {
      if (pointerButton === 2 || pointerButton === 1 || (tool === "select" && !movingAtom)) {
        ox = panStartOx + (p.x - dragStartScreen.x);
        oy = panStartOy + (p.y - dragStartScreen.y);
        draw();
        return;
      }
      if (tool === "select" && movingAtom) {
        movingAtom.x = mouseWorld.x;
        movingAtom.y = mouseWorld.y;
        dragMoved = true;
        draw();
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
          const snapAng = Math.round(rawAng / (Math.PI / 6)) * (Math.PI / 6); // aimanté tous les 30°
          dragEndWorld = {
            x: orig.x + BOND_LEN * Math.cos(snapAng),
            y: orig.y + BOND_LEN * Math.sin(snapAng),
          };
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
    mouseScreen = p;
    mouseWorld = s2w(p.x, p.y);
    canvas.style.cursor = tool === "select" ? "grab" : "crosshair";

    if (pointerButton === 2 || pointerButton === 1) return;

    if (tool === "select") {
      if (movingAtom && dragMoved) scheduleEmit();
      movingAtom = null;
      return;
    }

    // Outils Cycles
    if (tool in RING_META) {
      hoverAtom = findAtomAtScreen(p.x, p.y);
      hoverBond = hoverAtom ? null : findBondAtScreen(p.x, p.y);
      applyRingPreview(computeRingPreview(tool));
      return;
    }

    // Gomme
    if (tool === "erase") {
      if (hoverAtom) eraseAtom(hoverAtom);
      else if (hoverBond) eraseBond(hoverBond);
      return;
    }

    // Charges
    if (tool === "charge_plus" || tool === "charge_minus") {
      if (hoverAtom) {
        pushHistory();
        hoverAtom.charge = (hoverAtom.charge || 0) + (tool === "charge_plus" ? 1 : -1);
        hoverAtom.charge = Math.max(-3, Math.min(3, hoverAtom.charge));
        draw();
        scheduleEmit();
      }
      return;
    }

    // Outil Atome (C, N, O, S...)
    if (tool === "atom") {
      pushHistory();
      if (hoverAtom) {
        hoverAtom.el = activeEl;
      } else {
        const id = nextId++;
        atoms.push({ id, el: activeEl, x: mouseWorld.x, y: mouseWorld.y, charge: 0 });
      }
      draw();
      scheduleEmit();
      return;
    }

    // Outils Liaison (bond1, bond2, bond3, wedge, dash)
    const desiredOrder = tool === "bond2" ? 2 : tool === "bond3" ? 3 : 1;
    const desiredStereo = tool === "wedge" ? 1 : tool === "dash" ? 2 : 0;

    if (dragMoved && dragEndWorld) {
      pushHistory();
      let aStart = dragFromAtom;
      if (!aStart) {
        const id = nextId++;
        aStart = { id, el: "C", x: dragStartWorld.x, y: dragStartWorld.y, charge: 0 };
        atoms.push(aStart);
      }
      let aEnd = dragTargetAtom || findExistingAtomNearWorld(dragEndWorld.x, dragEndWorld.y, 0.32);
      if (!aEnd) {
        const id = nextId++;
        aEnd = { id, el: "C", x: dragEndWorld.x, y: dragEndWorld.y, charge: 0 };
        atoms.push(aEnd);
      }
      if (aStart.id !== aEnd.id) {
        const ex = findBond(aStart.id, aEnd.id);
        if (ex) {
          ex.order = desiredOrder;
          ex.stereo = desiredStereo;
          ex.a = aStart.id; ex.b = aEnd.id;
        } else {
          bonds.push({ a: aStart.id, b: aEnd.id, order: desiredOrder, stereo: desiredStereo });
        }
      }
      dragFromAtom = null; dragTargetAtom = null; dragEndWorld = null;
      draw();
      scheduleEmit();
      return;
    }

    // Simple clic sans glisser
    if (hoverBond) {
      pushHistory();
      if (tool === "bond1") {
        hoverBond.stereo = 0;
        hoverBond.order = (hoverBond.order % 3) + 1;
      } else if (tool === "bond2" || tool === "bond3") {
        hoverBond.stereo = 0;
        hoverBond.order = hoverBond.order === desiredOrder ? 1 : desiredOrder;
      } else if (tool === "wedge" || tool === "dash") {
        hoverBond.order = 1;
        if (hoverBond.stereo === desiredStereo) {
          const tmp = hoverBond.a; hoverBond.a = hoverBond.b; hoverBond.b = tmp;
        } else {
          hoverBond.stereo = desiredStereo;
        }
      }
      draw();
      scheduleEmit();
    } else if (hoverAtom) {
      // Prolonge automatiquement la chaîne à 120°
      pushHistory();
      const ang = bestOutgoingAngle(hoverAtom);
      const nx = hoverAtom.x + BOND_LEN * Math.cos(ang);
      const ny = hoverAtom.y + BOND_LEN * Math.sin(ang);
      let target = findExistingAtomNearWorld(nx, ny, 0.32);
      if (!target) {
        const id = nextId++;
        target = { id, el: "C", x: nx, y: ny, charge: 0 };
        atoms.push(target);
      }
      if (!findBond(hoverAtom.id, target.id)) {
        bonds.push({ a: hoverAtom.id, b: target.id, order: desiredOrder, stereo: desiredStereo });
      }
      draw();
      scheduleEmit();
    } else {
      // Clic dans le vide : crée une liaison initiale orientée à -30°
      pushHistory();
      const id1 = nextId++, id2 = nextId++;
      const ang = -Math.PI / 6;
      atoms.push({ id: id1, el: "C", x: mouseWorld.x, y: mouseWorld.y, charge: 0 });
      atoms.push({ id: id2, el: "C", x: mouseWorld.x + BOND_LEN * Math.cos(ang), y: mouseWorld.y + BOND_LEN * Math.sin(ang), charge: 0 });
      bonds.push({ a: id1, b: id2, order: desiredOrder, stereo: desiredStereo });
      draw();
      scheduleEmit();
    }
  });

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const p = getPos(e);
    const before = s2w(p.x, p.y);
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    scale = Math.max(16, Math.min(95, scale * factor));
    ox = p.x - before.x * scale;
    oy = p.y + before.y * scale;
    draw();
  }, { passive: false });

  // Boutons d'action
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
    pushHistory();
    atoms = []; bonds = [];
    draw();
    scheduleEmit();
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
        setStatus(true, "Géométrie 2D nettoyée");
        scheduleEmit();
      } else {
        setStatus(false, data.message || "Valence invalide");
      }
    } catch (err) {
      setStatus(false, "Erreur réseau");
    }
  });

  // Raccourcis clavier (façon ChemDraw au survol)
  window.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      document.getElementById("btn-undo").click();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
      e.preventDefault();
      document.getElementById("btn-redo").click();
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (hoverAtom) { e.preventDefault(); eraseAtom(hoverAtom); }
      else if (hoverBond) { e.preventDefault(); eraseBond(hoverBond); }
      return;
    }
    const k = e.key.toLowerCase();
    const keyElMap = { c: "C", n: "N", o: "O", s: "S", p: "P", f: "F", l: "Cl", r: "Br", i: "I" };
    if (hoverAtom && keyElMap[k]) {
      pushHistory();
      hoverAtom.el = keyElMap[k];
      draw();
      scheduleEmit();
      return;
    }
    if (hoverBond && (k === "1" || k === "2" || k === "3")) {
      pushHistory();
      hoverBond.order = Number(k);
      hoverBond.stereo = 0;
      draw();
      scheduleEmit();
      return;
    }
    if (k === "v") setTool("select");
    else if (k === "b") setTool("bond1");
    else if (k === "2") setTool("bond2");
    else if (k === "3") setTool("bond3");
    else if (k === "w") setTool("wedge");
    else if (k === "d") setTool("dash");
    else if (k === "e") setTool("erase");
  });

  // Communication descendante depuis ChemStudio (thème, chargement externe, statut)
  window.addEventListener("message", ev => {
    const d = ev.data;
    if (!d || typeof d !== "object") return;
    if (d.type === "cs_load_graph" && d.graph) {
      loadGraph(d.graph, false);
      setStatus(true, "Structure chargée");
    } else if (d.type === "cs_set_theme") {
      document.documentElement.setAttribute("data-theme", d.dark ? "dark" : "light");
      styleElButtons();
      draw();
    } else if (d.type === "cs_sketcher_status") {
      setStatus(!!d.ok, String(d.text || (d.ok ? "Valence OK" : "Valence ?")));
    }
  });

  window.addEventListener("resize", draw);
  styleElButtons();
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
    )