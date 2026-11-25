import html
import json
from typing import Any
import urllib.parse

from nicegui import run, ui
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from chem_engine import (
    REACTION_TEMPLATES,
    clean_val,
    compute_dipole_moment_3d,
    compute_quantum_orbitals,
    enumerate_tautomers_svgs,
    estimate_chemprop,
    extract_single_smiles,
    generate_3d_mol,
    get_bond_length_3d,
    get_chemical_identifiers,
    overlay_molecules_3d,
    parse_smiles,
    predict_reaction_product,
    render_2d_svg,
    render_reaction_svg,
    resolve_query_to_smiles,
    run_retrosynthesis_engine,
    search_pubchem_markush_analogues,
)
from sketcher_2d import mol_to_graph
from theme import safe_html


def _btn(label_html: str, classes: str = "glass-btn", on_click=None, style: str = "", props: str = ""):
    el = ui.element("button").classes(classes)
    if style:
        el.style(style)
    if props:
        el.props(props)
    if on_click is not None:
        el.on("click", on_click)
    with el:
        safe_html(label_html)
    return el


def _sync_rxn_smiles(state: dict[str, Any]):
    r_list = [s for s in state.get("rxn_reactants", []) if clean_val(s)]
    p_list = [s for s in state.get("rxn_products", []) if clean_val(s)]
    r_part = ".".join(r_list) if r_list else "CC(=O)O"
    p_part = ".".join(p_list) if p_list else "CC(=O)OCC"
    state["rxn_smiles"] = f"{r_part}>>{p_part}"


def build_studio_views(state: dict[str, Any], on_molecule_change, on_basket_change):
    # Modale Table de Fukui (Onglet 1 · Quantique)
    fukui_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
    with fukui_dialog:
        fukui_body = ui.element("div").classes("cb-modal-card")

    def open_fukui_modal(fukui_rows: list[dict[str, Any]], omega_val: float):
        fukui_body.clear()
        with fukui_body:
            with ui.element("div").classes("modal-head"):
                with ui.element("div"):
                    safe_html('<div class="section-kicker">CHIMIE QUANTIQUE · HÜCKEL ÉTENDU</div>')
                    safe_html(
                        f'<h2 style="margin:4px 0 0 0;font-size:1.25rem;">Table des Indices de Fukui</h2>'
                        f'<span class="muted-small">Indice d\'électrophilie global ω = {omega_val} eV</span>'
                    )
                _btn("Fermer ✕", on_click=lambda _e: fukui_dialog.close())

            trs = []
            for r in fukui_rows:
                trs.append(
                    f"<tr>"
                    f"<td><strong>{html.escape(str(r['atome']))}</strong></td>"
                    f"<td style='font-family:\"JetBrains Mono\",monospace;'>{r['coeff_homo']}</td>"
                    f"<td style='font-family:\"JetBrains Mono\",monospace;'>{r['coeff_lumo']}</td>"
                    f"<td style='font-family:\"JetBrains Mono\",monospace;color:var(--cyan);'>{r['f_minus']}</td>"
                    f"<td style='font-family:\"JetBrains Mono\",monospace;color:var(--accent);'>{r['f_plus']}</td>"
                    f"</tr>"
                )
            safe_html(
                '<div class="data-table-wrap" style="max-height:420px;overflow-y:auto;">'
                '<table class="data-table">'
                '<thead><tr><th>Atome</th><th>Coeff HOMO</th><th>Coeff LUMO</th><th>Fukui f⁻ (Nucléophile)</th><th>Fukui f⁺ (Électrophile)</th></tr></thead>'
                f'<tbody>{"".join(trs)}</tbody></table></div>'
            )
        fukui_dialog.open()

    # ==================================================================
    # ONGLET 1 : ATELIER 2D MAISON (PERSISTANT), 3D, QUANTIQUE & OVERLAY
    # ==================================================================
    def render_tab_2d3d():
        view_cfg: dict[str, Any] = {
            "mode_3d": "standard",  # standard | quantum | overlay
            "style": "Ball & Stick",
            "ff": "MMFF94",
            "seed": 42,
            "show_h": True,
            "charges": False,
            "spin": False,
            "bond_ai": 0,
            "bond_aj": 1,
            "bond_dist": 0.0,
            "quantum_surface": "HOMO",
            "overlay_target": "",
            "overlay_custom": "",
        }

        init_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
        init_dark = 1 if state.get("dark", True) else 0
        state["last_sketcher_smi"] = init_smi

        with ui.element("div").classes("modal-grid"):
            # --- COLONNE GAUCHE : TOILE 2D PERSISTANTE (NE SE RECHARGE JAMAIS AU DESSIN) ---
            with ui.element("div").classes("hud-card"):
                safe_html(
                    '<div class="section-kicker">ATELIER MOLÉCULAIRE 2D · SYNCHRO EN DIRECT</div>'
                    '<h2>Dessin &amp; Édition de Structure</h2>'
                    '<p class="muted-small" style="margin-bottom:10px;">Chaque modification met automatiquement à jour la 3D, la formule et les spectres.</p>'
                )
                sketcher_src = f"/api/studio/sketcher?smiles={urllib.parse.quote(init_smi)}&dark={init_dark}"
                safe_html(
                    f'<iframe id="cs-sketcher-iframe" class="sketcher-frame" '
                    f'src="{sketcher_src}" title="Dessinateur 2D ChemStudio"></iframe>'
                )

                with ui.element("div").style("display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-top:10px;"):
                    in_direct_smi = ui.input(
                        label="SMILES direct (édition ou collage)",
                        value=init_smi,
                    ).props("outlined dense")

                    def apply_direct_smi(_e=None):
                        val = clean_val(in_direct_smi.value)
                        if not val:
                            return
                        if parse_smiles(val) is None:
                            ui.notify("SMILES invalide.", type="negative")
                            return
                        on_molecule_change(val)

                    in_direct_smi.on("keydown.enter", apply_direct_smi)
                    _btn("Charger SMILES", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=apply_direct_smi)

            # --- COLONNE DROITE : 3D / QUANTIQUE / SUPERPOSITION (RAFRAÎCHIE EN DIRECT) ---
            right_card = ui.element("div").classes("hud-card")

        def refresh_tab():
            raw_smi = clean_val(state.get("current_smiles"))
            single_smi = extract_single_smiles(raw_smi)
            mol_main = parse_smiles(single_smi)
            d_flag = 1 if state.get("dark", True) else 0

            # Synchronise le champ SMILES et pousse vers la toile uniquement si changement externe
            if in_direct_smi.value != single_smi:
                in_direct_smi.value = single_smi

            if single_smi != state.get("last_sketcher_smi"):
                state["last_sketcher_smi"] = single_smi
                graph_payload = mol_to_graph(single_smi)
                ui.run_javascript(f"window.csPushGraphToSketcher({json.dumps(graph_payload)});")

            right_card.clear()
            with right_card:
                safe_html('<div class="section-kicker">VISUALISATION 3D &amp; CHIMIE COMPUTATIONNELLE</div>')
                with ui.element("div").classes("sub-tabs").style("margin-top:8px;"):
                    for m_key, m_lbl in (
                        ("standard", "Modèle 3D"),
                        ("quantum", "Quantique (HOMO / LUMO)"),
                        ("overlay", "Superposition 3D"),
                    ):
                        cls = "glass-btn glass-btn-accent" if view_cfg["mode_3d"] == m_key else "glass-btn"
                        _btn(
                            m_lbl,
                            classes=cls,
                            style="padding:6px 12px;font-size:0.79rem;",
                            on_click=lambda _e, k=m_key: (view_cfg.update(mode_3d=k), refresh_tab()),
                        )

                if not mol_main:
                    safe_html('<p class="muted-text">Structure chimique invalide. Dessinez ou chargez une molécule valide.</p>')
                    return

                # --- SOUS-MODE 1 : MODÈLE 3D STANDARD ---
                if view_cfg["mode_3d"] == "standard":
                    mol_3d = generate_3d_mol(single_smi, force_field=str(view_cfg["ff"]), seed=int(view_cfg["seed"]))
                    n_at = mol_3d.GetNumAtoms() if mol_3d else 2
                    ai_clamped = min(max(0, int(view_cfg["bond_ai"])), max(0, n_at - 1))
                    aj_clamped = min(max(0, int(view_cfg["bond_aj"])), max(0, n_at - 1))
                    if aj_clamped == ai_clamped and n_at > 1:
                        aj_clamped = 1 if ai_clamped == 0 else 0
                    cur_len = get_bond_length_3d(mol_3d, ai_clamped, aj_clamped) if mol_3d else 1.5

                    with ui.expansion("Réglages 3D, Conformères & Distances interatomiques", value=False).style("width:100%;margin-bottom:10px;"):
                        with ui.element("div").style("display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px;"):
                            sel_st = ui.select(["Ball & Stick", "Stick", "Spacefill"], value=view_cfg["style"], label="Style").props("outlined dense")
                            sel_ff = ui.select(["MMFF94", "UFF"], value=view_cfg["ff"], label="Champ de force").props("outlined dense")
                            in_seed = ui.number(label="Conformère #", value=int(view_cfg["seed"]), min=1, max=100, step=1).props("outlined dense")

                        with ui.element("div").style("display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;"):
                            chk_h = ui.checkbox("Hydrogènes", value=bool(view_cfg["show_h"]))
                            chk_q = ui.checkbox("Charges Gasteiger (δ+/δ-)", value=bool(view_cfg["charges"]))
                            chk_spin = ui.checkbox("Rotation continue", value=bool(view_cfg["spin"]))

                        safe_html(f'<div class="muted-small" style="margin-bottom:6px;">Modifier une distance 3D (distance actuelle A{ai_clamped}–A{aj_clamped} : <strong>{cur_len:.2f} Å</strong>)</div>')
                        with ui.element("div").style("display:grid;grid-template-columns:1fr 1fr 1.3fr auto;gap:8px;align-items:center;"):
                            in_ai = ui.number(label="Atome A", value=ai_clamped, min=0, max=max(0, n_at - 1), step=1).props("outlined dense")
                            in_aj = ui.number(label="Atome B", value=aj_clamped, min=0, max=max(0, n_at - 1), step=1).props("outlined dense")
                            in_d = ui.number(
                                label="Distance (Å)",
                                value=float(view_cfg["bond_dist"]) if float(view_cfg["bond_dist"]) > 0 else cur_len,
                                min=0.8,
                                max=4.0,
                                step=0.05,
                                format="%.2f",
                            ).props("outlined dense")

                            def apply_3d_settings(_e=None):
                                view_cfg.update(
                                    style=str(sel_st.value or "Ball & Stick"),
                                    ff=str(sel_ff.value or "MMFF94"),
                                    seed=int(in_seed.value or 42),
                                    show_h=bool(chk_h.value),
                                    charges=bool(chk_q.value),
                                    spin=bool(chk_spin.value),
                                    bond_ai=int(in_ai.value or 0),
                                    bond_aj=int(in_aj.value or 1),
                                    bond_dist=float(in_d.value or 0.0),
                                )
                                refresh_tab()

                            _btn("Appliquer", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=apply_3d_settings)

                        for ctrl in (sel_st, sel_ff, in_seed, chk_h, chk_q, chk_spin):
                            ctrl.on_value_change(
                                lambda _e: (
                                    view_cfg.update(
                                        style=str(sel_st.value or "Ball & Stick"),
                                        ff=str(sel_ff.value or "MMFF94"),
                                        seed=int(in_seed.value or 42),
                                        show_h=bool(chk_h.value),
                                        charges=bool(chk_q.value),
                                        spin=bool(chk_spin.value),
                                        bond_dist=0.0,
                                    ),
                                    refresh_tab(),
                                )
                            )

                    q_params = urllib.parse.urlencode({
                        "smiles": single_smi,
                        "style": str(view_cfg["style"]),
                        "ff": str(view_cfg["ff"]),
                        "seed": int(view_cfg["seed"]),
                        "show_h": 1 if view_cfg["show_h"] else 0,
                        "charges": 1 if view_cfg["charges"] else 0,
                        "spin": 1 if view_cfg["spin"] else 0,
                        "dark": d_flag,
                        "ai": int(view_cfg["bond_ai"]) if float(view_cfg["bond_dist"]) > 0 else -1,
                        "aj": int(view_cfg["bond_aj"]) if float(view_cfg["bond_dist"]) > 0 else -1,
                        "dist": float(view_cfg["bond_dist"]),
                    })
                    safe_html(f'<iframe class="viewer-3d-frame" src="/api/studio/3d/standard?{q_params}" title="Modèle 3D"></iframe>')

                # --- SOUS-MODE 2 : CHIMIE QUANTIQUE (HOMO / LUMO / ESP) ---
                elif view_cfg["mode_3d"] == "quantum":
                    q_res = compute_quantum_orbitals(mol_main)
                    mol_3d_q = generate_3d_mol(single_smi)
                    dip_d = compute_dipole_moment_3d(mol_3d_q) if mol_3d_q else 0.0

                    if not q_res or not mol_3d_q:
                        safe_html('<p class="muted-text">Calcul quantique indisponible pour cette structure.</p>')
                        return

                    safe_html(
                        f"""
                        <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:10px;">
                          <div class="kpi-card"><div class="kpi-label">HOMO</div><div class="kpi-value" style="font-size:1.05rem;color:var(--cyan);">{q_res['e_homo']} eV</div></div>
                          <div class="kpi-card"><div class="kpi-label">LUMO</div><div class="kpi-value" style="font-size:1.05rem;color:var(--accent);">{q_res['e_lumo']} eV</div></div>
                          <div class="kpi-card"><div class="kpi-label">Gap ΔE</div><div class="kpi-value" style="font-size:1.05rem;">{q_res['gap']} eV</div></div>
                          <div class="kpi-card"><div class="kpi-label">Dipôle</div><div class="kpi-value" style="font-size:1.05rem;">{dip_d} D</div></div>
                        </div>
                        """
                    )
                    with ui.element("div").style("display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-bottom:10px;"):
                        surf_opts = {
                            "HOMO": "HOMO (Orbitale Haute Occupée)",
                            "LUMO": "LUMO (Orbitale Basse Vacante)",
                            "ESP": "Potentiel Électrostatique (Surface ESP)",
                        }
                        sel_surf = ui.select(
                            surf_opts,
                            value=str(view_cfg["quantum_surface"]),
                            label="Surface quantique",
                        ).props("outlined dense")
                        sel_surf.on_value_change(
                            lambda e: (view_cfg.update(quantum_surface=str(e.value or "HOMO")), refresh_tab())
                        )
                        _btn(
                            "Table de Fukui",
                            style="height:40px;",
                            on_click=lambda _e, rows=q_res["fukui_rows"], om=q_res["omega"]: open_fukui_modal(rows, om),
                        )

                    q_params = urllib.parse.urlencode({
                        "smiles": single_smi,
                        "mode": str(view_cfg["quantum_surface"]),
                        "dark": d_flag,
                    })
                    safe_html(f'<iframe class="viewer-3d-frame" style="height:385px;" src="/api/studio/3d/quantum?{q_params}" title="Orbitale 3D"></iframe>')

                # --- SOUS-MODE 3 : SUPERPOSITION 3D (MCS / RMSD) ---
                else:
                    hist = list(state.get("history", []))
                    default_ref = hist[1] if len(hist) > 1 else (hist[0] if hist else single_smi)
                    if not view_cfg["overlay_target"] or view_cfg["overlay_target"] not in hist:
                        view_cfg["overlay_target"] = default_ref

                    with ui.element("div").style("display:grid;grid-template-columns:1fr 1fr auto;gap:8px;align-items:center;margin-bottom:10px;"):
                        sel_hist = ui.select(hist, value=view_cfg["overlay_target"], label="2e molécule (historique)").props("outlined dense")
                        in_cust = ui.input(
                            label="Ou CAS / Nom / SMILES",
                            value=str(view_cfg["overlay_custom"]),
                            placeholder="Ex : Ibuprofen",
                        ).props("outlined dense")

                        async def run_overlay_lookup(_e=None):
                            cust = clean_val(in_cust.value)
                            view_cfg["overlay_custom"] = cust
                            if cust:
                                resolved = await run.io_bound(resolve_query_to_smiles, cust)
                                if not resolved:
                                    ui.notify("2e composé introuvable dans PubChem.", type="warning")
                                    return
                                view_cfg["overlay_target"] = resolved
                            else:
                                view_cfg["overlay_target"] = str(sel_hist.value or default_ref)
                            refresh_tab()

                        sel_hist.on_value_change(
                            lambda e: (view_cfg.update(overlay_target=str(e.value or default_ref), overlay_custom=""), refresh_tab())
                        )
                        _btn("Superposer", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=run_overlay_lookup)

                    target_s2 = clean_val(view_cfg["overlay_target"]) or default_ref
                    m1_ov, m2_ov, mcs_atoms, rmsd_val = overlay_molecules_3d(single_smi, target_s2)
                    if m1_ov and m2_ov:
                        rmsd_txt = f"{rmsd_val:.3f} Å" if rmsd_val is not None else "N/A"
                        safe_html(
                            f"""
                            <div class="kpi-grid" style="grid-template-columns:repeat(2,1fr);margin-bottom:10px;">
                              <div class="kpi-card"><div class="kpi-label">Squelette commun (MCS)</div><div class="kpi-value" style="font-size:1.05rem;color:var(--accent);">{mcs_atoms} atomes</div></div>
                              <div class="kpi-card"><div class="kpi-label">Écart RMSD 3D (Vert vs Cyan)</div><div class="kpi-value" style="font-size:1.05rem;color:var(--cyan);">{rmsd_txt}</div></div>
                            </div>
                            """
                        )
                        q_params = urllib.parse.urlencode({"s1": single_smi, "s2": target_s2, "dark": d_flag})
                        safe_html(f'<iframe class="viewer-3d-frame" style="height:385px;" src="/api/studio/3d/overlay?{q_params}" title="Superposition 3D"></iframe>')
                    else:
                        safe_html('<p class="muted-text">Impossible de générer la superposition 3D pour ces deux structures.</p>')

        refresh_tab()
        return refresh_tab

    # ==================================================================
    # ONGLET 2 : RÉACTIONS, STŒCHIOMÉTRIE, RÉTROSENTHÈSE IA & MARKUSH
    # ==================================================================
    def render_tab_rxn():
        root = ui.element("div")
        rxn_cfg: dict[str, Any] = {
            "sub": "stoich",  # stoich | retro | markush
            "ref_mass_mg": 500.0,
            "exp_mass_mg": 0.0,
            "row_params": {},
            "markush_mode": "similarity",
            "markush_thresh": 85,
            "markush_max": 6,
            "markush_results": [],
        }

        def refresh_tab():
            root.clear()
            single_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
            mol_main = parse_smiles(single_smi)

            with root:
                with ui.element("div").classes("sub-tabs"):
                    for k, lbl in (
                        ("stoich", "1. Constructeur & Grille de Stœchiométrie"),
                        ("retro", "2. Rétrosynthèse IA (Déconnexions)"),
                        ("markush", "3. Recherche Markush & Analogues PubChem"),
                    ):
                        cls = "glass-btn glass-btn-accent" if rxn_cfg["sub"] == k else "glass-btn"
                        _btn(lbl, classes=cls, on_click=lambda _e, key=k: (rxn_cfg.update(sub=key), refresh_tab()))

                # --- SOUS-ONGLET 1 : CONSTRUCTEUR & STŒCHIOMÉTRIE ---
                if rxn_cfg["sub"] == "stoich":
                    with ui.element("div").classes("hud-card").style("margin-bottom:16px;"):
                        with ui.expansion("Modifier le panier réactionnel (Recherche CAS/Nom/SMILES ou Prédicteur automatique)", value=False).style("width:100%;"):
                            with ui.element("div").classes("modal-grid").style("padding-top:8px;"):
                                with ui.element("div"):
                                    safe_html('<div class="section-kicker" style="margin-bottom:6px;">AJOUTER AU PANIER</div>')
                                    add_q = ui.input(
                                        label="CAS, Nom ou SMILES",
                                        placeholder="Ex : Acetic anhydride, 108-24-7, CCO...",
                                    ).props("outlined dense").style("width:100%;margin-bottom:8px;")

                                    async def add_to_basket(as_product: bool):
                                        q = clean_val(add_q.value)
                                        if not q:
                                            return
                                        smi = await run.io_bound(resolve_query_to_smiles, q)
                                        if not smi:
                                            ui.notify("Composé introuvable dans PubChem.", type="negative")
                                            return
                                        if as_product:
                                            state["rxn_products"] = [smi]
                                        else:
                                            state["rxn_reactants"].append(smi)
                                        _sync_rxn_smiles(state)
                                        on_basket_change()
                                        refresh_tab()

                                    def reset_basket(_e=None):
                                        state["rxn_reactants"] = []
                                        state["rxn_products"] = []
                                        _sync_rxn_smiles(state)
                                        on_basket_change()
                                        refresh_tab()

                                    with ui.element("div").style("display:flex;gap:8px;flex-wrap:wrap;"):
                                        _btn("➕ Aux Réactifs", classes="glass-btn glass-btn-accent", on_click=lambda _e: add_to_basket(False))
                                        _btn("🎯 Comme Produit", on_click=lambda _e: add_to_basket(True))
                                        _btn("Vider le panier", classes="glass-btn glass-btn-danger", on_click=reset_basket)

                                with ui.element("div"):
                                    safe_html('<div class="section-kicker" style="margin-bottom:6px;">PRÉDICTEUR AUTOMATIQUE DE PRODUIT</div>')
                                    tpl_names = list(REACTION_TEMPLATES.keys())
                                    sel_tpl = ui.select(tpl_names, value=tpl_names[0], label="Transformation type").props("outlined dense").style("width:100%;margin-bottom:8px;")

                                    def run_predictor(_e=None):
                                        smarts = REACTION_TEMPLATES[str(sel_tpl.value)]
                                        pred_smi, msg = predict_reaction_product(list(state.get("rxn_reactants", [])), smarts)
                                        if pred_smi:
                                            state["rxn_products"] = [pred_smi]
                                            _sync_rxn_smiles(state)
                                            ui.notify(msg, type="positive")
                                            on_basket_change()
                                            refresh_tab()
                                        else:
                                            ui.notify(msg, type="warning")

                                    _btn("Générer le produit depuis les réactifs actuels", classes="glass-btn glass-btn-accent", on_click=run_predictor)

                        _sync_rxn_smiles(state)
                        r_list = [s for s in state.get("rxn_reactants", []) if parse_smiles(s) is not None]
                        p_list = [s for s in state.get("rxn_products", []) if parse_smiles(s) is not None]

                        if not r_list or not p_list:
                            safe_html('<p class="muted-text" style="margin-top:12px;">Ajoutez au moins un réactif et un produit dans le panier réactionnel pour afficher le schéma et la grille stœchiométrique.</p>')
                            return

                        rxn_can = f"{'.'.join(r_list)}>>{'.'.join(p_list)}"
                        rxn_svg = render_reaction_svg(rxn_can, width=720, height=205)

                        with ui.element("div").classes("modal-grid").style("margin-top:12px;align-items:center;"):
                            safe_html(f'<div class="svg-box-adaptive">{rxn_svg}</div>')
                            with ui.element("div"):
                                in_ref_mg = ui.number(
                                    label="Masse engagée du Réactif 1 (mg)",
                                    value=float(rxn_cfg["ref_mass_mg"]),
                                    min=1.0,
                                    step=50.0,
                                    format="%.1f",
                                ).props("outlined dense").style("width:100%;margin-bottom:10px;")
                                in_exp_mg = ui.number(
                                    label="Masse de Produit 1 isolée au labo (mg)",
                                    value=float(rxn_cfg["exp_mass_mg"]),
                                    min=0.0,
                                    step=10.0,
                                    format="%.1f",
                                ).props("outlined dense").style("width:100%;")

                        entries: list[dict[str, Any]] = []
                        for idx, smi in enumerate(r_list):
                            m = parse_smiles(smi)
                            if m is None:
                                continue
                            key = f"R{idx}_{smi}"
                            if key not in rxn_cfg["row_params"]:
                                rxn_cfg["row_params"][key] = {"eq": 1.0 if idx == 0 else 1.2, "dens": 1.0}
                            entries.append({
                                "key": key,
                                "is_reactant": True,
                                "role": f"Réactif {idx + 1}",
                                "smiles": smi,
                                "formula": rdMolDescriptors.CalcMolFormula(m),
                                "mw": round(float(Descriptors.MolWt(m)), 2),
                                "exact_mw": float(Descriptors.ExactMolWt(m)),
                            })
                        for idx, smi in enumerate(p_list):
                            m = parse_smiles(smi)
                            if m is None:
                                continue
                            key = f"P{idx}_{smi}"
                            if key not in rxn_cfg["row_params"]:
                                rxn_cfg["row_params"][key] = {"eq": 1.0, "dens": 1.0}
                            entries.append({
                                "key": key,
                                "is_reactant": False,
                                "role": f"Produit {idx + 1}",
                                "smiles": smi,
                                "formula": rdMolDescriptors.CalcMolFormula(m),
                                "mw": round(float(Descriptors.MolWt(m)), 2),
                                "exact_mw": float(Descriptors.ExactMolWt(m)),
                            })

                        name_slots: dict[str, Any] = {}
                        calc_slots: dict[str, Any] = {}
                        kpi_slot = ui.element("div").style("margin-top:14px;")

                        def recalc_stoich():
                            r_m = float(in_ref_mg.value or 500.0)
                            e_m = float(in_exp_mg.value or 0.0)
                            rxn_cfg["ref_mass_mg"] = r_m
                            rxn_cfg["exp_mass_mg"] = e_m

                            r1_mw = entries[0]["mw"] if entries else 1.0
                            ref_mmol = r_m / r1_mw if r1_mw > 0 else 0.0

                            for ent in entries:
                                p = rxn_cfg["row_params"][ent["key"]]
                                eq_v = max(0.0, float(p["eq"]))
                                dens_v = max(0.05, float(p["dens"]))
                                mmol = eq_v * ref_mmol
                                mass_mg = mmol * ent["mw"]
                                vol_ul = mass_mg / dens_v
                                slot = calc_slots.get(ent["key"])
                                if slot is not None:
                                    slot.clear()
                                    with slot:
                                        safe_html(
                                            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-family:\'JetBrains Mono\',monospace;font-size:0.80rem;">'
                                            f'<div><span class="muted-small">mmol</span><br><strong>{mmol:.3f}</strong></div>'
                                            f'<div><span class="muted-small">Masse</span><br><strong style="color:var(--cyan);">{mass_mg:.1f} mg</strong></div>'
                                            f'<div><span class="muted-small">Vol. liq.</span><br><strong>{vol_ul:.1f} µL</strong></div>'
                                            f'</div>'
                                        )

                            p1_ent = next((e for e in entries if not e["is_reactant"]), None)
                            sum_r_mw = sum(e["mw"] for e in entries if e["is_reactant"])
                            if p1_ent and sum_r_mw > 0:
                                theo_mg = ref_mmol * p1_ent["mw"]
                                y_pct = (e_m / theo_mg * 100.0) if (theo_mg > 0 and e_m > 0) else 0.0
                                atom_ec = (p1_ent["mw"] / sum_r_mw) * 100.0
                                dmz = p1_ent["exact_mw"] - entries[0]["exact_mw"]
                                y_txt = f"{y_pct:.1f} %" if e_m > 0 else "Renseigner masse"
                                kpi_slot.clear()
                                with kpi_slot:
                                    safe_html(
                                        f"""
                                        <div class="kpi-grid">
                                          <div class="kpi-card"><div class="kpi-label">Masse théorique (100 %)</div><div class="kpi-value" style="color:var(--cyan);">{theo_mg:.1f} mg</div></div>
                                          <div class="kpi-card"><div class="kpi-label">Rendement expérimental</div><div class="kpi-value" style="color:var(--accent);">{y_txt}</div></div>
                                          <div class="kpi-card"><div class="kpi-label">Économie d'atomes</div><div class="kpi-value">{atom_ec:.1f} %</div></div>
                                          <div class="kpi-card"><div class="kpi-label">Suivi MS (Δ m/z R1→P1)</div><div class="kpi-value">{dmz:+.4f} Da</div></div>
                                        </div>
                                        """
                                    )

                        in_ref_mg.on_value_change(lambda _e: recalc_stoich())
                        in_exp_mg.on_value_change(lambda _e: recalc_stoich())

                        safe_html('<div class="section-kicker" style="margin:14px 0 8px 0;">GRILLE STŒCHIOMÉTRIQUE INTERACTIVE (ÉQUIVALENTS &amp; DENSITÉS MODIFIABLES)</div>')
                        with ui.element("div").style("display:flex;flex-direction:column;gap:8px;"):
                            for ent in entries:
                                k = ent["key"]
                                p_cur = rxn_cfg["row_params"][k]
                                pill_cls = "stock-ok" if ent["is_reactant"] else "stock-low"
                                with ui.element("div").classes("kv-item").style("display:grid;grid-template-columns:1.5fr 0.7fr 0.7fr 1.6fr;gap:10px;align-items:center;"):
                                    with ui.element("div"):
                                        safe_html(
                                            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
                                            f'<span class="stock-pill {pill_cls}">{html.escape(ent["role"])}</span>'
                                            f'<span class="mono-tag">{html.escape(ent["formula"])} · {ent["mw"]} g/mol</span>'
                                            f'</div>'
                                        )
                                        ns = ui.element("div").style("margin-top:4px;")
                                        with ns:
                                            safe_html(f'<span class="muted-small">{html.escape(ent["smiles"])}</span>')
                                        name_slots[k] = ns

                                    in_eq = ui.number(label="Équiv.", value=float(p_cur["eq"]), min=0.01, step=0.1, format="%.2f").props("outlined dense")
                                    in_dn = ui.number(label="Densité", value=float(p_cur["dens"]), min=0.1, step=0.05, format="%.2f").props("outlined dense")

                                    def on_eq_ch(e, key_k=k):
                                        rxn_cfg["row_params"][key_k]["eq"] = float(e.value or 1.0)
                                        recalc_stoich()

                                    def on_dn_ch(e, key_k=k):
                                        rxn_cfg["row_params"][key_k]["dens"] = float(e.value or 1.0)
                                        recalc_stoich()

                                    in_eq.on_value_change(on_eq_ch)
                                    in_dn.on_value_change(on_dn_ch)
                                    calc_slots[k] = ui.element("div")

                        recalc_stoich()

                        async def enrich_basket_names():
                            for ent in entries:
                                inf = await run.io_bound(get_chemical_identifiers, ent["smiles"])
                                slot = name_slots.get(ent["key"])
                                if slot is not None:
                                    slot.clear()
                                    with slot:
                                        safe_html(
                                            f'<strong style="font-size:0.84rem;color:var(--txt);">{html.escape(str(inf["trivial"]))}</strong> '
                                            f'<span class="muted-small">· CAS {html.escape(str(inf["cas"]))}</span>'
                                        )

                        ui.timer(0.05, enrich_basket_names, once=True)

                # --- SOUS-ONGLET 2 : RÉTROSENTHÈSE IA ---
                elif rxn_cfg["sub"] == "retro":
                    if not mol_main:
                        safe_html('<div class="hud-card"><p class="muted-text">Chargez une molécule valide pour lancer la rétrosynthèse.</p></div>')
                        return

                    routes = run_retrosynthesis_engine(mol_main)
                    if not routes:
                        safe_html('<div class="hud-card"><div class="section-kicker">RÉTROSENTHÈSE IA</div><h2>Aucune coupure stratégique standard détectée</h2><p class="muted-text">Essayez une molécule portant un ester, un amide, un biaryle, un éther, un alcène, un triazole ou un alcool secondaire/tertiaire.</p></div>')
                        return

                    prec_name_slots: list[tuple[str, Any]] = []
                    with ui.element("div").style("display:flex;flex-direction:column;gap:14px;"):
                        for idx, rt in enumerate(routes):
                            rxn_svg = render_reaction_svg(rt["rxn_smiles"], width=620, height=185)
                            with ui.element("div").classes("hud-card"):
                                safe_html(
                                    f'<div class="section-kicker">VOIE DE DÉCONNEXION #{idx + 1}</div>'
                                    f'<h2>{html.escape(rt["strategy"])}</h2>'
                                )
                                with ui.element("div").classes("modal-grid").style("align-items:center;"):
                                    safe_html(f'<div class="svg-box-adaptive">{rxn_svg}</div>')
                                    with ui.element("div"):
                                        for p_i, p_smi in enumerate(rt["precursors"]):
                                            with ui.element("div").classes("kv-item").style("display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;"):
                                                lbl_slot = ui.element("div")
                                                with lbl_slot:
                                                    safe_html(f'<strong>Brique {p_i + 1}</strong> <span class="mono-tag">{html.escape(p_smi)}</span>')
                                                prec_name_slots.append((p_smi, lbl_slot))
                                                _btn(
                                                    "Éditer 2D",
                                                    style="padding:4px 10px;font-size:0.74rem;",
                                                    on_click=lambda _e, s=p_smi: on_molecule_change(s),
                                                )

                                        def load_route_stoich(_e=None, precs=tuple(rt["precursors"])):
                                            state["rxn_reactants"] = list(precs)
                                            state["rxn_products"] = [Chem.MolToSmiles(mol_main, canonical=True)]
                                            _sync_rxn_smiles(state)
                                            rxn_cfg["sub"] = "stoich"
                                            ui.notify("Voie chargée dans la grille de stœchiométrie !", type="positive")
                                            on_basket_change()
                                            refresh_tab()

                                        _btn(
                                            f"Charger la Voie #{idx + 1} dans la Stœchiométrie",
                                            classes="glass-btn glass-btn-accent",
                                            style="width:100%;margin-top:6px;",
                                            on_click=load_route_stoich,
                                        )

                    async def enrich_retro_names():
                        for p_smi, slot in prec_name_slots:
                            inf = await run.io_bound(get_chemical_identifiers, p_smi)
                            slot.clear()
                            with slot:
                                safe_html(
                                    f'<strong style="font-size:0.85rem;color:var(--txt);">{html.escape(str(inf["trivial"]))}</strong>'
                                    f'<div class="muted-small">CAS {html.escape(str(inf["cas"]))} · <code>{html.escape(p_smi)}</code></div>'
                                )

                    ui.timer(0.05, enrich_retro_names, once=True)

                # --- SOUS-ONGLET 3 : MARKUSH & ANALOGUES PUBCHEM ---
                else:
                    if not mol_main:
                        safe_html('<div class="hud-card"><p class="muted-text">Chargez une molécule valide pour rechercher des analogues.</p></div>')
                        return

                    with ui.element("div").classes("hud-card").style("margin-bottom:14px;"):
                        safe_html('<div class="section-kicker">EXPLORATION CHIMIQUE PUBCHEM</div><h2>Analogues &amp; Dérivés de Markush</h2>')
                        with ui.element("div").style("display:grid;grid-template-columns:1.5fr 1fr 0.8fr auto;gap:10px;align-items:center;"):
                            sel_mk = ui.select(
                                {"similarity": "Analogues proches (Similarité 2D)", "substructure": "Dérivés de Markush (Sous-structures)"},
                                value=str(rxn_cfg["markush_mode"]),
                                label="Type de recherche",
                            ).props("outlined dense")
                            in_thr = ui.number(label="Seuil similarité (%)", value=int(rxn_cfg["markush_thresh"]), min=70, max=98, step=1).props("outlined dense")
                            in_max = ui.number(label="Max résultats", value=int(rxn_cfg["markush_max"]), min=3, max=12, step=1).props("outlined dense")

                            async def do_markush(_e=None):
                                rxn_cfg["markush_mode"] = str(sel_mk.value or "similarity")
                                rxn_cfg["markush_thresh"] = int(in_thr.value or 85)
                                rxn_cfg["markush_max"] = int(in_max.value or 6)
                                ui.notify("Interrogation de PubChem en cours...", type="info")
                                found = await run.io_bound(
                                    search_pubchem_markush_analogues,
                                    single_smi,
                                    rxn_cfg["markush_mode"],
                                    rxn_cfg["markush_thresh"],
                                    rxn_cfg["markush_max"],
                                )
                                rxn_cfg["markush_results"] = found
                                if not found:
                                    ui.notify("Aucun analogue retourné par PubChem.", type="warning")
                                refresh_tab()

                            _btn("Rechercher sur PubChem", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=do_markush)

                    res_list = list(rxn_cfg.get("markush_results", []))
                    if res_list:
                        with ui.element("div").style("display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;"):
                            for item in res_list:
                                m_f = parse_smiles(item["smiles"])
                                if not m_f:
                                    continue
                                svg_f = render_2d_svg(m_f, width=250, height=165)
                                with ui.element("div").classes("hud-card").style("padding:14px;"):
                                    safe_html(f'<div class="svg-box-adaptive" style="margin-bottom:10px;">{svg_f}</div>')
                                    safe_html(
                                        f'<strong style="font-size:0.90rem;color:var(--txt);">{html.escape(str(item["nom"])[:38])}</strong>'
                                        f'<div class="muted-small" style="margin-bottom:10px;">MW : {html.escape(str(item.get("mw", "N/A")))} g/mol · CID {item.get("cid")}</div>'
                                    )
                                    with ui.element("div").style("display:flex;gap:8px;"):
                                        _btn(
                                            "Charger 2D",
                                            classes="glass-btn glass-btn-accent",
                                            style="flex:1;padding:6px 10px;font-size:0.76rem;",
                                            on_click=lambda _e, s=item["smiles"]: on_molecule_change(s),
                                        )

                                        def add_mk_rea(_e=None, s=item["smiles"]):
                                            state["rxn_reactants"].append(s)
                                            _sync_rxn_smiles(state)
                                            ui.notify("Analogue ajouté aux réactifs du panier !", type="positive")
                                            on_basket_change()

                                        _btn(
                                            "➕ Réactif",
                                            style="flex:1;padding:6px 10px;font-size:0.76rem;",
                                            on_click=add_mk_rea,
                                        )

        refresh_tab()
        return refresh_tab

    # ==================================================================
    # ONGLET 3 : STRUCTURE, STÉRÉOCHIMIE & CHEMPROP
    # ==================================================================
    def render_tab_struct():
        root = ui.element("div")

        def refresh_tab():
            root.clear()
            single_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
            mol_main = parse_smiles(single_smi)

            with root:
                if not mol_main:
                    safe_html('<div class="hud-card"><p class="muted-text">Structure chimique invalide.</p></div>')
                    return

                cprop = estimate_chemprop(mol_main)
                svg_annot = render_2d_svg(mol_main, width=490, height=345, atom_indices=True, bond_indices=True, stereo=True)

                lip_violations = sum([
                    1 if cprop["mw"] > 500 else 0,
                    1 if cprop["logp"] > 5 else 0,
                    1 if cprop["hbd"] > 5 else 0,
                    1 if cprop["hba"] > 10 else 0,
                ])
                lip_pill = (
                    '<span class="stock-pill stock-ok">Lipinski Ro5 : Conforme (0 violation)</span>'
                    if lip_violations == 0
                    else f'<span class="stock-pill stock-low">Lipinski Ro5 : {lip_violations} violation(s)</span>'
                )

                with ui.element("div").classes("modal-grid"):
                    with ui.element("div").classes("hud-card"):
                        safe_html(
                            '<div class="section-kicker">CARTOGRAPHIE TOPOLOGIQUE &amp; STÉRÉOCHIMIE CIP</div>'
                            '<h2>Carte 2D Annotée (Atomes, Liaisons, R/S, E/Z)</h2>'
                            f'<div class="svg-box" style="margin-bottom:8px;">{svg_annot}</div>'
                            '<div class="muted-small" style="margin-bottom:12px;">Noir : indices d\'atomes · Bleu : indices de liaisons (utiles pour la fragmentation MS et les distances 3D).</div>'
                        )

                        with ui.expansion("Explorer les formes tautomères (jusqu'à 6 formes)", value=False).style("width:100%;"):
                            tauts = enumerate_tautomers_svgs(mol_main, max_count=6)
                            if not tauts:
                                safe_html('<span class="muted-small">Aucun autre tautomère énuméré.</span>')
                            else:
                                with ui.element("div").style("display:grid;grid-template-columns:repeat(auto-fill,minmax(175px,1fr));gap:10px;padding-top:8px;"):
                                    for idx, t in enumerate(tauts):
                                        with ui.element("div").classes("kv-item").style("text-align:center;"):
                                            safe_html(f'<div class="svg-box-adaptive" style="padding:4px;margin-bottom:6px;">{t["svg"]}</div>')
                                            _btn(
                                                f"Charger #{idx + 1}",
                                                style="width:100%;padding:4px 8px;font-size:0.73rem;",
                                                on_click=lambda _e, s=t["smiles"]: on_molecule_change(s),
                                            )

                    with ui.element("div").classes("hud-card"):
                        safe_html(
                            '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">'
                            '<div><div class="section-kicker">PROPRIÉTÉS PHYSICO-CHIMIQUES ESTIMÉES</div>'
                            '<h2 style="margin:4px 0 0 0;">Panneau ChemProp &amp; Descripteurs</h2></div>'
                            f'{lip_pill}</div>'
                            f"""
                            <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
                              <div class="kpi-card"><div class="kpi-label">Ébullition estimée (Tb)</div><div class="kpi-value" style="font-size:1.1rem;color:var(--warn);">{cprop['tb_c']} °C</div></div>
                              <div class="kpi-card"><div class="kpi-label">Fusion estimée (Tf)</div><div class="kpi-value" style="font-size:1.1rem;color:var(--cyan);">{cprop['tm_c']} °C</div></div>
                              <div class="kpi-card"><div class="kpi-label">Solubilité (LogS)</div><div class="kpi-value" style="font-size:1.05rem;">{cprop['logs']}</div><div class="muted-small">{cprop['sol_mg_ml']} mg/mL</div></div>
                              <div class="kpi-card"><div class="kpi-label">LogP (Crippen)</div><div class="kpi-value" style="font-size:1.1rem;">{cprop['logp']}</div></div>
                              <div class="kpi-card"><div class="kpi-label">Réfringence Molaire</div><div class="kpi-value" style="font-size:1.05rem;">{cprop['mr']}</div><div class="muted-small">cm³/mol</div></div>
                              <div class="kpi-card"><div class="kpi-label">Surface Polaire (TPSA)</div><div class="kpi-value" style="font-size:1.1rem;color:var(--accent);">{cprop['tpsa']} Å²</div></div>
                              <div class="kpi-card"><div class="kpi-label">Insaturation (DBE)</div><div class="kpi-value" style="font-size:1.1rem;">{cprop['dbe']:g}</div></div>
                              <div class="kpi-card"><div class="kpi-label">Donneurs / Accepteurs H</div><div class="kpi-value" style="font-size:1.1rem;">{cprop['hbd']} / {cprop['hba']}</div></div>
                              <div class="kpi-card"><div class="kpi-label">Liaisons rotatives</div><div class="kpi-value" style="font-size:1.1rem;">{cprop['rot']}</div></div>
                            </div>
                            <div class="kv-grid" style="margin-top:8px;">
                              <div class="kv-item"><div class="kv-label">Formule brute</div><div class="kv-val" style="font-family:'JetBrains Mono',monospace;">{html.escape(cprop['formula'])}</div></div>
                              <div class="kv-item"><div class="kv-label">Masse exacte</div><div class="kv-val" style="font-family:'JetBrains Mono',monospace;">{cprop['exact_mw']} Da</div></div>
                              <div class="kv-item"><div class="kv-label">Cycles</div><div class="kv-val" style="font-family:'JetBrains Mono',monospace;">{cprop['rings']}</div></div>
                            </div>
                            """
                        )

        refresh_tab()
        return refresh_tab

    return render_tab_2d3d, render_tab_rxn, render_tab_struct