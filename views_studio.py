import html
import json
from typing import Any
import urllib.parse

from nicegui import run, ui
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from chem_engine import (
    REACTION_CATALOG,
    clean_val,
    compute_dipole_moment_3d,
    compute_quantum_orbitals,
    enumerate_tautomers_svgs,
    estimate_chemprop,
    extract_single_smiles,
    fetch_chembase_catalog,
    generate_3d_mol,
    get_bond_length_3d,
    get_chembase_config,
    get_chemical_identifiers,
    overlay_molecules_3d,
    parse_smiles,
    predict_reaction_candidates,
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


def _fmt_mass(mg: float) -> str:
    if mg >= 1000.0:
        return f"{mg / 1000.0:.3f} g"
    return f"{mg:.1f} mg"


def _fmt_vol_ml(ml: float) -> str:
    if ml >= 1.0:
        return f"{ml:.2f} mL"
    return f"{ml * 1000.0:.1f} µL"


def _copy_to_clipboard(text: str, label: str):
    js = f"""
    (async () => {{
      const txt = {json.dumps(text)};
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(txt);
          return;
        }}
      }} catch (e) {{}}
      const ta = document.createElement('textarea');
      ta.value = txt;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }})();
    """
    ui.run_javascript(js)
    ui.notify(f"Tableau copié ({label}) !", type="positive")


def build_studio_views(state: dict[str, Any], on_molecule_change, on_basket_change):
    # Référence partagée pour rafraîchir l'onglet Réactions depuis le bandeau d'édition de l'onglet 2D
    rxn_refresh_ref: dict[str, Any] = {"fn": None}

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
                # Bandeau contextuel affiché uniquement lors d'une édition issue de l'onglet Réactions
                rxn_edit_banner = ui.element("div")

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

        def refresh_rxn_edit_banner():
            rxn_edit_banner.clear()
            target = state.get("rxn_edit_target")
            if not target:
                return
            side = target.get("side", "rxn_reactants")
            idx = int(target.get("idx", 0))
            lbl = str(target.get("label", "Molécule"))
            with rxn_edit_banner:
                with ui.element("div").style(
                    "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;"
                    "padding:8px 12px;margin-bottom:10px;border-radius:10px;"
                    "border:1px solid var(--accent);background:rgba(16,185,129,0.10);"
                ):
                    safe_html(
                        f'<span style="font-size:0.80rem;">✎ Édition en cours : '
                        f'<strong>{html.escape(lbl)}</strong> (Réaction)</span>'
                    )
                    with ui.element("div").style("display:flex;gap:6px;"):
                        def _apply_replace(_e=None):
                            new_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
                            if parse_smiles(new_smi) is None:
                                ui.notify("Structure 2D invalide.", type="negative")
                                return
                            lst = list(state.get(side, []))
                            if 0 <= idx < len(lst):
                                lst[idx] = new_smi
                                state[side] = lst
                                _sync_rxn_smiles(state)
                                state["rxn_edit_target"] = None
                                on_basket_change()
                                refresh_rxn_edit_banner()
                                if callable(rxn_refresh_ref.get("fn")):
                                    rxn_refresh_ref["fn"]()
                                ui.notify(f"{lbl} remplacé dans la réaction !", type="positive")
                            else:
                                state["rxn_edit_target"] = None
                                refresh_rxn_edit_banner()

                        def _cancel_replace(_e=None):
                            state["rxn_edit_target"] = None
                            refresh_rxn_edit_banner()
                            if callable(rxn_refresh_ref.get("fn")):
                                rxn_refresh_ref["fn"]()

                        _btn(
                            "✓ Remplacer dans la réaction",
                            classes="glass-btn glass-btn-accent",
                            style="padding:4px 10px;font-size:0.75rem;",
                            on_click=_apply_replace,
                        )
                        _btn(
                            "✕",
                            style="padding:4px 8px;font-size:0.75rem;",
                            props='title="Annuler le mode remplacement"',
                            on_click=_cancel_replace,
                        )

        def refresh_tab():
            raw_smi = clean_val(state.get("current_smiles"))
            single_smi = extract_single_smiles(raw_smi)
            mol_main = parse_smiles(single_smi)
            d_flag = 1 if state.get("dark", True) else 0

            refresh_rxn_edit_banner()

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

        # Conditions réactionnelles légères persistées dans state
        state.setdefault("rxn_conditions", {"solvent": "", "temp": "", "duration": ""})

        rxn_cfg: dict[str, Any] = {
            "sub": "stoich",  # stoich | retro | markush
            "limiting_idx": 0,
            "start_qty": 500.0,
            "start_unit": "mg",  # mg | g | mmol | mL
            "exp_qty": 0.0,
            "exp_unit": "mg",  # mg | g
            "row_params": {},
            "smi_defaults": {},
            "name_cache": {},
            "pred_tpl": "__auto__",
            "markush_mode": "similarity",
            "markush_thresh": 85,
            "markush_max": 6,
            "markush_results": [],
            "render_gen": 0,
        }

        # Modale rapide "+ Ajouter" (créée une seule fois pour éviter les dialogues orphelins)
        add_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
        with add_dialog:
            add_dialog_body = ui.element("div").classes("cb-modal-card").style("max-width:600px;width:94vw;")
            # Modale 1 : Choix parmi plusieurs produits prédits compatibles
        pred_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
        with pred_dialog:
            pred_dialog_body = ui.element("div").classes("cb-modal-card").style("max-width:780px;width:95vw;")

        # Modale 2 : Catalogue complet des réactions (recherche + filtres par famille)
        catalog_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
        with catalog_dialog:
            catalog_dialog_body = ui.element("div").classes("cb-modal-card").style("max-width:740px;width:95vw;")

        def _apply_predicted_product(prod_smi: str, rxn_name: str):
            cur_p = list(state.get("rxn_products", []))
            short_nm = rxn_name.split("(")[0].strip()
            if prod_smi in cur_p:
                ui.notify("Ce produit est déjà présent dans la réaction.", type="info")
                pred_dialog.close()
                return
            if not cur_p:
                state["rxn_products"] = [prod_smi]
            else:
                cur_p.append(prod_smi)
                state["rxn_products"] = cur_p
            _sync_rxn_smiles(state)
            on_basket_change()
            pred_dialog.close()
            ui.notify(f"Produit ajouté ({short_nm}).", type="positive")
            refresh_tab()

        def open_pred_candidates_modal(candidates: list[dict[str, Any]]):
            pred_dialog_body.clear()
            with pred_dialog_body:
                with ui.element("div").classes("modal-head"):
                    with ui.element("div"):
                        safe_html('<div class="section-kicker">PRÉDICTION DE RÉACTION</div>')
                        safe_html(
                            f'<h2 style="margin:4px 0 0 0;font-size:1.18rem;">'
                            f'{len(candidates)} transformation(s) compatible(s)</h2>'
                            f'<span class="muted-small">Sélectionnez le produit attendu pour l\'ajouter au schéma.</span>'
                        )
                    _btn("Fermer ✕", on_click=lambda _e: pred_dialog.close())

                with ui.element("div").style(
                    "display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));"
                    "gap:10px;max-height:460px;overflow-y:auto;padding-right:4px;"
                ):
                    for cand in candidates:
                        m_c = parse_smiles(cand["smiles"])
                        if m_c is None:
                            continue
                        svg_c = render_2d_svg(m_c, width=190, height=115)
                        form_c = rdMolDescriptors.CalcMolFormula(m_c)
                        mw_c = round(float(Descriptors.MolWt(m_c)), 2)
                        with ui.element("div").classes("kv-item").style(
                            "display:flex;flex-direction:column;justify-content:space-between;gap:6px;padding:10px;"
                        ):
                            safe_html(
                                f'<span class="mono-tag" style="font-size:0.68rem;align-self:flex-start;">'
                                f'{html.escape(str(cand["category"]))}</span>'
                                f'<div style="font-weight:700;font-size:0.80rem;line-height:1.25;">'
                                f'{html.escape(str(cand["name"]))}</div>'
                                f'<div class="svg-box-adaptive" style="padding:3px;">{svg_c}</div>'
                                f'<div class="muted-small" style="font-family:\'JetBrains Mono\',monospace;font-size:0.71rem;">'
                                f'{html.escape(form_c)} · {mw_c} g/mol</div>'
                                f'<div class="muted-small" style="font-size:0.72rem;">'
                                f'Conditions : {html.escape(str(cand.get("reagents") or "—"))}</div>'
                            )
                            _btn(
                                "Choisir ce produit",
                                classes="glass-btn glass-btn-accent",
                                style="width:100%;padding:5px 8px;font-size:0.75rem;",
                                on_click=lambda _e, s=cand["smiles"], nm=cand["name"]: _apply_predicted_product(s, nm),
                            )
            pred_dialog.open()

        def run_prediction_flow(template_name: str = "__auto__"):
            r_cur = [s for s in state.get("rxn_reactants", []) if parse_smiles(s) is not None]
            if not r_cur:
                ui.notify("Ajoutez au moins un réactif avant de prédire.", type="warning")
                return
            cands = predict_reaction_candidates(r_cur, template_name)
            if not cands:
                if template_name == "__auto__":
                    ui.notify("Aucune réaction du catalogue ne correspond aux réactifs actuels.", type="warning")
                else:
                    ui.notify("Les réactifs actuels ne correspondent pas aux fonctions attendues par ce modèle.", type="warning")
                return
            if len(cands) == 1:
                _apply_predicted_product(cands[0]["smiles"], cands[0]["name"])
            else:
                open_pred_candidates_modal(cands)

        def open_catalog_modal():
            cat_state = {"cat": "Toutes", "q": ""}
            all_cats = ["Toutes"] + list(dict.fromkeys(it["category"] for it in REACTION_CATALOG))

            def _render_catalog():
                catalog_dialog_body.clear()
                q_low = cat_state["q"].lower().strip()
                filtered = [
                    it for it in REACTION_CATALOG
                    if (cat_state["cat"] == "Toutes" or it["category"] == cat_state["cat"])
                    and (
                        not q_low
                        or q_low in it["name"].lower()
                        or q_low in it["category"].lower()
                        or q_low in str(it.get("reagents", "")).lower()
                    )
                ]

                with catalog_dialog_body:
                    with ui.element("div").classes("modal-head"):
                        with ui.element("div"):
                            safe_html('<div class="section-kicker">CATALOGUE DE TRANSFORMATIONS</div>')
                            safe_html(
                                f'<h2 style="margin:4px 0 0 0;font-size:1.18rem;">'
                                f'Modèles de réactions ({len(REACTION_CATALOG)})</h2>'
                            )
                        with ui.element("div").style("display:flex;gap:6px;"):
                            _btn(
                                "Détection automatique",
                                classes="glass-btn glass-btn-accent",
                                style="padding:5px 10px;font-size:0.76rem;",
                                on_click=lambda _e: (catalog_dialog.close(), run_prediction_flow("__auto__")),
                            )
                            _btn("Fermer ✕", on_click=lambda _e: catalog_dialog.close())

                    in_search = ui.input(
                        label="Rechercher une réaction ou un réactif",
                        placeholder="Ex : Suzuki, amide, oxydation, Boc, NaBH4...",
                        value=cat_state["q"],
                    ).props("outlined dense clearable").style("width:100%;margin-bottom:8px;")

                    list_box = ui.element("div").style(
                        "max-height:360px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;margin-top:8px;padding-right:4px;"
                    )

                    def _fill_list():
                        list_box.clear()
                        ql = cat_state["q"].lower().strip()
                        rows = [
                            it for it in REACTION_CATALOG
                            if (cat_state["cat"] == "Toutes" or it["category"] == cat_state["cat"])
                            and (
                                not ql
                                or ql in it["name"].lower()
                                or ql in it["category"].lower()
                                or ql in str(it.get("reagents", "")).lower()
                            )
                        ]
                        with list_box:
                            if not rows:
                                safe_html('<span class="muted-small">Aucune réaction ne correspond à ce filtre.</span>')
                                return
                            for it in rows:
                                nr = int(it.get("n_reactants", 1))
                                nr_lbl = "1 réactif" if nr == 1 else f"{nr} réactifs"
                                with ui.element("div").classes("kv-item").style(
                                    "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 10px;"
                                ):
                                    safe_html(
                                        f'<div style="min-width:0;">'
                                        f'<div style="font-weight:700;font-size:0.83rem;color:var(--txt);">'
                                        f'{html.escape(it["name"])}</div>'
                                        f'<div class="muted-small" style="font-size:0.74rem;">'
                                        f'{html.escape(it["category"])} · {nr_lbl} · {html.escape(str(it.get("reagents", "")))}</div>'
                                        f'</div>'
                                    )
                                    _btn(
                                        "Appliquer",
                                        classes="glass-btn",
                                        style="padding:4px 10px;font-size:0.74rem;white-space:nowrap;",
                                        on_click=lambda _e, nm=it["name"]: (catalog_dialog.close(), run_prediction_flow(nm)),
                                    )

                    in_search.on_value_change(lambda e: (cat_state.update(q=str(e.value or "")), _fill_list()))

                    # Filtres par famille (boutons compacts)
                    with ui.element("div").style("display:flex;flex-wrap:wrap;gap:5px;"):
                        for c_name in all_cats:
                            cls = "glass-btn glass-btn-accent" if cat_state["cat"] == c_name else "glass-btn"
                            _btn(
                                html.escape(c_name),
                                classes=cls,
                                style="padding:3px 9px;font-size:0.72rem;",
                                on_click=lambda _e, cn=c_name: (cat_state.update(cat=cn), _render_catalog()),
                            )

                    _fill_list()

            _render_catalog()
            catalog_dialog.open()

        def _commit_mol_to_side(smi: str, side_key: str, known_name: str = "", known_cas: str = ""):
            m = parse_smiles(smi)
            if m is None:
                ui.notify("Structure SMILES invalide.", type="negative")
                return False
            can = Chem.MolToSmiles(m, canonical=True)
            if known_name:
                rxn_cfg["name_cache"][can] = {
                    "trivial": known_name,
                    "cas": known_cas or "Non trouvé",
                }
            state.setdefault(side_key, []).append(can)
            _sync_rxn_smiles(state)
            on_basket_change()
            add_dialog.close()
            refresh_tab()
            return True

        def open_add_modal(initial_side: str = "rxn_reactants"):
            modal_state = {"side": initial_side}

            def _render_modal():
                add_dialog_body.clear()
                is_reac = modal_state["side"] == "rxn_reactants"
                title_lbl = "Ajouter un Réactif" if is_reac else "Ajouter un Produit"
                active_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
                active_mol = parse_smiles(active_smi)

                with add_dialog_body:
                    # En-tête + bascule rapide Réactif / Produit
                    with ui.element("div").classes("modal-head"):
                        with ui.element("div"):
                            safe_html('<div class="section-kicker">AJOUT RAPIDE À LA RÉACTION</div>')
                            safe_html(f'<h2 style="margin:4px 0 0 0;font-size:1.2rem;">{title_lbl}</h2>')
                        with ui.element("div").style("display:flex;gap:6px;align-items:center;"):
                            _btn(
                                "Réactif",
                                classes="glass-btn glass-btn-accent" if is_reac else "glass-btn",
                                style="padding:5px 10px;font-size:0.76rem;",
                                on_click=lambda _e: (modal_state.update(side="rxn_reactants"), _render_modal()),
                            )
                            _btn(
                                "Produit",
                                classes="glass-btn glass-btn-accent" if not is_reac else "glass-btn",
                                style="padding:5px 10px;font-size:0.76rem;",
                                on_click=lambda _e: (modal_state.update(side="rxn_products"), _render_modal()),
                            )
                            _btn("✕", style="padding:5px 10px;", on_click=lambda _e: add_dialog.close())

                    # 1) Saisie directe Nom / CAS / SMILES
                    safe_html('<div class="section-kicker" style="margin:8px 0 6px 0;">1. SAISIR NOM, CAS OU SMILES</div>')
                    with ui.element("div").style("display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-bottom:14px;"):
                        in_q = ui.input(
                            label="Nom, n° CAS ou SMILES",
                            placeholder="Ex : Aspirin, 64-17-5, CC(=O)Cl...",
                        ).props("outlined dense autofocus")

                        async def _submit_query(_e=None):
                            q = clean_val(in_q.value)
                            if not q:
                                return
                            # Validation locale directe si c'est déjà un SMILES
                            if parse_smiles(q) is not None:
                                _commit_mol_to_side(q, modal_state["side"])
                                return
                            smi = await run.io_bound(resolve_query_to_smiles, q)
                            if not smi:
                                ui.notify("Composé introuvable dans PubChem.", type="warning")
                                return
                            _commit_mol_to_side(smi, modal_state["side"])

                        in_q.on("keydown.enter", _submit_query)
                        _btn("Ajouter", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=_submit_query)

                    # 2) Import depuis le dessinateur 2D actif
                    safe_html('<div class="section-kicker" style="margin:6px 0 6px 0;">2. MOLÉCULE ACTIVE DU DESSINATEUR 2D</div>')
                    with ui.element("div").classes("kv-item").style(
                        "display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px;"
                    ):
                        if active_mol:
                            mini_svg = render_2d_svg(active_mol, width=115, height=74)
                            f_act = rdMolDescriptors.CalcMolFormula(active_mol)
                            mw_act = round(float(Descriptors.MolWt(active_mol)), 2)
                            with ui.element("div").style("display:flex;align-items:center;gap:10px;min-width:0;"):
                                safe_html(f'<div class="svg-box-adaptive" style="padding:2px;flex-shrink:0;">{mini_svg}</div>')
                                safe_html(
                                    f'<div style="min-width:0;">'
                                    f'<div style="font-weight:700;font-size:0.86rem;">{html.escape(f_act)} · {mw_act} g/mol</div>'
                                    f'<div class="muted-small" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px;">'
                                    f'<code>{html.escape(active_smi)}</code></div></div>'
                                )
                            _btn(
                                "Importer du 2D",
                                classes="glass-btn glass-btn-accent",
                                style="white-space:nowrap;",
                                on_click=lambda _e, s=active_smi: _commit_mol_to_side(s, modal_state["side"]),
                            )
                        else:
                            safe_html('<span class="muted-small">Aucune molécule valide dans le dessinateur 2D.</span>')

                    # 3) Catalogue ChemBase (si connecté — sans indicateur de stock)
                    cb_url, cb_key = get_chembase_config(
                        clean_val(state.get("chembase_url", "")),
                        clean_val(state.get("chembase_api_key", "")),
                    )
                    safe_html('<div class="section-kicker" style="margin:6px 0 6px 0;">3. CATALOGUE CHEMBASE</div>')
                    if not (cb_url and cb_key):
                        safe_html(
                            '<div class="muted-small" style="padding:6px 0;">'
                            'ChemBase non connecté (variables <code>CHEMBASE_URL</code> / <code>CHEMBASE_API_KEY</code> non configurées).'
                            '</div>'
                        )
                    else:
                        with ui.element("div").style("display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-bottom:8px;"):
                            in_cb = ui.input(
                                label="Filtrer le catalogue ChemBase",
                                placeholder="Nom, CAS ou formule...",
                            ).props("outlined dense")
                            cb_list_box = ui.element("div").style(
                                "grid-column:1 / -1;order:3;max-height:210px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;"
                            )

                            async def _search_cb(_e=None):
                                cb_list_box.clear()
                                with cb_list_box:
                                    safe_html('<span class="muted-small">Chargement du catalogue ChemBase...</span>')
                                ok, items, msg = await run.io_bound(
                                    fetch_chembase_catalog,
                                    clean_val(in_cb.value),
                                    cb_url,
                                    cb_key,
                                )
                                cb_list_box.clear()
                                with cb_list_box:
                                    if not ok:
                                        safe_html(f'<span class="muted-small">{html.escape(msg)}</span>')
                                        return
                                    if not items:
                                        safe_html('<span class="muted-small">Aucune fiche trouvée dans ChemBase.</span>')
                                        return
                                    for it in items[:20]:
                                        it_name = clean_val(it.get("nom") or it.get("name") or it.get("title") or "Sans nom")
                                        it_cas = clean_val(it.get("cas") or it.get("CAS") or "")
                                        it_smi = clean_val(it.get("smiles") or it.get("SMILES") or "")
                                        it_form = clean_val(it.get("formule") or it.get("formula") or "")
                                        sub_bits = [b for b in (f"CAS {it_cas}" if it_cas else "", it_form) if b]

                                        with ui.element("div").classes("kv-item").style(
                                            "display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 10px;"
                                        ):
                                            safe_html(
                                                f'<div style="min-width:0;">'
                                                f'<div style="font-weight:600;font-size:0.84rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                                                f'{html.escape(it_name)}</div>'
                                                f'<div class="muted-small">{html.escape(" · ".join(sub_bits) or it_smi)}</div>'
                                                f'</div>'
                                            )

                                            async def _pick_cb(_e=None, s=it_smi, nm=it_name, cs=it_cas):
                                                resolved = s
                                                if not resolved or parse_smiles(resolved) is None:
                                                    q_fb = cs or nm
                                                    resolved = await run.io_bound(resolve_query_to_smiles, q_fb) or ""
                                                if not resolved:
                                                    ui.notify(f"Impossible de résoudre la structure pour {nm}.", type="warning")
                                                    return
                                                _commit_mol_to_side(resolved, modal_state["side"], known_name=nm, known_cas=cs)

                                            _btn(
                                                "＋ Ajouter",
                                                classes="glass-btn glass-btn-accent",
                                                style="padding:4px 10px;font-size:0.74rem;white-space:nowrap;",
                                                on_click=_pick_cb,
                                            )

                            in_cb.on("keydown.enter", _search_cb)
                            _btn("Chercher", style="height:40px;", on_click=_search_cb)
                        ui.timer(0.05, _search_cb, once=True)

            _render_modal()
            add_dialog.open()

        def refresh_tab():
            rxn_cfg["render_gen"] = int(rxn_cfg["render_gen"]) + 1
            cur_gen = rxn_cfg["render_gen"]
            root.clear()
            single_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
            mol_main = parse_smiles(single_smi)

            with root:
                with ui.element("div").classes("sub-tabs"):
                    for k, lbl in (
                        ("stoich", "1. Constructeur & Grille de Stœchiométrie"),
                        ("retro", "2. Rétrosynthèse (Déconnexions)"),
                        ("markush", "3. Recherche Markush & Analogues PubChem"),
                    ):
                        cls = "glass-btn glass-btn-accent" if rxn_cfg["sub"] == k else "glass-btn"
                        _btn(lbl, classes=cls, on_click=lambda _e, key=k: (rxn_cfg.update(sub=key), refresh_tab()))

                # ======================================================
                # SOUS-ONGLET 1 : CONSTRUCTEUR EN CARTES & STŒCHIOMÉTRIE
                # ======================================================
                if rxn_cfg["sub"] == "stoich":
                    _sync_rxn_smiles(state)
                    r_list = [s for s in state.get("rxn_reactants", []) if parse_smiles(s) is not None]
                    p_list = [s for s in state.get("rxn_products", []) if parse_smiles(s) is not None]

                    # Sécurise l'index du réactif limitant
                    if not r_list:
                        rxn_cfg["limiting_idx"] = 0
                    elif int(rxn_cfg["limiting_idx"]) >= len(r_list) or int(rxn_cfg["limiting_idx"]) < 0:
                        rxn_cfg["limiting_idx"] = 0

                    conds = state.setdefault("rxn_conditions", {"solvent": "", "temp": "", "duration": ""})
                    edit_target = state.get("rxn_edit_target")

                    # Slots de noms (cartes + lignes de stœchiométrie) pour enrichissement PubChem async
                    card_name_slots: dict[str, list[Any]] = {}
                    row_name_slots: dict[str, Any] = {}

                    # Actions sur les cartes molécules
                    def _delete_mol(side_key: str, idx: int):
                        lst = list(state.get(side_key, []))
                        if 0 <= idx < len(lst):
                            lst.pop(idx)
                            state[side_key] = lst
                            if side_key == "rxn_reactants":
                                lim = int(rxn_cfg["limiting_idx"])
                                if idx == lim:
                                    rxn_cfg["limiting_idx"] = 0
                                elif idx < lim:
                                    rxn_cfg["limiting_idx"] = lim - 1
                            if edit_target and edit_target.get("side") == side_key:
                                state["rxn_edit_target"] = None
                            _sync_rxn_smiles(state)
                            on_basket_change()
                            refresh_tab()

                    def _swap_mol(side_key: str, idx: int):
                        other_key = "rxn_products" if side_key == "rxn_reactants" else "rxn_reactants"
                        src = list(state.get(side_key, []))
                        if 0 <= idx < len(src):
                            smi = src.pop(idx)
                            state[side_key] = src
                            state.setdefault(other_key, []).append(smi)
                            if side_key == "rxn_reactants":
                                lim = int(rxn_cfg["limiting_idx"])
                                if idx == lim:
                                    rxn_cfg["limiting_idx"] = 0
                                elif idx < lim:
                                    rxn_cfg["limiting_idx"] = lim - 1
                            if edit_target and edit_target.get("side") == side_key:
                                state["rxn_edit_target"] = None
                            _sync_rxn_smiles(state)
                            on_basket_change()
                            refresh_tab()

                    def _send_to_2d(side_key: str, idx: int, smi: str, role_lbl: str):
                        state["rxn_edit_target"] = {
                            "side": side_key,
                            "idx": idx,
                            "orig_smiles": smi,
                            "label": role_lbl,
                        }
                        on_molecule_change(smi)
                        ui.notify(
                            f"{role_lbl} envoyé dans le dessinateur 2D — modifiez puis cliquez sur « Remplacer ».",
                            type="info",
                        )
                        refresh_tab()

                    def _apply_from_2d(side_key: str, idx: int, role_lbl: str):
                        new_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
                        if parse_smiles(new_smi) is None:
                            ui.notify("Molécule active du dessinateur 2D invalide.", type="negative")
                            return
                        lst = list(state.get(side_key, []))
                        if 0 <= idx < len(lst):
                            lst[idx] = new_smi
                            state[side_key] = lst
                            state["rxn_edit_target"] = None
                            _sync_rxn_smiles(state)
                            on_basket_change()
                            refresh_tab()
                            ui.notify(f"{role_lbl} mis à jour depuis le dessinateur 2D !", type="positive")

                    def _render_mol_card(side_key: str, idx: int, smi: str):
                        m = parse_smiles(smi)
                        if m is None:
                            return
                        is_reac = side_key == "rxn_reactants"
                        role_short = f"R{idx + 1}" if is_reac else f"P{idx + 1}"
                        role_full = f"Réactif {idx + 1}" if is_reac else f"Produit {idx + 1}"
                        formula = rdMolDescriptors.CalcMolFormula(m)
                        mw = round(float(Descriptors.MolWt(m)), 2)
                        svg = render_2d_svg(m, width=156, height=96)
                        is_editing = bool(
                            edit_target
                            and edit_target.get("side") == side_key
                            and int(edit_target.get("idx", -1)) == idx
                        )
                        is_limiting = is_reac and int(rxn_cfg["limiting_idx"]) == idx
                        border_col = (
                            "var(--accent)"
                            if is_editing
                            else ("var(--cyan)" if is_limiting else "rgba(148,163,184,0.22)")
                        )
                        swap_title = "Basculer vers les produits" if is_reac else "Basculer vers les réactifs"

                        with ui.element("div").style(
                            f"width:176px;display:flex;flex-direction:column;justify-content:space-between;"
                            f"padding:9px;border-radius:14px;border:1px solid {border_col};"
                            f"background:rgba(148,163,184,0.06);backdrop-filter:blur(10px);"
                        ):
                            # Barre haute : badge rôle + 3 micro-actions directes
                            with ui.element("div").style("display:flex;align-items:center;justify-content:space-between;gap:4px;margin-bottom:6px;"):
                                pill_cls = "stock-ok" if is_reac else "stock-low"
                                lim_dot = ' <span title="Réactif limitant" style="color:var(--cyan);font-size:0.72rem;">●</span>' if is_limiting else ""
                                safe_html(f'<span class="stock-pill {pill_cls}" style="padding:1px 7px;font-size:0.70rem;">{role_short}{lim_dot}</span>')

                                with ui.element("div").style("display:flex;gap:3px;"):
                                    micro_st = "padding:2px 6px;font-size:0.74rem;min-width:24px;line-height:1.2;border-radius:7px;"
                                    _btn(
                                        "✎",
                                        style=micro_st,
                                        props='title="Envoyer dans le dessinateur 2D"',
                                        on_click=lambda _e, sk=side_key, i=idx, s=smi, rf=role_full: _send_to_2d(sk, i, s, rf),
                                    )
                                    _btn(
                                        "⇄",
                                        style=micro_st,
                                        props=f'title="{swap_title}"',
                                        on_click=lambda _e, sk=side_key, i=idx: _swap_mol(sk, i),
                                    )
                                    _btn(
                                        "✕",
                                        classes="glass-btn glass-btn-danger",
                                        style=micro_st,
                                        props='title="Supprimer cette molécule"',
                                        on_click=lambda _e, sk=side_key, i=idx: _delete_mol(sk, i),
                                    )

                            # Miniature 2D
                            safe_html(f'<div class="svg-box-adaptive" style="padding:3px;margin-bottom:6px;">{svg}</div>')

                            # Nom (enrichi) + formule / MW
                            cached = rxn_cfg["name_cache"].get(smi)
                            init_name = str(cached["trivial"]) if cached else formula
                            ns = ui.element("div")
                            with ns:
                                safe_html(
                                    f'<div title="{html.escape(init_name)}" style="font-weight:700;font-size:0.80rem;'
                                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--txt);">'
                                    f'{html.escape(init_name)}</div>'
                                )
                            card_name_slots.setdefault(smi, []).append(ns)
                            safe_html(
                                f'<div class="muted-small" style="font-family:\'JetBrains Mono\',monospace;font-size:0.71rem;">'
                                f'{html.escape(formula)} · {mw} g/mol</div>'
                            )

                            # Action rapide de remplacement si cette carte est en édition 2D
                            if is_editing:
                                _btn(
                                    "✓ Appliquer le 2D",
                                    classes="glass-btn glass-btn-accent",
                                    style="width:100%;margin-top:6px;padding:4px 6px;font-size:0.72rem;",
                                    on_click=lambda _e, sk=side_key, i=idx, rf=role_full: _apply_from_2d(sk, i, rf),
                                )

                    def _render_add_card(side_key: str, label: str):
                        el = ui.element("button").classes("glass-btn").style(
                            "width:118px;min-height:178px;display:flex;flex-direction:column;align-items:center;"
                            "justify-content:center;gap:6px;border-radius:14px;"
                            "border:1.5px dashed rgba(148,163,184,0.38);background:transparent;cursor:pointer;"
                        )
                        el.on("click", lambda _e, sk=side_key: open_add_modal(sk))
                        with el:
                            safe_html(
                                f'<span style="font-size:1.45rem;line-height:1;color:var(--accent);">＋</span>'
                                f'<span style="font-size:0.76rem;font-weight:600;">{html.escape(label)}</span>'
                            )

                    # --- CARTE PRINCIPALE : CONSTRUCTEUR VISUEL ---
                    with ui.element("div").classes("hud-card").style("margin-bottom:16px;"):
                        with ui.element("div").style("display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-bottom:12px;"):
                            safe_html(
                                '<div><div class="section-kicker">CONSTRUCTEUR RÉACTIONNEL VISUEL</div>'
                                '<h2 style="margin:2px 0 0 0;font-size:1.15rem;">Schéma &amp; Conditions de Réaction</h2></div>'
                            )
                            if r_list or p_list:
                                def _clear_all(_e=None):
                                    state["rxn_reactants"] = []
                                    state["rxn_products"] = []
                                    state["rxn_edit_target"] = None
                                    rxn_cfg["limiting_idx"] = 0
                                    _sync_rxn_smiles(state)
                                    on_basket_change()
                                    refresh_tab()

                                _btn(
                                    "Vider",
                                    classes="glass-btn glass-btn-danger",
                                    style="padding:5px 11px;font-size:0.76rem;",
                                    on_click=_clear_all,
                                )

                        # Flux : [ Cartes Réactifs + Ajouter ] -> [ Flèche / Prédire / Conditions ] -> [ Cartes Produits + Ajouter ]
                        with ui.element("div").style(
                            "display:grid;grid-template-columns:minmax(220px,1fr) minmax(215px,245px) minmax(220px,1fr);"
                            "gap:16px;align-items:start;"
                        ):
                            # COLONNE GAUCHE : RÉACTIFS
                            with ui.element("div"):
                                safe_html(f'<div class="section-kicker" style="margin-bottom:8px;">RÉACTIFS ({len(r_list)})</div>')
                                with ui.element("div").style("display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;"):
                                    for i, smi in enumerate(r_list):
                                        _render_mol_card("rxn_reactants", i, smi)
                                    _render_add_card("rxn_reactants", "Ajouter")

                            # COLONNE CENTRALE : FLÈCHE, PRÉDICTION & CONDITIONS LÉGÈRES
                            with ui.element("div").style(
                                "display:flex;flex-direction:column;align-items:center;gap:8px;padding-top:22px;"
                            ):
                                # Flèche réactionnelle SVG
                                safe_html(
                                    '<svg viewBox="0 0 220 26" style="width:100%;max-width:220px;height:26px;overflow:visible;">'
                                    '<line x1="6" y1="13" x2="202" y2="13" stroke="var(--accent)" stroke-width="2.4" stroke-linecap="round"/>'
                                    '<polyline points="193,6 205,13 193,20" fill="none" stroke="var(--accent)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
                                    '</svg>'
                                )

                                # Bouton Prédire le produit + sélecteur discret de modèle (Auto par défaut)
                                with ui.element("div").style("display:flex;width:100%;gap:4px;"):
                                    _btn(
                                        "Prédire le produit",
                                        classes="glass-btn glass-btn-accent",
                                        style="flex:1;padding:7px 10px;font-size:0.79rem;",
                                        on_click=lambda _e: run_prediction_flow("__auto__"),
                                    )
                                    _btn(
                                        "▾",
                                        classes="glass-btn",
                                        style="padding:7px 10px;font-size:0.79rem;",
                                        props='title="Ouvrir le catalogue des réactions"',
                                        on_click=lambda _e: open_catalog_modal(),
                                    )

                                # Champs optionnels légers sous la flèche (Solvant, Température, Durée)
                                in_solv = ui.input(
                                    placeholder="Solvant (ex : THF, DCM)",
                                    value=str(conds.get("solvent", "")),
                                ).props("outlined dense").style("width:100%;")
                                in_solv.on_value_change(lambda e: conds.update(solvent=clean_val(e.value)))

                                with ui.element("div").style("display:grid;grid-template-columns:1fr 1fr;gap:6px;width:100%;"):
                                    in_temp = ui.input(
                                        placeholder="T° (ex : 25 °C)",
                                        value=str(conds.get("temp", "")),
                                    ).props("outlined dense")
                                    in_dur = ui.input(
                                        placeholder="Durée (ex : 2 h)",
                                        value=str(conds.get("duration", "")),
                                    ).props("outlined dense")
                                    in_temp.on_value_change(lambda e: conds.update(temp=clean_val(e.value)))
                                    in_dur.on_value_change(lambda e: conds.update(duration=clean_val(e.value)))

                            # COLONNE DROITE : PRODUITS
                            with ui.element("div"):
                                safe_html(f'<div class="section-kicker" style="margin-bottom:8px;">PRODUITS ({len(p_list)})</div>')
                                with ui.element("div").style("display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;"):
                                    for i, smi in enumerate(p_list):
                                        _render_mol_card("rxn_products", i, smi)
                                    _render_add_card("rxn_products", "Ajouter")

                    # --- CARTE SECONDAIRE : GRILLE DE STŒCHIOMÉTRIE SOUPLE ---
                    with ui.element("div").classes("hud-card"):
                        if not r_list:
                            safe_html(
                                '<div class="section-kicker">GRILLE DE STŒCHIOMÉTRIE</div>'
                                '<p class="muted-text" style="margin-top:8px;">Ajoutez au moins un réactif ci-dessus pour calculer les quantités stœchiométriques.</p>'
                            )
                            return

                        entries: list[dict[str, Any]] = []
                        for idx, smi in enumerate(r_list):
                            m = parse_smiles(smi)
                            if m is None:
                                continue
                            key = f"R{idx}_{smi}"
                            if key not in rxn_cfg["row_params"]:
                                smi_def = rxn_cfg["smi_defaults"].get(smi, {})
                                rxn_cfg["row_params"][key] = {
                                    "eq": 1.0 if idx == int(rxn_cfg["limiting_idx"]) else float(smi_def.get("eq", 1.2)),
                                    "dens": float(smi_def.get("dens", 1.0)),
                                    "qual_mode": str(smi_def.get("qual_mode", "%")),  # "%" | "M"
                                    "qual_val": float(smi_def.get("qual_val", 100.0)),
                                }
                            entries.append({
                                "key": key,
                                "idx": idx,
                                "is_reactant": True,
                                "role": f"R{idx + 1}",
                                "role_long": f"Réactif {idx + 1}",
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
                                rxn_cfg["row_params"][key] = {
                                    "eq": 1.0,
                                    "dens": 1.0,
                                    "qual_mode": "%",
                                    "qual_val": 100.0,
                                }
                            entries.append({
                                "key": key,
                                "idx": idx,
                                "is_reactant": False,
                                "role": f"P{idx + 1}",
                                "role_long": f"Produit {idx + 1}",
                                "smiles": smi,
                                "formula": rdMolDescriptors.CalcMolFormula(m),
                                "mw": round(float(Descriptors.MolWt(m)), 2),
                                "exact_mw": float(Descriptors.ExactMolWt(m)),
                            })

                        lim_idx = int(rxn_cfg["limiting_idx"])
                        lim_ent = next((e for e in entries if e["is_reactant"] and e["idx"] == lim_idx), entries[0])
                        rxn_cfg["row_params"][lim_ent["key"]]["eq"] = 1.0

                        # État calculé courant (utilisé par l'affichage et par "Copier le tableau")
                        computed_rows: list[dict[str, Any]] = []
                        computed_kpis: dict[str, Any] = {}
                        calc_slots: dict[str, Any] = {}
                        kpi_slot = ui.element("div")

                        def _compute_all():
                            computed_rows.clear()
                            computed_kpis.clear()

                            q_start = max(0.0, float(rxn_cfg["start_qty"] or 0.0))
                            u_start = str(rxn_cfg["start_unit"] or "mg")
                            lp = rxn_cfg["row_params"][lim_ent["key"]]
                            l_mw = max(1e-6, float(lim_ent["mw"]))
                            l_dens = max(0.05, float(lp.get("dens", 1.0)))
                            l_qmode = str(lp.get("qual_mode", "%"))
                            l_qval = max(1e-4, float(lp.get("qual_val", 100.0)))
                            l_purity_frac = min(1.0, l_qval / 100.0) if l_qmode == "%" else 1.0

                            # Calcul de n_lim (mmol) selon l'unité de départ + pureté (%) ou concentration (M)
                            if u_start == "mmol":
                                ref_mmol = q_start
                            elif u_start == "mg":
                                ref_mmol = (q_start * l_purity_frac) / l_mw
                            elif u_start == "g":
                                ref_mmol = (q_start * 1000.0 * l_purity_frac) / l_mw
                            elif u_start == "mL":
                                if l_qmode == "M":
                                    ref_mmol = q_start * l_qval  # mL * (mmol/mL)
                                else:
                                    ref_mmol = (q_start * l_dens * 1000.0 * l_purity_frac) / l_mw
                            else:
                                ref_mmol = q_start / l_mw

                            for ent in entries:
                                p = rxn_cfg["row_params"][ent["key"]]
                                is_lim = ent["key"] == lim_ent["key"]
                                eq_v = 1.0 if is_lim else max(0.0, float(p.get("eq", 1.0)))
                                dens_v = max(0.05, float(p.get("dens", 1.0)))
                                q_mode = str(p.get("qual_mode", "%"))
                                q_val = max(1e-4, float(p.get("qual_val", 100.0)))

                                mmol = eq_v * ref_mmol
                                pure_mg = mmol * ent["mw"]

                                if ent["is_reactant"] and q_mode == "M":
                                    weigh_mg = pure_mg
                                    vol_ml = mmol / q_val if q_val > 0 else 0.0
                                    qual_str = f"{q_val:g} M"
                                    mass_tag = "pur"
                                elif ent["is_reactant"]:
                                    p_frac = min(1.0, q_val / 100.0)
                                    weigh_mg = pure_mg / p_frac if p_frac > 0 else pure_mg
                                    vol_ml = (weigh_mg / dens_v) / 1000.0
                                    qual_str = f"{q_val:g} %"
                                    mass_tag = "brut" if q_val < 99.95 else ""
                                else:
                                    weigh_mg = pure_mg
                                    vol_ml = (weigh_mg / dens_v) / 1000.0
                                    qual_str = "—"
                                    mass_tag = "théo."

                                cached_nm = rxn_cfg["name_cache"].get(ent["smiles"], {}).get("trivial", "")
                                computed_rows.append({
                                    "role": ent["role"] + (" (Lim.)" if is_lim else ""),
                                    "name": cached_nm or ent["formula"],
                                    "formula": ent["formula"],
                                    "mw": ent["mw"],
                                    "eq": eq_v,
                                    "qual": qual_str,
                                    "dens": dens_v if (ent["is_reactant"] and q_mode != "M") else None,
                                    "mmol": mmol,
                                    "mass_mg": weigh_mg,
                                    "vol_ml": vol_ml,
                                    "is_reactant": ent["is_reactant"],
                                })

                                slot = calc_slots.get(ent["key"])
                                if slot is not None:
                                    slot.clear()
                                    with slot:
                                        m_lbl = f"Masse ({mass_tag})" if mass_tag else "Masse"
                                        v_lbl = "Vol. sol." if (ent["is_reactant"] and q_mode == "M") else "Vol. liq."
                                        vol_cell = (
                                            f'<div><span class="muted-small">{v_lbl}</span><br>'
                                            f'<strong>{_fmt_vol_ml(vol_ml)}</strong></div>'
                                            if ent["is_reactant"]
                                            else '<div><span class="muted-small">Base</span><br><span class="muted-small">100 %</span></div>'
                                        )
                                        safe_html(
                                            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;'
                                            f'font-family:\'JetBrains Mono\',monospace;font-size:0.79rem;">'
                                            f'<div><span class="muted-small">mmol</span><br><strong>{mmol:.3f}</strong></div>'
                                            f'<div><span class="muted-small">{m_lbl}</span><br><strong style="color:var(--cyan);">{_fmt_mass(weigh_mg)}</strong></div>'
                                            f'{vol_cell}'
                                            f'</div>'
                                        )

                            p1_ent = next((e for e in entries if not e["is_reactant"]), None)
                            sum_r_mw = sum(e["mw"] for e in entries if e["is_reactant"])
                            kpi_slot.clear()
                            if p1_ent and sum_r_mw > 0:
                                p1_eq = max(0.0, float(rxn_cfg["row_params"][p1_ent["key"]].get("eq", 1.0)))
                                theo_mg = ref_mmol * p1_eq * p1_ent["mw"]
                                exp_val = max(0.0, float(rxn_cfg["exp_qty"] or 0.0))
                                exp_mg = exp_val * 1000.0 if rxn_cfg["exp_unit"] == "g" else exp_val
                                y_pct = (exp_mg / theo_mg * 100.0) if (theo_mg > 0 and exp_mg > 0) else 0.0
                                atom_ec = (p1_ent["mw"] / sum_r_mw) * 100.0
                                dmz = p1_ent["exact_mw"] - lim_ent["exact_mw"]
                                y_txt = f"{y_pct:.1f} %" if exp_mg > 0 else "—"

                                computed_kpis.update(
                                    theo_mg=theo_mg,
                                    exp_mg=exp_mg,
                                    y_pct=y_pct,
                                    atom_ec=atom_ec,
                                    dmz=dmz,
                                    lim_role=lim_ent["role"],
                                )
                                with kpi_slot:
                                    safe_html(
                                        f"""
                                        <div class="kpi-grid" style="margin-top:14px;">
                                          <div class="kpi-card"><div class="kpi-label">Masse théorique P1 (100 %)</div><div class="kpi-value" style="color:var(--cyan);">{_fmt_mass(theo_mg)}</div></div>
                                          <div class="kpi-card"><div class="kpi-label">Rendement expérimental</div><div class="kpi-value" style="color:var(--accent);">{y_txt}</div></div>
                                          <div class="kpi-card"><div class="kpi-label">Économie d'atomes</div><div class="kpi-value">{atom_ec:.1f} %</div></div>
                                          <div class="kpi-card"><div class="kpi-label">Suivi MS (Δ m/z {html.escape(lim_ent["role"])}→P1)</div><div class="kpi-value">{dmz:+.4f} Da</div></div>
                                        </div>
                                        """
                                    )

                        def _build_export_text(fmt: str) -> str:
                            cond_parts = []
                            if clean_val(conds.get("solvent")):
                                cond_parts.append(f"Solvant : {clean_val(conds['solvent'])}")
                            if clean_val(conds.get("temp")):
                                cond_parts.append(f"T° : {clean_val(conds['temp'])}")
                            if clean_val(conds.get("duration")):
                                cond_parts.append(f"Durée : {clean_val(conds['duration'])}")
                            cond_line = " | ".join(cond_parts)

                            headers = ["Rôle", "Composé", "Formule", "MW (g/mol)", "Équiv.", "Pureté / Conc.", "mmol", "Masse (mg)", "Volume (mL)"]
                            rows_data = []
                            for r in computed_rows:
                                rows_data.append([
                                    str(r["role"]),
                                    str(r["name"]),
                                    str(r["formula"]),
                                    f"{r['mw']:.2f}",
                                    f"{r['eq']:.2f}",
                                    str(r["qual"]),
                                    f"{r['mmol']:.3f}",
                                    f"{r['mass_mg']:.1f}",
                                    f"{r['vol_ml']:.3f}" if r["is_reactant"] else "—",
                                ])

                            if fmt == "tsv":
                                lines = []
                                if cond_line:
                                    lines.append(f"Conditions\t{cond_line}")
                                lines.append("\t".join(headers))
                                for rd in rows_data:
                                    lines.append("\t".join(rd))
                                if computed_kpis:
                                    lines.append("")
                                    lines.append(f"Masse théorique P1 (mg)\t{computed_kpis['theo_mg']:.1f}")
                                    if computed_kpis["exp_mg"] > 0:
                                        lines.append(f"Rendement (%)\t{computed_kpis['y_pct']:.1f}")
                                    lines.append(f"Économie d'atomes (%)\t{computed_kpis['atom_ec']:.1f}")
                                return "\n".join(lines)

                            # Markdown
                            lines = []
                            if cond_line:
                                lines.append(f"**Conditions :** {cond_line}")
                                lines.append("")
                            lines.append("| " + " | ".join(headers) + " |")
                            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                            for rd in rows_data:
                                safe_cells = [c.replace("|", "/") for c in rd]
                                lines.append("| " + " | ".join(safe_cells) + " |")
                            if computed_kpis:
                                summary = f"Masse théorique P1 : **{_fmt_mass(computed_kpis['theo_mg'])}** · Économie d'atomes : **{computed_kpis['atom_ec']:.1f} %**"
                                if computed_kpis["exp_mg"] > 0:
                                    summary += f" · Rendement : **{computed_kpis['y_pct']:.1f} %**"
                                lines.append("")
                                lines.append(summary)
                            return "\n".join(lines)

                        # En-tête de la grille + bouton "Copier le tableau"
                        with ui.element("div").style(
                            "display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:12px;"
                        ):
                            safe_html(
                                '<div><div class="section-kicker">GRILLE DE STŒCHIOMÉTRIE SOUPLE</div>'
                                f'<span class="muted-small">Réactif limitant actuel : <strong style="color:var(--cyan);">{html.escape(lim_ent["role_long"])}</strong> (cliquez sur le bouton radio d\'une ligne pour changer)</span></div>'
                            )
                            copy_btn = _btn("Copier le tableau ▾", style="padding:6px 12px;font-size:0.79rem;")
                            with copy_btn:
                                with ui.menu():
                                    ui.menu_item(
                                        "Format Markdown (.md)",
                                        on_click=lambda: _copy_to_clipboard(_build_export_text("md"), "Markdown"),
                                    )
                                    ui.menu_item(
                                        "Format Excel / Tableur (TSV)",
                                        on_click=lambda: _copy_to_clipboard(_build_export_text("tsv"), "Excel / TSV"),
                                    )

                        # Bandeau quantité de départ du limitant + masse isolée produit 1
                        with ui.element("div").style(
                            "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:14px;"
                        ):
                            with ui.element("div").classes("kv-item").style("display:grid;grid-template-columns:1fr 95px;gap:8px;align-items:center;"):
                                in_start_qty = ui.number(
                                    label=f"Quantité engagée ({lim_ent['role']} · Limitant)",
                                    value=float(rxn_cfg["start_qty"]),
                                    min=0.001,
                                    step=10.0 if rxn_cfg["start_unit"] == "mg" else 0.5,
                                    format="%.3f" if rxn_cfg["start_unit"] in ("g", "mmol", "mL") else "%.1f",
                                ).props("outlined dense")
                                sel_start_unit = ui.select(
                                    ["mg", "g", "mmol", "mL"],
                                    value=str(rxn_cfg["start_unit"]),
                                    label="Unité",
                                ).props("outlined dense")

                                in_start_qty.on_value_change(
                                    lambda e: (rxn_cfg.update(start_qty=float(e.value or 0.0)), _compute_all())
                                )
                                sel_start_unit.on_value_change(
                                    lambda e: (rxn_cfg.update(start_unit=str(e.value or "mg")), _compute_all())
                                )

                            if p_list:
                                with ui.element("div").classes("kv-item").style("display:grid;grid-template-columns:1fr 85px;gap:8px;align-items:center;"):
                                    in_exp_qty = ui.number(
                                        label="Masse isolée de P1 (rendement)",
                                        value=float(rxn_cfg["exp_qty"]),
                                        min=0.0,
                                        step=10.0 if rxn_cfg["exp_unit"] == "mg" else 0.1,
                                        format="%.3f" if rxn_cfg["exp_unit"] == "g" else "%.1f",
                                    ).props("outlined dense")
                                    sel_exp_unit = ui.select(
                                        ["mg", "g"],
                                        value=str(rxn_cfg["exp_unit"]),
                                        label="Unité",
                                    ).props("outlined dense")

                                    in_exp_qty.on_value_change(
                                        lambda e: (rxn_cfg.update(exp_qty=float(e.value or 0.0)), _compute_all())
                                    )
                                    sel_exp_unit.on_value_change(
                                        lambda e: (rxn_cfg.update(exp_unit=str(e.value or "mg")), _compute_all())
                                    )

                        # Lignes de la grille de stœchiométrie
                        with ui.element("div").style("display:flex;flex-direction:column;gap:8px;"):
                            for ent in entries:
                                k = ent["key"]
                                p_cur = rxn_cfg["row_params"][k]
                                is_lim = ent["is_reactant"] and ent["idx"] == lim_idx
                                pill_cls = "stock-ok" if ent["is_reactant"] else "stock-low"
                                row_border = "border:1px solid var(--cyan);" if is_lim else ""

                                with ui.element("div").classes("kv-item").style(
                                    f"display:grid;grid-template-columns:34px minmax(165px,1.35fr) 78px 142px 78px minmax(195px,1.5fr);"
                                    f"gap:8px;align-items:center;{row_border}"
                                ):
                                    # Col 1 : Bouton radio Réactif limitant
                                    if ent["is_reactant"]:
                                        def _set_limiting(_e=None, target_i=ent["idx"], target_k=k):
                                            rxn_cfg["limiting_idx"] = target_i
                                            rxn_cfg["row_params"][target_k]["eq"] = 1.0
                                            refresh_tab()

                                        radio_el = ui.element("button").style(
                                            "width:22px;height:22px;border-radius:50%;display:flex;align-items:center;"
                                            "justify-content:center;cursor:pointer;padding:0;background:transparent;"
                                            f"border:2px solid {'var(--cyan)' if is_lim else 'rgba(148,163,184,0.45)'};"
                                        ).props(
                                            f'role="radio" aria-checked="{"true" if is_lim else "false"}" '
                                            f'title="Définir {ent["role_long"]} comme réactif limitant"'
                                        )
                                        radio_el.on("click", _set_limiting)
                                        with radio_el:
                                            if is_lim:
                                                safe_html('<span style="width:10px;height:10px;border-radius:50%;background:var(--cyan);display:block;"></span>')
                                    else:
                                        safe_html('<span class="muted-small" style="text-align:center;display:block;">→</span>')

                                    # Col 2 : Rôle, formule, MW, nom trivial
                                    with ui.element("div").style("min-width:0;"):
                                        lim_badge = (
                                            ' <span style="font-size:0.66rem;font-weight:700;color:var(--cyan);letter-spacing:0.04em;">LIMITANT</span>'
                                            if is_lim
                                            else ""
                                        )
                                        safe_html(
                                            f'<div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">'
                                            f'<span class="stock-pill {pill_cls}" style="padding:1px 7px;font-size:0.70rem;">{html.escape(ent["role"])}</span>'
                                            f'<span class="mono-tag">{html.escape(ent["formula"])} · {ent["mw"]}</span>'
                                            f'{lim_badge}</div>'
                                        )
                                        cached = rxn_cfg["name_cache"].get(ent["smiles"])
                                        ns = ui.element("div").style("margin-top:3px;")
                                        with ns:
                                            if cached:
                                                safe_html(
                                                    f'<strong style="font-size:0.82rem;color:var(--txt);">{html.escape(str(cached["trivial"]))}</strong> '
                                                    f'<span class="muted-small">· CAS {html.escape(str(cached["cas"]))}</span>'
                                                )
                                            else:
                                                safe_html(f'<span class="muted-small">{html.escape(ent["smiles"])}</span>')
                                        row_name_slots[k] = ns

                                    # Col 3 : Équivalents
                                    if is_lim:
                                        safe_html(
                                            '<div style="text-align:center;font-family:\'JetBrains Mono\',monospace;'
                                            'font-size:0.80rem;padding:6px 4px;border-radius:8px;background:rgba(56,189,248,0.10);color:var(--cyan);">'
                                            '<strong>1.00</strong><div style="font-size:0.65rem;opacity:0.85;">réf.</div></div>'
                                        )
                                    else:
                                        in_eq = ui.number(
                                            label="Équiv.",
                                            value=float(p_cur["eq"]),
                                            min=0.01,
                                            step=0.1,
                                            format="%.2f",
                                        ).props("outlined dense")

                                        def _on_eq_ch(e, key_k=k, s=ent["smiles"]):
                                            val = float(e.value or 1.0)
                                            rxn_cfg["row_params"][key_k]["eq"] = val
                                            rxn_cfg["smi_defaults"].setdefault(s, {})["eq"] = val
                                            _compute_all()

                                        in_eq.on_value_change(_on_eq_ch)

                                    # Col 4 : Pureté (%) OU Concentration (M)
                                    if ent["is_reactant"]:
                                        with ui.element("div").style("display:grid;grid-template-columns:1fr 56px;gap:4px;align-items:center;"):
                                            in_qv = ui.number(
                                                label="Pureté" if p_cur["qual_mode"] == "%" else "Conc.",
                                                value=float(p_cur["qual_val"]),
                                                min=0.01,
                                                max=100.0 if p_cur["qual_mode"] == "%" else 50.0,
                                                step=1.0 if p_cur["qual_mode"] == "%" else 0.1,
                                                format="%.1f" if p_cur["qual_mode"] == "%" else "%.2f",
                                            ).props("outlined dense")
                                            sel_qm = ui.select(
                                                ["%", "M"],
                                                value=str(p_cur["qual_mode"]),
                                            ).props("outlined dense")

                                            def _on_qv_ch(e, key_k=k, s=ent["smiles"]):
                                                val = float(e.value or 100.0)
                                                rxn_cfg["row_params"][key_k]["qual_val"] = val
                                                rxn_cfg["smi_defaults"].setdefault(s, {})["qual_val"] = val
                                                _compute_all()

                                            def _on_qm_ch(e, key_k=k, s=ent["smiles"]):
                                                new_m = str(e.value or "%")
                                                rxn_cfg["row_params"][key_k]["qual_mode"] = new_m
                                                rxn_cfg["row_params"][key_k]["qual_val"] = 100.0 if new_m == "%" else 1.0
                                                rxn_cfg["smi_defaults"].setdefault(s, {}).update(
                                                    qual_mode=new_m,
                                                    qual_val=rxn_cfg["row_params"][key_k]["qual_val"],
                                                )
                                                refresh_tab()

                                            in_qv.on_value_change(_on_qv_ch)
                                            sel_qm.on_value_change(_on_qm_ch)
                                    else:
                                        safe_html('<span class="muted-small" style="text-align:center;display:block;">—</span>')

                                    # Col 5 : Densité (uniquement utile hors mode concentration M)
                                    if ent["is_reactant"] and p_cur["qual_mode"] != "M":
                                        in_dn = ui.number(
                                            label="Dens.",
                                            value=float(p_cur["dens"]),
                                            min=0.1,
                                            step=0.05,
                                            format="%.2f",
                                        ).props("outlined dense")

                                        def _on_dn_ch(e, key_k=k, s=ent["smiles"]):
                                            val = float(e.value or 1.0)
                                            rxn_cfg["row_params"][key_k]["dens"] = val
                                            rxn_cfg["smi_defaults"].setdefault(s, {})["dens"] = val
                                            _compute_all()

                                        in_dn.on_value_change(_on_dn_ch)
                                    else:
                                        safe_html('<span class="muted-small" style="text-align:center;display:block;">—</span>')

                                    # Col 6 : Résultats calculés en direct
                                    calc_slots[k] = ui.element("div")

                        # Conteneur KPI en bas de la grille
                        kpi_slot.move(root.default_slot.children[-1])
                        _compute_all()

                        # Enrichissement asynchrone des noms PubChem (uniquement pour les SMILES absents du cache)
                        async def enrich_basket_names():
                            unique_smis = list(dict.fromkeys(e["smiles"] for e in entries))
                            for smi in unique_smis:
                                if rxn_cfg["render_gen"] != cur_gen:
                                    return
                                if smi in rxn_cfg["name_cache"]:
                                    continue
                                inf = await run.io_bound(get_chemical_identifiers, smi)
                                if rxn_cfg["render_gen"] != cur_gen:
                                    return
                                rxn_cfg["name_cache"][smi] = {
                                    "trivial": str(inf.get("trivial") or "Composé personnalisé"),
                                    "cas": str(inf.get("cas") or "Non trouvé"),
                                }
                                tr = rxn_cfg["name_cache"][smi]["trivial"]
                                cs = rxn_cfg["name_cache"][smi]["cas"]
                                for c_slot in card_name_slots.get(smi, []):
                                    c_slot.clear()
                                    with c_slot:
                                        safe_html(
                                            f'<div title="{html.escape(tr)}" style="font-weight:700;font-size:0.80rem;'
                                            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--txt);">'
                                            f'{html.escape(tr)}</div>'
                                        )
                                for ent in entries:
                                    if ent["smiles"] == smi:
                                        r_slot = row_name_slots.get(ent["key"])
                                        if r_slot is not None:
                                            r_slot.clear()
                                            with r_slot:
                                                safe_html(
                                                    f'<strong style="font-size:0.82rem;color:var(--txt);">{html.escape(tr)}</strong> '
                                                    f'<span class="muted-small">· CAS {html.escape(cs)}</span>'
                                                )
                            # Met à jour computed_rows avec les noms fraîchement résolus pour l'export
                            if rxn_cfg["render_gen"] == cur_gen:
                                _compute_all()

                        ui.timer(0.05, enrich_basket_names, once=True)

                # ======================================================
                # SOUS-ONGLET 2 : RÉTROSENTHÈSE IA
                # ======================================================
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
                                            rxn_cfg["limiting_idx"] = 0
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
                            if rxn_cfg["render_gen"] != cur_gen:
                                return
                            inf = await run.io_bound(get_chemical_identifiers, p_smi)
                            if rxn_cfg["render_gen"] != cur_gen:
                                return
                            slot.clear()
                            with slot:
                                safe_html(
                                    f'<strong style="font-size:0.85rem;color:var(--txt);">{html.escape(str(inf["trivial"]))}</strong>'
                                    f'<div class="muted-small">CAS {html.escape(str(inf["cas"]))} · <code>{html.escape(p_smi)}</code></div>'
                                )

                    ui.timer(0.05, enrich_retro_names, once=True)

                # ======================================================
                # SOUS-ONGLET 3 : MARKUSH & ANALOGUES PUBCHEM
                # ======================================================
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
                                            "+ Réactif",
                                            style="flex:1;padding:6px 10px;font-size:0.76rem;",
                                            on_click=add_mk_rea,
                                        )

        rxn_refresh_ref["fn"] = refresh_tab
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