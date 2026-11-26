import base64
import html
import json
import os
import secrets
import threading
import time
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from nicegui import app, run, ui
import requests
from rdkit import Chem

from chem_engine import (
    EXEMPLES_MOLECULES,
    build_3dmol_overlay_html,
    build_3dmol_quantum_html,
    build_3dmol_standard_html,
    build_chembase_send_url,
    clean_val,
    compute_quantum_orbitals,
    estimate_chemprop,
    extract_single_smiles,
    fetch_chembase_catalog,
    generate_3d_mol,
    get_chembase_config,
    get_chemical_identifiers,
    is_safe_public_url,
    lookup_chembase_by_cas,
    overlay_molecules_3d,
    parse_smiles,
    resolve_query_to_smiles,
    set_bond_length_3d,
)
from sketcher_2d import build_custom_sketcher_html, clean_graph_2d, graph_to_smiles
from theme import LOGO_SVG, THEME_TOGGLE_SVGS, inject_theme, safe_html
from views_spec_export import build_spec_and_export_views
from views_studio import build_studio_views

IS_RENDER = os.environ.get("RENDER") is not None
STORAGE_SECRET = os.environ.get("STORAGE_SECRET") or (
    secrets.token_urlsafe(32) if IS_RENDER else "chemstudio-local-dev"
)
_LAST_WAKE_PING = 0.0

DEFAULT_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">'
    '<defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#10b981"/>'
    '<stop offset="55%" stop-color="#38bdf8"/>'
    '<stop offset="100%" stop-color="#6366f1"/>'
    '</linearGradient></defs>'
    '<rect width="120" height="120" rx="26" fill="#081010"/>'
    '<polygon points="60,12 102,36 102,84 60,108 18,84 18,36" '
    'fill="none" stroke="url(#g)" stroke-width="9" stroke-linejoin="round"/>'
    '<text x="60" y="74" text-anchor="middle" font-family="Arial, sans-serif" '
    'font-size="40" font-weight="800" fill="#f8fafc">CS</text>'
    '</svg>'
)


def get_favicon_data_uri() -> str:
    b64 = base64.b64encode(DEFAULT_FAVICON_SVG.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _wake_chembase_async(base_url: str):
    """Réveille silencieusement ChemBase sur Render (au plus une fois toutes les 5 min)."""
    global _LAST_WAKE_PING
    url_clean = clean_val(base_url).rstrip("/")
    if not url_clean or not is_safe_public_url(url_clean):
        return
    now = time.time()
    if now - _LAST_WAKE_PING < 300:
        return
    _LAST_WAKE_PING = now

    def _ping():
        try:
            requests.get(f"{url_clean}/api/ping", timeout=15)
        except Exception:
            pass

    threading.Thread(target=_ping, daemon=True).start()


# ------------------------------------------------------------------
# ROUTES FASTAPI : PING, DESSINATEUR 2D MAISON & VISUALISEURS 3DMOL
# ------------------------------------------------------------------
@app.get("/api/ping")
def api_ping():
    return JSONResponse(content={"ok": True, "app": "ChemStudio"})


@app.get("/api/studio/sketcher", response_class=HTMLResponse)
def serve_custom_sketcher(smiles: str = "", dark: int = 1):
    return HTMLResponse(content=build_custom_sketcher_html(clean_val(smiles), dark_mode=bool(dark)))


@app.post("/api/studio/clean2d")
async def api_clean_2d(request: Request):
    try:
        graph = await request.json()
        cleaned, smi, msg, bad_ids = clean_graph_2d(graph if isinstance(graph, dict) else {})
        if cleaned is None:
            return JSONResponse(content={"ok": False, "message": msg, "invalid_ids": bad_ids})
        return JSONResponse(content={"ok": True, "graph": cleaned, "smiles": smi, "message": msg, "invalid_ids": []})
    except Exception as ex:
        return JSONResponse(content={"ok": False, "message": f"Erreur : {ex}", "invalid_ids": []})


@app.get("/api/studio/3d/standard", response_class=HTMLResponse)
def serve_3d_standard(
    smiles: str = "",
    style: str = "Ball & Stick",
    ff: str = "MMFF94",
    seed: int = 42,
    show_h: int = 1,
    charges: int = 0,
    spin: int = 0,
    dark: int = 1,
    ai: int = -1,
    aj: int = -1,
    dist: float = 0.0,
):
    mol_3d = generate_3d_mol(clean_val(smiles), force_field=ff, seed=seed)
    if not mol_3d:
        return HTMLResponse(content="<html><body style='color:#94a3b8;font-family:sans-serif;padding:20px;'>Structure 3D indisponible (ou molécule supérieure à 120 atomes lourds).</body></html>")
    if ai >= 0 and aj >= 0 and dist > 0 and ai != aj:
        set_bond_length_3d(mol_3d, ai, aj, dist)
    return HTMLResponse(
        content=build_3dmol_standard_html(
            mol_3d,
            style=style,
            show_h=bool(show_h),
            color_charges=bool(charges),
            spin=bool(spin),
            dark_mode=bool(dark),
        )
    )


@app.get("/api/studio/3d/quantum", response_class=HTMLResponse)
def serve_3d_quantum(smiles: str = "", mode: str = "HOMO", dark: int = 1):
    mol_main = parse_smiles(clean_val(smiles))
    mol_3d = generate_3d_mol(clean_val(smiles))
    if not mol_main or not mol_3d:
        return HTMLResponse(content="<html><body style='color:#94a3b8;font-family:sans-serif;padding:20px;'>Calcul quantique indisponible.</body></html>")
    q_res = compute_quantum_orbitals(mol_main)
    if not q_res:
        return HTMLResponse(content="<html><body style='color:#94a3b8;font-family:sans-serif;padding:20px;'>Calcul quantique indisponible (ou molécule supérieure à 100 atomes lourds).</body></html>")
    coeffs = q_res["c_homo"] if mode == "HOMO" else q_res["c_lumo"]
    return HTMLResponse(content=build_3dmol_quantum_html(mol_3d, coeffs, mode=mode, dark_mode=bool(dark)))


@app.get("/api/studio/3d/overlay", response_class=HTMLResponse)
def serve_3d_overlay(s1: str = "", s2: str = "", dark: int = 1):
    m1, m2, _mcs, _rmsd = overlay_molecules_3d(clean_val(s1), clean_val(s2))
    if not m1 or not m2:
        return HTMLResponse(content="<html><body style='color:#94a3b8;font-family:sans-serif;padding:20px;'>Superposition indisponible.</body></html>")
    return HTMLResponse(content=build_3dmol_overlay_html(m1, m2, dark_mode=bool(dark)))


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


# ------------------------------------------------------------------
# PAGE PRINCIPALE CHEMSTUDIO v2.0
# ------------------------------------------------------------------
@ui.page("/")
def index_page(smiles: str = "", cas: str = ""):
    inject_theme()

    saved_dark = True
    dark = ui.dark_mode(value=saved_dark)
    default_cb_url, default_cb_key = get_chembase_config()

    initial_smi = clean_val(smiles) or "CC(=O)OC1=CC=CC=C1C(=O)O"
    if parse_smiles(initial_smi) is None:
        initial_smi = "CC(=O)OC1=CC=CC=C1C(=O)O"

    state: dict[str, Any] = {
        "tab": "t2d3d",
        "dark": saved_dark,
        "current_smiles": initial_smi,
        "last_sketcher_smi": initial_smi,
        "history": [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(=O)NC1=CC=C(O)C=C1",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        ],
        "rxn_reactants": ["CC(=O)O", "OCC"],
        "rxn_products": ["CC(=O)OCC"],
        "rxn_smiles": "CC(=O)O.OCC>>CC(=O)OCC",
        "ident": {"iupac": "Chargement...", "cas": "...", "trivial": "Chargement...", "cid": None},
        "chembase_match": None,
        "ident_seq": 0,
        "pubchem_timer": None,
        "chembase_url": default_cb_url,
        "chembase_api_key": default_cb_key,
    }
    if initial_smi not in state["history"]:
        state["history"].insert(0, initial_smi)

    def get_cb_creds() -> tuple[str, str]:
        u = clean_val(state.get("chembase_url", "")).rstrip("/")
        k = clean_val(state.get("chembase_api_key", ""))
        return u, k

    def sync_rxn_smiles():
        r_part = ".".join(state["rxn_reactants"]) if state["rxn_reactants"] else "CC(=O)O"
        p_part = ".".join(state["rxn_products"]) if state["rxn_products"] else "CC(=O)OCC"
        state["rxn_smiles"] = f"{r_part}>>{p_part}"

    # Réveil croisé silencieux de ChemBase sur Render (uniquement si configuré)
    cb_url_boot, _cb_key_boot = get_cb_creds()
    if cb_url_boot:
        _wake_chembase_async(cb_url_boot)

    init_mode = "dark" if saved_dark else "light"
    ui.timer(0.05, lambda: ui.run_javascript(f'window.cbApplyTheme("{init_mode}");'), once=True)

    tab_buttons: dict[str, Any] = {}
    tab_panels: dict[str, Any] = {}
    tab_refreshers: dict[str, Any] = {}

    # Modales globales
    examples_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
    with examples_dialog:
        examples_body = ui.element("div").classes("cb-modal-card")

    iupac_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
    with iupac_dialog:
        iupac_body = ui.element("div").classes("cb-modal-card")

    chembase_dialog = ui.dialog().props('transition-show="fade" transition-hide="fade"')
    with chembase_dialog:
        chembase_body = ui.element("div").classes("cb-modal-card").style("width:min(760px, 95vw) !important;")

    with ui.element("div").classes("app-shell"):
        # --- EN-TÊTE ---
        with ui.element("header").classes("top-header"):
            with ui.element("div").classes("brand-group"):
                with ui.element("div").classes("cb-logo-wrap"):
                    safe_html(LOGO_SVG)
                with ui.element("div").classes("brand-text"):
                    safe_html('<div class="section-kicker">COMPUTATIONAL CHEMISTRY &amp; ANALYTICAL SUITE</div>')
                    safe_html('<h1>ChemStudio <span class="version-pill">v2.0</span></h1>')

            with ui.element("div").style("flex:1;min-width:250px;max-width:460px;display:flex;gap:8px;align-items:center;"):
                search_input = ui.input(
                    placeholder="Rechercher CAS, Nom ou SMILES (ex : 50-78-2, Ibuprofen...)"
                ).props("outlined dense clearable").style("flex:1;")

                async def run_top_search(_e=None):
                    q = clean_val(search_input.value)
                    if not q:
                        return
                    ui.notify(f"Recherche de « {q} »...", type="info")
                    smi = await run.io_bound(resolve_query_to_smiles, q)
                    if smi:
                        set_active_molecule(smi)
                    else:
                        ui.notify("Composé introuvable dans PubChem.", type="negative")

                search_input.on("keydown.enter", run_top_search)
                _btn("Charger", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=run_top_search)

            with ui.element("div").classes("header-actions"):
                _btn("Exemples & Historique", on_click=lambda _e: open_examples_modal())
                basket_btn_slot = ui.element("div")
                cb_btn_slot = ui.element("div")

                def toggle_theme(_e=None):
                    state["dark"] = not state["dark"]
                    mode = "dark" if state["dark"] else "light"
                    ui.run_javascript(f'localStorage.setItem("cs_theme", "{mode}");')
                    ui.run_javascript(f'window.cbToggleThemeWave("{mode}");')

                    def after_wave():
                        dark.value = state["dark"]
                        ref = tab_refreshers.get(state["tab"])
                        if callable(ref):
                            ref()

                    ui.timer(0.1, after_wave, once=True)

                with (
                    ui.element("button")
                    .classes("glass-btn icon-btn")
                    .props('id="theme-toggle" title="Changer de thème" aria-label="Changer de thème"')
                    .on("click", toggle_theme)
                ):
                    safe_html(THEME_TOGGLE_SVGS)

        # --- BARRE DES 5 ONGLETS ---
        nav_defs = [
            ("t2d3d", "1. Atelier 2D, 3D & Quantique"),
            ("trxn", "2. Réactions, Rétrosynthèse & Markush"),
            ("tstruct", "3. Structure & ChemProp"),
            ("tspec", "4. Spectrométrie & RMN 2D"),
            ("texport", "5. Sécurité & Publication"),
        ]
        with ui.element("nav").classes("lg-nav").props('id="lg-nav"'):
            ui.element("div").classes("lg-pill").props('id="lg-pill"')
            for k, lbl in nav_defs:
                cls = "lg-tab active" if k == state["tab"] else "lg-tab"
                with ui.element("button").classes(cls).on("click", lambda _e, key=k: switch_tab(key)) as b:
                    safe_html(f"<span>{lbl}</span>")
                tab_buttons[k] = b

        ui.element("div").classes("hud-divider")

        # --- BANDEAU MOLÉCULE COMPACT (COMMUN AUX 5 ONGLETS) ---
        banner_slot = ui.element("div")

        def refresh_header_badges():
            basket_btn_slot.clear()
            with basket_btn_slot:
                r_n = len(state["rxn_reactants"])
                p_n = len(state["rxn_products"])
                _btn(f"Réaction ({r_n}R → {p_n}P)", on_click=lambda _e: switch_tab("trxn"))

            cb_url, cb_key = get_cb_creds()
            cb_connected = bool(cb_url and cb_key)
            cb_btn_slot.clear()
            with cb_btn_slot:
                dot_html = '<span class="status-dot status-ok"></span>' if cb_connected else ""
                _btn(
                    f'{dot_html}<span>ChemBase</span>',
                    on_click=lambda _e: open_chembase_modal(),
                )

        def render_molecule_banner():
            banner_slot.clear()
            single_smi = extract_single_smiles(state["current_smiles"])
            mol_main = parse_smiles(single_smi)
            ident = state["ident"]
            cb_match = state["chembase_match"]
            cb_url, cb_key = get_cb_creds()
            cb_connected = bool(cb_url and cb_key)

            with banner_slot:
                if not mol_main:
                    safe_html('<div class="mol-banner"><strong style="color:var(--danger);">Structure chimique invalide.</strong></div>')
                    return

                cprop = estimate_chemprop(mol_main)
                cas_val = clean_val(ident.get("cas")) or "Non trouvé"
                triv_val = clean_val(ident.get("trivial")) or "Composé personnalisé"
                copy_cas_js = html.escape(f"window.cbCopyText({json.dumps(cas_val)}, 'CAS')", quote=True)

                cb_badge_html = ""
                if cb_match:
                    cb_prod = clean_val(cb_match.get("Produit final") or cb_match.get("produit")) or triv_val
                    cb_badge_html = f'<span class="stock-pill stock-ok" title="Présent dans votre catalogue ChemBase">En base ChemBase · {html.escape(cb_prod)}</span>'
                elif cb_connected and cas_val not in ("...", "Non trouvé"):
                    cb_badge_html = '<span class="stock-pill stock-low">Absent de ChemBase</span>'

                with ui.element("div").classes("mol-banner"):
                    with ui.element("div").style("display:flex;align-items:center;gap:10px;flex-wrap:wrap;"):
                        safe_html(
                            f'<strong style="font-size:1rem;color:var(--txt);">{html.escape(triv_val)}</strong>'
                            f'<span class="mono-tag copyable-tag" title="Cliquer pour copier le CAS" onclick="{copy_cas_js}">CAS {html.escape(cas_val)}</span>'
                            f'<span class="mono-tag">{html.escape(cprop["formula"])}</span>'
                            f'<span class="mono-tag">{cprop["mw"]} g/mol</span>'
                            f'{cb_badge_html}'
                        )

                    with ui.element("div").style("display:flex;align-items:center;gap:7px;flex-wrap:wrap;"):
                        _btn("IUPAC & SMILES", style="padding:5px 10px;font-size:0.76rem;", on_click=lambda _e: open_iupac_modal(cprop))

                        def quick_add_r(_e=None):
                            state["rxn_reactants"].append(cprop["canonical_smiles"])
                            sync_rxn_smiles()
                            refresh_header_badges()
                            if state["tab"] == "trxn" and callable(tab_refreshers.get("trxn")):
                                tab_refreshers["trxn"]()
                            ui.notify(f"« {triv_val} » ajouté aux réactifs.", type="positive")

                        def quick_set_p(_e=None):
                            state["rxn_products"] = [cprop["canonical_smiles"]]
                            sync_rxn_smiles()
                            refresh_header_badges()
                            if state["tab"] == "trxn" and callable(tab_refreshers.get("trxn")):
                                tab_refreshers["trxn"]()
                            ui.notify(f"« {triv_val} » défini comme produit cible.", type="positive")

                        _btn("+ Réactif", style="padding:5px 10px;font-size:0.76rem;", on_click=quick_add_r)
                        _btn("Définir Produit", style="padding:5px 10px;font-size:0.76rem;", on_click=quick_set_p)

                        send_url = (
                            build_chembase_send_url(cas_val, triv_val, cprop["canonical_smiles"], override_url=cb_url)
                            if cb_connected
                            else ""
                        )
                        if send_url:
                            safe_html(
                                f'<a class="glass-btn glass-btn-accent" style="padding:5px 11px;font-size:0.76rem;" '
                                f'href="{html.escape(send_url)}" target="_blank" rel="noopener noreferrer">Envoyer vers ChemBase ↗</a>'
                            )

        async def resolve_banner_async(target_smi: str, seq_id: int):
            ident = await run.io_bound(get_chemical_identifiers, target_smi)
            if state["ident_seq"] != seq_id:
                return
            state["ident"] = ident
            cb_url, cb_key = get_cb_creds()
            cb_match = None
            if cb_url and cb_key and ident.get("cas") not in (None, "", "Non trouvé"):
                cb_match = await run.io_bound(lookup_chembase_by_cas, str(ident["cas"]), cb_url, cb_key)
            if state["ident_seq"] != seq_id:
                return
            state["chembase_match"] = cb_match
            render_molecule_banner()

        def set_active_molecule(new_smiles: str, switch_to_2d: bool = False, from_sketcher: bool = False):
            raw = clean_val(new_smiles)
            if not raw:
                return
            if ">>" in raw or ">" in raw:
                parts = raw.split(">")
                r_list = [s.strip() for s in parts[0].split(".") if parse_smiles(s.strip()) is not None]
                p_list = [s.strip() for s in parts[-1].split(".") if parse_smiles(s.strip()) is not None]
                if r_list:
                    state["rxn_reactants"] = r_list
                if p_list:
                    state["rxn_products"] = p_list
                sync_rxn_smiles()
                refresh_header_badges()
                single = r_list[0] if r_list else extract_single_smiles(raw)
                ui.notify("Réaction détectée et chargée dans l'onglet Réactions.", type="info")
            else:
                m = parse_smiles(raw)
                if m is None:
                    ui.notify("Structure SMILES invalide.", type="negative")
                    return
                single = Chem.MolToSmiles(m, canonical=True)

            state["current_smiles"] = single
            if from_sketcher:
                state["last_sketcher_smi"] = single
            else:
                # Force la mise à jour de la toile 2D lors d'un chargement externe
                state["last_sketcher_smi"] = ""

            if single not in state["history"]:
                state["history"].insert(0, single)
                state["history"] = state["history"][:16]

            state["ident_seq"] += 1
            seq_now = state["ident_seq"]
            state["ident"] = {"iupac": "Recherche PubChem...", "cas": "...", "trivial": "Identification...", "cid": None}
            state["chembase_match"] = None

            render_molecule_banner()
            if switch_to_2d and state["tab"] != "t2d3d":
                switch_tab("t2d3d")
            else:
                ref = tab_refreshers.get(state["tab"])
                if callable(ref):
                    ref()

            # Double cadence : différé (0.85 s) lors du dessin en direct, quasi-immédiat sinon
            prev_timer = state.get("pubchem_timer")
            if prev_timer is not None:
                try:
                    prev_timer.cancel()
                except Exception:
                    pass
            delay_s = 0.85 if from_sketcher else 0.08
            state["pubchem_timer"] = ui.timer(
                delay_s,
                lambda s=single, sid=seq_now: resolve_banner_async(s, sid),
                once=True,
            )

        # Réception en temps réel depuis la toile 2D maison
        def handle_sketcher_change(e):
            graph = (e.args or {}).get("graph", {}) if isinstance(e.args, dict) else {}
            if not isinstance(graph, dict):
                return
            smi, msg, bad_ids = graph_to_smiles(graph)
            if not smi:
                state["last_sketcher_smi"] = ""
                ui.run_javascript(
                    f"window.csPushSketcherStatus(false, {json.dumps(msg)}, {json.dumps(bad_ids)}, '');"
                )
                return

            ui.run_javascript(
                f'window.csPushSketcherStatus(true, "Valence OK", [], {json.dumps(smi)});'
            )
            cur_single = extract_single_smiles(state["current_smiles"])
            if smi == cur_single:
                state["last_sketcher_smi"] = smi
                return
            set_active_molecule(smi, switch_to_2d=False, from_sketcher=True)

        ui.on("cs_sketcher_change", handle_sketcher_change)

        # --- CONTENU DES 5 ONGLETS ---
        render_t2d3d, render_trxn, render_tstruct = build_studio_views(
            state,
            on_molecule_change=lambda s: set_active_molecule(s, switch_to_2d=True, from_sketcher=False),
            on_basket_change=refresh_header_badges,
        )
        render_tspec, render_texport = build_spec_and_export_views(state)

        with ui.element("main"):
            with ui.element("section").classes("tab-panel active") as p1:
                tab_refreshers["t2d3d"] = render_t2d3d()
            tab_panels["t2d3d"] = p1

            with ui.element("section").classes("tab-panel") as p2:
                tab_refreshers["trxn"] = render_trxn()
            tab_panels["trxn"] = p2

            with ui.element("section").classes("tab-panel") as p3:
                tab_refreshers["tstruct"] = render_tstruct()
            tab_panels["tstruct"] = p3

            with ui.element("section").classes("tab-panel") as p4:
                tab_refreshers["tspec"] = render_tspec()
            tab_panels["tspec"] = p4

            with ui.element("section").classes("tab-panel") as p5:
                tab_refreshers["texport"] = render_texport()
            tab_panels["texport"] = p5

        def switch_tab(tab_key: str):
            if state["tab"] == tab_key:
                return
            state["tab"] = tab_key
            for k, b in tab_buttons.items():
                b.classes(add="active") if k == tab_key else b.classes(remove="active")
            for k, p in tab_panels.items():
                p.classes(add="active") if k == tab_key else p.classes(remove="active")
            ref = tab_refreshers.get(tab_key)
            if callable(ref):
                ref()
            ui.run_javascript("window.cbMovePill(false);")

        # --- MODALES ---
        def open_examples_modal():
            examples_body.clear()
            with examples_body:
                with ui.element("div").classes("modal-head"):
                    with ui.element("div"):
                        safe_html('<div class="section-kicker">ACCÈS RAPIDE</div><h2 style="margin:4px 0 0 0;">Exemples &amp; Historique de Session</h2>')
                    _btn("Fermer ✕", on_click=lambda _e: examples_dialog.close())

                safe_html('<div class="section-kicker" style="margin-bottom:8px;">MOLÉCULES DE DÉMONSTRATION</div>')
                with ui.element("div").style("display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin-bottom:16px;"):
                    for name, smi in EXEMPLES_MOLECULES.items():
                        _btn(
                            html.escape(name),
                            style="justify-content:flex-start;",
                            on_click=lambda _e, s=smi: (examples_dialog.close(), set_active_molecule(s)),
                        )

                safe_html('<div class="section-kicker" style="margin-bottom:8px;">HISTORIQUE RÉCENT</div>')
                with ui.element("div").style("display:flex;flex-direction:column;gap:6px;max-height:220px;overflow-y:auto;"):
                    for h_smi in state["history"]:
                        _btn(
                            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.76rem;">{html.escape(h_smi)}</span>',
                            style="justify-content:flex-start;",
                            on_click=lambda _e, s=h_smi: (examples_dialog.close(), set_active_molecule(s)),
                        )
            examples_dialog.open()

        def open_iupac_modal(cprop: dict[str, Any]):
            iupac_body.clear()
            iupac_name = clean_val(state["ident"].get("iupac")) or "Non répertorié"
            can_smi = cprop["canonical_smiles"]
            ik = cprop["inchikey"]
            cp_iup = html.escape(f"window.cbCopyText({json.dumps(iupac_name)}, 'IUPAC')", quote=True)
            cp_smi = html.escape(f"window.cbCopyText({json.dumps(can_smi)}, 'SMILES')", quote=True)
            cp_ik = html.escape(f"window.cbCopyText({json.dumps(ik)}, 'InChIKey')", quote=True)

            with iupac_body:
                with ui.element("div").classes("modal-head"):
                    with ui.element("div"):
                        safe_html('<div class="section-kicker">IDENTIFIANTS CHIMIQUES (CLIC POUR COPIER)</div><h2 style="margin:4px 0 0 0;">IUPAC, SMILES &amp; InChIKey</h2>')
                    _btn("Fermer ✕", on_click=lambda _e: iupac_dialog.close())

                safe_html(
                    f'<div class="kv-item copyable-tag" style="margin-bottom:10px;" onclick="{cp_iup}">'
                    f'<div class="kv-label">Nom IUPAC</div><div class="kv-val">{html.escape(iupac_name)}</div></div>'
                    f'<div class="kv-item copyable-tag" style="margin-bottom:10px;" onclick="{cp_smi}">'
                    f'<div class="kv-label">SMILES Canonique</div><div class="kv-val" style="font-family:\'JetBrains Mono\',monospace;">{html.escape(can_smi)}</div></div>'
                    f'<div class="kv-item copyable-tag" onclick="{cp_ik}">'
                    f'<div class="kv-label">InChIKey</div><div class="kv-val" style="font-family:\'JetBrains Mono\',monospace;">{html.escape(ik)}</div></div>'
                )
            iupac_dialog.open()

        def open_chembase_modal():
            chembase_body.clear()
            cur_url, cur_key = get_cb_creds()

            with chembase_body:
                with ui.element("div").classes("modal-head"):
                    with ui.element("div"):
                        safe_html('<div class="section-kicker">INTÉGRATION OPTIONNELLE</div><h2 style="margin:4px 0 0 0;">Liaison avec ChemBase</h2>')
                    _btn("Fermer ✕", on_click=lambda _e: chembase_dialog.close())

                safe_html(
                    '<p class="muted-small" style="margin:0 0 10px 0;">'
                    'Optionnel : reliez votre espace ChemBase pour importer vos fiches ou y envoyer vos molécules. '
                    'Vos identifiants restent enregistrés uniquement dans ce navigateur.'
                    '</p>'
                )

                grid_cols = "1.2fr 1.2fr auto auto" if cur_key else "1.2fr 1.2fr auto"
                with ui.element("div").style(f"display:grid;grid-template-columns:{grid_cols};gap:8px;align-items:center;margin-bottom:12px;"):
                    in_cb_url = ui.input(
                        label="Adresse ChemBase (https://...)",
                        value=cur_url,
                    ).props("outlined dense")
                    in_cb_key = ui.input(
                        label="Clé d'accès du laboratoire",
                        value=cur_key,
                        password=True,
                        password_toggle_button=True,
                    ).props("outlined dense")

                    def save_cb_cfg(_e=None):
                        new_u = clean_val(in_cb_url.value).rstrip("/")
                        new_k = clean_val(in_cb_key.value)
                        if new_u and not is_safe_public_url(new_u):
                            ui.notify("Adresse invalide : utilisez une URL publique en https://", type="negative")
                            return
                        state["chembase_url"] = new_u
                        state["chembase_api_key"] = new_k
                        payload = json.dumps(json.dumps({"url": new_u, "key": new_k}))
                        ui.run_javascript(f'localStorage.setItem("cs_chembase_cfg", {payload});')
                        if new_u:
                            _wake_chembase_async(new_u)
                        refresh_header_badges()
                        render_molecule_banner()
                        if state["tab"] == "trxn" and callable(tab_refreshers.get("trxn")):
                            tab_refreshers["trxn"]()
                        ui.notify("Connexion ChemBase enregistrée sur ce navigateur.", type="positive")
                        ui.timer(0.05, load_cb_catalog, once=True)

                    def disconnect_cb(_e=None):
                        default_u, _ = get_chembase_config()
                        state["chembase_url"] = default_u
                        state["chembase_api_key"] = ""
                        state["chembase_match"] = None
                        ui.run_javascript('localStorage.removeItem("cs_chembase_cfg");')
                        refresh_header_badges()
                        render_molecule_banner()
                        if state["tab"] == "trxn" and callable(tab_refreshers.get("trxn")):
                            tab_refreshers["trxn"]()
                        ui.notify("ChemBase déconnecté.", type="info")
                        open_chembase_modal()

                    _btn("Connecter", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=save_cb_cfg)
                    if cur_key:
                        _btn("Déconnecter", classes="glass-btn glass-btn-danger", style="height:40px;", on_click=disconnect_cb)

                with ui.element("div").style("display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-bottom:10px;"):
                    in_cb_q = ui.input(placeholder="Filtrer votre catalogue ChemBase par nom ou CAS...").props("outlined dense clearable")
                    _btn("Actualiser", style="height:40px;", on_click=lambda _e: load_cb_catalog())

                cb_list_slot = ui.element("div").style("max-height:330px;overflow-y:auto;")

                async def load_cb_catalog():
                    u_val, k_val = get_cb_creds()
                    cb_list_slot.clear()
                    if not u_val or not k_val:
                        with cb_list_slot:
                            safe_html('<p class="muted-text">Renseignez l\'adresse de votre espace ChemBase et votre clé d\'accès pour parcourir votre catalogue.</p>')
                        return
                    with cb_list_slot:
                        safe_html('<p class="muted-small">Interrogation de ChemBase en cours...</p>')
                    ok, items, msg = await run.io_bound(fetch_chembase_catalog, clean_val(in_cb_q.value), u_val, k_val)
                    cb_list_slot.clear()
                    with cb_list_slot:
                        if not ok:
                            safe_html(f'<p style="color:var(--danger);font-size:0.86rem;">{html.escape(msg)}</p>')
                            return
                        if not items:
                            safe_html('<p class="muted-text">Aucune fiche trouvée dans ChemBase pour ce filtre.</p>')
                            return
                        for it in items[:40]:
                            prod_n = clean_val(it.get("produit") or it.get("nom") or "Sans nom")
                            cas_n = clean_val(it.get("cas"))
                            rx_n = clean_val(it.get("reaction_1"))

                            async def import_from_cb(_e=None, c=cas_n, p=prod_n):
                                q_try = c if c and c not in ("N/A", "-") else p
                                smi = await run.io_bound(resolve_query_to_smiles, q_try)
                                if not smi and p != q_try:
                                    smi = await run.io_bound(resolve_query_to_smiles, p)
                                if smi:
                                    chembase_dialog.close()
                                    set_active_molecule(smi, switch_to_2d=True)
                                    ui.notify(f"« {p} » importé depuis ChemBase !", type="positive")
                                else:
                                    ui.notify(f"Structure SMILES introuvable sur PubChem pour « {p} ».", type="warning")

                            with ui.element("div").classes("kv-item").style("display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:7px;flex-wrap:wrap;"):
                                safe_html(
                                    f'<div><strong style="color:var(--txt);">{html.escape(prod_n)}</strong> '
                                    f'<span class="mono-tag">CAS {html.escape(cas_n or "N/A")}</span>'
                                    f'<div class="muted-small">{html.escape(rx_n)}</div></div>'
                                )
                                _btn("Ouvrir dans Studio", classes="glass-btn glass-btn-accent", style="padding:4px 10px;font-size:0.75rem;", on_click=import_from_cb)

                in_cb_q.on("keydown.enter", lambda _e: load_cb_catalog())
                ui.timer(0.05, load_cb_catalog, once=True)

            chembase_dialog.open()

        safe_html(
            """
            <footer class="app-footer">
              <span>CHEMSTUDIO v2.0 · RDKIT + DESSINATEUR 2D LIQUID GLASS</span>
              <span>SUITE COMPUTATIONNELLE · COMPATIBLE CHEMBASE</span>
            </footer>
            """
        )

    refresh_header_badges()
    render_molecule_banner()
    ui.timer(0.08, lambda: ui.run_javascript("window.cbMovePill(true);"), once=True)

    async def boot_resolve():
        try:
            saved_theme = await ui.run_javascript('localStorage.getItem("cs_theme")', timeout=2.0)
            if saved_theme in ("dark", "light"):
                want_dark = saved_theme == "dark"
                if want_dark != state["dark"]:
                    state["dark"] = want_dark
                    dark.value = want_dark
                    ui.run_javascript(f'window.cbApplyTheme("{saved_theme}");')
                    ref = tab_refreshers.get(state["tab"])
                    if callable(ref):
                        ref()
        except Exception:
            pass

        try:
            saved_cb_raw = await ui.run_javascript('localStorage.getItem("cs_chembase_cfg")', timeout=2.0)
            if isinstance(saved_cb_raw, str) and saved_cb_raw:
                saved_cb = json.loads(saved_cb_raw)
                if isinstance(saved_cb, dict):
                    u_s = clean_val(saved_cb.get("url")).rstrip("/")
                    k_s = clean_val(saved_cb.get("key"))
                    if u_s and is_safe_public_url(u_s):
                        state["chembase_url"] = u_s
                        state["chembase_api_key"] = k_s
                        _wake_chembase_async(u_s)
                        refresh_header_badges()
        except Exception:
            pass

        if clean_val(cas) and not clean_val(smiles):
            found = await run.io_bound(resolve_query_to_smiles, clean_val(cas))
            if found:
                set_active_molecule(found)
                return
        state["ident_seq"] += 1
        await resolve_banner_async(extract_single_smiles(state["current_smiles"]), state["ident_seq"])

    ui.timer(0.12, boot_resolve, once=True)


if __name__ in {"__main__", "__mp_main__"}:
    port = int(os.environ.get("PORT", "8081"))
    ui.run(
        host="0.0.0.0",
        port=port,
        title="ChemStudio v2.0 — Chimie Computationnelle & Spectrométrie",
        favicon=get_favicon_data_uri(),
        dark=True,
        storage_secret=STORAGE_SECRET,
        reload=not IS_RENDER,
        show=not IS_RENDER,
    )