import html
import json
from typing import Any

from nicegui import run, ui
from rdkit import Chem
from rdkit.Chem import Descriptors

from chem_engine import (
    NMR_SOLVENTS,
    build_lorentzian_points,
    calc_isotopic_spectrum,
    clean_val,
    extract_single_smiles,
    fragment_on_bond,
    generate_3d_mol,
    get_acyclic_bonds,
    get_chemical_identifiers,
    get_ghs_safety,
    parse_smiles,
    render_2d_svg,
    render_annotated_reaction_svg,
    simulate_advanced_nmr_1d_2d,
    compute_clean_2d_coords,
)
from theme import safe_html

PUB_STYLES = [
    "ACS Document 1996 (Noir & Blanc)",
    "RSC Standard (Couleurs sobres)",
    "Wiley Clean (Trait épais N&B)",
    "Mode Sombre (Présentation)",
]


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


def _chart_theme(is_dark: bool) -> dict[str, str]:
    return {
        "ax_txt": "#94a3b8" if is_dark else "#475569",
        "ax_line": "rgba(148,163,184,0.32)" if is_dark else "rgba(15,23,42,0.24)",
        "split": "rgba(148,163,184,0.14)" if is_dark else "rgba(15,23,42,0.09)",
        "tip_bg": "rgba(11,20,22,0.95)" if is_dark else "rgba(255,255,255,0.96)",
        "tip_txt": "#e6edf3" if is_dark else "#0f172a",
        "cyan": "#38bdf8" if is_dark else "#0284c7",
        "green": "#10b981" if is_dark else "#059669",
        "warn": "#f59e0b" if is_dark else "#d97706",
        "rose": "#f43f5e" if is_dark else "#e11d48",
    }


def build_spec_and_export_views(state: dict[str, Any]):
    # ==================================================================
    # ONGLET 4 : SPECTROMÉTRIE (HRMS, FRAGMENTATION, RMN 1D & RMN 2D)
    # ==================================================================
    def render_tab_spec():
        root = ui.element("div")
        spec_cfg: dict[str, Any] = {
            "sub": "hrms",  # hrms | frag | nmr1d | nmr2d
            "bond_idx": None,
            "freq_mhz": 400.0,
            "solvent": "CDCl3",
            "h_min": 0.0,
            "h_max": 12.5,
            "c_min": 0.0,
            "c_max": 215.0,
            "obs_mz": "",
        }

        def refresh_tab():
            root.clear()
            single_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
            mol_main = parse_smiles(single_smi)
            ct = _chart_theme(bool(state.get("dark", True)))

            with root:
                with ui.element("div").classes("sub-tabs"):
                    for k, lbl in (
                        ("hrms", "1. Masse Haute Résolution (HRMS & Isotopes)"),
                        ("frag", "2. Fragmentation MS par coupure"),
                        ("nmr1d", "3. RMN 1D Haute Précision (¹H & ¹³C)"),
                        ("nmr2d", "4. RMN 2D Corrélations (COSY, HSQC & HMBC)"),
                    ):
                        cls = "glass-btn glass-btn-accent" if spec_cfg["sub"] == k else "glass-btn"
                        _btn(lbl, classes=cls, on_click=lambda _e, key=k: (spec_cfg.update(sub=key), refresh_tab()))

                if not mol_main:
                    safe_html('<div class="hud-card"><p class="muted-text">Structure chimique invalide.</p></div>')
                    return

                # --- SOUS-ONGLET 1 : HRMS & MASSIF ISOTOPIQUE ---
                if spec_cfg["sub"] == "hrms":
                    exact_m = float(Descriptors.ExactMolWt(mol_main))
                    adducts, iso_rows = calc_isotopic_spectrum(mol_main)

                    with ui.element("div").classes("modal-grid"):
                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                '<div class="section-kicker">SPECTROMÉTRIE DE MASSE HAUTE RÉSOLUTION</div>'
                                '<h2>Adduits Ioniques Attendus (ESI⁺ / ESI⁻)</h2>'
                                '<div class="kpi-card" style="margin-bottom:14px;">'
                                '<div class="kpi-label">Masse Exacte Monoisotopique (M)</div>'
                                f'<div class="kpi-value" style="color:var(--cyan);">{exact_m:.5f} Da</div>'
                                '</div>'
                            )
                            ad_trs = "".join(
                                f"<tr><td><strong>{html.escape(a['ion'])}</strong></td>"
                                f"<td style='font-family:\"JetBrains Mono\",monospace;color:var(--accent);'>{a['mz']:.5f}</td></tr>"
                                for a in adducts
                            )
                            safe_html(
                                '<div class="data-table-wrap"><table class="data-table">'
                                '<thead><tr><th>Ion / Adduit</th><th>m/z exact</th></tr></thead>'
                                f'<tbody>{ad_trs}</tbody></table></div>'
                            )
                            safe_html(
                                '<div class="section-kicker" style="margin:14px 0 6px 0;">VÉRIFICATEUR HRMS (ÉCART EN PPM)</div>'
                            )
                            ppm_result_slot = ui.element("div")
                            with ui.element("div").style("display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-bottom:8px;"):
                                in_obs = ui.input(
                                    label="m/z mesuré (optionnel)",
                                    placeholder=f"Ex : {adducts[0]['mz']:.4f}",
                                    value=str(spec_cfg.get("obs_mz") or ""),
                                ).props("outlined dense clearable")

                                def _eval_hrms_ppm(_e=None):
                                    raw_v = clean_val(in_obs.value).replace(",", ".")
                                    spec_cfg["obs_mz"] = raw_v
                                    ppm_result_slot.clear()
                                    if not raw_v:
                                        return
                                    try:
                                        obs_val = float(raw_v)
                                        if obs_val <= 0:
                                            raise ValueError()
                                    except ValueError:
                                        with ppm_result_slot:
                                            safe_html('<span class="muted-small" style="color:var(--danger);">Saisissez une valeur numérique positive.</span>')
                                        return

                                    best = min(adducts, key=lambda a: abs(obs_val - float(a["mz"])))
                                    theo = float(best["mz"])
                                    err_ppm = ((obs_val - theo) / theo) * 1e6
                                    abs_ppm = abs(err_ppm)
                                    if abs_ppm <= 5.0:
                                        badge_cls, verdict = "stock-ok", "Conforme (≤ 5 ppm)"
                                    elif abs_ppm <= 15.0:
                                        badge_cls, verdict = "stock-low", "Écart modéré"
                                    else:
                                        badge_cls, verdict = "stock-empty", "Hors tolérance"
                                    with ppm_result_slot:
                                        safe_html(
                                            f'<div class="kv-item" style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">'
                                            f'<span style="font-size:0.82rem;">Plus proche : <strong>{html.escape(str(best["ion"]))}</strong> '
                                            f'(<span style="font-family:\'JetBrains Mono\',monospace;">{err_ppm:+.2f} ppm</span>)</span>'
                                            f'<span class="stock-pill {badge_cls}">{verdict}</span>'
                                            f'</div>'
                                        )

                                in_obs.on("keydown.enter", _eval_hrms_ppm)
                                _btn("Vérifier", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=_eval_hrms_ppm)
                            if ppm_result_slot.parent_slot is not None:
                                ppm_result_slot.move(ppm_result_slot.parent_slot.parent)
                            _eval_hrms_ppm()

                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                '<div class="section-kicker">DISTRIBUTION ISOTOPIQUE SIMULÉE</div>'
                                '<h2>Massif Isotopique (M, M+1, M+2...)</h2>'
                            )
                            if not iso_rows:
                                safe_html('<p class="muted-text">Massif isotopique désactivé pour les structures de plus de 120 atomes lourds (les adduits exacts à gauche restent disponibles).</p>')
                            else:
                             ui.echart({
                                "backgroundColor": "transparent",
                                "tooltip": {
                                    "trigger": "axis",
                                    "backgroundColor": ct["tip_bg"],
                                    "borderColor": ct["cyan"],
                                    "textStyle": {"color": ct["tip_txt"]},
                                },
                                "grid": {"left": 48, "right": 18, "top": 18, "bottom": 34},
                                "xAxis": {
                                    "type": "category",
                                    "data": [f"{r['pic']}\n({r['mz']:.2f})" for r in iso_rows],
                                    "axisLabel": {"color": ct["ax_txt"], "fontSize": 11},
                                    "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                },
                                "yAxis": {
                                    "type": "value",
                                    "max": 100,
                                    "axisLabel": {"color": ct["ax_txt"], "formatter": "{value} %"},
                                    "splitLine": {"lineStyle": {"color": ct["split"]}},
                                },
                                "series": [{
                                    "name": "Intensité relative (%)",
                                    "type": "bar",
                                    "barWidth": "42%",
                                    "data": [r["rel"] for r in iso_rows],
                                    "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": ct["cyan"]},
                                }],
                            }).style("height:245px;width:100%;")

                            with ui.expansion("Table détaillée des pics isotopiques", value=False).style("width:100%;margin-top:8px;"):
                                iso_trs = "".join(
                                    f"<tr><td><strong>{html.escape(r['pic'])}</strong></td>"
                                    f"<td style='font-family:\"JetBrains Mono\",monospace;'>{r['mz']:.5f}</td>"
                                    f"<td style='font-family:\"JetBrains Mono\",monospace;color:var(--cyan);'>{r['rel']:.2f} %</td></tr>"
                                    for r in iso_rows
                                )
                                safe_html(
                                    '<div class="data-table-wrap"><table class="data-table">'
                                    '<thead><tr><th>Pic</th><th>m/z exact</th><th>Intensité Relative (%)</th></tr></thead>'
                                    f'<tbody>{iso_trs}</tbody></table></div>'
                                )

                # --- SOUS-ONGLET 2 : FRAGMENTATION MS ---
                elif spec_cfg["sub"] == "frag":
                    svg_b = render_2d_svg(mol_main, width=460, height=340, atom_indices=True, bond_indices=True, stereo=False)
                    bonds = get_acyclic_bonds(mol_main)

                    with ui.element("div").classes("modal-grid"):
                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                '<div class="section-kicker">REPÉRAGE DES LIAISONS</div>'
                                '<h2>Carte des Indices de Liaisons (Bleu)</h2>'
                                f'<div class="svg-box">{svg_b}</div>'
                            )

                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                '<div class="section-kicker">CLIVAGE IN SILICO</div>'
                                '<h2>Fragmentation MS par Coupure de Liaison</h2>'
                            )
                            if not bonds:
                                safe_html('<p class="muted-text">Toutes les liaisons de cette molécule appartiennent à un cycle (aucune liaison acyclique à couper).</p>')
                                return

                            bond_opts = {int(b["idx"]): str(b["label"]) for b in bonds}
                            cur_b = spec_cfg["bond_idx"]
                            if cur_b not in bond_opts:
                                cur_b = next(iter(bond_opts.keys()))
                                spec_cfg["bond_idx"] = cur_b

                            sel_b = ui.select(
                                bond_opts,
                                value=cur_b,
                                label="Choisir la liaison acyclique à couper",
                            ).props("outlined dense").style("width:100%;margin-bottom:14px;")
                            sel_b.on_value_change(
                                lambda e: (spec_cfg.update(bond_idx=int(e.value)), refresh_tab())
                            )

                            frags = fragment_on_bond(mol_main, int(cur_b))
                            with ui.element("div").style("display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;"):
                                for fr in frags:
                                    with ui.element("div").classes("kv-item").style("text-align:center;padding:12px;"):
                                        safe_html(
                                            f'<div class="svg-box-adaptive" style="margin-bottom:8px;">{fr["svg"]}</div>'
                                            f'<div class="kpi-label">Fragment {fr["index"]} · {html.escape(str(fr["formula"]))}</div>'
                                            f'<div class="kpi-value" style="color:var(--accent);font-size:1.1rem;">{fr["exact_mass"]:.4f} Da</div>'
                                        )

                # --- SOUS-ONGLET 3 : RMN 1D (¹H & ¹³C) ---
                elif spec_cfg["sub"] == "nmr1d":
                    h_rows, c_rows, _cosy, _hsqc, _hmbc = simulate_advanced_nmr_1d_2d(mol_main)
                    solv_key = str(spec_cfg.get("solvent") or "CDCl3")
                    if solv_key not in NMR_SOLVENTS:
                        solv_key = "CDCl3"
                    solv_info = NMR_SOLVENTS[solv_key]
                    solv_opts = {k: str(v["label"]) for k, v in NMR_SOLVENTS.items()}

                    with ui.element("div").classes("hud-card").style("margin-bottom:14px;padding:14px 18px;"):
                        with ui.element("div").style("display:grid;grid-template-columns:1.35fr 0.95fr 0.75fr 0.75fr 0.75fr 0.75fr auto;gap:8px;align-items:center;"):
                            sel_solv = ui.select(
                                solv_opts,
                                value=solv_key,
                                label="Solvant RMN (repères)",
                            ).props("outlined dense")
                            sel_freq = ui.select(
                                {300.0: "300 MHz", 400.0: "400 MHz", 500.0: "500 MHz", 600.0: "600 MHz"},
                                value=float(spec_cfg["freq_mhz"]),
                                label="Spectromètre",
                            ).props("outlined dense")
                            in_hmin = ui.number(label="¹H min", value=float(spec_cfg["h_min"]), min=0.0, max=14.0, step=0.5).props("outlined dense")
                            in_hmax = ui.number(label="¹H max", value=float(spec_cfg["h_max"]), min=0.5, max=15.0, step=0.5).props("outlined dense")
                            in_cmin = ui.number(label="¹³C min", value=float(spec_cfg["c_min"]), min=0.0, max=210.0, step=5.0).props("outlined dense")
                            in_cmax = ui.number(label="¹³C max", value=float(spec_cfg["c_max"]), min=10.0, max=230.0, step=5.0).props("outlined dense")

                            def apply_nmr_zoom(_e=None):
                                h0 = float(in_hmin.value or 0.0)
                                h1 = float(in_hmax.value or 12.5)
                                c0 = float(in_cmin.value or 0.0)
                                c1 = float(in_cmax.value or 215.0)
                                spec_cfg.update(
                                    solvent=str(sel_solv.value or "CDCl3"),
                                    freq_mhz=float(sel_freq.value or 400.0),
                                    h_min=min(h0, h1 - 0.5),
                                    h_max=max(h1, h0 + 0.5),
                                    c_min=min(c0, c1 - 5.0),
                                    c_max=max(c1, c0 + 5.0),
                                )
                                refresh_tab()

                            sel_solv.on_value_change(apply_nmr_zoom)
                            sel_freq.on_value_change(apply_nmr_zoom)
                            _btn("Actualiser", classes="glass-btn glass-btn-accent", style="height:40px;", on_click=apply_nmr_zoom)

                        safe_html(
                            '<div class="muted-small" style="margin-top:8px;">'
                            'Spectre estimé par tables d\'incréments empiriques (type Pretsch / Shoolery) · '
                            'Les lignes pointillées ambre indiquent les pics résiduels du solvant sélectionné.'
                            '</div>'
                        )

                    with ui.element("div").classes("modal-grid"):
                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                f'<div class="section-kicker">RMN PROTON · {int(spec_cfg["freq_mhz"])} MHZ · {html.escape(str(solv_info["label"]))}</div>'
                                '<h2>Spectre RMN ¹H (zoom molette actif)</h2>'
                            )
                            if h_rows:
                                pts_h = build_lorentzian_points(
                                    h_rows,
                                    spec_cfg["h_min"],
                                    spec_cfg["h_max"],
                                    width=0.0035,
                                    freq_mhz=float(spec_cfg["freq_mhz"]),
                                    num_points=2400,
                                )
                                h_solv_marks = [
                                    {"xAxis": float(p["ppm"]), "name": str(p["label"])}
                                    for p in solv_info.get("h_peaks", [])
                                    if float(spec_cfg["h_min"]) <= float(p["ppm"]) <= float(spec_cfg["h_max"])
                                ]
                                h_series: dict[str, Any] = {
                                    "name": "RMN ¹H",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": pts_h,
                                    "lineStyle": {"width": 1.6, "color": ct["cyan"]},
                                }
                                if h_solv_marks:
                                    h_series["markLine"] = {
                                        "symbol": ["none", "none"],
                                        "silent": True,
                                        "lineStyle": {"type": "dashed", "color": ct["warn"], "width": 1.2, "opacity": 0.85},
                                        "label": {
                                            "show": True,
                                            "position": "insideEndTop",
                                            "color": ct["warn"],
                                            "fontSize": 10,
                                            "formatter": "{b}",
                                        },
                                        "data": h_solv_marks,
                                    }

                                ui.echart({
                                    "backgroundColor": "transparent",
                                    "tooltip": {
                                        "trigger": "axis",
                                        "backgroundColor": ct["tip_bg"],
                                        "borderColor": ct["cyan"],
                                        "textStyle": {"color": ct["tip_txt"]},
                                    },
                                    "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": 6}],
                                    "grid": {"left": 38, "right": 16, "top": 22, "bottom": 52},
                                    "xAxis": {
                                        "type": "value",
                                        "inverse": True,
                                        "min": float(spec_cfg["h_min"]),
                                        "max": float(spec_cfg["h_max"]),
                                        "name": "δ (ppm)",
                                        "axisLabel": {"color": ct["ax_txt"]},
                                        "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                        "splitLine": {"lineStyle": {"color": ct["split"]}},
                                    },
                                    "yAxis": {
                                        "type": "value",
                                        "axisLabel": {"show": False},
                                        "splitLine": {"show": False},
                                    },
                                    "series": [h_series],
                                }).style("height:265px;width:100%;")

                                h_trs_list = []
                                for r in h_rows:
                                    j1 = float(r.get("j_hz") or 0.0)
                                    j2 = float(r.get("j2_hz") or 0.0)
                                    j_txt = f"{j1:.1f}, {j2:.1f}" if (j1 > 0 and j2 > 0) else (f"{j1:.1f}" if j1 > 0 else "—")
                                    h_trs_list.append(
                                        f"<tr><td><strong>{html.escape(str(r['porte_par']))}</strong></td>"
                                        f"<td>{html.escape(str(r['integration']))}</td>"
                                        f"<td>{html.escape(str(r['multiplicite']))}</td>"
                                        f"<td style='font-family:\"JetBrains Mono\",monospace;'>{j_txt}</td>"
                                        f"<td>{html.escape(str(r['environnement']))}</td>"
                                        f"<td style='font-family:\"JetBrains Mono\",monospace;color:var(--cyan);font-weight:700;'>{float(r['shift']):.2f}</td></tr>"
                                    )
                                with ui.expansion(f"Table des attributions ¹H ({len(h_rows)} signaux)", value=True).style("width:100%;margin-top:8px;"):
                                    safe_html(
                                        '<div class="data-table-wrap"><table class="data-table">'
                                        '<thead><tr><th>Porté par</th><th>Intég.</th><th>Mult.</th><th>J (Hz)</th><th>Environnement</th><th>δ ¹H (ppm)</th></tr></thead>'
                                        f'<tbody>{"".join(h_trs_list)}</tbody></table></div>'
                                    )
                            else:
                                safe_html('<p class="muted-text">Aucun proton détecté (ou molécule supérieure à 120 atomes lourds).</p>')

                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                f'<div class="section-kicker">RMN CARBONE · DÉCOUPLAGE PROTON · {html.escape(str(solv_info["label"]))}</div>'
                                '<h2>Spectre RMN ¹³C {¹H} (zoom molette actif)</h2>'
                            )
                            if c_rows:
                                pts_c = build_lorentzian_points(
                                    c_rows,
                                    spec_cfg["c_min"],
                                    spec_cfg["c_max"],
                                    width=0.28,
                                    freq_mhz=float(spec_cfg["freq_mhz"]),
                                    num_points=1600,
                                )
                                c_solv_marks = [
                                    {"xAxis": float(p["ppm"]), "name": str(p["label"])}
                                    for p in solv_info.get("c_peaks", [])
                                    if float(spec_cfg["c_min"]) <= float(p["ppm"]) <= float(spec_cfg["c_max"])
                                ]
                                c_series: dict[str, Any] = {
                                    "name": "RMN ¹³C",
                                    "type": "line",
                                    "showSymbol": False,
                                    "data": pts_c,
                                    "lineStyle": {"width": 1.6, "color": ct["green"]},
                                }
                                if c_solv_marks:
                                    c_series["markLine"] = {
                                        "symbol": ["none", "none"],
                                        "silent": True,
                                        "lineStyle": {"type": "dashed", "color": ct["warn"], "width": 1.2, "opacity": 0.85},
                                        "label": {
                                            "show": True,
                                            "position": "insideEndTop",
                                            "color": ct["warn"],
                                            "fontSize": 10,
                                            "formatter": "{b}",
                                        },
                                        "data": c_solv_marks,
                                    }

                                ui.echart({
                                    "backgroundColor": "transparent",
                                    "tooltip": {
                                        "trigger": "axis",
                                        "backgroundColor": ct["tip_bg"],
                                        "borderColor": ct["green"],
                                        "textStyle": {"color": ct["tip_txt"]},
                                    },
                                    "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": 6}],
                                    "grid": {"left": 38, "right": 16, "top": 22, "bottom": 52},
                                    "xAxis": {
                                        "type": "value",
                                        "inverse": True,
                                        "min": float(spec_cfg["c_min"]),
                                        "max": float(spec_cfg["c_max"]),
                                        "name": "δ (ppm)",
                                        "axisLabel": {"color": ct["ax_txt"]},
                                        "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                        "splitLine": {"lineStyle": {"color": ct["split"]}},
                                    },
                                    "yAxis": {
                                        "type": "value",
                                        "axisLabel": {"show": False},
                                        "splitLine": {"show": False},
                                    },
                                    "series": [c_series],
                                }).style("height:265px;width:100%;")

                                c_trs = "".join(
                                    f"<tr><td><strong>{html.escape(str(r['carbones']))}</strong></td>"
                                    f"<td>{html.escape(str(r['environnement']))}</td>"
                                    f"<td>{r['nb_c']}</td>"
                                    f"<td style='font-family:\"JetBrains Mono\",monospace;color:var(--accent);font-weight:700;'>{float(r['shift']):.1f}</td></tr>"
                                    for r in c_rows
                                )
                                with ui.expansion(f"Table des attributions ¹³C ({len(c_rows)} signaux)", value=True).style("width:100%;margin-top:8px;"):
                                    safe_html(
                                        '<div class="data-table-wrap"><table class="data-table">'
                                        '<thead><tr><th>Carbone(s) #</th><th>Environnement</th><th>Nb C</th><th>δ ¹³C (ppm)</th></tr></thead>'
                                        f'<tbody>{c_trs}</tbody></table></div>'
                                    )
                            else:
                                safe_html('<p class="muted-text">Aucun carbone détecté (ou molécule supérieure à 120 atomes lourds).</p>')

                # --- SOUS-ONGLET 4 : RMN 2D (COSY, HSQC & HMBC) ---
                else:
                    _h, _c, cosy_pts, hsqc_pts, hmbc_pts = simulate_advanced_nmr_1d_2d(mol_main)

                    with ui.element("div").classes("modal-grid"):
                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                '<div class="section-kicker">CORRÉLATION HOMONUCLÉAIRE</div>'
                                '<h2>Carte 2D ¹H–¹H COSY (Contours &amp; Diagonale)</h2>'
                            )
                            if cosy_pts:
                                diag_pts = [[p["f2"], p["f1"]] for p in cosy_pts if p["type"] == "Diagonale"]
                                cross_pts = [[p["f2"], p["f1"]] for p in cosy_pts if p["type"] != "Diagonale"]
                                ui.echart({
                                    "backgroundColor": "transparent",
                                    "tooltip": {
                                        "trigger": "item",
                                        "backgroundColor": ct["tip_bg"],
                                        "borderColor": ct["cyan"],
                                        "textStyle": {"color": ct["tip_txt"]},
                                    },
                                    "legend": {"top": 4, "textStyle": {"color": ct["ax_txt"]}},
                                    "grid": {"left": 48, "right": 20, "top": 34, "bottom": 42},
                                    "xAxis": {
                                        "type": "value",
                                        "inverse": True,
                                        "min": 0,
                                        "max": 12.5,
                                        "name": "δ ¹H F2 (ppm)",
                                        "nameLocation": "middle",
                                        "nameGap": 25,
                                        "axisLabel": {"color": ct["ax_txt"]},
                                        "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                        "splitLine": {"lineStyle": {"color": ct["split"]}},
                                    },
                                    "yAxis": {
                                        "type": "value",
                                        "inverse": True,
                                        "min": 0,
                                        "max": 12.5,
                                        "name": "δ ¹H F1 (ppm)",
                                        "axisLabel": {"color": ct["ax_txt"]},
                                        "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                        "splitLine": {"lineStyle": {"color": ct["split"]}},
                                    },
                                    "series": [
                                        {
                                            "name": "Ligne diagonale",
                                            "type": "line",
                                            "showSymbol": False,
                                            "silent": True,
                                            "data": [[11, 11], [0, 0]],
                                            "lineStyle": {"type": "dashed", "color": ct["ax_line"], "width": 1},
                                        },
                                        {
                                            "name": "Diagonale",
                                            "type": "scatter",
                                            "symbolSize": 10,
                                            "data": diag_pts,
                                            "itemStyle": {"color": ct["ax_txt"], "opacity": 0.65},
                                        },
                                        {
                                            "name": "Corrélation ³J_HH",
                                            "type": "scatter",
                                            "symbolSize": 16,
                                            "data": cross_pts,
                                            "itemStyle": {"color": ct["cyan"], "opacity": 0.85, "borderColor": ct["cyan"], "borderWidth": 2},
                                        },
                                    ],
                                }).style("height:360px;width:100%;")

                                cosy_trs = "".join(
                                    f"<tr><td style='font-family:\"JetBrains Mono\",monospace;'>{p['f2']:.2f}</td>"
                                    f"<td style='font-family:\"JetBrains Mono\",monospace;'>{p['f1']:.2f}</td>"
                                    f"<td>{html.escape(p['type'])}</td></tr>"
                                    for p in cosy_pts
                                )
                                with ui.expansion(f"Taches de corrélation COSY ({len(cosy_pts)})", value=False).style("width:100%;margin-top:8px;"):
                                    safe_html(
                                        '<div class="data-table-wrap"><table class="data-table">'
                                        '<thead><tr><th>δ ¹H F2 (ppm)</th><th>δ ¹H F1 (ppm)</th><th>Type</th></tr></thead>'
                                        f'<tbody>{cosy_trs}</tbody></table></div>'
                                    )
                            else:
                                safe_html('<p class="muted-text">Aucune corrélation COSY détectée.</p>')

                        with ui.element("div").classes("hud-card"):
                            safe_html(
                                '<div class="section-kicker">CORRÉLATION HÉTÉRONUCLÉAIRE</div>'
                                '<h2>Carte 2D ¹H–¹³C HSQC (¹J_CH) &amp; HMBC (²˒³J_CH)</h2>'
                            )
                            het_all = hsqc_pts + hmbc_pts
                            if het_all:
                                ui.echart({
                                    "backgroundColor": "transparent",
                                    "tooltip": {
                                        "trigger": "item",
                                        "backgroundColor": ct["tip_bg"],
                                        "borderColor": ct["green"],
                                        "textStyle": {"color": ct["tip_txt"]},
                                    },
                                    "legend": {"top": 4, "textStyle": {"color": ct["ax_txt"]}},
                                    "grid": {"left": 52, "right": 20, "top": 34, "bottom": 42},
                                    "xAxis": {
                                        "type": "value",
                                        "inverse": True,
                                        "min": 0,
                                        "max": 11,
                                        "name": "δ ¹H (ppm)",
                                        "nameLocation": "middle",
                                        "nameGap": 25,
                                        "axisLabel": {"color": ct["ax_txt"]},
                                        "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                        "splitLine": {"lineStyle": {"color": ct["split"]}},
                                    },
                                    "yAxis": {
                                        "type": "value",
                                        "inverse": True,
                                        "min": 0,
                                        "max": 215,
                                        "name": "δ ¹³C (ppm)",
                                        "axisLabel": {"color": ct["ax_txt"]},
                                        "axisLine": {"lineStyle": {"color": ct["ax_line"]}},
                                        "splitLine": {"lineStyle": {"color": ct["split"]}},
                                    },
                                    "series": [
                                        {
                                            "name": "HSQC (¹J_CH directe)",
                                            "type": "scatter",
                                            "symbolSize": 15,
                                            "data": [[p["h"], p["c"]] for p in hsqc_pts],
                                            "itemStyle": {"color": ct["green"], "opacity": 0.90},
                                        },
                                        {
                                            "name": "HMBC (²˒³J_CH longue distance)",
                                            "type": "scatter",
                                            "symbolSize": 11,
                                            "data": [[p["h"], p["c"]] for p in hmbc_pts],
                                            "itemStyle": {"color": ct["warn"], "opacity": 0.72},
                                        },
                                    ],
                                }).style("height:360px;width:100%;")

                                het_trs = "".join(
                                    f"<tr><td><strong>{html.escape(p['carbone'])}</strong></td>"
                                    f"<td style='font-family:\"JetBrains Mono\",monospace;'>{p['h']:.2f}</td>"
                                    f"<td style='font-family:\"JetBrains Mono\",monospace;'>{p['c']:.1f}</td>"
                                    f"<td>{html.escape(p['type'])}</td></tr>"
                                    for p in het_all
                                )
                                with ui.expansion(f"Taches HSQC / HMBC ({len(het_all)})", value=False).style("width:100%;margin-top:8px;"):
                                    safe_html(
                                        '<div class="data-table-wrap" style="max-height:300px;overflow-y:auto;"><table class="data-table">'
                                        '<thead><tr><th>Carbone</th><th>δ ¹H (ppm)</th><th>δ ¹³C (ppm)</th><th>Corrélation</th></tr></thead>'
                                        f'<tbody>{het_trs}</tbody></table></div>'
                                    )
                            else:
                                safe_html('<p class="muted-text">Aucune corrélation HSQC / HMBC détectée.</p>')

        return refresh_tab

    # ==================================================================
    # ONGLET 5 : SÉCURITÉ SGH & STUDIO DE PUBLICATION (ACS / RSC / WILEY)
    # ==================================================================
    def render_tab_export():
        root = ui.element("div")
        pub_cfg: dict[str, Any] = {
            "style": PUB_STYLES[0],
            "indices": False,
            "ghs_gen": 0,
            "target": "mol",  # mol | rxn
        }

        def refresh_tab():
            pub_cfg["ghs_gen"] = int(pub_cfg.get("ghs_gen", 0)) + 1
            cur_gen = pub_cfg["ghs_gen"]
            root.clear()
            single_smi = extract_single_smiles(clean_val(state.get("current_smiles")))
            mol_main = parse_smiles(single_smi)

            with root:
                if not mol_main:
                    safe_html('<div class="hud-card"><p class="muted-text">Structure chimique invalide.</p></div>')
                    return

                with ui.element("div").classes("modal-grid"):
                    # --- COLONNE GAUCHE : SÉCURITÉ SGH PUBCHEM ---
                    with ui.element("div").classes("hud-card"):
                        safe_html(
                            '<div class="section-kicker">CLASSIFICATION HSE · PUBCHEM</div>'
                            '<h2>Sécurité SGH / GHS</h2>'
                        )
                        ghs_slot = ui.element("div")
                        with ghs_slot:
                            safe_html('<p class="muted-small">Interrogation des données SGH sur PubChem...</p>')

                        async def load_ghs():
                            ident = state.get("ident")
                            cid = ident.get("cid") if isinstance(ident, dict) else None
                            if not cid:
                                fresh_id = await run.io_bound(get_chemical_identifiers, single_smi)
                                if pub_cfg["ghs_gen"] != cur_gen:
                                    return
                                cid = fresh_id.get("cid")
                            pictos, hazards = await run.io_bound(get_ghs_safety, cid)
                            if pub_cfg["ghs_gen"] != cur_gen:
                                return
                            ghs_slot.clear()
                            with ghs_slot:
                                if not pictos and not hazards:
                                    safe_html('<p class="muted-text">Aucun pictogramme SGH répertorié sur PubChem pour ce composé.</p>')
                                    return
                                if pictos:
                                    pics_html = "".join(
                                        f'<div style="text-align:center;">'
                                        f'<img src="{html.escape(p["url"])}" alt="{html.escape(p["code"])}" style="width:76px;height:76px;background:#fff;border-radius:10px;padding:4px;"/>'
                                        f'<div class="muted-small" style="font-family:\'JetBrains Mono\',monospace;margin-top:3px;">{html.escape(p["code"])}</div></div>'
                                        for p in pictos
                                    )
                                    safe_html(f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 14px 0;">{pics_html}</div>')
                                if hazards:
                                    hz_html = "".join(
                                        f'<div class="kv-item" style="margin-bottom:6px;font-size:0.81rem;">{html.escape(h)}</div>'
                                        for h in hazards
                                    )
                                    safe_html(hz_html)

                        ui.timer(0.05, load_ghs, once=True)

                    # --- COLONNE DROITE : STUDIO DE PUBLICATION ---
                    with ui.element("div").classes("hud-card"):
                        safe_html(
                            '<div class="section-kicker">EXPORT HAUTE DÉFINITION</div>'
                            '<h2>Studio de Publication Scientifique (ACS / RSC / Wiley)</h2>'
                        )
                        with ui.element("div").classes("sub-tabs").style("margin-bottom:10px;"):
                            for t_key, t_lbl in (("mol", "Molécule active"), ("rxn", "Schéma de réaction")):
                                cls = "glass-btn glass-btn-accent" if pub_cfg["target"] == t_key else "glass-btn"
                                _btn(
                                    t_lbl,
                                    classes=cls,
                                    style="padding:5px 11px;font-size:0.77rem;",
                                    on_click=lambda _e, tk=t_key: (pub_cfg.update(target=tk), refresh_tab()),
                                )

                        is_dark_style = "Sombre" in str(pub_cfg["style"])
                        box_bg = "#121a1e" if is_dark_style else "#ffffff"

                        if pub_cfg["target"] == "mol":
                            with ui.element("div").style("display:grid;grid-template-columns:1.6fr auto;gap:12px;align-items:center;margin-bottom:12px;"):
                                sel_pub = ui.select(PUB_STYLES, value=str(pub_cfg["style"]), label="Feuille de style").props("outlined dense")
                                chk_idx = ui.checkbox("Numéroter les atomes", value=bool(pub_cfg["indices"]))

                                sel_pub.on_value_change(lambda e: (pub_cfg.update(style=str(e.value or PUB_STYLES[0])), refresh_tab()))
                                chk_idx.on_value_change(lambda e: (pub_cfg.update(indices=bool(e.value)), refresh_tab()))

                            svg_pub = render_2d_svg(
                                mol_main,
                                width=520,
                                height=365,
                                atom_indices=bool(pub_cfg["indices"]),
                                stereo=True,
                                style_name=str(pub_cfg["style"]),
                                transparent=not is_dark_style,
                            )
                            safe_html(f'<div class="svg-box" style="background:{box_bg};margin-bottom:14px;">{svg_pub}</div>')

                            def copy_mol_png(_e=None):
                                ui.run_javascript(f"window.csCopySvgAsPng({json.dumps(svg_pub)});")

                            def dl_svg(_e=None):
                                ui.download(svg_pub.encode("utf-8"), "ChemStudio_publication.svg")

                            def dl_mol(_e=None):
                                m2d = Chem.Mol(mol_main)
                                compute_clean_2d_coords(m2d)
                                mol_2d = Chem.MolToMolBlock(m2d)
                                ui.download(mol_2d.encode("utf-8"), "ChemStudio_structure_2d.mol")

                            def dl_sdf(_e=None):
                                m3d = generate_3d_mol(single_smi)
                                if not m3d:
                                    ui.notify("Impossible de générer la géométrie 3D.", type="negative")
                                    return
                                sdf_str = Chem.MolToMolBlock(m3d)
                                ui.download(sdf_str.encode("utf-8"), "ChemStudio_structure_3d.sdf")

                            with ui.element("div").style("display:grid;grid-template-columns:repeat(2,1fr);gap:8px;"):
                                _btn("Copier image (PNG)", classes="glass-btn glass-btn-accent", on_click=copy_mol_png)
                                _btn("Vectoriel (.svg)", on_click=dl_svg)
                                _btn("Structure 2D (.mol)", on_click=dl_mol)
                                _btn("Géométrie 3D (.sdf)", on_click=dl_sdf)
                        else:
                            sel_pub_r = ui.select(PUB_STYLES, value=str(pub_cfg["style"]), label="Feuille de style").props("outlined dense").style("width:100%;margin-bottom:12px;")
                            sel_pub_r.on_value_change(lambda e: (pub_cfg.update(style=str(e.value or PUB_STYLES[0])), refresh_tab()))

                            r_list = [s for s in state.get("rxn_reactants", []) if parse_smiles(s) is not None]
                            p_list = [s for s in state.get("rxn_products", []) if parse_smiles(s) is not None]
                            raw_conds = state.get("rxn_conditions")
                            conds: dict[str, Any] = raw_conds if isinstance(raw_conds, dict) else {}

                            if not r_list and not p_list:
                                safe_html('<p class="muted-text">Construisez d\'abord une réaction dans l\'onglet 2 pour générer un schéma de publication.</p>')
                            else:
                                svg_rxn = render_annotated_reaction_svg(
                                    r_list,
                                    p_list,
                                    solvent=clean_val(conds.get("solvent")),
                                    temp=clean_val(conds.get("temp")),
                                    duration=clean_val(conds.get("duration")),
                                    style_name=str(pub_cfg["style"]),
                                    transparent=not is_dark_style,
                                )
                                safe_html(
                                    f'<div class="svg-box" style="background:{box_bg};margin-bottom:14px;overflow-x:auto;">'
                                    f'{svg_rxn}</div>'
                                )

                                def copy_rxn_pub_png(_e=None):
                                    ui.run_javascript(f"window.csCopySvgAsPng({json.dumps(svg_rxn)});")

                                def dl_rxn_pub_svg(_e=None):
                                    ui.download(svg_rxn.encode("utf-8"), "ChemStudio_reaction_publication.svg")

                                with ui.element("div").style("display:grid;grid-template-columns:repeat(2,1fr);gap:10px;"):
                                    _btn("Copier image (PNG)", classes="glass-btn glass-btn-accent", on_click=copy_rxn_pub_png)
                                    _btn("Vectoriel (.svg)", on_click=dl_rxn_pub_svg)

        return refresh_tab

    return render_tab_spec, render_tab_export