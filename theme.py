import inspect
from nicegui import ui

LOGO_SVG = """
<svg viewBox="0 0 120 120" width="54" height="54" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="csGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10b981"/>
      <stop offset="55%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#6366f1"/>
    </linearGradient>
    <filter id="csGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <polygon points="60,10 103,35 103,85 60,110 17,85 17,35"
           fill="rgba(16, 185, 129, 0.08)" stroke="url(#csGrad)"
           stroke-width="6.5" stroke-linejoin="round" filter="url(#csGlow)"/>
  <circle cx="60" cy="10" r="4.5" fill="#10b981"/>
  <circle cx="103" cy="85" r="4.5" fill="#38bdf8"/>
  <circle cx="17" cy="85" r="4.5" fill="#6366f1"/>
  <text x="60" y="72" text-anchor="middle" font-family="'Plus Jakarta Sans', Arial, sans-serif"
        font-size="34" font-weight="800" fill="currentColor" letter-spacing="-1">CS</text>
</svg>
"""

THEME_TOGGLE_SVGS = """
<svg class="icon-sun" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="4"/>
  <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
</svg>
<svg class="icon-moon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
</svg>
"""

_SUPPORTS_SANITIZE = "sanitize" in inspect.signature(ui.html).parameters


def safe_html(content: str):
    """Compatibilité NiceGUI 2.x et 3.x sans avertissement de sanitisation."""
    if _SUPPORTS_SANITIZE:
        return ui.html(content, sanitize=False)
    return ui.html(content)


THEME_HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #081010;
  --bg-grad-1: rgba(16, 185, 129, 0.14);
  --bg-grad-2: rgba(56, 189, 248, 0.12);
  --bg-grad-3: rgba(99, 102, 241, 0.10);
  --txt: #e6edf3;
  --sub-txt: #94a3b8;
  --card-glass: rgba(15, 25, 26, 0.68);
  --sub-card: rgba(10, 18, 19, 0.62);
  --modal-bg: rgba(11, 20, 22, 0.92);
  --border: rgba(148, 163, 184, 0.20);
  --border-strong: rgba(56, 189, 248, 0.42);
  --accent: #10b981;
  --accent-soft: rgba(16, 185, 129, 0.16);
  --cyan: #38bdf8;
  --cyan-soft: rgba(56, 189, 248, 0.16);
  --warn: #f59e0b;
  --danger: #ef4444;
  --pill-bg: rgba(56, 189, 248, 0.16);
  --pill-border: rgba(56, 189, 248, 0.42);
  --pill-shadow: 0 4px 18px rgba(56, 189, 248, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.14);
  --svg-bg: rgba(255, 255, 255, 0.96);
}

html[data-theme="light"] {
  --bg: #f1f5f9;
  --bg-grad-1: rgba(16, 185, 129, 0.12);
  --bg-grad-2: rgba(2, 132, 199, 0.11);
  --bg-grad-3: rgba(99, 102, 241, 0.08);
  --txt: #0f172a;
  --sub-txt: #475569;
  --card-glass: rgba(255, 255, 255, 0.78);
  --sub-card: rgba(248, 250, 252, 0.85);
  --modal-bg: rgba(255, 255, 255, 0.94);
  --border: rgba(15, 23, 42, 0.12);
  --border-strong: rgba(2, 132, 199, 0.36);
  --accent: #059669;
  --accent-soft: rgba(5, 150, 105, 0.12);
  --cyan: #0284c7;
  --cyan-soft: rgba(2, 132, 199, 0.12);
  --warn: #d97706;
  --danger: #dc2626;
  --pill-bg: rgba(255, 255, 255, 0.95);
  --pill-border: rgba(2, 132, 199, 0.32);
  --pill-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
  --svg-bg: #ffffff;
}

html, body {
  margin: 0;
  padding: 0;
  min-height: 100vh;
  background-color: var(--bg) !important;
  color: var(--txt) !important;
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background-image:
    radial-gradient(circle at 12% 10%, var(--bg-grad-1), transparent 36%),
    radial-gradient(circle at 86% 14%, var(--bg-grad-2), transparent 34%),
    radial-gradient(circle at 52% 92%, var(--bg-grad-3), transparent 40%);
  background-attachment: fixed;
}

.nicegui-content {
  padding: 0 !important;
  max-width: none !important;
}

::view-transition-old(root),
::view-transition-new(root) {
  animation: none;
  mix-blend-mode: normal;
}
::view-transition-new(root) {
  z-index: 9999;
}

.app-shell {
  width: min(1440px, 96vw);
  margin: 0 auto;
  padding: 16px 0 36px 0;
  position: relative;
  z-index: 1;
}

.top-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.brand-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.cb-logo-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--txt);
  transition: transform 0.25s ease;
}
.cb-logo-wrap:hover {
  transform: scale(1.05);
}

.brand-text h1 {
  margin: 0;
  font-size: 1.48rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  display: flex;
  align-items: center;
  gap: 8px;
  line-height: 1.15;
}

.version-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  border: 1px solid rgba(16, 185, 129, 0.35);
}

.section-kicker {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--cyan);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.glass-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 8px 13px;
  border-radius: 11px;
  background: var(--card-glass);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid var(--border);
  color: var(--txt);
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 0.81rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
}
.glass-btn:hover {
  border-color: var(--cyan);
  box-shadow: 0 0 14px var(--cyan-soft);
  transform: translateY(-1px);
}
.glass-btn-accent {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.24), rgba(56, 189, 248, 0.22));
  border-color: rgba(16, 185, 129, 0.48);
  color: var(--txt);
}
.glass-btn-accent:hover {
  border-color: var(--accent);
  box-shadow: 0 0 18px var(--accent-soft);
}
.glass-btn-danger {
  background: rgba(239, 68, 68, 0.14);
  border-color: rgba(239, 68, 68, 0.42);
  color: var(--danger);
}
.glass-btn-danger:hover {
  background: rgba(239, 68, 68, 0.22);
  border-color: var(--danger);
}
.icon-btn {
  width: 38px;
  height: 38px;
  padding: 0;
}

html[data-theme="dark"] .icon-sun { display: block; }
html[data-theme="dark"] .icon-moon { display: none; }
html[data-theme="light"] .icon-sun { display: none; }
html[data-theme="light"] .icon-moon { display: block; }

.lg-nav {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px;
  border-radius: 15px;
  background: var(--card-glass);
  backdrop-filter: blur(18px);
  border: 1px solid var(--border);
  max-width: 100%;
  overflow-x: auto;
}
.lg-pill {
  position: absolute;
  top: 5px;
  left: 5px;
  height: calc(100% - 10px);
  border-radius: 11px;
  background: var(--pill-bg);
  border: 1px solid var(--pill-border);
  box-shadow: var(--pill-shadow);
  pointer-events: none;
  transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1), width 0.32s cubic-bezier(0.22, 1, 0.36, 1);
  z-index: 0;
}
.lg-tab {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 8px 14px;
  border-radius: 11px;
  border: none;
  background: transparent;
  color: var(--sub-txt);
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 0.83rem;
  font-weight: 600;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.2s ease;
}
.lg-tab.active, .lg-tab:hover {
  color: var(--txt);
}

.hud-divider {
  height: 1px;
  width: 100%;
  margin: 12px 0 16px 0;
  background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
}

.hud-card {
  background: var(--card-glass);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.14);
}
.hud-card h2 {
  margin: 4px 0 10px 0;
  font-size: 1.15rem;
  font-weight: 700;
}

.mol-banner {
  background: var(--card-glass);
  backdrop-filter: blur(18px);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.modal-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
@media (max-width: 980px) {
  .modal-grid { grid-template-columns: 1fr; }
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}
.kpi-card {
  background: var(--sub-card);
  border: 1px solid var(--border);
  border-radius: 13px;
  padding: 12px 14px;
}
.kpi-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--sub-txt);
  font-weight: 600;
}
.kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.25rem;
  font-weight: 700;
  margin-top: 4px;
  color: var(--txt);
}

.kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
}
.kv-item {
  background: var(--sub-card);
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: 9px 12px;
}
.kv-label {
  font-size: 0.69rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--sub-txt);
  font-weight: 600;
}
.kv-val {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--txt);
  margin-top: 2px;
  word-break: break-word;
}

.mono-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 7px;
  background: var(--cyan-soft);
  color: var(--cyan);
  border: 1px solid rgba(56, 189, 248, 0.30);
}
.copyable-tag {
  cursor: pointer;
  transition: transform 0.15s ease, border-color 0.15s ease;
}
.copyable-tag:hover {
  border-color: var(--accent);
  transform: translateY(-1px);
}
.stock-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.stock-ok {
  background: rgba(16, 185, 129, 0.16);
  color: var(--accent);
  border: 1px solid rgba(16, 185, 129, 0.38);
}
.stock-low {
  background: rgba(245, 158, 11, 0.16);
  color: var(--warn);
  border: 1px solid rgba(245, 158, 11, 0.40);
}
.stock-empty {
  background: rgba(239, 68, 68, 0.16);
  color: var(--danger);
  border: 1px solid rgba(239, 68, 68, 0.40);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 11px;
  border-radius: 999px;
  background: var(--card-glass);
  border: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.73rem;
  color: var(--sub-txt);
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.status-ok { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.status-warn { background: var(--warn); box-shadow: 0 0 8px var(--warn); }
.status-err { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

.svg-box {
  background: var(--svg-bg);
  border: 1px solid var(--border);
  border-radius: 13px;
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.svg-box svg {
  max-width: 100%;
  height: auto;
  display: block;
}
.svg-box-adaptive {
  background: var(--sub-card);
  border: 1px solid var(--border);
  border-radius: 13px;
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.svg-box-adaptive svg {
  max-width: 100%;
  height: auto;
  display: block;
}
html[data-theme="dark"] .svg-box-adaptive svg {
  filter: invert(0.90) hue-rotate(180deg) contrast(1.08);
}

.viewer-3d-frame {
  width: 100%;
  height: 465px;
  border: 1px solid var(--border);
  border-radius: 13px;
  background: var(--sub-card);
}
.sketcher-frame {
  width: 100%;
  height: 520px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--sub-card);
  display: block;
}

.sub-tabs {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.data-table-wrap {
  width: 100%;
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--sub-card);
}
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
.data-table th {
  text-align: left;
  padding: 9px 11px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  text-transform: uppercase;
  color: var(--cyan);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.data-table td {
  padding: 8px 11px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.10);
  color: var(--txt);
}
.data-table tr:last-child td {
  border-bottom: none;
}

.cb-modal-card {
  width: min(640px, 94vw) !important;
  max-width: 94vw !important;
  background: var(--modal-bg) !important;
  backdrop-filter: blur(26px) saturate(175%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 24px;
  color: var(--txt);
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
}
.modal-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
}

.muted-text {
  color: var(--sub-txt);
  font-size: 0.86rem;
  line-height: 1.5;
  margin: 0;
}
.muted-small {
  color: var(--sub-txt);
  font-size: 0.76rem;
}

.tab-panel { display: none; }
.tab-panel.active { display: block; animation: csFadeIn 0.22s ease; }
@keyframes csFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.app-footer {
  margin-top: 26px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  color: var(--sub-txt);
}
</style>

<script>
window.csPushThemeToSketcher = function(isDark) {
  const f = document.getElementById("cs-sketcher-iframe");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({ type: "cs_set_theme", dark: !!isDark }, "*");
  }
};

window.csPushGraphToSketcher = function(graphObj) {
  const f = document.getElementById("cs-sketcher-iframe");
  if (f && f.contentWindow && graphObj) {
    f.contentWindow.postMessage({ type: "cs_load_graph", graph: graphObj }, "*");
  }
};

window.csPushSketcherStatus = function(ok, text, invalidIds, smiles) {
  const f = document.getElementById("cs-sketcher-iframe");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage(
      { type: "cs_sketcher_status", ok: !!ok, text: text || "", invalidIds: invalidIds || [], smiles: smiles || "" },
      "*"
    );
  }
};

window.cbApplyTheme = function(mode) {
  const isLight = mode === "light";
  document.documentElement.setAttribute("data-theme", isLight ? "light" : "dark");
  window.csPushThemeToSketcher(!isLight);
};

window.cbToggleThemeWave = function(mode) {
  const target = mode === "light" ? "light" : "dark";
  const btn = document.getElementById("theme-toggle");
  window.csPushThemeToSketcher(target !== "light");
  if (!document.startViewTransition || !btn) {
    window.cbApplyTheme(target);
    return;
  }
  const rect = btn.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  const endRadius = Math.hypot(
    Math.max(x, window.innerWidth - x),
    Math.max(y, window.innerHeight - y)
  );
  const transition = document.startViewTransition(() => {
    window.cbApplyTheme(target);
  });
  transition.ready.then(() => {
    document.documentElement.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${endRadius}px at ${x}px ${y}px)`] },
      { duration: 480, easing: "cubic-bezier(0.22, 1, 0.36, 1)", pseudoElement: "::view-transition-new(root)" }
    );
  });
};

window.cbMovePill = function(instant) {
  const nav = document.getElementById("lg-nav");
  const pill = document.getElementById("lg-pill");
  if (!nav || !pill) return;
  const active = nav.querySelector(".lg-tab.active");
  if (!active) return;
  if (instant) pill.style.transition = "none";
  pill.style.width = active.offsetWidth + "px";
  pill.style.transform = `translateX(${active.offsetLeft - 5}px)`;
  if (instant) {
    pill.offsetHeight;
    pill.style.transition = "";
  }
};

window.cbCopyText = function(text, label) {
  if (!navigator.clipboard) return;
  navigator.clipboard.writeText(text).then(() => {
    const toast = document.createElement("div");
    toast.textContent = "Copié : " + (label || text);
    toast.style.cssText =
      "position:fixed;bottom:22px;right:22px;z-index:99999;padding:9px 15px;border-radius:10px;" +
      "background:#0f172a;color:#38bdf8;border:1px solid rgba(56,189,248,0.45);" +
      "font-family:'JetBrains Mono',monospace;font-size:0.78rem;box-shadow:0 10px 28px rgba(0,0,0,0.4);";
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 1800);
  });
};

window.addEventListener("resize", () => window.cbMovePill(true));

// === Toast rapide, export PNG/SVG & Tableau Périodique plein écran ===
window.csToast = function(msg) {
  let t = document.getElementById("cs-float-toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "cs-float-toast";
    t.style.cssText = "position:fixed;bottom:22px;right:22px;z-index:10050;padding:10px 16px;border-radius:12px;font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:600;color:#e6edf3;background:rgba(10,20,24,0.94);border:1px solid rgba(56,189,248,0.5);box-shadow:0 12px 32px rgba(0,0,0,0.45);backdrop-filter:blur(12px);transition:opacity 0.25s ease;pointer-events:none;";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = "1";
  clearTimeout(t._tm);
  t._tm = setTimeout(() => { t.style.opacity = "0"; }, 2400);
};

window.csDownloadSvg = function(svgText) {
  const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "molecule_chemstudio.svg";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  window.csToast("Fichier molecule_chemstudio.svg téléchargé");
};

window.csCopySvgAsPng = async function(svgText) {
  const makeBlob = () => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth || img.width || 300;
      const h = img.naturalHeight || img.height || 240;
      const c = document.createElement("canvas");
      const dpr = 3;
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      const ctx = c.getContext("2d");
      ctx.scale(dpr, dpr);
      ctx.drawImage(img, 0, 0, w, h);
      c.toBlob(b => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png");
    };
    img.onerror = () => reject(new Error("SVG load failed"));
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgText);
  });

  try {
    if (navigator.clipboard && typeof window.ClipboardItem === "function") {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": makeBlob() })]);
      window.csToast("Image transparente HD copiée (prête pour Word)");
      return;
    }
  } catch (_err) {
    // Secours : téléchargement direct en PNG transparent si le navigateur bloque le presse-papiers
  }
  try {
    const blob = await makeBlob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "molecule_chemstudio.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    window.csToast("Image PNG transparente téléchargée (insérable dans Word)");
  } catch (_e2) {
    window.csToast("Impossible de générer l'image PNG");
  }
};

window.csOpenPeriodicTable = (function() {
  const PT_DATA = [
    [1,"H","Hydrogène",1,1,"nm"],[2,"He","Hélium",18,1,"ng"],
    [3,"Li","Lithium",1,2,"alk"],[4,"Be","Béryllium",2,2,"ae"],[5,"B","Bore",13,2,"mt"],[6,"C","Carbone",14,2,"nm"],[7,"N","Azote",15,2,"nm"],[8,"O","Oxygène",16,2,"nm"],[9,"F","Fluor",17,2,"hal"],[10,"Ne","Néon",18,2,"ng"],
    [11,"Na","Sodium",1,3,"alk"],[12,"Mg","Magnésium",2,3,"ae"],[13,"Al","Aluminium",13,3,"pm"],[14,"Si","Silicium",14,3,"mt"],[15,"P","Phosphore",15,3,"nm"],[16,"S","Soufre",16,3,"nm"],[17,"Cl","Chlore",17,3,"hal"],[18,"Ar","Argon",18,3,"ng"],
    [19,"K","Potassium",1,4,"alk"],[20,"Ca","Calcium",2,4,"ae"],[21,"Sc","Scandium",3,4,"tm"],[22,"Ti","Titane",4,4,"tm"],[23,"V","Vanadium",5,4,"tm"],[24,"Cr","Chrome",6,4,"tm"],[25,"Mn","Manganèse",7,4,"tm"],[26,"Fe","Fer",8,4,"tm"],[27,"Co","Cobalt",9,4,"tm"],[28,"Ni","Nickel",10,4,"tm"],[29,"Cu","Cuivre",11,4,"tm"],[30,"Zn","Zinc",12,4,"tm"],[31,"Ga","Gallium",13,4,"pm"],[32,"Ge","Germanium",14,4,"mt"],[33,"As","Arsenic",15,4,"mt"],[34,"Se","Sélénium",16,4,"nm"],[35,"Br","Brome",17,4,"hal"],[36,"Kr","Krypton",18,4,"ng"],
    [37,"Rb","Rubidium",1,5,"alk"],[38,"Sr","Strontium",2,5,"ae"],[39,"Y","Yttrium",3,5,"tm"],[40,"Zr","Zirconium",4,5,"tm"],[41,"Nb","Niobium",5,5,"tm"],[42,"Mo","Molybdène",6,5,"tm"],[43,"Tc","Technétium",7,5,"tm"],[44,"Ru","Ruthénium",8,5,"tm"],[45,"Rh","Rhodium",9,5,"tm"],[46,"Pd","Palladium",10,5,"tm"],[47,"Ag","Argent",11,5,"tm"],[48,"Cd","Cadmium",12,5,"tm"],[49,"In","Indium",13,5,"pm"],[50,"Sn","Étain",14,5,"pm"],[51,"Sb","Antimoine",15,5,"mt"],[52,"Te","Tellure",16,5,"mt"],[53,"I","Iode",17,5,"hal"],[54,"Xe","Xénon",18,5,"ng"],
    [55,"Cs","Césium",1,6,"alk"],[56,"Ba","Baryum",2,6,"ae"],[72,"Hf","Hafnium",4,6,"tm"],[73,"Ta","Tantale",5,6,"tm"],[74,"W","Tungstène",6,6,"tm"],[75,"Re","Rhénium",7,6,"tm"],[76,"Os","Osmium",8,6,"tm"],[77,"Ir","Iridium",9,6,"tm"],[78,"Pt","Platine",10,6,"tm"],[79,"Au","Or",11,6,"tm"],[80,"Hg","Mercure",12,6,"tm"],[81,"Tl","Thallium",13,6,"pm"],[82,"Pb","Plomb",14,6,"pm"],[83,"Bi","Bismuth",15,6,"pm"],[84,"Po","Polonium",16,6,"mt"],[85,"At","Astate",17,6,"hal"],[86,"Rn","Radon",18,6,"ng"],
    [87,"Fr","Francium",1,7,"alk"],[88,"Ra","Radium",2,7,"ae"],[104,"Rf","Rutherfordium",4,7,"tm"],[105,"Db","Dubnium",5,7,"tm"],[106,"Sg","Seaborgium",6,7,"tm"],[107,"Bh","Bohrium",7,7,"tm"],[108,"Hs","Hassium",8,7,"tm"],[109,"Mt","Meitnerium",9,7,"tm"],[110,"Ds","Darmstadtium",10,7,"tm"],[111,"Rg","Roentgenium",11,7,"tm"],[112,"Cn","Copernicium",12,7,"tm"],[113,"Nh","Nihonium",13,7,"pm"],[114,"Fl","Flérovium",14,7,"pm"],[115,"Mc","Moscovium",15,7,"pm"],[116,"Lv","Livermorium",16,7,"pm"],[117,"Ts","Tennesse",17,7,"hal"],[118,"Og","Oganesson",18,7,"ng"],
    [57,"La","Lanthane",4,9,"lan"],[58,"Ce","Cérium",5,9,"lan"],[59,"Pr","Praséodyme",6,9,"lan"],[60,"Nd","Néodyme",7,9,"lan"],[61,"Pm","Prométhium",8,9,"lan"],[62,"Sm","Samarium",9,9,"lan"],[63,"Eu","Europium",10,9,"lan"],[64,"Gd","Gadolinium",11,9,"lan"],[65,"Tb","Terbium",12,9,"lan"],[66,"Dy","Dysprosium",13,9,"lan"],[67,"Ho","Holmium",14,9,"lan"],[68,"Er","Erbium",15,9,"lan"],[69,"Tm","Thulium",16,9,"lan"],[70,"Yb","Ytterbium",17,9,"lan"],[71,"Lu","Lutécium",18,9,"lan"],
    [89,"Ac","Actinium",4,10,"act"],[90,"Th","Thorium",5,10,"act"],[91,"Pa","Protactinium",6,10,"act"],[92,"U","Uranium",7,10,"act"],[93,"Np","Neptunium",8,10,"act"],[94,"Pu","Plutonium",9,10,"act"],[95,"Am","Américium",10,10,"act"],[96,"Cm","Curium",11,10,"act"],[97,"Bk","Berkélium",12,10,"act"],[98,"Cf","Californium",13,10,"act"],[99,"Es","Einsteinium",14,10,"act"],[100,"Fm","Fermium",15,10,"act"],[101,"Md","Mendélévium",16,10,"act"],[102,"No","Nobélium",17,10,"act"],[103,"Lr","Lawrencium",18,10,"act"]
  ];
  const FAM_NAMES = {
    nm:"Non-métal", ng:"Gaz noble", alk:"Métal alcalin", ae:"Alcalino-terreux",
    tm:"Métal de transition", pm:"Métal pauvre", mt:"Métalloïde",
    hal:"Halogène", lan:"Lanthanide", act:"Actinide"
  };

  let overlay = null, searchEl = null, infoEl = null, cells = [];

  function norm(s) {
    return String(s || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
  }

  function selectSym(sym) {
    const f = document.getElementById("cs-sketcher-iframe");
    if (f && f.contentWindow) {
      f.contentWindow.postMessage({ type: "cs_select_element", el: sym }, "*");
    }
    if (overlay) overlay.classList.remove("open");
  }

  function applyFilter(raw) {
    const q = norm(raw);
    let firstMatch = null, exactMatch = null;
    cells.forEach(c => {
      const ok = !q || c.dataset.search.includes(q);
      c.classList.toggle("dim", !ok);
      c.classList.remove("best-match");
      if (ok && q) {
        if (!firstMatch) firstMatch = c;
        if (norm(c.dataset.sym) === q || c.dataset.z === q) exactMatch = c;
      }
    });
    const pick = exactMatch || firstMatch;
    if (pick && q) {
      pick.classList.add("best-match");
      infoEl.textContent = `Entrée ↵ pour choisir : Z = ${pick.dataset.z} · ${pick.dataset.sym} — ${pick.dataset.name} (${FAM_NAMES[pick.dataset.fam] || ""})`;
    } else if (q) {
      infoEl.textContent = "Aucun élément trouvé pour cette recherche.";
    } else {
      infoEl.textContent = "Survolez un élément pour ses détails · Cliquez ou tapez son symbole puis Entrée";
    }
    return pick;
  }

  function ensureModal() {
    if (overlay) return;
    const st = document.createElement("style");
    st.textContent = `
      #cs-pt-overlay {
        position: fixed; inset: 0; z-index: 9999; display: none;
        align-items: center; justify-content: center; padding: 18px;
        background: rgba(4, 10, 14, 0.76); backdrop-filter: blur(10px);
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      }
      #cs-pt-overlay.open { display: flex; }
      #cs-pt-card {
        width: 100%; max-width: 1180px; max-height: 92vh; overflow: auto;
        border-radius: 18px; padding: 18px 20px;
        background: rgba(11, 20, 25, 0.96); color: #e6edf3;
        border: 1px solid rgba(148, 163, 184, 0.26);
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.58);
        display: flex; flex-direction: column; gap: 12px;
      }
      html[data-theme="light"] #cs-pt-card {
        background: rgba(255, 255, 255, 0.98); color: #0f172a;
        border-color: rgba(15, 23, 42, 0.14);
      }
      .cs-pt-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
      .cs-pt-title { font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #38bdf8; font-weight: 700; }
      html[data-theme="light"] .cs-pt-title { color: #0284c7; }
      .cs-pt-search-row { display: flex; align-items: center; gap: 8px; }
      #cs-pt-search {
        flex: 1; height: 40px; padding: 0 14px; border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.30);
        background: rgba(255, 255, 255, 0.05); color: inherit;
        font-size: 14px; font-family: inherit; outline: none;
      }
      html[data-theme="light"] #cs-pt-search { background: #f8fafc; border-color: rgba(15, 23, 42, 0.18); }
      #cs-pt-search:focus { border-color: #38bdf8; box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.18); }
      .cs-pt-close {
        height: 36px; padding: 0 12px; border-radius: 9px; cursor: pointer;
        border: 1px solid rgba(148, 163, 184, 0.28); background: transparent;
        color: inherit; font-weight: 700; font-size: 13px;
      }
      .cs-pt-close:hover { border-color: #f43f5e; color: #f43f5e; }
      .cs-pt-grid {
        display: grid; grid-template-columns: repeat(18, minmax(48px, 1fr));
        grid-template-rows: repeat(7, 50px) 10px repeat(2, 50px);
        gap: 4px; min-width: 920px;
      }
      .cs-pt-cell {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(255, 255, 255, 0.04); color: inherit; cursor: pointer;
        padding: 2px; transition: transform 0.12s ease, border-color 0.12s ease, opacity 0.12s ease;
        overflow: hidden;
      }
      .cs-pt-cell:hover { transform: scale(1.12); z-index: 5; border-color: #38bdf8; box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
      .cs-pt-cell.dim { opacity: 0.16; }
      .cs-pt-cell.cur-el { box-shadow: inset 0 0 0 2px #10b981; }
      .cs-pt-cell.best-match { border-color: #38bdf8 !important; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.65); transform: scale(1.06); z-index: 4; }
      .cs-pt-z { font-family: 'JetBrains Mono', monospace; font-size: 9px; opacity: 0.72; line-height: 1; }
      .cs-pt-sym { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700; line-height: 1.15; }
      .cs-pt-nm { font-size: 8.5px; opacity: 0.78; line-height: 1.05; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .cs-pt-gap {
        display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace; font-size: 9.5px; opacity: 0.55;
        border-radius: 7px; border: 1px dashed rgba(148, 163, 184, 0.28);
      }
      .cs-fam-nm  { background: rgba(16, 185, 129, 0.18); border-color: rgba(16, 185, 129, 0.45); }
      .cs-fam-ng  { background: rgba(139, 92, 246, 0.18); border-color: rgba(139, 92, 246, 0.42); }
      .cs-fam-alk { background: rgba(244, 63, 94, 0.16); border-color: rgba(244, 63, 94, 0.40); }
      .cs-fam-ae  { background: rgba(249, 115, 22, 0.16); border-color: rgba(249, 115, 22, 0.40); }
      .cs-fam-tm  { background: rgba(56, 189, 248, 0.13); border-color: rgba(56, 189, 248, 0.34); }
      .cs-fam-pm  { background: rgba(148, 163, 184, 0.16); border-color: rgba(148, 163, 184, 0.36); }
      .cs-fam-mt  { background: rgba(20, 184, 166, 0.18); border-color: rgba(20, 184, 166, 0.45); }
      .cs-fam-hal { background: rgba(234, 179, 8, 0.18); border-color: rgba(234, 179, 8, 0.45); }
      .cs-fam-lan { background: rgba(236, 72, 153, 0.15); border-color: rgba(236, 72, 153, 0.38); }
      .cs-fam-act { background: rgba(217, 70, 239, 0.15); border-color: rgba(217, 70, 239, 0.38); }
      .cs-pt-foot { font-family: 'JetBrains Mono', monospace; font-size: 12px; opacity: 0.82; min-height: 18px; }
    `;
    document.head.appendChild(st);

    overlay = document.createElement("div");
    overlay.id = "cs-pt-overlay";
    overlay.innerHTML = `
      <div id="cs-pt-card">
        <div class="cs-pt-top">
          <span class="cs-pt-title">Tableau Périodique des Éléments (118)</span>
          <button class="cs-pt-close" id="cs-pt-close">Fermer (Échap)</button>
        </div>
        <div class="cs-pt-search-row">
          <input id="cs-pt-search" type="text" placeholder="Rechercher un élément par symbole, nom ou numéro atomique (ex : Pd, Bore, 26, Cuivre)..."/>
          <button class="cs-pt-close" id="cs-pt-clear-q" title="Effacer la recherche">✕</button>
        </div>
        <div style="overflow-x:auto;padding:4px 2px;">
          <div class="cs-pt-grid" id="cs-pt-grid"></div>
        </div>
        <div class="cs-pt-foot" id="cs-pt-info">Survolez un élément pour ses détails · Cliquez ou tapez son symbole puis Entrée</div>
      </div>
    `;
    document.body.appendChild(overlay);

    const grid = overlay.querySelector("#cs-pt-grid");
    searchEl = overlay.querySelector("#cs-pt-search");
    infoEl = overlay.querySelector("#cs-pt-info");

    const addGap = (txt, col, row) => {
      const g = document.createElement("div");
      g.className = "cs-pt-gap";
      g.style.gridColumn = String(col);
      g.style.gridRow = String(row);
      g.textContent = txt;
      grid.appendChild(g);
    };
    addGap("57–71", 3, 6);
    addGap("89–103", 3, 7);

    PT_DATA.forEach(([z, sym, name, col, row, fam]) => {
      const btn = document.createElement("button");
      btn.className = "cs-pt-cell cs-fam-" + fam;
      btn.style.gridColumn = String(col);
      btn.style.gridRow = String(row);
      btn.dataset.z = String(z);
      btn.dataset.sym = sym;
      btn.dataset.name = name;
      btn.dataset.fam = fam;
      btn.dataset.search = norm(`${z} ${sym} ${name} ${FAM_NAMES[fam] || ""}`);
      btn.innerHTML = `<span class="cs-pt-z">${z}</span><span class="cs-pt-sym">${sym}</span><span class="cs-pt-nm">${name}</span>`;
      btn.addEventListener("mouseenter", () => {
        infoEl.textContent = `Z = ${z} · ${sym} — ${name} · ${FAM_NAMES[fam] || ""}`;
      });
      btn.addEventListener("click", () => selectSym(sym));
      grid.appendChild(btn);
      cells.push(btn);
    });

    searchEl.addEventListener("input", () => applyFilter(searchEl.value));
    searchEl.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        const pick = applyFilter(searchEl.value);
        if (pick) selectSym(pick.dataset.sym);
      } else if (e.key === "Escape") {
        overlay.classList.remove("open");
      }
    });
    overlay.querySelector("#cs-pt-clear-q").addEventListener("click", () => {
      searchEl.value = "";
      applyFilter("");
      searchEl.focus();
    });
    overlay.querySelector("#cs-pt-close").addEventListener("click", () => overlay.classList.remove("open"));
    overlay.addEventListener("pointerdown", e => {
      if (e.target === overlay) overlay.classList.remove("open");
    });
    window.addEventListener("keydown", e => {
      if (e.key === "Escape" && overlay && overlay.classList.contains("open")) {
        overlay.classList.remove("open");
      }
    });
  }

  return function(activeEl) {
    ensureModal();
    searchEl.value = "";
    applyFilter("");
    cells.forEach(c => c.classList.toggle("cur-el", c.dataset.sym === activeEl));
    overlay.classList.add("open");
    setTimeout(() => searchEl.focus(), 40);
  };
})();

// Pont temps réel du dessinateur 2D maison -> NiceGUI + presse-papiers + tableau périodique
window.addEventListener("message", (event) => {
  if (!event.data || typeof event.data !== "object") return;
  const d = event.data;
  if (d.type === "cs_sketcher_change" && typeof emitEvent === "function") {
    emitEvent("cs_sketcher_change", { graph: d.graph || {} });
  } else if (d.type === "cs_copy_text" && typeof window.cbCopyText === "function") {
    window.cbCopyText(String(d.text || ""), String(d.label || "Texte"));
  } else if (d.type === "cs_copy_png" && d.svg) {
    window.csCopySvgAsPng(String(d.svg));
  } else if (d.type === "cs_download_svg" && d.svg) {
    window.csDownloadSvg(String(d.svg));
  } else if (d.type === "cs_toast" && d.text) {
    window.csToast(String(d.text));
  } else if (d.type === "cs_open_periodic_table") {
    window.csOpenPeriodicTable(String(d.activeEl || "C"));
  }
});
</script>
"""


def inject_theme():
    ui.add_head_html(THEME_HEAD_HTML)