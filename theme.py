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

window.csPushSketcherStatus = function(ok, text) {
  const f = document.getElementById("cs-sketcher-iframe");
  if (f && f.contentWindow) {
    f.contentWindow.postMessage({ type: "cs_sketcher_status", ok: !!ok, text: text || "" }, "*");
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

// Pont temps réel du dessinateur 2D maison -> NiceGUI
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "cs_sketcher_change" && typeof emitEvent === "function") {
    emitEvent("cs_sketcher_change", { graph: event.data.graph || {} });
  }
});
</script>
"""


def inject_theme():
    ui.add_head_html(THEME_HEAD_HTML)