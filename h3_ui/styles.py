"""Design tokens and responsive styles for the MiniMax H3 Gradio UI."""

H3_UI_CSS = """
:root {
  --h3-accent: #6757f5;
  --h3-accent-soft: color-mix(in srgb, var(--h3-accent) 12%, transparent);
  --h3-panel: color-mix(in srgb, var(--background-fill-primary) 95%, var(--h3-accent) 5%);
  --h3-space-2: .6rem;
  --h3-space-3: .85rem;
  --h3-space-4: 1.15rem;
  --h3-radius-md: 14px;
  --h3-radius-lg: 18px;
  --h3-shadow: 0 12px 32px rgba(0, 0, 0, .11);
}

.gradio-container { max-width: 1540px !important; }
.gradio-container button { min-height: 42px; }
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container [role="tab"]:focus-visible {
  outline: 3px solid color-mix(in srgb, var(--h3-accent) 58%, transparent) !important;
  outline-offset: 2px;
}

.h3-hero {
  padding: var(--h3-space-4) 1.3rem;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--h3-radius-lg);
  background: linear-gradient(135deg, var(--h3-accent-soft), transparent 62%);
  margin-bottom: var(--h3-space-3);
}
.h3-hero h1 { margin: 0 0 .25rem; font-size: clamp(1.55rem, 2.4vw, 2.25rem); letter-spacing: -.025em; }
.h3-hero p { margin: 0; opacity: .8; }

.h3-system-ready,
.h3-system-warning {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: .35rem .65rem;
  min-height: 42px;
  padding: .55rem .8rem;
  border-radius: var(--h3-radius-md);
  border: 1px solid var(--border-color-primary);
  margin: .25rem 0 .65rem;
}
.h3-system-ready { background: color-mix(in srgb, #22c55e 10%, transparent); }
.h3-system-warning { background: color-mix(in srgb, #f59e0b 12%, transparent); }
.h3-system-ready strong, .h3-system-warning strong { white-space: nowrap; }
.h3-system-ready span, .h3-system-warning span { opacity: .82; }

.h3-generator-shell { gap: 1rem; align-items: flex-start; }
.h3-composer { gap: var(--h3-space-3); }
.h3-composer > .h3-advanced-block { order: 90; }
.h3-mode-row { gap: var(--h3-space-3); }
.h3-run-panel { gap: var(--h3-space-2); }
.h3-section-intro h3, .h3-section-intro p { margin-bottom: 0; }
.h3-section-intro p { opacity: .72; }

.h3-action-dock {
  position: sticky;
  top: .65rem;
  z-index: 8;
  padding: .75rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 16px;
  background: color-mix(in srgb, var(--h3-panel) 96%, transparent);
  box-shadow: var(--h3-shadow);
  backdrop-filter: blur(12px);
}
.h3-primary-action button { min-height: 48px; font-size: 1.02rem; font-weight: 700; }
.h3-status textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.h3-settings-panel { gap: var(--h3-space-2); }
.h3-settings-summary { border: 1px solid color-mix(in srgb, var(--h3-accent) 22%, var(--border-color-primary)); border-radius: var(--h3-radius-md); background: var(--h3-panel); box-shadow: 0 8px 24px rgba(0, 0, 0, .08); overflow: hidden; }
.h3-setup-card { font-size: .93rem; }
.h3-tone-purple { --h3-tone: #8b5cf6; --h3-tone-bg: color-mix(in srgb, #8b5cf6 16%, var(--background-fill-primary)); }
.h3-tone-blue { --h3-tone: #3b82f6; --h3-tone-bg: color-mix(in srgb, #3b82f6 16%, var(--background-fill-primary)); }
.h3-tone-cyan { --h3-tone: #0891b2; --h3-tone-bg: color-mix(in srgb, #06b6d4 16%, var(--background-fill-primary)); }
.h3-tone-green { --h3-tone: #16a34a; --h3-tone-bg: color-mix(in srgb, #22c55e 15%, var(--background-fill-primary)); }
.h3-tone-amber { --h3-tone: #d97706; --h3-tone-bg: color-mix(in srgb, #f59e0b 17%, var(--background-fill-primary)); }
.h3-tone-pink { --h3-tone: #db2777; --h3-tone-bg: color-mix(in srgb, #ec4899 15%, var(--background-fill-primary)); }
.h3-tone-neutral { --h3-tone: var(--body-text-color-subdued); --h3-tone-bg: color-mix(in srgb, var(--border-color-primary) 24%, var(--background-fill-primary)); }
.h3-setup-heading { display: flex; align-items: center; justify-content: space-between; gap: .75rem; padding: .8rem .9rem; color: #fff; background: linear-gradient(118deg, #4f46e5, #7c3aed 58%, #a855f7); }
.h3-setup-title { display: flex; align-items: center; gap: .65rem; min-width: 0; }
.h3-setup-symbol { display: grid; place-items: center; width: 2.15rem; height: 2.15rem; flex: 0 0 auto; border: 1px solid rgba(255,255,255,.35); border-radius: 10px; background: rgba(255,255,255,.17); box-shadow: inset 0 1px 0 rgba(255,255,255,.16); font-size: .8rem; }
.h3-setup-title > span:last-child { display: flex; flex-direction: column; min-width: 0; }
.h3-setup-title small { opacity: .78; font-size: .66rem; font-weight: 750; letter-spacing: .09em; line-height: 1.2; text-transform: uppercase; }
.h3-setup-title strong { overflow: hidden; font-size: 1.02rem; letter-spacing: -.01em; line-height: 1.35; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-result { padding: .32rem .62rem; border: 1px solid rgba(255,255,255,.48); border-radius: 999px; color: #fff; background: rgba(255,255,255,.16); font-size: .71rem; font-weight: 800; letter-spacing: .055em; text-transform: uppercase; }
.h3-setup-profile { display: flex; align-items: center; gap: .55rem; padding: .62rem .9rem; border-bottom: 1px solid var(--border-color-primary); background: color-mix(in srgb, #f59e0b 7%, var(--background-fill-primary)); }
.h3-setup-profile-badge { padding: .25rem .5rem; border: 1px solid color-mix(in srgb, var(--h3-tone) 42%, transparent); border-radius: 7px; color: var(--h3-tone); background: var(--h3-tone-bg); font-size: .66rem; font-weight: 850; letter-spacing: .06em; text-transform: uppercase; }
.h3-setup-profile > strong { min-width: 0; overflow: hidden; font-size: .79rem; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .45rem; padding: .7rem .85rem .55rem; }
.h3-setup-metric { display: flex; align-items: center; gap: .48rem; min-width: 0; padding: .5rem; border: 1px solid color-mix(in srgb, var(--h3-tone) 28%, var(--border-color-primary)); border-radius: 10px; background: var(--h3-tone-bg); }
.h3-setup-metric-icon { display: grid; place-items: center; width: 1.6rem; height: 1.6rem; flex: 0 0 auto; border-radius: 7px; color: #fff; background: var(--h3-tone); font-size: .78rem; font-weight: 900; }
.h3-setup-metric-copy { display: flex; flex-direction: column; min-width: 0; }
.h3-setup-metric-copy strong { overflow: hidden; color: var(--body-text-color); font-size: .82rem; line-height: 1.18; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-metric-copy span { color: var(--body-text-color-subdued); font-size: .61rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.h3-setup-pills { display: flex; flex-wrap: wrap; gap: .34rem; padding: 0 .85rem .75rem; }
.h3-setup-pill { display: inline-flex; align-items: center; max-width: 100%; overflow: hidden; border: 1px solid color-mix(in srgb, var(--h3-tone) 32%, var(--border-color-primary)); border-radius: 999px; color: var(--h3-tone); background: var(--h3-tone-bg); font-size: .68rem; line-height: 1; }
.h3-setup-pill > span { padding: .31rem .38rem .31rem .5rem; opacity: .76; font-weight: 700; }
.h3-setup-pill > strong { overflow: hidden; padding: .31rem .5rem .31rem .4rem; border-left: 1px solid color-mix(in srgb, var(--h3-tone) 25%, transparent); font-weight: 850; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-disclosure { border-top: 1px solid var(--border-color-primary); }
.h3-setup-disclosure summary { display: flex; align-items: center; gap: .4rem; padding: .62rem .85rem; color: #6d5ee8; background: color-mix(in srgb, var(--h3-accent) 5%, var(--background-fill-primary)); cursor: pointer; font-size: .78rem; font-weight: 800; list-style: none; user-select: none; }
.h3-setup-disclosure summary::-webkit-details-marker { display: none; }
.h3-setup-disclosure summary:focus-visible { outline: 3px solid color-mix(in srgb, var(--h3-accent) 48%, transparent); outline-offset: -3px; }
.h3-setup-disclosure summary:hover { background: var(--h3-accent-soft); }
.h3-setup-hide { display: none; }
.h3-setup-disclosure[open] .h3-setup-show { display: none; }
.h3-setup-disclosure[open] .h3-setup-hide { display: inline; }
.h3-setup-chevron { width: .45rem; height: .45rem; margin-left: auto; border-right: 2px solid currentColor; border-bottom: 2px solid currentColor; transform: rotate(45deg); transition: transform .16s ease; }
.h3-setup-disclosure[open] .h3-setup-chevron { transform: rotate(225deg); }
.h3-setup-detail-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: .55rem; padding: .7rem .75rem .8rem; background: color-mix(in srgb, var(--h3-accent) 3%, var(--background-fill-primary)); }
.h3-setup-group { padding: .35rem .6rem .5rem; border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--background-fill-primary); }
.h3-setup-detail-grid h4 { display: flex; align-items: center; gap: .38rem; margin: .2rem 0 .4rem; color: var(--body-text-color); font-size: .7rem; font-weight: 850; letter-spacing: .065em; text-transform: uppercase; }
.h3-setup-detail-grid h4 span { color: var(--h3-accent); }
.h3-setup-detail-grid dl { margin: 0; }
.h3-setup-detail { display: grid; grid-template-columns: minmax(5rem, .8fr) minmax(0, 1.2fr); align-items: center; gap: .55rem; padding: .32rem 0; border-top: 1px solid color-mix(in srgb, var(--border-color-primary) 58%, transparent); line-height: 1.3; }
.h3-setup-detail dt { color: var(--body-text-color-subdued); font-size: .7rem; font-weight: 650; }
.h3-setup-detail dd { margin: 0; overflow: hidden; text-align: right; }
.h3-setup-value { display: inline-block; max-width: 100%; overflow: hidden; padding: .2rem .4rem; border: 1px solid color-mix(in srgb, var(--h3-tone) 26%, var(--border-color-primary)); border-radius: 6px; color: var(--h3-tone); background: var(--h3-tone-bg); font-size: .69rem; font-weight: 800; overflow-wrap: anywhere; text-align: left; text-overflow: ellipsis; vertical-align: middle; }
.h3-settings-section { margin-top: .2rem; border-radius: var(--h3-radius-md); overflow: hidden; }
.h3-settings-section > div { gap: .75rem; }
.h3-danger-zone { border-color: color-mix(in srgb, #ef4444 55%, var(--border-color-primary)); }

.h3-gallery-shell { gap: .8rem; }
.h3-gallery-heading { padding: 1rem 1.15rem; border: 1px solid var(--border-color-primary); border-radius: var(--h3-radius-lg); background: linear-gradient(135deg, var(--h3-accent-soft), transparent 68%); }
.h3-gallery-heading h2, .h3-gallery-heading p, .h3-gallery-section-title h3, .h3-gallery-section-title p { margin-bottom: 0; }
.h3-gallery-heading p, .h3-gallery-section-title p { opacity: .72; }
.h3-gallery-toolbar { align-items: center; gap: .8rem; }
.h3-gallery-toolbar button { min-height: 42px; font-weight: 650; }
.h3-gallery-status { min-height: 42px; display: flex; align-items: center; padding: .45rem .75rem; border: 1px solid var(--border-color-primary); border-radius: 12px; background: var(--h3-panel); }
.h3-gallery-status p { margin: 0; }
.h3-gallery-card { border-radius: 15px; overflow: hidden; border-color: var(--border-color-primary); }
.h3-gallery-import { align-items: end; }
.h3-gallery-import button { min-height: 46px; margin-bottom: 2px; font-weight: 700; }
.h3-gallery-workspace { gap: 1rem; align-items: stretch; }
.h3-gallery-section-title { min-height: 58px; padding: .15rem .2rem; }
.h3-gallery-grid { border-radius: 14px; overflow: hidden; }
.h3-gallery-player video { border-radius: 12px; background: #08090c; }
.h3-gallery-download { min-height: 34px; }
.h3-gallery-ai-settings { padding: .25rem; border-radius: 12px; background: color-mix(in srgb, var(--h3-panel) 72%, transparent); }
.h3-gallery-actions button { min-height: 48px; font-weight: 700; }
.h3-gallery-danger { border-color: color-mix(in srgb, #ef4444 45%, var(--border-color-primary)); }
.h3-gallery-danger-actions button { min-height: 44px; }
.h3-gallery-post-status { min-height: 24px; }

@media (max-width: 1100px) {
  .h3-run-panel { min-width: 360px !important; }
  .h3-gallery-workspace > div { min-width: 0 !important; }
}

@media (max-width: 900px) {
  .gradio-container { padding-left: .65rem !important; padding-right: .65rem !important; }
  .h3-generator-shell { flex-direction: column; }
  .h3-generator-shell > div, .h3-run-panel, .h3-gallery-workspace > div { width: 100% !important; min-width: 0 !important; }
  .h3-action-dock { top: .4rem; }
  .h3-gallery-workspace { gap: .55rem; flex-direction: column; }
  .h3-gallery-heading { padding: .85rem; }
}

@media (max-width: 600px) {
  .gradio-container { padding-left: .45rem !important; padding-right: .45rem !important; }
  .h3-hero { padding: .85rem; border-radius: 14px; }
  .h3-hero p { font-size: .92rem; }
  .h3-mode-row { flex-direction: column; }
  .h3-mode-row > div { width: 100% !important; min-width: 0 !important; }
  .h3-action-dock {
    position: sticky;
    top: auto;
    bottom: .35rem;
    padding: .55rem;
  }
  .h3-action-dock button { min-width: 0 !important; }
  .h3-system-ready, .h3-system-warning { align-items: flex-start; flex-direction: column; }
  .h3-setup-metrics, .h3-setup-detail-grid { grid-template-columns: 1fr; }
  .h3-setup-profile > strong, .h3-setup-title strong { white-space: normal; }
}

@media (prefers-reduced-motion: reduce) {
  .gradio-container *, .gradio-container *::before, .gradio-container *::after {
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
  }
}
"""

# This is the only stylesheet mounted by the Gradio 6 application. Keep it
# strictly scoped to the generated setup summary so it cannot alter Gradio's
# accordions, tabs, forms, or page layout.
H3_SETUP_CSS = """
.h3-settings-summary {
  container-type: inline-size;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
}
.h3-settings-summary > div { padding: 0 !important; }
.h3-setup-card {
  overflow: hidden;
  border: 1px solid #374151;
  border-radius: 14px;
  color: #e5e7eb;
  background: #111827;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .22);
  font-family: inherit;
  font-size: .9rem;
}
.h3-tone-purple { --h3-tone: #c4b5fd; --h3-tone-solid: #7c3aed; --h3-tone-bg: rgba(124, 58, 237, .18); }
.h3-tone-blue { --h3-tone: #93c5fd; --h3-tone-solid: #2563eb; --h3-tone-bg: rgba(37, 99, 235, .18); }
.h3-tone-cyan { --h3-tone: #67e8f9; --h3-tone-solid: #0891b2; --h3-tone-bg: rgba(8, 145, 178, .18); }
.h3-tone-green { --h3-tone: #86efac; --h3-tone-solid: #16a34a; --h3-tone-bg: rgba(22, 163, 74, .17); }
.h3-tone-amber { --h3-tone: #fcd34d; --h3-tone-solid: #d97706; --h3-tone-bg: rgba(217, 119, 6, .19); }
.h3-tone-pink { --h3-tone: #f9a8d4; --h3-tone-solid: #db2777; --h3-tone-bg: rgba(219, 39, 119, .17); }
.h3-tone-neutral { --h3-tone: #cbd5e1; --h3-tone-solid: #64748b; --h3-tone-bg: rgba(100, 116, 139, .18); }
.h3-setup-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: .75rem;
  padding: .78rem .85rem;
  color: #fff;
  background: linear-gradient(118deg, #4338ca, #7c3aed 62%, #9333ea);
}
.h3-setup-title { display: flex; align-items: center; gap: .62rem; min-width: 0; }
.h3-setup-symbol {
  display: grid;
  place-items: center;
  width: 2rem;
  height: 2rem;
  flex: 0 0 auto;
  border: 1px solid rgba(255,255,255,.3);
  border-radius: 9px;
  background: rgba(255,255,255,.14);
  font-size: .72rem;
}
.h3-setup-title > span:last-child { display: flex; flex-direction: column; min-width: 0; }
.h3-setup-title small { opacity: .76; font-size: .62rem; font-weight: 800; letter-spacing: .08em; line-height: 1.2; text-transform: uppercase; }
.h3-setup-title strong { overflow: hidden; font-size: .98rem; line-height: 1.3; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-result {
  padding: .3rem .58rem;
  border: 1px solid rgba(255,255,255,.38);
  border-radius: 999px;
  color: #fff;
  background: rgba(255,255,255,.13);
  font-size: .66rem;
  font-weight: 850;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.h3-setup-profile {
  display: flex;
  align-items: center;
  gap: .5rem;
  padding: .58rem .85rem;
  border-bottom: 1px solid #293548;
  background: #172033;
}
.h3-setup-profile-badge {
  padding: .22rem .46rem;
  border: 1px solid rgba(245, 158, 11, .42);
  border-radius: 6px;
  color: #fcd34d;
  background: rgba(217, 119, 6, .17);
  font-size: .62rem;
  font-weight: 850;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.h3-setup-profile > strong { min-width: 0; overflow: hidden; color: #f8fafc; font-size: .76rem; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .42rem; padding: .65rem .75rem .52rem; }
.h3-setup-metric {
  display: flex;
  align-items: center;
  gap: .42rem;
  min-width: 0;
  padding: .46rem;
  border: 1px solid color-mix(in srgb, var(--h3-tone-solid) 48%, #374151);
  border-radius: 9px;
  background: var(--h3-tone-bg);
}
.h3-setup-metric-icon {
  display: grid;
  place-items: center;
  width: 1.48rem;
  height: 1.48rem;
  flex: 0 0 auto;
  border-radius: 6px;
  color: #fff;
  background: var(--h3-tone-solid);
  font-size: .7rem;
  font-weight: 900;
}
.h3-setup-metric-copy { display: flex; flex-direction: column; min-width: 0; }
.h3-setup-metric-copy strong { overflow: hidden; color: #f8fafc; font-size: .78rem; line-height: 1.2; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-metric-copy span { color: #94a3b8; font-size: .56rem; font-weight: 750; letter-spacing: .045em; text-transform: uppercase; }
.h3-setup-pills { display: flex; flex-wrap: wrap; gap: .32rem; padding: 0 .75rem .68rem; }
.h3-setup-pill {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--h3-tone-solid) 52%, #374151);
  border-radius: 999px;
  color: var(--h3-tone);
  background: var(--h3-tone-bg);
  font-size: .64rem;
  line-height: 1;
}
.h3-setup-pill > span { padding: .29rem .34rem .29rem .46rem; opacity: .78; font-weight: 700; }
.h3-setup-pill > strong { overflow: hidden; padding: .29rem .46rem .29rem .34rem; border-left: 1px solid rgba(255,255,255,.1); font-weight: 850; text-overflow: ellipsis; white-space: nowrap; }
.h3-setup-disclosure { border-top: 1px solid #293548; }
.h3-setup-disclosure summary {
  display: flex;
  align-items: center;
  gap: .4rem;
  padding: .58rem .78rem;
  color: #c4b5fd;
  background: #151b2b;
  cursor: pointer;
  font-size: .72rem;
  font-weight: 800;
  list-style: none;
  user-select: none;
}
.h3-setup-disclosure summary::-webkit-details-marker { display: none; }
.h3-setup-disclosure summary:focus-visible { outline: 2px solid #a78bfa; outline-offset: -2px; }
.h3-setup-disclosure summary:hover { background: #1c2540; }
.h3-setup-hide { display: none; }
.h3-setup-disclosure[open] .h3-setup-show { display: none; }
.h3-setup-disclosure[open] .h3-setup-hide { display: inline; }
.h3-setup-chevron { width: .42rem; height: .42rem; margin-left: auto; border-right: 2px solid currentColor; border-bottom: 2px solid currentColor; transform: rotate(45deg); transition: transform .16s ease; }
.h3-setup-disclosure[open] .h3-setup-chevron { transform: rotate(225deg); }
.h3-setup-detail-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: .48rem; padding: .62rem; background: #0f172a; }
.h3-setup-group { padding: .34rem .56rem .48rem; border: 1px solid #334155; border-radius: 9px; background: #151e30; }
.h3-setup-detail-grid h4 { display: flex; align-items: center; gap: .36rem; margin: .18rem 0 .36rem; color: #f8fafc; font-size: .66rem; font-weight: 850; letter-spacing: .065em; text-transform: uppercase; }
.h3-setup-detail-grid h4 span { color: #a78bfa; }
.h3-setup-detail-grid dl { margin: 0; }
.h3-setup-detail {
  display: grid;
  grid-template-columns: minmax(5rem, .8fr) minmax(0, 1.2fr);
  align-items: center;
  gap: .5rem;
  padding: .3rem 0;
  border-top: 1px solid #293548;
  line-height: 1.3;
}
.h3-setup-detail dt { color: #94a3b8; font-size: .66rem; font-weight: 650; }
.h3-setup-detail dd { margin: 0; overflow: hidden; text-align: right; }
.h3-setup-value {
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  padding: .18rem .38rem;
  border: 1px solid color-mix(in srgb, var(--h3-tone-solid) 48%, #334155);
  border-radius: 6px;
  color: var(--h3-tone);
  background: var(--h3-tone-bg);
  font-size: .65rem;
  font-weight: 800;
  overflow-wrap: anywhere;
  text-align: left;
  text-overflow: ellipsis;
  vertical-align: middle;
}
@container (max-width: 430px) {
  .h3-setup-metrics { grid-template-columns: minmax(0, 1fr); }
  .h3-setup-title strong, .h3-setup-profile > strong { white-space: normal; }
}
@media (prefers-reduced-motion: reduce) {
  .h3-setup-card *, .h3-setup-card *::before, .h3-setup-card *::after { transition-duration: .01ms !important; }
}
"""
