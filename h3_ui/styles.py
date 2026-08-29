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
.h3-settings-summary { padding: .7rem .85rem; border: 1px solid var(--border-color-primary); border-radius: var(--h3-radius-md); background: var(--h3-panel); }
.h3-settings-summary p { margin: 0; line-height: 1.45; }
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
