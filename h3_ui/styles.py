"""Design tokens and responsive styles for the Prism Studio Gradio UI."""

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
  border: 1px solid #303b4d;
  border-radius: 10px;
  color: #e5e7eb;
  background: #111827;
  font-family: inherit;
  font-size: 13px;
  line-height: 1.45;
}
.h3-setup-card .h3-setup-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 16px;
  border-bottom: 1px solid #303b4d;
  background: #182132;
}
.h3-setup-heading > strong { color: #f8fafc; font-size: 14px; font-weight: 650; }
.h3-setup-heading > span {
  padding: 2px 9px;
  border: 1px solid #554681;
  border-radius: 6px;
  background: #2b2442;
  color: #d4c5ff;
  font-size: 12px;
  font-weight: 600;
}
.h3-setup-card .h3-setup-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0;
  margin: 0;
  padding: 14px 16px 10px;
}
.h3-setup-metrics .h3-setup-detail {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 0 16px;
  border: 0;
  border-left: 1px solid #303b4d;
  min-width: 0;
}
.h3-setup-metrics .h3-setup-detail:first-child { padding-left: 0; border-left: 0; }
.h3-setup-card dt { color: #94a3b8; font-size: 11px; font-weight: 500; }
.h3-setup-card dd { margin: 0; color: #edf2f7; overflow-wrap: anywhere; }
.h3-setup-metrics .h3-setup-detail dd { text-align: left; font-size: 14px; font-weight: 600; }
.h3-setup-context {
  display: flex;
  flex-wrap: wrap;
  gap: 5px 18px;
  padding: 0 16px 12px;
  color: #aab7c9;
  font-size: 12px;
}
.h3-setup-card .h3-setup-disclosure { border-top: 1px solid #293548; }
.h3-setup-card .h3-setup-disclosure summary {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  color: #bac5d6;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  list-style: none;
}
.h3-setup-disclosure summary::-webkit-details-marker { display: none; }
.h3-setup-disclosure summary::after { content: "+"; margin-left: auto; color: #94a3b8; }
.h3-setup-disclosure[open] summary::after { content: "−"; }
.h3-setup-card .h3-setup-disclosure summary:hover { color: #ede9fe; background: #1b2537; }
.h3-setup-disclosure summary:focus-visible { outline: 2px solid #a78bfa; outline-offset: -2px; }
.h3-setup-card .h3-setup-detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 24px;
  padding: 0 16px 12px;
  background: transparent;
}
.h3-setup-detail-grid dl, .h3-setup-changes { margin: 0; }
.h3-setup-detail-grid .h3-setup-detail, .h3-setup-changes .h3-setup-detail {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.5fr);
  gap: 12px;
  padding: 6px 0;
  border-top: 1px solid #293548;
  font-size: 12px;
}
.h3-setup-detail-grid dd, .h3-setup-changes dd { text-align: right; }
.h3-setup-changes { padding: 0 16px 12px; }
.h3-setup-card ul { margin: 0; padding: 0 16px 12px 32px; font-size: 12px; color: #aab7c9; }
.h3-setup-card [role="alert"] { padding: 10px 16px; color: #fcd34d; }
@container (max-width: 520px) {
  .h3-setup-card .h3-setup-metrics { grid-template-columns: 1fr; gap: 9px; }
  .h3-setup-metrics .h3-setup-detail { flex-direction: row; justify-content: space-between; align-items: baseline; border: 0; padding: 0; gap: 12px; }
  .h3-setup-metrics .h3-setup-detail dd { text-align: right; font-size: 13px; }
  .h3-setup-card .h3-setup-detail-grid { grid-template-columns: 1fr; gap: 0; }
}
"""

# Workspace rules target only owned classes/IDs and semantic roles. They are
# applied by the workspace composition root.
H3_WORKSPACE_CSS = """
.h3-compare-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.h3-compare-pair figure { margin: 0; min-width: 0; }
.h3-compare-pair video { width: 100%; max-height: 420px; }
.h3-compare-video input[type=range] { width: 100%; }
@media (max-width: 600px) { .h3-compare-pair { grid-template-columns: 1fr; } }
.h3-workspace {
  --layout-gap: 8px;
  --h3-surface: #ffffff;
  --h3-muted-surface: #f2f5f3;
  --h3-text: #202e2b;
  --h3-muted: #596963;
  --h3-border: #dbe3df;
  --h3-accent: #087f6c;
  --h3-focus: #087f6c;
  --h3-disclosure: #edf4f1;
  --h3-disclosure-hover: #e0eee7;
  --h3-disclosure-border: #b7ccc2;
  --body-text-color: var(--h3-text);
  --body-text-color-subdued: var(--h3-muted);
  --input-placeholder-color: #65766e;
  --block-background-fill: var(--h3-surface);
  --block-border-color: var(--h3-border);
  --input-background-fill: var(--h3-surface);
  --input-border-color: var(--h3-border);
  --button-primary-background-fill: #087f6c;
  --button-primary-background-fill-hover: #066555;
  --button-primary-border-color: #087f6c;
  --button-primary-border-color-hover: #066555;
  --button-primary-text-color: #fff;
  --button-primary-text-color-hover: #fff;
  --button-secondary-background-fill: var(--h3-surface);
  --button-secondary-background-fill-hover: var(--h3-muted-surface);
  --button-secondary-text-color: var(--h3-text);
  --button-secondary-border-color: var(--h3-border);
  --checkbox-background-color-selected: var(--h3-accent);
  --checkbox-border-color-selected: var(--h3-accent);
  --slider-color: var(--h3-accent);
  --link-text-color: var(--h3-accent);
  --radius-lg: 12px;
  --radius-md: 8px;
  max-width: none;
  margin: 0 auto;
  padding: 12px 24px 28px;
  background: var(--h3-muted-surface);
  color: var(--h3-text);
}
.dark .h3-workspace {
  --h3-surface: #192522;
  --h3-muted-surface: #111b18;
  --h3-text: #e9f1ed;
  --h3-muted: #a9bdb3;
  --h3-border: #34463e;
  --h3-accent: #66d8bb; --h3-focus: #66d8bb;
  --h3-disclosure: #20362f;
  --h3-disclosure-hover: #2b473e;
  --h3-disclosure-border: #4b6b5e;
  --input-placeholder-color: #94a3b8;
}
body:has(.h3-workspace), .gradio-container:has(.h3-workspace) { background: #f2f5f3; }
/* The workspace fills the window; Gradio's own 32px side padding would only add empty margins. */
.gradio-container:has(.h3-workspace) .main.app { padding: 8px clamp(0px, 1vw, 16px); }
:is(.dark body, body.dark):has(.h3-workspace), .dark .gradio-container:has(.h3-workspace) { background: #111b18; }
.h3-workspace .h3-hero { display: flex; align-items: center; gap: 16px; padding: 0; }
.h3-workspace .h3-brand-mark { display: grid; place-items: center; width: 54px; height: 54px; flex-shrink: 0; border-radius: 16px; background: #087f6c; color: white; font-size: 22px; font-weight: 750; letter-spacing: -.06em; }
.h3-workspace .h3-hero h1 { font-size: 25px; line-height: 1.25; letter-spacing: -.035em; margin: 4px 0; }
.h3-workspace .h3-hero p { color: var(--h3-muted); margin: 0; }
.h3-workspace .h3-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--h3-accent); margin: 0; }
.h3-workspace .h3-workspace-header { align-items: center; gap: 24px; padding: 0 0 8px; }
.h3-workspace .html-container { padding: 0; }
.h3-workspace .h3-header-status p { margin: 0; }
.h3-workspace .h3-header-status { gap: 2px; min-width: 240px; border-left: 1px solid var(--h3-border); padding-left: 20px; font-size: 12px; }
.h3-workspace .h3-header-status :is(.h3-system-ready, .h3-system-warning, .h3-system-status) { padding: 0; margin: 0; }
.h3-workspace #h3-main-tabs > div > [role="tablist"] { gap: 6px; border-bottom: 1px solid var(--h3-border); padding: 0 0 6px; }
.h3-workspace #h3-main-tabs > div > [role="tablist"] > button { min-height: 44px; padding: 10px 22px; border: 1px solid transparent; border-radius: 9px; color: var(--h3-muted); font-weight: 600; background: transparent; }
.h3-workspace #h3-main-tabs > div > [role="tablist"] > button:hover { background: var(--h3-surface); color: var(--h3-text); }
.h3-workspace #h3-main-tabs > div > [role="tablist"] > button[aria-selected="true"] { color: var(--h3-accent); border-color: var(--h3-border); background: var(--h3-surface); box-shadow: 0 2px 4px #122e2110; }
.h3-workspace #h3-main-tabs > [role="tabpanel"] { border: 0; padding: 0; background: transparent; }
.h3-workspace #h3-engine-tabs > [role="tabpanel"] { border: 0; padding: 0; background: transparent; }
.h3-workspace .h3-mobile-nav-container { display: none; }
.h3-workspace .h3-task-picker { padding: 6px 10px; align-items: center; background: var(--h3-surface); border: 1px solid var(--h3-border); border-radius: 14px; }
.h3-workspace .h3-task-picker > .form { border: 0; box-shadow: none; }
.h3-workspace .h3-task-picker label:has(input[type="radio"]) { min-height: 40px; padding-inline: 18px; border-radius: 8px; }
.h3-workspace .h3-task-picker label:has(input:checked) { border-color: var(--h3-accent); color: var(--h3-accent); background: var(--h3-muted-surface); }
.h3-workspace .h3-engine-help { color: var(--h3-muted); font-size: 12px; padding-inline: 4px; }
.h3-workspace .h3-engine-help p { margin: 0; }
.h3-workspace #h3-engine-tabs > div:has(> [role="tablist"]) { display: none; }
.h3-workspace .h3-generator-shell { gap: 20px; align-items: flex-start; }
.h3-workspace .h3-composer, .h3-workspace .h3-preview-panel { min-width: 0; }
.h3-workspace :is(.h3-composer, .h3-preview-panel) { border: 1px solid var(--h3-border); border-radius: 16px; padding: 16px; background: var(--h3-surface); box-shadow: 0 4px 18px #122e2106; }
.h3-workspace .h3-panel-heading { display: flex; align-items: center; gap: 12px; padding: 0 0 4px; }
.h3-workspace .h3-panel-heading h3 { margin: 0; font-size: 18px; letter-spacing: -.025em; }
.h3-workspace .h3-panel-heading p { margin: 3px 0 0; font-size: 12px; color: var(--h3-muted); }
.h3-workspace .h3-step { display: grid; place-items: center; width: 32px; height: 32px; flex-shrink: 0; border: 1px solid var(--h3-border); border-radius: 10px; color: var(--h3-accent); background: var(--h3-muted-surface); font-size: 11px; font-weight: 700; }
.h3-workspace .h3-prompt-editor textarea { min-height: 150px; font-size: 15px; line-height: 1.65; }
.h3-workspace #h3-output-settings { border-top: 1px solid var(--h3-border); padding-top: 16px; margin-top: 4px; }
.h3-workspace .h3-result-player { min-height: 260px; background: var(--h3-muted-surface); border-style: dashed; }
.h3-workspace .h3-result-player .empty { flex-direction: column; gap: 10px; color: var(--h3-muted); }
.h3-workspace .h3-result-player .empty::after { content: "Your video will appear here"; font-size: 13px; }
/* Small buttons in Compose: an action, and links that open a Fine-tune tab. */
.h3-workspace .h3-settings-links { flex-wrap: wrap; align-items: center; gap: 6px; }
.h3-workspace .h3-settings-links > button { flex: 0 1 auto !important; min-width: 0 !important; min-height: 36px; padding: 6px 12px; border-radius: 8px; font-size: 13px; font-weight: 600; }
.h3-workspace .h3-settings-links > button:not(.h3-settings-link) { border: 1px solid var(--h3-border); }
.h3-workspace button.h3-settings-link { border-color: transparent; background: transparent; color: var(--h3-accent); box-shadow: none; }
.h3-workspace button.h3-settings-link:hover { background: var(--h3-muted-surface); }
/* Give every Gradio disclosure the same visible, clickable header. The accent
   rail lives on the container so it runs down the open content too, tying the
   revealed controls to their header instead of the surrounding page. */
.h3-workspace .gr-accordion {
  padding: 0;
  overflow: hidden;
  border: 1px solid var(--h3-disclosure-border);
  border-radius: 10px;
  background: var(--h3-surface);
  box-shadow: inset 3px 0 var(--h3-accent), 0 1px 2px #122e210d;
  transition: box-shadow 120ms ease;
}
.h3-workspace .gr-accordion:has(> button.label-wrap:hover) {
  box-shadow: inset 3px 0 var(--h3-accent), 0 3px 10px #122e211a;
}
.h3-workspace .gr-accordion + .gr-accordion { margin-top: 2px; }
.h3-workspace .gr-accordion > button.label-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 48px;
  padding: 10px 12px 10px 17px;
  border: 0;
  border-radius: 0;
  background: var(--h3-disclosure);
  box-shadow: inset 3px 0 var(--h3-accent);
  color: var(--h3-accent);
  text-align: start;
  cursor: pointer;
  transition: background-color 120ms ease;
}
.h3-workspace .gr-accordion > button.label-wrap > span:first-child {
  flex: 1;
  min-width: 0;
  font-size: 14px;
  font-weight: 650;
  line-height: 1.45;
}
/* Spell out the action; the empty alt text keeps it out of the accessible name. */
.h3-workspace .gr-accordion > button.label-wrap::after {
  content: "Show" / "";
  order: 1;
  color: var(--h3-muted);
  font-size: 12px;
  font-weight: 600;
}
.h3-workspace .gr-accordion > button.label-wrap.open::after { content: "Hide" / ""; }
.h3-workspace .gr-accordion > button.label-wrap > .icon {
  display: grid;
  place-items: center;
  order: 2;
  flex: 0 0 26px;
  width: 26px;
  height: 26px;
  border: 1px solid var(--h3-accent);
  border-radius: 50%;
  background: var(--h3-accent);
  color: var(--h3-surface);
  font-size: 11px;
  transform: rotate(-90deg) !important;
  transition: transform 160ms ease, background-color 120ms ease, color 120ms ease;
}
.h3-workspace .gr-accordion > button.label-wrap.open > .icon {
  background: var(--h3-surface);
  color: var(--h3-accent);
  transform: rotate(0deg) !important;
}
.h3-workspace .gr-accordion > button.label-wrap:hover {
  background: var(--h3-disclosure-hover);
}
.h3-workspace .gr-accordion > button.label-wrap:hover::after { color: var(--h3-accent); }
.h3-workspace .gr-accordion > button.label-wrap.open {
  box-shadow: inset 3px 0 var(--h3-accent), 0 1px var(--h3-disclosure-border);
}
.h3-workspace .gr-accordion > button.label-wrap:focus-visible {
  outline: 2px solid var(--h3-focus);
  outline-offset: -3px;
}
.h3-workspace .gr-accordion > div[data-testid="accordion-content"] {
  padding: 14px 14px 14px 17px;
}
@media (prefers-reduced-motion: reduce) {
  .h3-workspace .gr-accordion,
  .h3-workspace .gr-accordion > button.label-wrap,
  .h3-workspace .gr-accordion > button.label-wrap > .icon { transition: none; }
}
.h3-workspace .h3-engine-view { padding: 20px; border: 1px solid var(--h3-border); border-radius: 16px; background: var(--h3-surface); }
.h3-workspace .h3-essentials { border: 1px solid var(--h3-border); border-radius: 12px; padding: 12px; background: var(--h3-surface); }
.h3-workspace .h3-action-dock { padding: 14px; border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-muted-surface); }
.h3-workspace .h3-generation-buttons { gap: 8px; }
.h3-workspace .h3-generation-buttons > button { min-width: 100px; min-height: 44px; white-space: nowrap; }
.h3-workspace .h3-generation-buttons > button:not(.h3-primary-action) { border: 1px solid var(--h3-border); }
.h3-workspace .h3-primary-action { font-weight: 650; }
.h3-workspace .h3-primary-action:not(:disabled):not(.h3-busy) { box-shadow: 0 4px 14px #087f6c40; }
.h3-workspace button:disabled { cursor: not-allowed; opacity: .6; }
/* A GPU button while its job runs (job_admission.busy_updates). */
button.h3-busy { pointer-events: none; opacity: .6; }
.h3-workspace .h3-shortcut-hint { margin: 6px 0 0; font-size: 12px; color: var(--h3-muted); }
@media (pointer: coarse) { .h3-workspace .h3-shortcut-hint { display: none; } }
.h3-workspace kbd { padding: 1px 6px; border: 1px solid var(--h3-border); border-bottom-width: 2px; border-radius: 5px; background: var(--h3-surface); font: 600 11px/1.5 ui-monospace, monospace; }
.h3-workspace .h3-action-dock .h3-status { margin-top: 8px; }
.h3-workspace :is(.h3-actions-slot, .h3-action-dock) { scroll-margin-top: 16px; }
.h3-workspace .h3-gallery-workspace { gap: 16px; align-items: flex-start; }
.h3-workspace .h3-gallery-filters { align-items: center; }
.h3-workspace .h3-compare-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.h3-workspace .h3-compare-pair figure { margin: 0; }
.h3-workspace .h3-compare-pair video { width: 100%; }
.h3-workspace .h3-job-details pre { max-height: 360px; overflow: auto; }
.h3-workspace .h3-section-intro { padding: 0; }
.h3-workspace .h3-section-intro :is(h3, p) { margin: 0 0 6px; }
.h3-workspace .h3-section-intro p { color: var(--h3-muted); }
.h3-workspace .h3-system-status { display: flex; gap: 8px; align-items: center; }
.h3-workspace :is(.h3-system-ready, .h3-system-warning) { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; padding: 8px 0; }
.h3-workspace .h3-mobile-nav a { display: inline-flex; align-items: center; min-height: 44px; padding: 0 12px; }
.h3-workspace .h3-readiness { display: flex; gap: 6px; padding: 8px 0; }
.h3-workspace .h3-setup-card { color: var(--h3-text); background: var(--h3-surface); border-color: var(--h3-border); font-size: 14px; }
.h3-workspace .h3-setup-card .h3-setup-heading { background: var(--h3-muted-surface); border-color: var(--h3-border); }
.h3-workspace .h3-setup-heading > strong, .h3-workspace .h3-setup-card dd { color: var(--h3-text); }
.h3-workspace .h3-setup-heading > span { color: var(--h3-accent); background: var(--h3-surface); border-color: var(--h3-border); }
.h3-workspace .h3-setup-card dt, .h3-workspace .h3-setup-context, .h3-workspace .h3-setup-card ul { color: var(--h3-muted); font-size: 13px; }
.h3-workspace .h3-setup-card .h3-setup-disclosure, .h3-workspace .h3-setup-metrics .h3-setup-detail, .h3-workspace .h3-setup-detail-grid .h3-setup-detail, .h3-workspace .h3-setup-changes .h3-setup-detail { border-color: var(--h3-border); }
.h3-workspace .h3-setup-card .h3-setup-disclosure summary { min-height: 40px; color: var(--h3-accent); background: var(--h3-disclosure); font-size: 13px; font-weight: 600; box-shadow: inset 3px 0 var(--h3-accent); }
.h3-workspace .h3-setup-card .h3-setup-disclosure summary:hover { background: var(--h3-disclosure-hover); }
.h3-workspace .h3-setup-disclosure summary::after { content: "▾" / ""; display: grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; color: var(--h3-surface); background: var(--h3-accent); font-size: 11px; transform: rotate(-90deg); transition: transform 160ms ease; }
.h3-workspace .h3-setup-disclosure[open] summary::after { content: "▾" / ""; color: var(--h3-accent); background: var(--h3-surface); box-shadow: inset 0 0 0 1px var(--h3-accent); transform: none; }
.h3-workspace .h3-setup-disclosure[open] summary { box-shadow: inset 3px 0 var(--h3-accent), 0 1px var(--h3-disclosure-border); }
.h3-workspace .h3-setup-disclosure[open] > :not(summary) { padding-top: 10px; box-shadow: inset 3px 0 var(--h3-accent); }
@media (prefers-reduced-motion: reduce) { .h3-workspace .h3-setup-disclosure summary::after { transition: none; } }
.h3-workspace :is(button, input, textarea, select, summary, a):focus-visible { outline: 2px solid var(--h3-focus); outline-offset: 3px; scroll-margin: 100px 0; }
.h3-workspace .h3-job-table { overflow-x: auto; }
.h3-workspace .h3-job-table table { width: 100%; border-collapse: collapse; font-size: 14px; }
.h3-workspace .h3-job-table :is(th, td) { text-align: left; padding: 10px; border-bottom: 1px solid var(--h3-border); }
.h3-workspace .h3-view-heading { align-items: center; margin-bottom: 8px; }
.h3-workspace :is(.h3-gallery-shell, .h3-jobs-shell, .h3-projects-shell) { background: transparent; border: 0; padding: 0; gap: 16px; }
.h3-workspace .h3-view-heading h2 { font-size: 24px; letter-spacing: -.025em; margin: 0 0 6px; }
.h3-workspace .h3-view-heading p { color: var(--h3-muted); margin: 0; max-width: 760px; }
.h3-workspace .h3-gallery-toolbar { align-items: center; gap: 12px; padding: 12px; border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-muted-surface); }
.h3-workspace .h3-library-button { width: 150px !important; min-width: 150px !important; flex: 0 0 150px !important; }
.h3-workspace .h3-gallery-status { color: var(--h3-muted); font-size: 13px; min-width: 200px; }
.h3-workspace .h3-gallery-filters { align-items: center; gap: 12px; padding-bottom: 8px; }
.h3-workspace .h3-gallery-section-title h3 { margin: 0 0 4px; font-size: 17px; }
.h3-workspace .h3-gallery-section-title p { margin: 0; font-size: 13px; color: var(--h3-muted); }
.h3-workspace .h3-gallery-grid { border-radius: 12px; }
.h3-workspace .h3-gallery-grid button:focus-visible { outline-offset: -3px; }
.h3-workspace .h3-gallery-card { border: 1px solid var(--h3-border); border-radius: 10px; }
.h3-workspace .h3-gallery-download { overflow-wrap: anywhere; }
.h3-workspace .h3-gallery-danger-actions { gap: 8px; }
.h3-workspace .h3-jobs-workspace { align-items: flex-start; gap: 20px; }
.h3-workspace .h3-job-history { min-width: 0; }
.h3-workspace .h3-job-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 0 0 16px; }
.h3-workspace .h3-job-metrics > div { padding: 14px 12px; border: 1px solid var(--h3-border); border-radius: 10px; background: var(--h3-muted-surface); }
.h3-workspace .h3-job-metrics dt { font-size: 12px; color: var(--h3-muted); }
.h3-workspace .h3-job-metrics dd { font-size: 26px; font-weight: 650; margin: 4px 0 0; line-height: 1.2; }
.h3-workspace .h3-job-table { border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-surface); }
.h3-workspace .h3-job-table table { border: 0 !important; margin: 0; }
.h3-workspace .h3-job-table :is(th, td) { border: 0 !important; border-bottom: 1px solid var(--h3-border) !important; }
.h3-workspace .h3-job-table caption { text-align: left; padding: 12px; font-size: 12px; color: var(--h3-muted); }
.h3-workspace .h3-job-table th { background: var(--h3-muted-surface); font-size: 12px; font-weight: 600; white-space: nowrap; }
.h3-workspace .h3-job-table td { font-size: 13px; vertical-align: top; overflow-wrap: anywhere; }
.h3-workspace .h3-job-table td:nth-child(3) { min-width: 140px; }
.h3-workspace .h3-job-table small { display: block; margin-top: 5px; color: var(--h3-muted); }
.h3-workspace .h3-job-table tbody tr:last-child td { border-bottom: 0 !important; }
.h3-workspace .h3-job-table tbody tr:hover { background: var(--h3-muted-surface); }
.h3-workspace .h3-job-table [data-job-id] { cursor: pointer; scroll-margin: 24px; }
.h3-workspace .h3-job-table [aria-selected="true"] { background: var(--h3-muted-surface); box-shadow: inset 3px 0 var(--h3-accent); }
.h3-workspace .h3-job-table [data-job-id]:focus-visible { outline: 2px solid var(--h3-focus); outline-offset: -2px; }
.h3-workspace .h3-compare-selection-actions { gap: 8px; }
.h3-workspace .h3-compare-slots { gap: 16px; }
.h3-workspace .h3-compare-status { color: var(--h3-muted); font-size: 13px; }
.h3-workspace .h3-job-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 6px; background: var(--h3-muted-surface); font-size: 12px; font-weight: 600; white-space: nowrap; }
.h3-workspace .h3-job-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.h3-workspace .h3-job-success { color: #166534; background: #dcfce7; }
.h3-workspace .h3-job-danger { color: #991b1b; background: #fee2e2; }
.h3-workspace .h3-job-waiting { color: #854d0e; background: #fef9c3; }
.h3-workspace .h3-job-active { color: #1e40af; background: #dbeafe; }
.dark .h3-workspace .h3-job-success { color: #bbf7d0; background: #14532d; }
.dark .h3-workspace .h3-job-danger { color: #fecaca; background: #7f1d1d; }
.dark .h3-workspace .h3-job-waiting { color: #fef08a; background: #713f12; }
.dark .h3-workspace .h3-job-active { color: #bfdbfe; background: #1e3a8a; }
.h3-workspace .h3-job-detail-heading { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.h3-workspace .h3-job-details { overflow-wrap: anywhere; }
.h3-workspace .h3-job-details details { padding: 10px; background: var(--h3-muted-surface); border-radius: 8px; font-size: 13px; }
.h3-workspace .h3-empty-state { padding: 32px 20px; text-align: center; border: 1px dashed var(--h3-border); border-radius: 12px; background: var(--h3-muted-surface); }
.h3-workspace .h3-empty-state strong { font-size: 16px; }
.h3-workspace .h3-empty-state p { color: var(--h3-muted); font-size: 13px; margin: 8px auto 0; max-width: 360px; }
@media (max-width: 767px) {
  .h3-workspace .h3-jobs-workspace { flex-direction: column; }
  .h3-workspace .h3-jobs-workspace > .column { width: 100%; min-width: 0 !important; }
  .h3-workspace .h3-job-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .h3-workspace :is(.h3-gallery-toolbar, .h3-gallery-filters) { flex-direction: column; align-items: stretch; }
  .h3-workspace :is(.h3-gallery-toolbar, .h3-gallery-filters) > * { width: 100%; min-width: 0 !important; }
  .h3-workspace .h3-gallery-status { overflow-wrap: anywhere; }
  .h3-workspace .h3-library-button { width: 100% !important; flex-basis: auto !important; }
}
/* Song: four steps (Idea, Lyrics, Generate, Listen & refine); each pairs a working
   card with an action card that holds the step's next move. */
.h3-workspace .h3-song .styler { gap: 12px; background: transparent; }
.h3-workspace .h3-song.h3-engine-view { border: 0; box-shadow: none; padding: 8px; }
.h3-workspace .h3-song-intro { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 8px 22px; border-bottom: 1px solid var(--h3-border); margin-bottom: 10px; }
.h3-workspace .h3-song-intro h2 { margin: 5px 0 8px; font-size: clamp(24px, 2.4vw, 34px); letter-spacing: -.045em; font-weight: 650; }
.h3-workspace .h3-song-intro p:not(.h3-eyebrow) { margin: 0; color: var(--h3-muted); font-size: 14px; }
.h3-workspace .h3-song-badge { border: 1px solid var(--h3-border); border-radius: 999px; padding: 9px 14px; font-size: 10px; letter-spacing: .14em; color: var(--h3-accent); white-space: nowrap; }
/* Fine-tune (.h3-settings-tabs) shares the stepper's look, without numbers. Plain selector
   lists, not :is(): Gradio's scoping splits on commas inside :is() and inflates specificity. */
.h3-workspace .h3-song-steps, .h3-workspace .h3-settings-tabs { border: 0; background: transparent; padding: 0; }
.h3-workspace .h3-song-steps { counter-reset: h3-song-step; }
.h3-workspace .h3-song-steps > .tab-wrapper, .h3-workspace .h3-settings-tabs > .tab-wrapper { border: 0; margin-bottom: 14px; padding: 0; }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"], .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; border: 0; padding: 0; }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button { counter-increment: h3-song-step; gap: 10px; min-height: 54px; }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button, .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] > button { display: flex; align-items: center; min-width: 0; margin: 0; padding: 8px 14px; border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-surface); color: var(--h3-muted); font-size: 14px; font-weight: 600; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: border-color .15s, background .15s, color .15s; }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button::before { content: counter(h3-song-step); display: grid; place-items: center; flex: none; width: 28px; height: 28px; border: 1px solid var(--h3-border); border-radius: 999px; background: var(--h3-muted-surface); color: var(--h3-accent); font-size: 12px; font-weight: 700; }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button::after, .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] > button::after { display: none; }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button:hover, .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] > button:hover { color: var(--h3-text); border-color: var(--h3-disclosure-border); }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button[aria-selected="true"], .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] > button[aria-selected="true"] { color: var(--h3-text); border-color: var(--h3-accent); background: color-mix(in srgb, var(--h3-accent) 7%, var(--h3-surface)); box-shadow: inset 0 -3px var(--h3-accent); }
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button[aria-selected="true"]::before { border-color: var(--h3-accent); background: var(--h3-accent); color: var(--h3-surface); }
/* Steps before the open one read as visited. */
.h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button:has(~ button[aria-selected="true"])::before { content: "✓"; background: color-mix(in srgb, var(--h3-accent) 14%, var(--h3-surface)); border-color: color-mix(in srgb, var(--h3-accent) 40%, var(--h3-border)); }
.h3-workspace .h3-song-steps > [role="tabpanel"] { border: 0; padding: 0; background: transparent; }
.h3-workspace #h3-fine-tune { gap: 12px; margin-top: 20px; padding: 16px; border: 1px solid var(--h3-border); border-radius: 16px; background: var(--h3-surface); box-shadow: 0 4px 18px #122e2106; scroll-margin-top: 16px; }
.h3-workspace .h3-settings-tabs { scroll-margin-top: 16px; }
.h3-workspace .h3-settings-tabs > .tab-wrapper { margin-bottom: 12px; }
.h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] { grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); }
.h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] > button { min-height: 48px; }
.h3-workspace .h3-settings-tabs > [role="tabpanel"] { border: 1px solid var(--h3-border); border-radius: 12px; padding: 16px; background: var(--h3-surface); }
.h3-workspace .h3-settings-tab > .column { gap: 12px; }
.h3-workspace .h3-song-card-heading h3 { margin: 0; font-size: 18px; letter-spacing: -.025em; }
.h3-workspace .h3-song-card-heading p { margin: 4px 0 0; font-size: 13px; color: var(--h3-muted); }
.h3-workspace .h3-song-stage button.h3-song-nav, .h3-workspace .h3-song-stage .h3-primary-action { min-height: 44px; border-radius: 10px; }
.h3-workspace .h3-song-stage button.h3-song-skip { border-color: transparent; background: transparent; color: var(--h3-accent); }
.h3-workspace .h3-song-stage button.h3-song-skip:hover { background: var(--h3-muted-surface); }
.h3-workspace .h3-song-step-nav { gap: 8px; margin-top: 4px; padding-top: 12px; border-top: 1px solid var(--h3-border); }
.h3-workspace .h3-song-step-nav > button { flex: 1 1 0 !important; min-width: 0 !important; }
.h3-workspace .h3-song-hint p { margin: 0; font-size: 12px; color: var(--h3-muted); }
.h3-workspace .h3-song-writer-status textarea { font-size: 13px; color: var(--h3-muted); }
.h3-workspace .h3-song-recap dl { display: grid; gap: 8px; margin: 0 0 12px; }
.h3-workspace .h3-song-recap dl > div { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: 12px; padding: 10px 12px; border: 1px solid var(--h3-border); border-radius: 10px; }
.h3-workspace .h3-song-recap dt { padding-top: 2px; font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--h3-muted); }
.h3-workspace .h3-song-recap dd { margin: 0; font-size: 14px; color: var(--h3-text); overflow-wrap: anywhere; }
.h3-workspace .h3-song-recap em { color: var(--h3-muted); }
.h3-workspace .h3-song-lyrics textarea { font-size: 15px; line-height: 1.7; }
.h3-workspace .h3-song-editor-tabs { min-width: 0; }
.h3-workspace .h3-song-duration-presets { gap: 6px; }
.h3-workspace .h3-song-duration-presets > button { flex: 1 1 50px !important; min-width: 0 !important; border-radius: 8px; }
.h3-workspace .h3-song button.secondary { border: 1px solid var(--h3-border); }
.h3-workspace .h3-song-stage { gap: 12px; border-top: 3px solid var(--h3-accent); }
.h3-workspace .h3-song-stage .h3-action-dock { border: 0; padding: 10px 0 0; background: transparent; }
.h3-workspace .h3-song-stage .h3-generation-buttons > button { border-radius: 10px; }
.h3-workspace .h3-song,.h3-workspace .h3-song > .styler { overflow: clip !important; }
.h3-workspace .h3-song-choices { gap: 8px; flex-wrap: wrap; }
.h3-workspace .h3-song-choices > .block { flex: 1 1 150px !important; }
.h3-workspace .h3-song-quick-fixes { flex-wrap: wrap; gap: 6px; margin-top: -4px; }
.h3-workspace .h3-song-take-details { margin: 0; font-size: 13px; color: var(--h3-muted); }
.h3-workspace .h3-song-take-details strong { color: var(--h3-text); }
.h3-workspace .h3-song-take-actions { flex-wrap: wrap; gap: 6px; }
.h3-workspace .h3-song-take-actions > button { flex: 1 1 140px !important; border-radius: 8px; }
.h3-workspace .h3-song-quick-fixes > button { flex: 0 1 auto !important; border-radius: 999px; padding: 4px 12px; font-size: 12px; }
.h3-workspace .h3-song-check { border: 1px solid var(--h3-border); border-radius: 10px; padding: 10px 12px; background: var(--h3-muted-surface); }
.h3-workspace .h3-song-check-caption { margin: 0 0 8px; font-size: 13px; font-weight: 600; color: var(--h3-text); }
.h3-workspace .h3-song-check-caption span { font-weight: 400; color: var(--h3-muted); }
.h3-workspace .h3-song-timeline { display: flex; gap: 2px; min-height: 26px; border-radius: 6px; overflow: hidden; }
.h3-workspace .h3-song-chip { flex: 1 1 0; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 5px 4px; font-size: 11px; line-height: 16px; text-align: center; color: var(--h3-text); background: color-mix(in srgb, var(--h3-accent) 18%, var(--h3-surface)); }
.h3-workspace .h3-song-chip-edge { background: color-mix(in srgb, var(--h3-accent) 8%, var(--h3-surface)); }
.h3-workspace .h3-song-chip-lift { background: color-mix(in srgb, var(--h3-accent) 32%, var(--h3-surface)); }
.h3-workspace .h3-song-chip-hook { background: color-mix(in srgb, var(--h3-accent) 55%, var(--h3-surface)); font-weight: 600; }
.h3-workspace .h3-song-chip-contrast { background: color-mix(in srgb, var(--h3-accent) 26%, var(--h3-muted-surface)); font-style: italic; }
.h3-workspace .h3-song-chip-inst { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--h3-accent) 14%, var(--h3-surface)) 0 6px, var(--h3-surface) 6px 10px); }
.h3-workspace .h3-song-chip-spare { background: transparent; border: 1px dashed var(--h3-border); }
.h3-workspace .h3-song-check ul { margin: 8px 0 0; padding-left: 18px; font-size: 13px; color: var(--h3-text); }
.h3-workspace .h3-song-check li { margin: 3px 0; }
.h3-workspace .h3-song-check li.h3-song-check-warn::marker { content: "! "; color: #b4541a; font-weight: 700; }
.h3-workspace .h3-song-check li.h3-song-check-tip::marker { content: "› "; color: var(--h3-accent); }
.h3-workspace .h3-song-check li.h3-song-check-ok::marker { content: "✓ "; color: var(--h3-accent); }
@media (min-width: 768px) {
  .h3-workspace .h3-song-stage { position: sticky; top: 12px; }
}
@media (max-width: 767px) {
  .h3-workspace .h3-song .h3-generation-buttons { flex-wrap: wrap; }
  .h3-workspace .h3-song .h3-generation-buttons > button { flex: 1 1 100%; min-width: 0; white-space: normal; }
  .h3-workspace .h3-song-badge { display: none; }
  .h3-workspace .h3-song-intro { padding: 8px 0 16px; }
  .h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] { gap: 4px; }
  .h3-workspace #h3-fine-tune { padding: 12px; }
  .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
  .h3-workspace .h3-settings-tabs > .tab-wrapper > [role="tablist"] > button { min-height: 44px; padding: 6px 10px; font-size: 13px; white-space: normal; }
  .h3-workspace .h3-settings-tabs > [role="tabpanel"] { padding: 10px; }
  .h3-workspace .h3-song-steps > .tab-wrapper > [role="tablist"] > button { flex-direction: column; justify-content: center; gap: 4px; min-height: 64px; padding: 6px 2px; font-size: 11px; line-height: 1.2; text-align: center; white-space: normal; }
}
/* Dub & Translate: Song's stepper with three steps (Source, Review lines, Dub). */
.h3-workspace .h3-dub .h3-song-steps > .tab-wrapper > [role="tablist"] { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.h3-workspace .h3-dub-summary p { margin: 0; padding: 10px 12px; border: 1px solid var(--h3-border); border-radius: 10px; background: var(--h3-muted-surface); font-size: 13px; color: var(--h3-muted); }
.h3-workspace .h3-dub-summary strong { color: var(--h3-text); }
.h3-workspace .h3-dub-status p { margin: 0; font-size: 13px; color: var(--h3-muted); overflow-wrap: anywhere; }
.h3-workspace .h3-dub-status p:empty { display: none; }
.h3-workspace .h3-dub-lines { border-radius: 12px; }
.h3-workspace .h3-dub-lines tbody td:first-child { color: var(--h3-muted); font-variant-numeric: tabular-nums; }
.h3-workspace .h3-dub-lines tbody td:is(:nth-child(2), :nth-child(3)) { font-variant-numeric: tabular-nums; }
.h3-workspace .h3-dub-line-actions { flex-wrap: wrap; gap: 6px; }
.h3-workspace .h3-dub-line-actions > button { flex: 1 1 120px !important; min-width: 0 !important; border-radius: 8px; }
.h3-workspace .h3-dub-library > button { align-self: flex-end; min-height: 40px; }
/* Soundstage: Song's four-step strip (Script, Cast & sounds, Speak, Review lines) with sticky action
   cards, and visible outlines on the small secondary buttons. */
.h3-workspace .h3-drama3 .styler { gap: 8px; background: transparent; }
.h3-workspace .h3-drama3-script textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 14px; line-height: 1.65; }
/* Syntax colors after VS Code's Light+ and Dark+ themes; the mirror sits under the textarea. */
.h3-workspace .h3-drama3-script {
  --h3-hl-text: #1f2328; --h3-hl-punct: #800000; --h3-hl-cue: #008000; --h3-hl-pause: #098658;
  --h3-hl-keyword: #af00db; --h3-hl-tag: #800000; --h3-hl-effect: #267f99; --h3-hl-attr: #e50000;
  --h3-hl-value: #0000ff; --h3-hl-sound: #795e26; --h3-hl-sequence: #b35c00; --h3-hl-narrator: #5b6b80;
  /* Character names: none is a tag color, and each sits on a faint tint of itself. */
  --h3-hl-name0: #c92a2a; --h3-hl-name1: #1864ab; --h3-hl-name2: #2b8a3e; --h3-hl-name3: #6741d9;
  --h3-hl-name4: #d9480f; --h3-hl-name5: #9c36b5; --h3-hl-name6: #c2255c; --h3-hl-name7: #a07800;
}
.dark .h3-workspace .h3-drama3-script {
  --h3-hl-text: #d4d4d4; --h3-hl-punct: #808080; --h3-hl-cue: #6a9955; --h3-hl-pause: #b5cea8;
  --h3-hl-keyword: #c586c0; --h3-hl-tag: #569cd6; --h3-hl-effect: #4ec9b0; --h3-hl-attr: #9cdcfe;
  --h3-hl-value: #ce9178; --h3-hl-sound: #dcdcaa; --h3-hl-sequence: #d7ba7d; --h3-hl-narrator: #94a3b8;
  --h3-hl-name0: #ff8787; --h3-hl-name1: #74c0fc; --h3-hl-name2: #a9e34b; --h3-hl-name3: #b197fc;
  --h3-hl-name4: #ffa94d; --h3-hl-name5: #da77f2; --h3-hl-name6: #f783ac; --h3-hl-name7: #ffd43b;
}
.h3-workspace .h3-drama3-script :has(> textarea.h3-drama3-highlighted) { position: relative; }
.h3-workspace .h3-drama3-script textarea.h3-drama3-highlighted {
  position: relative; z-index: 1; background: transparent !important; color: transparent;
  caret-color: var(--h3-hl-text); white-space: pre-wrap; overflow-wrap: break-word; word-break: normal;
}
.h3-workspace .h3-drama3-script textarea.h3-drama3-highlighted::placeholder { color: var(--input-placeholder-color); }
.h3-workspace .h3-drama3-script textarea.h3-drama3-highlighted::selection { color: transparent; background: rgba(38, 132, 255, .3); }
.h3-workspace .h3-drama3-highlight {
  position: absolute; z-index: 0; margin: 0; box-sizing: border-box; overflow: hidden; pointer-events: none;
  border-style: solid; border-color: transparent; background: var(--input-background-fill);
  color: var(--h3-hl-text); white-space: pre-wrap; overflow-wrap: break-word; word-break: normal;
}
.h3-drama3-highlight .h3-hl-punct { color: var(--h3-hl-punct); }
.h3-drama3-highlight .h3-hl-cue { color: var(--h3-hl-cue); }
.h3-drama3-highlight .h3-hl-pause { color: var(--h3-hl-pause); }
.h3-drama3-highlight .h3-hl-keyword { color: var(--h3-hl-keyword); }
.h3-drama3-highlight .h3-hl-sequence { color: var(--h3-hl-sequence); }
.h3-drama3-highlight .h3-hl-narrator { color: var(--h3-hl-narrator); }
/* A tint and shadow, never padding, so the mirror's text stays under the textarea's. */
.h3-drama3-highlight .h3-hl-name {
  border-radius: 3px; background: color-mix(in srgb, currentColor 16%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, currentColor 16%, transparent);
}
.h3-drama3-highlight .h3-hl-tag { color: var(--h3-hl-tag); }
.h3-drama3-highlight .h3-hl-effect { color: var(--h3-hl-effect); }
.h3-drama3-highlight .h3-hl-attr { color: var(--h3-hl-attr); }
.h3-drama3-highlight .h3-hl-value { color: var(--h3-hl-value); }
.h3-drama3-highlight .h3-hl-sound { color: var(--h3-hl-sound); }
.h3-drama3-highlight .h3-hl-name0 { color: var(--h3-hl-name0); }
.h3-drama3-highlight .h3-hl-name1 { color: var(--h3-hl-name1); }
.h3-drama3-highlight .h3-hl-name2 { color: var(--h3-hl-name2); }
.h3-drama3-highlight .h3-hl-name3 { color: var(--h3-hl-name3); }
.h3-drama3-highlight .h3-hl-name4 { color: var(--h3-hl-name4); }
.h3-drama3-highlight .h3-hl-name5 { color: var(--h3-hl-name5); }
.h3-drama3-highlight .h3-hl-name6 { color: var(--h3-hl-name6); }
.h3-drama3-highlight .h3-hl-name7 { color: var(--h3-hl-name7); }
.h3-workspace .h3-drama3 button.secondary { border: 1px solid var(--h3-border); background: var(--h3-surface); }
.h3-workspace .h3-drama3 button.secondary:not(:disabled):hover { border-color: var(--h3-accent); color: var(--h3-accent); background: var(--h3-muted-surface); }
.h3-workspace .h3-drama3-inline-actions { flex-wrap: wrap; gap: 8px; align-items: center; }
.h3-workspace .h3-drama3-inline-actions > :is(button, .row) { flex: 0 1 auto !important; width: auto; min-width: 0 !important; }
.h3-workspace .h3-drama3-inline-actions > .row { gap: 8px; }
.h3-workspace .h3-drama3-slot-count { padding-top: 4px; }
.h3-workspace .h3-drama3-note, .h3-workspace .h3-drama3-note p { color: var(--h3-muted); font-size: 12px; margin: 0; }
.h3-workspace .h3-drama3-note:empty { display: none; }
.h3-workspace .h3-drama3-toolbar { padding: 10px; gap: 8px; border: 1px solid var(--h3-border); border-radius: 10px; background: var(--h3-muted-surface); }
.h3-workspace .h3-drama3-toolbar + .h3-drama3-toolbar { margin-top: 4px; }
.h3-workspace .h3-drama3-clip { padding: 8px; border: 1px dashed var(--h3-border); border-radius: 10px; background: var(--h3-muted-surface); }
.h3-workspace .h3-drama3 .gr-accordion.h3-drama3-slot { box-shadow: inset 3px 0 var(--h3-border); border-color: var(--h3-border); }
.h3-workspace .h3-drama3 .gr-accordion.h3-drama3-slot > button.label-wrap { min-height: 40px; padding-block: 6px; background: var(--h3-surface); box-shadow: inset 3px 0 var(--h3-border); color: var(--h3-text); }
.h3-workspace .h3-drama3 .gr-accordion.h3-drama3-slot > button.label-wrap.open { box-shadow: inset 3px 0 var(--h3-accent), 0 1px var(--h3-border); }
.h3-workspace .h3-drama3 .gr-accordion.h3-drama3-slot > button.label-wrap > span:first-child { font-size: 13px; }
.h3-workspace .h3-drama3-stage { gap: 12px; }
.h3-workspace .h3-drama3-auto { gap: 8px; padding: 14px; border: 1px dashed var(--h3-disclosure-border); border-radius: 12px; background: var(--h3-disclosure); }
.h3-workspace .h3-drama3-auto .h3-generation-buttons { flex-wrap: wrap; }
.h3-workspace .h3-drama3-auto .h3-generation-buttons > button { flex: 1 1 200px; min-height: 40px; white-space: normal; }
.h3-workspace .h3-drama3-auto-heading strong { color: var(--h3-accent); font-size: 14px; }
.h3-workspace .h3-drama3-auto-heading p { margin: 2px 0 0; color: var(--h3-muted); font-size: 12px; }
.h3-workspace .h3-drama3-subtitles ol { position: relative; max-height: 220px; overflow-y: auto; margin: 0; padding: 6px; list-style: none; border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-muted-surface); }
.h3-workspace .h3-drama3-subtitles li { display: flex; gap: 10px; margin: 0; padding: 6px 10px; border-radius: 8px; cursor: pointer; color: var(--h3-muted); font-size: 14px; line-height: 1.5; transition: background 160ms ease, color 160ms ease; }
.h3-workspace .h3-drama3-subtitles li:hover { background: var(--h3-surface); }
.h3-workspace .h3-drama3-subtitles li.past { opacity: .7; }
.h3-workspace .h3-drama3-subtitles li.active { color: var(--h3-text); background: var(--h3-surface); box-shadow: inset 3px 0 var(--h3-accent); }
.h3-workspace .h3-subtitle-speaker { flex: 0 0 auto; max-width: 30%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--h3-accent); font-size: 12px; font-weight: 650; padding-top: 2px; }
.h3-workspace .h3-subtitle-text { min-width: 0; }
.h3-workspace .h3-drama3-lines { gap: 10px; }
.h3-workspace .h3-drama3-lines .h3-generation-buttons > button { flex: 1 1 140px; min-height: 40px; }
/* Clip, unlike hidden, does not create the scroll container that would pin the stage in place.
   Top level, because Gradio scopes rules inside @media under .gradio-container, which then never match it. */
.gradio-container:has(.h3-drama3-stage), .h3-workspace .h3-drama3, .h3-workspace .h3-drama3 > .styler { overflow: clip !important; }
@media (min-width: 768px) {
  .h3-workspace .h3-drama3-stage { position: sticky; top: 12px; }
}
@media (max-width: 767px) {
  .h3-workspace .h3-drama3.h3-engine-view { padding: 6px; }
  .h3-workspace .h3-drama3 :is(.h3-composer, .h3-preview-panel) { padding: 8px; }
  .h3-workspace .h3-drama3 .gr-accordion > div[data-testid="accordion-content"] { padding: 8px 6px 8px 9px; }
  .h3-workspace .h3-drama3 :is(.h3-drama3-toolbar, .h3-drama3-clip) { padding: 6px; }
}
.h3-workspace .h3-mobile-nav { display: none; }
@media (max-width: 767px) {
  .h3-workspace { padding: 12px 6px 24px; }
  .h3-workspace .h3-hero h1 { font-size: 22px; }
  .h3-workspace .h3-hero p { font-size: 12px; }
  .h3-workspace .h3-header-status { border-left: 0; border-top: 1px solid var(--h3-border); padding: 10px 0 0; margin-top: 12px; min-width: 0; }
  .h3-workspace #h3-main-tabs > div > [role="tablist"] { gap: 2px; }
  .h3-workspace #h3-main-tabs > div > [role="tablist"] > button { padding: 8px 12px; }
  .h3-workspace :is(.h3-composer, .h3-engine-view) { padding: 12px; }
  .h3-workspace .h3-task-picker > * { min-width: 0 !important; }
  .h3-workspace .h3-generator-shell { flex-direction: column; }
  .h3-workspace .h3-composer, .h3-workspace .h3-preview-panel { width: 100%; min-width: 0 !important; }
}
@media (max-width: 767px) {
  .h3-workspace .h3-workspace-header { flex-direction: column; gap: 0; align-items: stretch; }
  .h3-workspace .h3-mobile-nav-container { display: block; }
  .h3-workspace .h3-mobile-nav { display: flex; gap: 8px; }
  .h3-workspace .h3-mode-row { flex-direction: column; }
  .h3-workspace .h3-task-picker { padding: 8px; }
  .h3-workspace .h3-preview-panel { padding: 8px; }
  .h3-workspace .h3-drama3 .h3-generation-buttons > button {
    flex: 1 1 100%; min-width: 0; white-space: normal;
  }
  .h3-workspace .h3-gallery-workspace { flex-direction: column; }
  .h3-workspace .h3-gallery-workspace > .column { width: 100%; min-width: 0 !important; }
  .h3-workspace .h3-setup-metrics { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .h3-workspace *, .h3-workspace *::before, .h3-workspace *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
"""
