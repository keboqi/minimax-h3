"""Design tokens and responsive styles for the MiniMax H3 Gradio UI."""

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
  --layout-gap: 10px;
  --body-text-color-subdued: #475569;
  --input-placeholder-color: #64748b;
  --h3-surface: var(--block-background-fill, #ffffff);
  --h3-muted-surface: var(--background-fill-secondary, #f4f6f8);
  --h3-text: var(--body-text-color, #202b38);
  --h3-muted: var(--body-text-color-subdued, #546171);
  --h3-border: var(--block-border-color, #d9e0e7);
  --h3-accent: #2563eb;
  --h3-focus: #2563eb;
  --button-primary-background-fill: #2563eb;
  --button-primary-background-fill-hover: #1d4ed8;
  --button-primary-text-color: #fff;
  color: var(--h3-text);
}
.dark .h3-workspace {
  --h3-accent: #93c5fd; --h3-focus: #93c5fd;
  --body-text-color-subdued: #cbd5e1;
  --input-placeholder-color: #94a3b8;
}
.h3-workspace .h3-hero { padding: 0; }
.h3-workspace .h3-hero h1 { font-size: 26px; letter-spacing: -.02em; margin: 0 0 8px; }
.h3-workspace .h3-hero p { color: var(--h3-muted); margin: 0; }
.h3-workspace .h3-workspace-header { align-items: center; gap: 16px; }
.h3-workspace .h3-header-status { gap: 0; min-width: 280px; }
.h3-workspace .h3-header-status :is(.h3-system-ready, .h3-system-warning, .h3-system-status) { padding: 0; margin: 0; }
.h3-workspace .h3-mobile-nav-container { display: none; }
.h3-workspace .h3-task-picker { padding: 8px; background: var(--h3-muted-surface); border-radius: 12px; }
.h3-workspace #h3-engine-tabs > div:has(> [role="tablist"]) { display: none; }
.h3-workspace .h3-generator-shell { gap: 16px; align-items: flex-start; }
.h3-workspace .h3-composer, .h3-workspace .h3-preview-panel { min-width: 0; }
.h3-workspace .h3-preview-panel { border: 1px solid var(--h3-border); border-radius: 12px; padding: 12px; background: var(--h3-surface); }
.h3-workspace .h3-essentials { border: 1px solid var(--h3-border); border-radius: 12px; padding: 12px; background: var(--h3-surface); }
.h3-workspace .h3-action-dock { padding: 12px; border: 1px solid var(--h3-border); border-radius: 12px; background: var(--h3-surface); }
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
.h3-workspace .h3-setup-card .h3-setup-disclosure summary { color: var(--h3-text); font-size: 13px; }
.h3-workspace .h3-setup-card .h3-setup-disclosure summary:hover { color: var(--h3-accent); background: var(--h3-muted-surface); }
.h3-workspace :is(button, input, textarea, select, summary, a):focus-visible { outline: 2px solid var(--h3-focus); outline-offset: 3px; scroll-margin: 100px 0; }
.h3-workspace .h3-job-table { overflow-x: auto; }
.h3-workspace .h3-job-table table { width: 100%; border-collapse: collapse; font-size: 14px; }
.h3-workspace .h3-job-table :is(th, td) { text-align: left; padding: 10px; border-bottom: 1px solid var(--h3-border); }
.h3-workspace .h3-view-heading { align-items: center; margin-bottom: 8px; }
.h3-workspace :is(.h3-gallery-shell, .h3-jobs-shell) { background: transparent; border: 0; padding: 0; gap: 16px; }
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
.h3-workspace .h3-mobile-nav { display: none; }
@media (max-width: 767px) {
  .h3-workspace .h3-generator-shell { flex-direction: column; }
  .h3-workspace .h3-composer, .h3-workspace .h3-preview-panel { width: 100%; }
}
@media (max-width: 767px) {
  .h3-workspace .h3-workspace-header { flex-direction: column; gap: 0; align-items: stretch; }
  .h3-workspace .h3-mobile-nav-container { display: block; }
  .h3-workspace .h3-mobile-nav { display: flex; gap: 8px; }
  .h3-workspace .h3-mode-row { flex-direction: column; }
  .h3-workspace .h3-task-picker { padding: 8px; }
  .h3-workspace .h3-preview-panel { padding: 8px; }
  .h3-workspace .h3-gallery-workspace { flex-direction: column; }
  .h3-workspace .h3-gallery-workspace > .column { width: 100%; min-width: 0 !important; }
  .h3-workspace .h3-setup-metrics { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) {
  .h3-workspace *, .h3-workspace *::before, .h3-workspace *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; }
}
"""
