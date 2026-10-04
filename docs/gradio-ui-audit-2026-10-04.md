# Gradio and current UI/UX audit

Baseline reviewed 4 October 2026, checkout `a7471a5`, branch `codex/ui-ux-redesign`. The implementation follow-up below describes the subsequent working-tree changes.

The current workspace already implements the large redesign: task/engine navigation, compact H3 settings, prompt review, progressive references, durable jobs, ownership/recovery, saved projects, asset annotations and comparison. The next pass should improve everyday interaction and reliability. Continue with Gradio; there is no demonstrated need to replace the frontend.

The original audit was read-only. The requested implementation now updates the application and shared Gradio pins. An ignored, isolated test environment and baseline/follow-up artifacts are under `.cache/gradio-audit-2026-10-04/`.

## Latest Gradio and relevance

The baseline pinned **6.27.0**; the implementation updates `h3_requirements.py` and `requirements-test.txt` to **6.29.1**. Local setup and Modal consume the shared runtime constant. The latest stable release is **6.29.1**, published **2 October 2026**. Its patch fixes Gallery empty-state behavior. [PyPI release](https://pypi.org/project/gradio/6.29.1/), [official changelog](https://gradio.app/changelog).

Changes since the pin that apply here:

| Capability | Application to this UI |
| --- | --- |
| 6.28: configurable tab overflow and alignment | Evaluate `gr.Tabs(overflow_behavior="wrap")` for the five main sections on mobile. `gr.Tab(alignment="right")` can group System/API on desktop. The default menu remains an option if wrapping consumes too much space. |
| 6.29: `height`/`max_height` for Accordion and Tab | Bound expanded inventory, job ledger and technical panels; avoid making the whole composer a nested scrolling region. |
| 6.28: keyword event inputs | Use `inputs_kwargs` for new internal handlers to reduce positional wiring mistakes. Preserve the existing public API names and positional contracts. |
| 6.29: clearer rejected-upload errors | Improve feedback for incompatible media uploads. The application still needs its own engine-specific validation. |
| Audio fixes in 6.28/6.29 | Upstream streaming and recording fixes may help future microphone/streaming features. Current uploaded voice references do not establish a streaming performance benefit. |

These items come from the [official changelog](https://gradio.app/changelog); supported layout parameters were also inspected in the installed 6.29.1 package. [Tab documentation](https://gradio.app/docs/gradio/tab), [event parameters](https://gradio.app/docs/gradio/timer).

Several useful capabilities are **already available at the current pin**: scoped HTML templates/CSS/JavaScript, BrowserState, reactive rendering, shared queue groups and lazy tab children. Their availability is not a reason by itself to upgrade. In particular, HTML components can support a compact job card or richer media actions without a separate frontend toolchain. [Custom HTML guide](https://gradio.app/guides/custom-HTML-components), [HTML reference](https://gradio.app/docs/gradio/html).

## Prioritized improvements

### 1. Refresh selected job details and actions together — first correctness fix

**Observed in the browser on both versions:** after selecting a running mock job, the Jobs table changes to `completed`, but **Cancel selected job stays enabled** after another six seconds.

`jobs_view.py:120` refreshes the table, dropdown choices and header count. `jobs_view.py:281` computes details, output files, failed variants and button availability separately, through `selected.change` at line 328. Keeping the same selected ID does not refresh those controls as the job changes.

Have one owner-scoped refresh compute both the list and the selected job presentation. Refresh cancellation/retry/finishing availability and retained files whenever the job revision changes. Keep targeted cancellation and existing recovery semantics.

**Acceptance:** select a running job and let it complete or fail without reselecting it. Cancel disables, output files appear, and applicable retry actions update. Verify the same behavior after reconnecting.

### 2. Put Generate beside the composition and make it reachable on mobile — largest UX gain

`h3_view.py:642–693` places the action group after the result preview and next-run summary. `styles.py` only makes that group sticky below 768 pixels, within its existing parent; it does not bring it into the initial viewport.

Measured with the default mock workspace, a short prompt and a 900-pixel viewport height:

| Viewport width | Generate starts at page Y | Page height |
| --- | ---: | ---: |
| 1440 | 1062 px | 1536 px |
| 1024 | 2075 px | 2376 px |
| 768 | 2115 px | 2415 px |
| 390 | 3005 px | 3383 px |

Move the single Generate/progress action group into the composer, adjacent to the prompt/output essentials. On small screens, provide a viewport dock with safe-area spacing and enough content padding. Keep one action component and one binding. Make idle Interrupt unavailable and label running cancellation clearly. Consider a keyboard shortcut with a visible hint after the click path works well.

Reduce initial vertical space: collapse Base model explanations into advanced controls, shorten the task/engine help, and use a useful compact result empty state. The 1199-pixel stacking breakpoint also warrants reevaluation at tablet widths.

**Acceptance:** Generate can be reached without scrolling through the result player, with a long prompt and open reference controls, at 390/768/1024 pixels and 200% zoom. The dock must not cover controls or keyboard focus.

### 3. Reduce settings and preference traffic — measurable optimization

The H3 settings controller already uses a small two-output prompt-readiness path and skips some unchanged properties. Preserve those improvements.

Remaining breadth:

- A normal H3 settings event reads **84 inputs** and declares **75 outputs** (`settings_controller.py:90–117, 338–409`).
- Preference persistence reads controller memory plus **149 controls** across all engines: **150 inputs** for a one-output save (`persistence.py:184–268`). Each controller event has a save continuation; other engine inputs also submit the entire preference snapshot.
- The composed mock app has **838 components**, **362 event dependencies**, and approximately **1.13 MB of serialized configuration**. This is an uncompressed serialized size, not a measured network transfer size.
- Five duration edits took **307–356 ms** on 6.27 and **312–360 ms** on 6.29.1, including Playwright round trips. The new version did not establish a faster settings path.

Use a per-session transition state and revision, submit the changed control plus the smallest required context, and return only affected properties. Keep the Python settings resolver authoritative. Split preference saves by namespace and coalesce rapid changes; keep prompts, uploads and credentials excluded. Introduce this incrementally and verify mode memory, presets and first-frame resolution before changing broader wiring.

Do not raise the global queue concurrency to fix UI responsiveness. Retain the shared single-GPU concurrency group. Pure UI handlers can avoid GPU dispatch, but settings transitions still need ordering per session. [Gradio queuing guide](https://gradio.app/guides/queuing).

**Acceptance:** measure p50/p95 latency, request count and payload size for prompt, duration, recipe, decoder, image upload and rapid edits. Confirm another browser session cannot block settings editing unnecessarily. Use the existing 250 ms local responsiveness target where appropriate.

### 4. Unify Media search, selection and preview — reduce repeated work

`media_view.py:99–192` provides a thumbnail gallery and selected preview. `library_tools.py:68–193` adds a separate collapsed search panel with an Indexed asset dropdown; search updates asset/comparison dropdowns rather than the main gallery. Users manage two selectors and two media-type filters.

Put search/type/favorites above the thumbnail grid. Use one selected asset identity for preview, downloads, annotations, lineage, enhancement and comparison. Add Compare A/B and Use as input actions at selection. Reuse existing managed-path validation, annotations and lineage storage.

Index new imports and outputs incrementally. Move **Index next 200 files** and **Rebuild media index** into System maintenance; routine browsing should not depend on those implementation controls. Retain paged thumbnails and lazy full-media preview.

**Acceptance:** searching filters the same grid, selection updates all associated actions, and importing/generating an asset makes it searchable without a manual index operation.

### 5. Bring the other engines into the same interaction pattern

Qwen, LTX, Music and YuE2 remain independent 3:2 forms with actions beneath their output player and long status boxes (`qwen_view.py:107–169`, `ltx_view.py:77–176`, `music_view.py:59–106`, `yue2_view.py:62–101`). Qwen's initial Text to image form shows editing controls, two resolution selectors, reference settings and a negative prompt even when their generation role is inactive.

Share a small composer/action/result/status layout across engines while retaining engine-specific inputs. Put routine output settings near the prompt, collapse technical settings, and show or clearly explain inactive controls based on mode/CFG. Preserve uploaded reference values and their optional use by the prompt writer when adjusting visibility. Add the same readiness feedback used by H3.

Also clarify exact-size presentation: `canvas_controls.py:112–142` switches unknown dimensions to `custom` but retains the previous aspect-ratio value. Display the actual ratio or a clear Custom state so the selector cannot imply a canvas it does not control.

**Acceptance:** users can compose, see the output intent, submit and interpret status in the same places across all five engines. Mode changes preserve drafts and relevant uploads.

### 6. Repair browser reliability and then simplify presentation

The auto-resolution acceptance test exposes **two DOM elements with ID `h3-output-settings` after recipe/mode transitions**, on both Gradio versions. Investigate conditional layout updates and element lifecycle; do not simply add `.first` to hide the duplicate.

The first 6.29.1 settings run also failed when a step edit stayed at eight instead of ten; a repeat passed. Treat this as a timing-sensitive transition or test synchronization issue until repeated runs isolate its cause. A resolved summary alone may not imply that all chained settings/persistence updates have settled.

After those fixes, use supported layout parameters and component-scoped CSS for consistent spacing, focus, light/dark colors and bounded technical disclosures. Replace the hidden engine-tab-bar selector only when a supported alternative preserves navigation, draft state and component identity. Preserve native lazy tab rendering: the audit saw **zero Gradio POSTs during 6.5 idle seconds in Create before visiting Jobs**; there is no evidence for an always-running poll on initial load. In Jobs, avoid rebuilding unchanged tables/dropdown choices every two seconds, but continue updating elapsed time and live actions as needed. [Timer reference](https://gradio.app/docs/gradio/timer), [Tab rendering reference](https://gradio.app/docs/gradio/tab).

## Baseline upgrade validation and recommendation

6.29.1 is a reasonable upgrade candidate after the browser reliability work. Update the runtime constant and test requirements together, then validate local and Modal packaging with existing dependency constraints. The tested constrained environment used NumPy **1.26.4**, Hugging Face Hub **1.33.0** (below 2), and Gradio Client **2.7.2**. A fresh unconstrained install initially selected Hub 2.x; that would violate the project's Transformers compatibility policy. Do not use that dependency resolution as deployment evidence.

| Check performed in this audit | Result |
| --- | --- |
| Existing CPU suite, 6.29.1, fresh and subsequently constrained environments | 284 tests run; 3 optional checks skipped in each run |
| Custom mock browser audit, 6.27 and 6.29.1 | Both completed with no page JavaScript errors; stale Jobs action reproduced on both |
| Existing workspace browser suite, 6.29.1 | Passed, including engine actions, ownership/projects, comparison and responsive checks |
| Existing voice-reference browser suite, 6.29.1 | Passed |
| Existing settings browser suite | Pinned version passed; initial latest run failed a step edit, latest repeat passed |
| Existing auto-resolution browser suite | Failed on both versions at the duplicate-ID assertion |

Indicative mock startup readiness was 1.90 seconds on the pinned version and 1.80 seconds on the candidate. These are single local samples from separately resolved environments, not a controlled performance comparison. Real GPU inference, deployment performance, large-library throughput and a complete accessibility/usability audit were not performed.

Keep the current job scheduler, public APIs, workflow graphs, seeds/retry ledger, reference-tag mapping, explicit project retention and preference privacy rules. SSR/PWA, a workflow canvas, and a separate frontend are lower-priority experiments; none addresses the observed interaction problems directly. SSR also requires Node 20+ and deployment validation. [Blocks deployment options](https://gradio.app/docs/gradio/blocks).

Recommended sequence: **selected-job freshness and browser transition reliability → action placement/mobile layout → settings/persistence optimization → unified Media and engine alignment → constrained Gradio upgrade**. The layout and event optimizations can start on 6.27.0.

Audit script, browser JSON measurements/screenshots and test logs are retained under `.cache/gradio-audit-2026-10-04/`. Existing browser suites also write their normal artifacts under `.cache/ui-redesign/`.

## Implementation follow-up — 4 October 2026

The requested pass upgrades the shared runtime/test pins to **Gradio 6.29.1** and implements the observed correctness, layout and interaction improvements. The runtime constant still supplies both local and Modal setup. NumPy 1.26.4 and Hugging Face Hub below 2 remain constrained.

### Resulting behavior

- **Compose and generate:** H3's single Generate/progress group and readiness message sit in the right results column directly below Next run, following the final placement request. Base model selection lives in Advanced settings. LTX, Qwen, Music and YuE2 keep their composer/action/result arrangement, with concise progress and settings-used metadata beside the output. Empty or incomplete requests receive engine-specific readiness feedback. Idle Interrupt is disabled and becomes available after an owned request is accepted.
- **Responsive layout:** two columns remain available at tablet widths down to 768 pixels; smaller screens stack. The mobile navigation links directly to Compose, Generate and Preview. It uses an anchor instead of a fixed action dock, so actions do not cover controls or keyboard focus. Main tabs use the supported wrapping behavior. Qwen editing controls follow mode/CFG, while uploaded drafts remain available; unfamiliar H3 dimensions show Custom rather than a stale aspect ratio.
- **Jobs:** the owner-scoped refresh updates the selected job's details, files and cancellation/retry availability with the table. Unchanged presentation is skipped. Terminal jobs disable cancellation without requiring reselection; active elapsed time continues to refresh.
- **Settings:** regular transitions submit the edited value, current media and session state, with at most **24 inputs**. Independent scalar edits update only memory, summary, readiness and Generate: **four outputs**. Coupled presets/policies retain the complete resolver-driven presentation. A per-session lock and authoritative memory preserve other recent edits without sharing a CPU settings queue between browsers. Numeric user events avoid delayed echoes from programmatic preset updates. Restoration cannot seed a partial settings snapshot or prematurely enable Generate.
- **Preferences:** seven namespace saves merge into one session snapshot, using at most 24 inputs per save instead of 150. H3 saves authoritative resolved settings and mode memory. Explicit saves follow unqueued resolution/canvas refreshes because that path does not reliably emit Gradio State.change. Prompts, uploaded paths and provider credentials remain excluded; the existing storage key, validation and migration policy are preserved.
- **Media:** type, text search and Favorites sit above the paged thumbnail grid. Selection synchronizes preview, download, settings, tags, lineage and comparison A. Empty searches clear the selection and inspector. New or changed managed files/sidecars are indexed on browsing; unchanged metadata is reused. Rebuild moves to System maintenance and preserves annotations. Search includes managed matches beyond the former 200-item dropdown limit. Explicit callback continuations keep selection synchronized after unqueued events. Comparison B remains available outside the filtered grid; unchanged comparison choices are skipped, avoiding native dropdown filtering races during search/type transitions. Hidden image comparison retains its last valid pair during video switches.

`h3-output-settings` now uses one native Column, removing the duplicate Group DOM ID. Existing GPU scheduling, immutable request capture, managed-path validation, deletion confirmation, project retention and ownership/recovery behavior remain intact. The **13 published API contracts** and **15 normalized workflow fixtures** are preserved by the CPU suite.

### Follow-up measurements

The same short-prompt mock audit uses a 900-pixel-high viewport:

| Viewport width | Baseline Generate Y | Implemented Generate Y |
| --- | ---: | ---: |
| 1440 | 1062 px | 768 px |
| 1024 | 2075 px | 768 px |
| 768 | 2115 px | 860 px |
| 390 | 3005 px | 1175 px |

All five measured widths (390/768/1024/1280/1440) have zero horizontal overflow. The composed mock configuration has **327 dependencies**, down from 362; serialized size is approximately **1.03 MB**, down from 1.13 MB. Component count increases from 838 to 877 to support the additional readiness and layout controls. These are uncompressed configuration measurements. They do not establish a network-transfer or production latency improvement. Local timing varies with concurrent browser load; a p50/p95 latency improvement has not been established. The optimization claim is narrower handler payloads and fewer redundant update paths.

The final mock sample became ready in **1.95 seconds**, made **zero Gradio POSTs during 6.5 idle seconds** before visiting Jobs, and recorded duration-edit round trips of **311–480 ms**. Duration edits remain above the suggested 250 ms responsiveness target. Those timings include browser automation and event processing; further latency work needs separate profiling. The selected completed job's cancellation control is disabled, and the final audit records no page JavaScript errors.

### Validation and limits

The follow-up CPU suite runs **288 tests**, with **three optional PyTorch checks skipped**. Browser acceptance covers settings overrides and restoration, voice references, automatic first-frame resolution, all five engine actions, immutable queued requests, ownership/projects, terminal-job freshness, filtered/empty Media selection, annotations, image/video comparison, dark/reduced-motion presentation, narrow widths, keyboard focus and 200% zoom. Standalone self-test preserves graph/default contracts. Lint, changed-file formatting, Python compilation and Git whitespace checks also run against this pass.

Evidence is under `.cache/gradio-audit-2026-10-04/implemented-*.log` and `implemented/results.json`, with screenshots in that directory and the browser suites' normal artifact locations. The isolated constrained environment passes dependency checks. Real GPU inference, Modal deployment, large-library throughput and a full screen-reader/usability audit remain untested. Initial indexing still scans the managed library and reads changed sidecars; its behavior at very large library sizes needs a separate benchmark.

The measurements above precede the final H3 action placement and gallery performance changes. H3 actions now sit directly below Next run in the right results column; desktop/mobile browser geometry and 38 UI contract checks verify that placement. The subsequent [gallery investigation](gallery-performance-2026-10-04.md) records a local 1,060-asset benchmark and fixes for repeated scans, per-file commits and sequential posters. Final pre-push CPU validation runs 293 tests with three optional PyTorch checks skipped.
