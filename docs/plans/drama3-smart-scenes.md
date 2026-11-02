# Smart scene workflow

Implemented 2026-11-02 as the opt-in **Guided (smart scenes)** writer workflow.

The goal is to give smaller models enough context to make good local decisions while asking for short, specific answers. Regular Guided remains available; its scene-grouping behavior is unchanged.

## Budget profiles

The Smart-only **Smart budget profile** control offers Auto (default), Request optimized and Token optimized. Auto uses the active provider's model: Gemini Flash Lite selects Request optimized, including when exposed through an OpenAI-compatible gateway. GPT-6 Luna and unknown/custom models select Token optimized. Explicit overrides always win. The writer status reports the resolved profile. This routes by configured model ID, not live pricing or account quotas; it does not enforce the user's daily request allowance.

| Policy | Request optimized | Token optimized |
| --- | --- | --- |
| Maximum target lines per window | 32 | 24 |
| Target text character budget | 9,000 | 6,500 |
| Context neighbors on each side | 2 | 1 |
| Voice designs per call | 4 | 3 |
| Downstream story context | Full compact story map | Summary, relevant scene information and stage-specific character profiles |

The Guided batch-lines setting can lower either line limit. Direction and sound windows also use a simple output-work estimate: long lines and performance/effect controls consume more of the budget. Short scenes share a request when they fit, retaining separate scene IDs and ownership checks. Two windows of the same scene never appear in one request. Supplied-source attribution stays scene-local; brief writing stays part-local. These are initial heuristics, not measured optimal settings or exact token accounting. Both profiles retain the same validation and fidelity requirements.

The optional `smart_budget_profile` API input is appended after `strict_source`, preserving existing positional argument order. The selector is shared by manual Rewrite and Auto. Single-pass remains the default workflow; original Guided is still available and ignores this setting.

## Processing

1. Segment supplied text into stable source units. Give the planner the full source, but ask only for a compact title, summary, canonical cast, scene starts and optional shared score palette. Validate ordered boundaries starting at the first source unit. For briefs, ask for a scene outline with short ordered part instructions instead of dialogue.
2. Process supplied text one scene at a time, in bounded windows using the selected profile. Send story and neighboring context, asking only for local speaker assignments and permitted annotations; code retains the actual words. Validate merged assignments and pronunciation consistency. For briefs, write one outlined part at a time using the profile's line/character limits and previous spoken lines as continuity context.
3. Batch missing voice designs using the profile's voice limit. Existing references remain authoritative.
4. Direct and stage each scene in bounded windows, with shared profiles, story map, palette and neighboring lines. Validate sound, pause and effect references against the target window. A long scene is still one scene in the compiled script: merge window events and pauses in source order, retaining the first window's scene beds, room and transition.
5. Use the existing compiler and shared final validation. Required failures retain the original text; optional direction/sound failures produce notes. Each malformed stage gets the existing single targeted retry.

## Boundaries and tradeoffs

- More requests, shorter answers. No fixed call-count promise; actual calls appear in the status.
- The planner still reads the entire source. This is not a solution for input exceeding a provider's context window. The plan is limited to 64 scenes and 16,000 JSON characters; each brief scene may have up to 16 writing parts.
- The character budget is a deterministic proxy, not a tokenizer or provider output-token setting. An indivisible supplied source unit can exceed its profile's character budget; it stays intact to protect fidelity.
- Context neighbors are read-only. Sound anchors must remain inside their target window. Cross-window score spans are not generated; scene-wide beds persist, while local score events stay local.
- Only the first sound window chooses scene-wide beds and acoustics. Later windows add events, pauses, placement and voice effects. Existing structured scripts retain authored sound staging and skip new sound design.
- The existing scene event budget, optional-stage degradation, final strict-source setting and lack of persisted stage retries remain unchanged.
- The added prompt and window limits need real-model evaluation. Mocked contract tests establish routing, bounded work, preservation, retries and merging; they do not establish audio quality or a measured reliability gain.

## Focused verification

`tests.test_drama3_smart` covers oversized scenes, character budgets, invalid scene maps, cross-scene references, sound-plan merging, preserved source, brief part continuity, local retries, structured-script retention and workflow dispatch. Existing Guided and writer integrity/guidance tests cover compatibility.

Budget regressions cover active-model selection, overrides, custom-model fallback, short-scene grouping, reduced token-mode context, dense-content splitting, grouped voice designs and reported profile status. The focused Sound Stage UI contract verifies the appended API parameter and the selector's default/choices. Real-model comparisons of calls, tokens, retries and listening quality remain to be performed; retries currently use the existing bounded targeted retry rather than recursively splitting failures.

## Review follow-up

Window splitting, request metadata and staging merges now live in `guided/batching.py`. Acting and sound use the same prepared batch: target IDs, speaker names, neighboring context and continuation scene IDs. Voice design uses one batching loop for both Guided modes. Content requests are built from explicit context rather than recovering context by splitting a previously formatted prompt.

The review reproduced and fixed three losses: duplicate scene plans overwrote earlier events, later windows changed a speaker's position throughout earlier speech, and brief part assembly discarded dramatic beats. Scene envelopes now require each requested scene exactly once; initial speaker positions win with a diagnostic on conflicts; beats survive part assembly.

Continuation windows cannot set scene-wide beds, room or transitions, even when the opening request failed. When a batch contains both a continuation and a fresh scene, this restriction applies only to the continuation. Focused regressions cover both failure and mixed-batch cases. No profile limits or workflow defaults changed in this refactor.
