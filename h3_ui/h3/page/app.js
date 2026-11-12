"use strict";
// The MiniMax H3 page. The server owns every setting and its policy; this draws what it answers.
(() => {
  const API = "/h3/api";
  const $ = (id) => document.getElementById(id);
  const h = (tag, props = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key in node && key !== "list") node[key] = value;
      else node.setAttribute(key, value === true ? "" : value);
    }
    node.append(...children.flat().filter((child) => child !== null && child !== undefined && child !== false));
    return node;
  };

  const app = {boot: null, s: null, runs: {}, progress: [], files: [], take: 0, suggestion: null, previous: null,
               enhanceContext: null, polling: false, picking: null,
               input: {slots: [], preset: "1920 × 1920", width: 1920, height: 1920, model: "", seed: -1, force_offload: false}};

  const request = async (method, path, body, {form = false} = {}) => {
    const options = {method, credentials: "same-origin"};
    if (form) options.body = body;
    else if (body !== undefined) {
      options.headers = {"Content-Type": "application/json"};
      options.body = JSON.stringify(body);
    }
    const response = await fetch(`${API}${path}`, options);
    let data = null;
    try { data = await response.json(); } catch { /* A proxy's error page has no JSON. */ }
    if (!response.ok) throw new Error(data?.error || data?.detail || `The server answered ${response.status}.`);
    return data;
  };
  const post = (path, body, extra) => request("POST", path, body, extra);

  const visibleBottom = () => {
    const frame = window.frameElement;
    if (!frame) return innerHeight;
    try { return Math.min(innerHeight, window.parent.innerHeight - frame.getBoundingClientRect().top); } catch { return innerHeight; }
  };
  let toastTimer;
  const toast = (message, error = false) => {
    const node = $("toast");
    node.textContent = message;
    node.className = error ? "toast error" : "toast";
    node.style.bottom = `${Math.max(20, innerHeight - visibleBottom() + 20)}px`;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, error ? 8000 : 4000);
  };
  // Every action that can fail reports here instead of leaving the page silent.
  const guard = (action) => async (...args) => {
    try { return await action(...args); } catch (error) { toast(error.message, true); return undefined; }
  };

  const take = (answer) => {
    if (!answer) return;
    if (answer.state) app.s = answer.state;
    if (answer.runs) app.runs = answer.runs;
    if (answer.progress) app.progress = answer.progress;
    if (answer.files) app.files = answer.files;
    render();
    if (Object.values(app.runs).some((run) => run?.active)) poll();
  };
  const change = guard(async (action, changes = {}, size) => take(await post("/settings", {action, changes, size})));
  const changeWriter = guard(async (changes) => take(await post("/writer", {changes})));
  const saveUi = (patch) => {
    app.s.ui = {...app.s.ui, ...patch};
    post("/ui", {ui: app.s.ui}).catch(() => { /* Remembering the open group is a convenience. */ });
  };

  // Controls ----------------------------------------------------------------
  const seg = (node, options, current, pick, {disabled = false, multi = false} = {}) => {
    node.replaceChildren(...options.map((option) => {
      const [label, value] = Array.isArray(option) ? option : [String(option), option];
      const on = multi ? current.includes(value) : value === current;
      return h("button", {type: "button", "aria-pressed": String(on), disabled, text: label, onclick: () => pick(value, on)});
    }));
    return node;
  };
  const number = (value, commit, {disabled = false, label} = {}) => h("input", {
    class: "field num mono", type: "text", inputMode: "decimal", value: String(value), disabled, "aria-label": label,
    onchange: (event) => {
      const parsed = Number(event.target.value);
      if (event.target.value.trim() === "" || Number.isNaN(parsed)) { toast(`${label} must be a number.`, true); render(); return; }
      commit(parsed);
    },
  });

  // Fine-tune ---------------------------------------------------------------
  const GROUPS = [
    ["Model & sampling", "Which checkpoint runs, Normal or Turbo, and how many steps it samples."],
    ["Size & seed", "Exact width and height, the start-frame size cap and a fixed seed."],
    ["Prompt writer", "The service Generate / enhance prompt uses, with its model and keys."],
    ["Speed & attention", "Attention backend and sampling acceleration. The defaults suit most jobs."],
    ["Memory & decoding", "Text encoder, model offloading, input reuse, and the video encoder and decoder."],
    ["Upscale & finish", "Raise resolution during sampling, or run an upscaling pass after generation."],
    ["Input upscaling", "Upscale uploaded frames or reference pictures with SeedVR2 before generating."],
  ];
  // The Next run card names the groups as the settings policy knows them.
  const GROUP_KEYS = ["Model & sampling", "Size, seed & canvas", "Prompt writer", "Speed & attention", "Memory & decoding", "Upscale & finish", "Input upscaling"];
  const v = (key) => app.s.values[key];
  const w = (key) => app.s.writer[key];
  const is = (key, ...values) => () => values.includes(v(key));
  const writerIs = (name) => () => w("prompt_writer_backend") === name;
  const post_ = () => v("postprocess");
  const LOCAL = "Local MiniMax-H3 8B";
  // [group, key, label, kind, extras]. Kinds: choice, bool, num, text, password, head, button, multi.
  const FIELDS = [
    [0, "model_profile", "Base model", "choice", {hint: "Speed uses the rebuilt single-pass NVFP4 files. Quality uses the mixed NVFP4/FP8/INT8 ConvRot files. Original uses the official BF16 files. Singularity uses the fine-tuned pruned v1.3 INT8 checkpoint. Speed, Original and Singularity download when first selected."}],
    [0, "generation_mode", "Generation", "choice", {hint: "Turbo uses the Turbo implementation below. LightX2V uses the matching reference adapter; Larry reference mode is experimental."}],
    [0, "turbo_variant", "Turbo implementation", "choice", {when: is("generation_mode", "Turbo"), hint: "Each variant supplies its trained step count; steps remain adjustable except Pika H3 Flex's fixed six-step LCM schedule. Adapters download on first use. PDMD and DMAD 4-step adapters are experimental."}],
    [0, "steps", "Steps", "num", {range: "4 – 30"}],
    [0, "scheduler", "Scheduler", "choice", {hint: "Controls how sampling steps are distributed across denoising."}],
    [0, "semantic_bridge", "Semantic Bridge (experimental)", "bool", {hint: "May improve prompt adherence. Downloads an 11 MB adapter on first use. FL2VA only; disabled for reference media."}],
    [0, "semantic_bridge_alpha", "Semantic Bridge strength", "num", {range: "0 – 1", when: () => v("semantic_bridge") && !("semantic_bridge" in app.s.locked), hint: "Start at 0.10; try 0.15 for a stronger effect. Higher values can reduce quality."}],

    [1, "width", "Width", "num", {hint: "Follows Aspect ratio and Size in Compose. Typing here switches them to Custom and Exact dimensions."}],
    [1, "height", "Height", "num", {}],
    [1, "auto_megapixels", "Start-frame auto cap", "choice", {hint: "Caps the automatic video size taken from a first frame."}],
    [1, "seed", "Seed", "num", {hint: "-1 chooses a random seed. Multiple videos always use independent seeds."}],

    [2, "prompt_writer_backend", "Prompt writer", "choice", {store: "writer", hint: "The local MiniMax-H3 8B writer supports T2VA, I2VA, L2VA, and FL2VA. Gemini supports all Reference media; OpenAI-compatible models support text and image enhancement."}],
    [2, "local_prompt_base_model", "Local base model", "choice", {store: "writer", when: writerIs(LOCAL), hint: "BF16 is the default full-precision checkpoint. FP8 is available as the lower-memory alternative."}],
    [2, "local_prompt_greedy", "Greedy decoding", "bool", {store: "writer", when: writerIs(LOCAL)}],
    [2, "local_prompt_max_tokens", "Max new tokens", "num", {store: "writer", when: writerIs(LOCAL), range: "256 – 8192"}],
    [2, "local_prompt_temperature", "Temperature (sampling)", "num", {store: "writer", when: writerIs(LOCAL), range: "0.1 – 2.0"}],
    [2, "local_prompt_top_p", "Top-p (sampling)", "num", {store: "writer", when: writerIs(LOCAL), range: "0.05 – 1.0"}],
    [2, "local_prompt_seed", "Seed", "num", {store: "writer", when: writerIs(LOCAL)}],
    [2, "gemini_prompt_model", "Gemini model", "choice", {store: "writer", when: writerIs("Gemini"), hint: "Uses the active inputs with prompt.txt. Set GEMINI_API_KEY on the server or enter a temporary key."}],
    [2, "gemini_api_key", "Temporary Gemini API key", "password", {store: "writer", when: writerIs("Gemini"), placeholder: "Uses GEMINI_API_KEY when blank", hint: "The server does not store keys entered here."}],
    [2, "openai_model", "OpenAI-compatible model", "text", {store: "writer", when: writerIs("OpenAI-compatible"), hint: "Uses the active text and images plus prompt.txt. Video and audio references require Gemini."}],
    [2, "openai_api_key", "Temporary OpenAI API key", "password", {store: "writer", when: writerIs("OpenAI-compatible"), placeholder: "Uses OPENAI_API_KEY when blank", hint: "The server does not store keys entered here."}],
    [2, "openai_base_url", "Base URL", "text", {store: "writer", when: writerIs("OpenAI-compatible"), placeholder: "Uses OPENAI_BASE_URL when blank"}],
    [2, null, "Result review", "head"],
    [2, "review", "Score videos and tag them", "bool", {store: "writer", hint: "After each video is saved, Gemini Decision scores how well it fits the prompt and adds Media library tags, in the background. Storyboard takes are scored too. Uses the Gemini key here, else GEMINI_API_KEY."}],
    [2, "gemini_api_key", "Temporary Gemini API key", "password", {store: "writer", when: () => w("review") && !writerIs("Gemini")(), placeholder: "Uses GEMINI_API_KEY when blank", hint: "Only for video scores and tags. The server does not store keys entered here."}],

    [3, "attention_mode", "Attention", "choice", {locked: () => app.boot.attention_locked, hint: "SLA is the default: block-sparse, audio-safe, and dense for short sequences. Veda is experimental learned sparse attention. Auto enables Sol-Attn for Reference mode or large jobs and uses Kitchen for smaller ones. Sage 2 applies the pinned KJNodes override."}],
    [3, "sla_preset", "SLA / Veda preset", "choice", {when: is("attention_mode", "SLA", "Veda"), hint: "Fast uses validated 0.90 sparsity. Balanced uses the LoRA-distilled 0.85 sparsity. Quality also runs the final sampling step dense to recover fine detail."}],
    [3, "sol_tau", "Sol-Attn tau", "num", {when: is("attention_mode", "Sol-Attn", "Auto"), range: "0.5 – 1.5"}],
    [3, "sol_thresh_type", "Sol threshold", "choice", {when: is("attention_mode", "Sol-Attn", "Auto"), hint: "diag is faster; exact calculates a more precise routing threshold."}],
    [3, "sol_exact_mode", "Exact H3 prefix mode", "choice", {when: is("attention_mode", "Sol-Attn", "Auto"), hint: "exact_kv preserves text/condition/reference/audio KV rows at low cost. exact_kv_and_rows also keeps prefix query rows dense for maximum audio/conditioning fidelity."}],
    [3, "sol_dense_steps", "Dense final transformer blocks", "num", {when: is("attention_mode", "Sol-Attn", "Auto"), range: "0 – 4", hint: "Keep the final N H3 transformer blocks dense. The final block is the most approximation-sensitive."}],
    [3, null, "Sampling acceleration", "head"],
    [3, "cache_mode", "Acceleration mode", "choice", {hint: "Spectrum is the normal H3 default. FirstBlockCache is the lower-memory fallback. Modes are mutually exclusive, and acceleration is off while native latent upscale is on."}],
    [3, "fbcache_preset", "FirstBlockCache preset", "choice", {when: is("cache_mode", "FirstBlockCache"), hint: "Fast is the recommended default. Named presets use a protected 10–95% denoising window and at most two consecutive cache hits."}],
    [3, "fbcache_threshold", "FirstBlock threshold", "num", {when: is("cache_mode", "FirstBlockCache"), range: "0 – 0.25", locked: () => v("fbcache_preset") !== "Custom"}],
    [3, "fbcache_max_hits", "Max consecutive cache hits", "num", {when: is("cache_mode", "FirstBlockCache"), range: "1 – 8", locked: () => v("fbcache_preset") !== "Custom"}],
    [3, "fbcache_start", "Cache start percent", "num", {when: is("cache_mode", "FirstBlockCache"), range: "0 – 0.90", locked: () => v("fbcache_preset") !== "Custom"}],
    [3, "fbcache_end", "Cache end percent", "num", {when: is("cache_mode", "FirstBlockCache"), range: "0.10 – 1.0", locked: () => v("fbcache_preset") !== "Custom"}],
    [3, "fbcache_temporal_guard", "Temporal frame guard", "bool", {when: is("cache_mode", "FirstBlockCache"), hint: "Checks the most-changed target-video latent frame in addition to the global residual average."}],
    [3, "easycache_threshold", "Reuse threshold", "num", {when: is("cache_mode", "EasyCache"), range: "0 – 0.5", hint: "Higher skips more steps. Start at 0.10 for H3; ComfyUI's generic default is 0.20."}],
    [3, "easycache_start", "Start percent", "num", {when: is("cache_mode", "EasyCache"), range: "0 – 0.9"}],
    [3, "easycache_end", "End percent", "num", {when: is("cache_mode", "EasyCache"), range: "0.1 – 1.0"}],
    [3, "easycache_verbose", "Log EasyCache decisions", "bool", {when: is("cache_mode", "EasyCache"), hint: "Logs skipped-step counts and estimated speedup in ComfyUI."}],

    [4, "text_encoder", "Text encoder", "choice", {hint: "BF16 is approximately 51.5 GB. NVFP4/AWQ and INT8 ConvRot download on first use."}],
    [4, "stage_model_offload", "Offload models between H3 stages", "bool", {hint: "Unload resident models between text encoding, diffusion, latent upscaling, and VAE decoding. Required by the BF16 encoder."}],
    [4, "reuse_unchanged_inputs", "Reuse unchanged prompt and media", "bool", {hint: "Reuse matching prompt/media encoding and unchanged workflow nodes. Off forces fresh conditioning at every stage, including refinement. Sampling still reruns when the seed changes."}],
    [4, "encoder_small_input", "Qwen small input attention", "bool", {hint: "On: upstream PyTorch/basic attention. Off: the server attention backend. Applies to Qwen text and vision encoding. Changing this rebuilds conditioning on the next generation."}],
    [4, "video_encoder", "Video encoder", "choice", {hint: "LightVAE downloads on first use. First/last-frame encoding is experimental; compare keyframe and reference fidelity."}],
    [4, "video_decoder", "Video decoder", "decoder", {hint: "Sol INT8 decodes whole clips with more VRAM and may change fine detail. One decoder per run. TensorRT prepares a local engine when needed."}],
    [4, "compile", "TensorRT engine", "button", {text: "Compile TensorRT VAE engine", run: () => startRun("/compile-trt"), status: () => app.runs.compile}],

    [5, null, "Native latent upscale", "head"],
    [5, "latent_upscale", "Generate at half resolution, then latent upscale 2x", "bool", {hint: "Runs inside H3 sampling, not after video generation. Width and height remain the H3 output resolution and must both be divisible by 64. Only the video latent is upscaled; H3 audio is preserved."}],
    [5, "latent_upscaler_model", "Latent upscaler model", "choice", {when: () => v("latent_upscale"), hint: "Balanced uses BF16 and is the default. Fast uses FP16; Quality uses FP32 and needs more memory. Downloaded on first use."}],
    [5, "latent_upscale_refine_steps", "High-resolution refinement steps", "num", {when: () => v("latent_upscale"), range: "1 – 6", locked: () => app.boot.postprocess_kinds.regenerate.includes(v("latent_upscale_method")), hint: "The clean 2x latent is lightly re-noised and refined. Two expensive high-resolution steps is the default."}],
    [5, "latent_upscale_refine_lora", "High-resolution refinement LoRA", "choice", {when: () => v("latent_upscale"), locked: () => app.boot.postprocess_kinds.regenerate.includes(v("latent_upscale_method")), hint: "Same as generation preserves the generation LoRA. Choose a different LoRA for just the high-resolution pass; downloaded on first use."}],
    [5, "latent_upscale_method", "High-resolution refinement method", "choice", {when: () => v("latent_upscale"), hint: "Full-frame is the normal path. Regenerate rank 16/384 uses a dedicated refiner plus LightX2V with three Euler steps, overriding refinement steps and LoRA above. Rank 16 downloads 155 MB; rank 384 downloads 5.87 GB."}],
    [5, null, "After generation", "head"],
    [5, "postprocess", "After generation", "choice", {hint: "Optionally run SeedVR2, H3 Regenerate or LTX-2.5 upscaling immediately after the base H3 video finishes. The source video remains in the gallery."}],
    [5, "upscale_resolution", "Output resolution", "choice", {when: () => app.boot.postprocess_kinds.ai.includes(post_()), hint: "Fits the source inside the selected square while preserving aspect ratio."}],
    [5, "seedvr2_model", "SeedVR2 model", "choice", {when: () => post_() === app.boot.postprocess_kinds.seedvr2, hint: "Downloaded on first use. 7B INT8 is the default quality/VRAM balance; FP16 favors fidelity, 7B Sharp favors stronger detail, and MXFP8/NVFP4 are experimental speed options."}],
    [5, "ltx25_model", "Generation finishing LTX model", "choice", {when: () => app.boot.postprocess_kinds.ltx.includes(post_()), hint: "Used for H3 finishing independently of the LTX generation workspace, with the H3 generation prompt. The gated 2x IC-LoRA downloads on first use."}],
    [5, "upscale_force_offload", "Unload H3 models before upscaling", "bool", {when: () => app.boot.postprocess_kinds.ai.includes(post_()), hint: "Reduces peak VRAM at the cost of reloading H3 for the next generation."}],
    [5, null, "Face refine", "head"],
    [5, "face_refine", "Refine small faces after generation", "bool", {hint: "Turn on for wide shots with distant faces. Re-renders the largest face up close and blends it back before any upscaling, adding an extra H3 pass per video. The detector and 8-step LoRA download on first use."}],
    [5, "face_refine_strength", "Face refine strength", "num", {when: () => v("face_refine"), range: () => app.boot.limits.face_refine_strength.join(" – "), hint: "How much of the face H3 may redraw. Higher fixes more but can drift from the original face."}],

    [6, "slots", "Images to upscale", "multi", {store: "input", options: () => app.boot.choices.input_slots, hint: "Fits smaller images upward into a target frame before generation. Aspect ratio is preserved and images already at or beyond the frame are not downscaled."}],
    [6, "preset", "Target frame preset", "choice", {store: "input", options: () => app.boot.choices.input_preset}],
    [6, "width", "Frame width", "num", {store: "input", when: () => app.input.preset === "Custom"}],
    [6, "height", "Frame height", "num", {store: "input", when: () => app.input.preset === "Custom"}],
    [6, "model", "SeedVR2 model", "choice", {store: "input", options: () => app.boot.choices.seedvr2_model, hint: "7B INT8 is the default quality/VRAM balance. FP16 favors fidelity; MXFP8 and NVFP4 are experimental speed options."}],
    [6, "seed", "Seed (-1 random)", "num", {store: "input"}],
    [6, "force_offload", "Unload resident models before input upscaling", "bool", {store: "input", hint: "Useful when H3 or another large model is already resident in VRAM."}],
    [6, "run", "Upscale", "button", {text: "Upscale selected inputs to frame", run: () => startRun("/input-upscale", app.input), status: () => app.runs["input-upscale"], files: true}],
  ];

  const fieldValue = (key, store) => store === "writer" ? w(key) : store === "input" ? app.input[key] : v(key);
  const fieldCommit = (key, store) => (value) => {
    if (store === "writer") return changeWriter({[key]: value});
    if (store === "input") {
      app.input[key] = value;
      if (key === "preset" && app.boot.input_presets[value]) [app.input.width, app.input.height] = app.boot.input_presets[value];
      return render();
    }
    if (key === "width" || key === "height") saveUi({aspect: "Custom", tier: "custom"});
    return change(key, {[key]: value});
  };

  const fineRow = ([, key, label, kind, extra = {}]) => {
    if (kind === "head") return h("h4", {class: "fine-head", text: label});
    const store = extra.store;
    const lockedValue = !store && key in app.s.locked ? app.s.locked[key] : undefined;
    const disabled = lockedValue !== undefined || Boolean(extra.locked?.());
    const value = lockedValue !== undefined ? lockedValue : kind === "decoder" ? app.s.decoder : fieldValue(key, store);
    const commit = fieldCommit(key, store);
    const before = !store && app.s.differences[key];
    const control = h("div", {class: "control"});
    if (kind === "choice" || kind === "decoder") {
      const options = extra.options?.() ?? app.boot.choices[key] ?? [];
      const pick = kind === "decoder" ? (chosen) => change(`decoder:${chosen}`) : commit;
      control.append(seg(h("div", {class: "seg", role: "group", "aria-label": label}), options, value, pick, {disabled}));
    } else if (kind === "bool") {
      control.append(seg(h("div", {class: "seg", role: "group", "aria-label": label}), [["Off", false], ["On", true]], Boolean(value), commit, {disabled}));
    } else if (kind === "multi") {
      control.append(seg(h("div", {class: "seg", role: "group", "aria-label": label}), extra.options(), value,
        (chosen, on) => commit(on ? value.filter((item) => item !== chosen) : [...value, chosen]), {multi: true}));
    } else if (kind === "num") {
      const range = typeof extra.range === "function" ? extra.range() : extra.range;
      control.append(number(value, commit, {disabled, label}), range ? h("span", {class: "small muted", text: range}) : null);
    } else if (kind === "text" || kind === "password") {
      control.append(h("input", {class: "field text", type: kind, value: value ?? "", placeholder: extra.placeholder, autocomplete: "off",
                                 "aria-label": label, onchange: (event) => commit(event.target.value)}));
    } else if (kind === "button") {
      const run = extra.status();
      control.append(h("button", {type: "button", class: "btn", text: extra.text, disabled: !app.boot.gpu || Boolean(run?.active), onclick: extra.run}),
        run?.active && key === "run" ? h("button", {type: "button", class: "btn", text: "Interrupt", onclick: () => interrupt("input-upscale")}) : null,
        run ? h("span", {class: "small muted", role: "status", text: run.error || run.status}) : null);
      if (extra.files && app.files.length) {
        control.append(h("div", {class: "files"}, app.files.map((file) => h("a", {href: file.url, download: file.name, text: file.name}))));
      }
    }
    return h("div", {class: "fine-row"},
      h("div", {class: "about"},
        h("span", {class: "label"}, label,
          before ? h("span", {class: "chip warn", text: `recipe: ${before[0]}`}) : null,
          lockedValue !== undefined ? h("span", {class: "chip", text: "set automatically"}) : null),
        extra.hint ? h("span", {class: "hint", text: extra.hint}) : null),
      control);
  };

  const openGroup = (index) => {
    saveUi({fine: true, group: index});
    render();
    $("fine").scrollIntoView({block: "nearest", behavior: "smooth"});
  };

  const renderFine = () => {
    const open = Boolean(app.s.ui.fine);
    const group = Math.min(GROUPS.length - 1, Math.max(0, Number(app.s.ui.group) || 0));
    let counts = {};
    try { counts = JSON.parse($("summary").querySelector("[data-changed-tabs]")?.dataset.changedTabs || "{}"); } catch { /* No card, no counts. */ }
    const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
    $("fine-count").hidden = !total;
    $("fine-count").textContent = `${total} changed`;
    $("fine-toggle").setAttribute("aria-expanded", String(open));
    $("fine").hidden = !open;
    if (!open) return;
    $("fine-nav").replaceChildren(...GROUPS.map(([label], index) => h("button", {
      type: "button", "aria-pressed": String(index === group), onclick: () => { saveUi({group: index}); render(); },
    }, label, counts[GROUP_KEYS[index]] ? h("span", {class: "chip warn", "aria-hidden": "true", text: String(counts[GROUP_KEYS[index]])}) : null)));
    $("fine-title").textContent = GROUPS[group][0];
    $("fine-detail").textContent = GROUPS[group][1];
    // A field keeps focus across a redraw only if its row is rebuilt after its change committed.
    $("fine-rows").replaceChildren(...FIELDS.filter((field) => field[0] === group && (!field[4]?.when || field[4].when())).map(fineRow));
  };

  // Media -------------------------------------------------------------------
  const KINDS = [["Picture", "ref_image", 9, "image/*"], ["Video", "ref_video", 3, "video/*"], ["Audio", "ref_audio", 3, "audio/*"]];
  const acceptOf = (slot) => slot.includes("audio") ? "audio/*" : slot.includes("video") ? "video/*" : "image/*";
  const upload = guard(async (slot, file) => {
    const form = new FormData();
    form.append("file", file);
    take(await post(`/media/upload?slot=${encodeURIComponent(slot)}`, form, {form: true}));
  });
  const pickFile = (slot) => {
    app.picking = slot;
    $("file-input").accept = acceptOf(slot);
    $("file-input").value = "";
    $("file-input").click();
  };
  const clearSlot = guard(async (slot) => {
    clearTimeout(promptTimer);
    await savePrompt();
    take(await post("/media/clear", {slot}));
  });
  const openLibrary = guard(async (slot) => {
    const dialog = $("library");
    const grid = h("div", {class: "library-grid"});
    const load = guard(async (query) => {
      const {items} = await request("GET", `/library?query=${encodeURIComponent(query)}&limit=60`);
      grid.replaceChildren(...items.map((item) => h("button", {type: "button", title: item.caption, onclick: guard(async () => {
        dialog.close();
        take(await post("/media/library", {slot, source: item.path}));
      })}, h("img", {src: item.thumb, alt: item.caption, loading: "lazy"}))));
      if (!items.length) grid.replaceChildren(h("p", {class: "small muted", text: "No pictures match."}));
    });
    dialog.replaceChildren(
      h("div", {class: "row between"}, h("h2", {text: "Media library"}), h("button", {type: "button", class: "btn quiet", text: "Close", onclick: () => dialog.close()})),
      h("input", {class: "field", type: "search", placeholder: "Search pictures", "aria-label": "Search pictures", oninput: (event) => load(event.target.value)}),
      grid);
    dialog.showModal();
    await load("");
  });
  const insertTag = (tag) => {
    const area = $("prompt");
    const start = area.selectionStart ?? area.value.length, end = area.selectionEnd ?? area.value.length;
    const before = area.value.slice(0, start), after = area.value.slice(end);
    const text = (before && !/\s$/.test(before) ? " " : "") + tag + (/^\s/.test(after) ? "" : " ");
    area.focus();
    area.setRangeText(text, start, end, "end");
    promptEdited();
  };
  const dropTarget = (node, slotFor) => {
    node.addEventListener("dragover", (event) => { event.preventDefault(); node.classList.add("over"); });
    node.addEventListener("dragleave", () => node.classList.remove("over"));
    node.addEventListener("drop", (event) => {
      event.preventDefault();
      node.classList.remove("over");
      const file = event.dataTransfer?.files?.[0];
      const slot = file && slotFor(file);
      if (slot) upload(slot, file); else if (file) toast("That file doesn't fit here.", true);
    });
    return node;
  };
  const thumb = (item, label) => item?.kind === "Picture"
    ? h("img", {class: "thumb", src: item.url, alt: label})
    : h("span", {class: "thumb", "aria-hidden": "true", text: item?.kind || label});
  const nextSlot = (prefix, limit) => {
    for (let index = 1; index <= limit; index += 1) if (!app.s.media[`${prefix}_${index}`]) return `${prefix}_${index}`;
    return null;
  };

  const mediaCard = (slot, title, {tag, library = false} = {}) => {
    const item = app.s.media[slot];
    const binding = tag ? app.s.references.find((reference) => reference.tag === tag && reference.active) : null;
    const used = tag ? app.s.prompt.includes(tag) : false;
    const chip = !tag ? null : binding?.unresolved ? h("span", {class: "chip warn", text: "replaced · confirm"})
      : h("span", {class: used ? "chip good" : "chip warn", text: used ? "in prompt" : "not in prompt yet"});
    return dropTarget(h("div", {class: "pic"}, thumb(item, title),
      h("div", {class: "body"},
        h("div", {class: "row tightrow"}, h("strong", {class: "small", text: title}), chip,
          binding ? h("span", {class: "small muted mono", text: `engine ${binding.engine}`}) : null),
        h("span", {class: "small muted detail", text: item.detail}),
        h("div", {class: "row tightrow"},
          tag ? h("button", {type: "button", class: "btn compact tag", text: `+ ${tag}`, title: "Add this tag to the prompt", disabled: Boolean(binding?.unresolved), onclick: () => insertTag(tag)}) : null,
          h("button", {type: "button", class: "btn compact", text: "Replace", onclick: () => pickFile(slot)}),
          library ? h("button", {type: "button", class: "btn compact", text: "Media library", onclick: () => openLibrary(slot)}) : null,
          h("button", {type: "button", class: "btn compact quiet", text: "Remove", onclick: () => clearSlot(slot)})))),
      () => slot);
  };

  const renderReferences = () => {
    const cards = [], adders = [];
    let count = 0;
    for (const [kind, prefix, limit] of KINDS) {
      let filled = 0;
      for (let index = 1; index <= limit; index += 1) {
        const slot = `${prefix}_${index}`;
        if (!app.s.media[slot]) continue;
        filled += 1;
        cards.push(mediaCard(slot, `${kind} ${index}`, {tag: `<${kind} ${index}>`, library: kind === "Picture"}));
      }
      count += filled;
      const free = nextSlot(prefix, limit);
      adders.push(h("button", {type: "button", class: "empty-slot", disabled: !free, text: `+ ${kind} · ${filled} of ${limit}`, onclick: () => pickFile(free)}));
      if (kind === "Picture") adders.push(h("button", {type: "button", class: "empty-slot", disabled: !free, text: "+ From Media library", onclick: () => openLibrary(free)}));
    }
    $("reference-count").textContent = count ? `${count} added` : "";
    $("reference-list").replaceChildren(...cards);
    $("reference-add").replaceChildren(...adders);
    const waiting = app.s.references.filter((reference) => reference.unresolved && (reference.active || app.s.prompt.includes(reference.tag)));
    const replaced = waiting.filter((reference) => reference.active);
    const removed = waiting.filter((reference) => !reference.active);
    $("reference-confirm").hidden = !waiting.length;
    $("confirm-references").hidden = !waiting.length;
    $("confirm-references").textContent = removed.length ? "Clear missing references" : "Confirm replaced references";
    $("reference-confirm-line").textContent = [
      removed.length ? `${removed.map((reference) => reference.tag).join(", ")} ${removed.length > 1 ? "have" : "has"} no attached media. Add a replacement or remove ${removed.length > 1 ? "these tags" : "this tag"} from the prompt.` : "",
      replaced.length ? `${replaced.map((reference) => reference.tag).join(", ")} ${replaced.length > 1 ? "were" : "was"} replaced. Confirm before using ${replaced.length > 1 ? "these tags" : "this tag"}.` : "",
    ].filter(Boolean).join(" ");
    seg($("ref-size"), app.boot.choices.ref_image_size, v("ref_image_size"), (value) => change("ref_image_size", {ref_image_size: value}));
  };

  const frameTile = (slot, title, note) => {
    const item = app.s.media[slot];
    if (!item) {
      return dropTarget(h("div", {class: "frame empty-slot"}, h("strong", {text: title}), h("span", {text: "Drop a picture"}),
        h("div", {class: "row tightrow"},
          h("button", {type: "button", class: "btn compact", text: "Upload", onclick: () => pickFile(slot)}),
          h("button", {type: "button", class: "btn compact", text: "Media library", onclick: () => openLibrary(slot)}))), () => slot);
    }
    return dropTarget(h("div", {class: "frame"}, h("img", {src: item.url, alt: title}),
      h("div", {class: "row tightrow between"}, h("strong", {class: "small", text: title}), note ? h("span", {class: "chip good", text: note}) : null),
      h("span", {class: "small muted detail", text: item.detail}),
      h("div", {class: "row tightrow"},
        h("button", {type: "button", class: "btn compact", text: "Replace", onclick: () => pickFile(slot)}),
        h("button", {type: "button", class: "btn compact", text: "Media library", onclick: () => openLibrary(slot)}),
        h("button", {type: "button", class: "btn compact quiet", text: "Remove", onclick: () => clearSlot(slot)}))), () => slot);
  };

  const renderFrames = () => {
    $("frame-list").replaceChildren(frameTile("first_image", "First frame", "auto resolution"), frameTile("last_image", "Last frame"));
    const voices = [1, 2, 3].map((index) => {
      const slot = `fl2va_audio_${index}`;
      const title = `Voice ${index} · <Audio ${index}>`;
      return app.s.media[slot] ? mediaCard(slot, title)
        : dropTarget(h("button", {type: "button", class: "empty-slot", text: `+ ${title}`, onclick: () => pickFile(slot)}), () => slot);
    });
    $("voice-list").replaceChildren(...voices);
    const filled = [1, 2, 3].filter((index) => app.s.media[`fl2va_audio_${index}`]).length;
    $("voice-count").textContent = `· ${filled} of 3`;
  };

  // Output ------------------------------------------------------------------
  const TIERS = [["768p", "draft"], ["1080p", "fast"], ["2K", "large"], ["Exact dimensions", "custom"]];
  const ratioOf = (name) => name.split("·", 1)[0].trim();
  const sizeChoice = () => {
    const {aspect, tier} = app.s.ui;
    if (aspect === "Custom" || tier === "custom") return {aspect: "Custom", tier: "custom"};
    // A remembered choice holds while the size is still its own, give or take the latent-upscale grid.
    const named = aspect && tier && (app.boot.resolutions[tier] || []).find((candidate) => ratioOf(candidate) === aspect);
    const [nameWidth, nameHeight] = named ? named.split("·").pop().trim().split("×").map(Number) : [];
    if (named && Math.abs(nameWidth - v("width")) < 64 && Math.abs(nameHeight - v("height")) < 64) return {aspect, tier};
    const sought = `${v("width")}×${v("height")}`;
    for (const [, key] of TIERS) {
      const name = (app.boot.resolutions[key] || []).find((candidate) => candidate.endsWith(sought));
      if (name) return {aspect: ratioOf(name), tier: key};
    }
    return {aspect: "Custom", tier: "custom"};
  };
  const pickSize = (aspect, tier) => {
    if (aspect === "Custom" || tier === "custom") {
      saveUi({aspect: "Custom", tier: "custom", fine: true, group: 1});
      return render();
    }
    const name = app.boot.resolutions[tier].find((candidate) => ratioOf(candidate) === aspect);
    if (!name) return toast(`${aspect} has no ${tier} size.`, true);
    saveUi({aspect, tier});
    return change("resolution", {}, {tier, name});
  };
  const renderOutput = () => {
    const busy = false;
    $("preset").replaceChildren(...app.boot.choices.preset.map((name) => h("button", {
      type: "button", "aria-pressed": String(name === v("preset")), text: name, disabled: busy, onclick: () => change("preset", {preset: name}),
    })));
    if (document.activeElement !== $("duration")) $("duration").value = v("duration");
    $("duration-value").textContent = `${Number($("duration").value).toFixed(1)} s`;
    const [low, high] = app.boot.limits.batch;
    seg($("batch"), Array.from({length: high - low + 1}, (_, index) => low + index), v("batch_count"), (value) => change("batch_count", {batch_count: value}));
    const chosen = sizeChoice();
    const ratios = [...new Set(Object.values(app.boot.resolutions).flat().map(ratioOf))];
    seg($("aspect"), [...ratios, "Custom"], chosen.aspect, (value) => pickSize(value, chosen.tier === "custom" ? "draft" : chosen.tier));
    seg($("tier"), TIERS, chosen.tier, (value) => pickSize(chosen.aspect === "Custom" ? "16:9" : chosen.aspect, value));
    const auto = v("mode") === "First / last frame" && app.s.media.first_image;
    $("resolution").replaceChildren(h("span", {class: "mono", text: `${v("width")} × ${v("height")}`}),
      h("span", {class: "muted", text: ` · ${auto ? "from the first frame's ratio and the start-frame cap · " : ""}${app.s.resolution.replace(/\d+×\d+ · /, "")}`}));
  };

  // Prompt ------------------------------------------------------------------
  let promptTimer;
  let pendingSave = Promise.resolve();
  const savePrompt = guard(async () => {
    clearTimeout(promptTimer);
    const sent = $("prompt").value;
    const save = pendingSave.then(async () => {
      const answer = await post("/prompt", {prompt: sent});
      // Typing may have gone on while the save was in flight; the newer text stays.
      if ($("prompt").value === sent) take(answer);
    });
    pendingSave = save.catch(() => {});
    await save;
  });
  const promptEdited = () => {
    app.s.prompt = $("prompt").value;
    clearTimeout(promptTimer);
    promptTimer = setTimeout(savePrompt, 500);
    renderReadiness();
    if (v("mode") === "Reference media") renderReferences();
  };
  const renderPrompt = () => {
    if (document.activeElement !== $("prompt") && $("prompt").value !== app.s.prompt) $("prompt").value = app.s.prompt;
    const run = app.runs.enhance;
    $("writer-line").textContent = w("prompt_writer_backend");
    $("enhance").disabled = Boolean(run?.active);
    $("enhance").textContent = run?.active ? "Writing…" : "Generate / enhance prompt";
    $("enhance-status").textContent = app.enhanceNote || run?.error || (run ? run.status : "Drafts a prompt from your idea and media, or enhances the one above. You review the suggestion before it replaces your prompt.");
    $("suggestion-box").hidden = !app.suggestion;
    $("suggestion").textContent = app.suggestion?.text || "";
    $("undo").hidden = app.previous === null;
  };

  // Runs --------------------------------------------------------------------
  const startRun = guard(async (path, body) => {
    await savePrompt();
    take(await post(path, body));
  });
  const interrupt = guard(async (kind) => {
    const answer = await post("/interrupt", {kind});
    if (answer.message) toast(answer.message);
    take(answer);
  });
  const poll = async () => {
    if (app.polling) return;
    app.polling = true;
    try {
      for (;;) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const before = app.runs;
        const answer = await request("GET", "/run");
        const settled = ["generate", "input-upscale"].some((kind) => before[kind]?.active && !answer.runs[kind]?.active);
        const enhanced = before.enhance?.active && !answer.runs.enhance?.active;
        take(settled ? await request("GET", "/run?state=true") : answer);
        if (enhanced) enhanceFinished(app.runs.enhance);
        if (settled && app.runs.generate && !app.runs.generate.error) app.take = 0;
        if (!Object.values(app.runs).some((run) => run?.active)) break;
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      app.polling = false;
    }
  };
  const enhanceFinished = (run) => {
    if (app.enhanceContext === null) return;
    app.enhanceNote = "";
    if (run?.suggestion) app.suggestion = {text: run.suggestion, original: run.original, context: app.enhanceContext};
    render();
  };
  const promptContext = () => {
    const mode = v("mode");
    const media = Object.entries(app.s.media).filter(([slot]) => mode === "Reference media" ? slot.startsWith("ref_")
      : mode === "First / last frame" && (slot === "first_image" || slot === "last_image" || slot.startsWith("fl2va_audio_")))
      .map(([slot, item]) => [slot, item.path]).sort(([a], [b]) => a.localeCompare(b));
    return JSON.stringify([mode, v("width"), v("height"), v("duration"),
      mode === "Reference media" ? app.s.reference_version : null, media]);
  };

  const renderReadiness = () => {
    const mode = v("mode");
    const hasPrompt = app.s.prompt.trim().length > 0;
    const hasMedia = mode === "Text to video" ? null
      : mode === "First / last frame" ? Boolean(app.s.media.first_image || app.s.media.last_image)
      : Object.keys(app.s.media).some((slot) => slot.startsWith("ref_"));
    const steps = [["Prompt", hasPrompt ? "done" : "now"], ["Media", hasMedia === null ? "" : hasMedia ? "done" : "now"],
                   ["Settings", app.s.issues.length ? "now" : "done"]];
    $("steps").replaceChildren(...steps.map(([label, state]) => h("span", {class: `step ${state}`}, h("span", {class: "dot", "aria-hidden": "true"}), label)));
    const run = app.runs.generate;
    const waiting = mode === "Reference media"
      ? app.s.references.filter((reference) => reference.unresolved && app.s.prompt.includes(reference.tag)) : [];
    const missing = waiting.find((reference) => !reference.active);
    const ready = app.s.readiness.ready && hasPrompt && hasMedia !== false && !waiting.length;
    $("readiness").textContent = run?.active ? "Generating. You can keep editing the next run."
      : !hasPrompt ? "Before generating: write a prompt."
      : missing ? `No media attached for ${missing.tag}. Add a replacement or remove the tag from the prompt.`
      : waiting.length ? `Confirm the replaced reference ${waiting[0].tag} first.`
      : app.s.readiness.message;
    $("run").disabled = !app.boot.gpu || !ready || Boolean(run?.active);
    $("stop").hidden = !run?.active;
    const status = $("run-status");
    status.hidden = !run || (!run.active && !run.error && !run.message);
    status.className = run?.error ? "status error" : "status";
    status.textContent = run?.error || run?.status || "";
  };


  // The result's prompt-fit score and tags from Gemini Decision; while one is pending the page asks again.
  let reviewTimer = null;
  const renderReview = (current) => {
    const box = $("result-review");
    const review = current?.review;
    box.hidden = !review;
    clearTimeout(reviewTimer);
    // Every result of a batch is scored on its own; keep asking while any of them is pending.
    if (current && app.s.results.videos.some((item) => item.review?.status === "pending")) {
      reviewTimer = setTimeout(guard(async () => take(await request("GET", "/run?state=true"))), 4000);
    }
    if (!review) return;
    if (review.status === "pending") {
      box.replaceChildren(h("span", {class: "muted", text: "Scoring the result against the prompt…"}));
    } else if (review.status === "done") {
      box.replaceChildren(
        h("span", {class: `chip ${review.fit >= 70 ? "good" : review.fit < 45 ? "warn" : ""}`, title: "How well the result fits its prompt, scored by Gemini Decision", text: `Prompt fit ${review.fit} · ${review.label}`}),
        ...review.tags.map((tag) => h("span", {class: "chip", title: "Added to Media library tags", text: tag})));
    } else {
      box.replaceChildren(h("span", {class: "muted", title: review.error || "", text: "No prompt-fit score: the review failed."}));
    }
  };

  const renderViewer = () => {
    const run = app.runs.generate;
    const videos = run?.active || (run && app.progress.length && !app.s.results.videos.length) ? app.progress : app.s.results.videos;
    const expected = run?.active ? v("batch_count") : videos.length;
    app.take = Math.min(app.take, Math.max(0, expected - 1));
    seg($("takes"), Array.from({length: Math.max(1, expected)}, (_, index) => [`Video ${index + 1}`, index]), app.take, (index) => { app.take = index; render(); });
    const current = videos.find((video, index) => (video.index ?? index) === app.take);
    const player = $("result-video");
    if (current) {
      if (player.dataset.path !== current.path) { player.src = current.url; player.dataset.path = current.path; }
    } else if (player.dataset.path) {
      player.removeAttribute("src");
      delete player.dataset.path;
      player.load();
    }
    player.hidden = !current;
    $("stage-empty").hidden = Boolean(current);
    $("empty-title").textContent = run?.active ? `Rendering video ${app.take + 1}` : "Nothing generated yet";
    $("empty-line").textContent = run?.active ? run.status : "Write a prompt on the right, then press Generate video.";
    $("viewer-line").textContent = current ? `${videos.length} ${videos.length === 1 ? "video" : "videos"}` : "No result yet";
    $("result-name").textContent = current?.name || "";
    renderReview(run?.active ? null : current);
    $("download").hidden = !current;
    if (current) { $("download").href = `${current.url}&download=true`; $("download").download = current.name; }
    const used = app.s.results.settings || "Settings used will appear with the generated result.";
    if ($("settings-used").dataset.html !== used) { $("settings-used").innerHTML = used; $("settings-used").dataset.html = used; }
  };

  const HELP = {
    "Text to video": "Prompt-only generation with native stereo audio.",
    "First / last frame": "Upload a first frame, a last frame, or both.",
    "Reference media": "Tag references in the prompt: <Picture 1–9>, <Video 1–3>, <Audio 1–3>.",
  };
  const render = () => {
    if (!app.s) return;
    const mode = v("mode");
    seg($("mode"), app.boot.choices.mode, mode, (value) => change("mode", {mode: value}));
    $("mode-help").textContent = HELP[mode] || "";
    $("references").hidden = mode !== "Reference media";
    $("frames").hidden = mode !== "First / last frame";
    if (mode === "Reference media") renderReferences();
    if (mode === "First / last frame") renderFrames();
    // The card is the settings policy's own rendering, so the two can't disagree.
    const summary = app.s.summary.replaceAll("Size, seed &amp; canvas", "Size &amp; seed");
    if ($("summary").dataset.html !== summary) { $("summary").innerHTML = summary; $("summary").dataset.html = summary; }
    renderPrompt();
    renderOutput();
    renderReadiness();
    renderViewer();
    // Typing in a Fine-tune box must not lose the box to a redraw from another answer.
    if (!$("fine-rows").contains(document.activeElement) || document.activeElement.tagName !== "INPUT") renderFine();
    queueFit();
  };

  // Frame and theme, as on the other pages ----------------------------------
  const fitFrame = () => {
    const frame = window.frameElement;
    if (!frame) return;
    const height = Math.max(860, document.body.scrollHeight + 24);
    if (Math.abs(frame.offsetHeight - height) > 4) frame.style.height = `${height}px`;
  };
  let fitQueued = false;
  const queueFit = () => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fitFrame(); });
  };
  const applyTheme = () => {
    let dark = matchMedia("(prefers-color-scheme: dark)").matches;
    try {
      if (window.parent !== window) dark = window.parent.document.body.classList.contains("dark");
    } catch { /* Framed by another origin. */ }
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  };

  const bind = () => {
    $("file-input").addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (file && app.picking) upload(app.picking, file);
    });
    $("prompt").addEventListener("input", promptEdited);
    $("empty").addEventListener("click", guard(async () => {
      clearTimeout(promptTimer);
      await pendingSave;
      const answer = await post("/empty");
      app.suggestion = null;
      app.previous = null;
      app.enhanceContext = null;
      app.enhanceNote = "";
      $("prompt").value = "";
      $("file-input").value = "";
      take(answer);
    }));
    $("prompt").addEventListener("blur", savePrompt);
    $("duration").addEventListener("input", () => { $("duration-value").textContent = `${Number($("duration").value).toFixed(1)} s`; });
    $("duration").addEventListener("change", () => change("duration", {duration: Number($("duration").value)}));
    $("run").addEventListener("click", () => { app.progress = []; app.take = 0; startRun("/generate"); });
    $("stop").addEventListener("click", () => interrupt("generate"));
    $("refresh").addEventListener("click", guard(async () => { take(await request("GET", "/run?state=true")); toast("Status refreshed."); }));
    $("restore").addEventListener("click", () => change("restore"));
    $("fine-toggle").addEventListener("click", () => { saveUi({fine: !app.s.ui.fine}); render(); });
    $("fine-close").addEventListener("click", () => { saveUi({fine: false}); render(); });
    $("open-writer").addEventListener("click", () => openGroup(2));
    $("open-model").addEventListener("click", () => openGroup(0));
    $("open-size").addEventListener("click", () => openGroup(1));
    $("confirm-references").addEventListener("click", guard(async () => {
      await savePrompt();
      take(await post("/references/confirm"));
    }));
    $("enhance").addEventListener("click", guard(async () => {
      app.suggestion = null;
      app.enhanceNote = "";
      await savePrompt();
      app.enhanceContext = promptContext();
      take(await post("/enhance"));
      if (!app.runs.enhance?.active) enhanceFinished(app.runs.enhance);
    }));
    $("accept").addEventListener("click", guard(async () => {
      const proposal = app.suggestion;
      app.suggestion = null;
      if ($("prompt").value !== proposal.original || promptContext() !== proposal.context) {
        app.enhanceNote = "The prompt or input media changed. Enhance the current draft again.";
        return render();
      }
      app.previous = proposal.original;
      app.enhanceNote = "Suggested prompt accepted.";
      $("prompt").value = proposal.text;
      app.s.prompt = proposal.text;
      return savePrompt();
    }));
    $("keep").addEventListener("click", () => { app.suggestion = null; app.enhanceNote = "Original prompt kept."; render(); });
    $("undo").addEventListener("click", () => {
      $("prompt").value = app.previous;
      app.s.prompt = app.previous;
      app.previous = null;
      app.enhanceNote = "";
      savePrompt();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey) || event.repeat) return;
      event.preventDefault();
      if (!$("run").disabled) $("run").click();
    });
  };

  const start = async () => {
    applyTheme();
    if (window.frameElement) document.documentElement.dataset.framed = "true";
    try {
      new MutationObserver(applyTheme).observe(window.parent.document.body, {attributes: true, attributeFilter: ["class"]});
    } catch { /* Framed by another origin. */ }
    new MutationObserver(queueFit).observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["hidden"]});
    document.addEventListener("toggle", queueFit, true);
    window.addEventListener("resize", queueFit);
    try {
      app.boot = await request("GET", "/boot");
      app.input.model = app.boot.default_seedvr2 || app.boot.choices.seedvr2_model[0] || "";
      bind();
      $("workshop-note").hidden = app.boot.gpu;
      $("loading").hidden = true;
      $("editor").hidden = false;
      take(app.boot);
    } catch (error) {
      $("loading").textContent = `MiniMax H3 couldn't open: ${error.message}`;
    }
  };
  start();
})();
