# Drama 3 speech studio

## Compose and generate

1. Enter a draft or a tagged script in **Speech text**. Use `<|speaker:0|>`
   through `<|speaker:15|>` to assign voices and cues such as `[laughs]` for delivery.
2. Choose **Basic** for dialogue or **Advanced** (the default) for sound effects,
   music and vocal effects. Basic keeps those tags in the script but ignores them.
3. Use **Auto: rewrite, voice, sound and speak** for the full workflow, or open
   **Rewrite as a Drama 3 script** and review the result before proceeding.
4. In the **Cast** column, use **Speaker voices** to choose a saved group, manage presets,
   upload clips, or generate a cast. Every speaker in the script needs a voice. The panel title shows the selected group and how many
   speakers it holds. In Advanced mode, review **Sound effects and music** too.
5. In the **Speak** column, choose a **Speech preset** (**Expressive** by default).
   **Generate speech** renders the current script and references without rewriting them.

Script and Cast stack on the left with Speak beside them; Speak stays in view while
you scroll. **Auto** sits under the speech text.

Under the player, the result's lines scroll as subtitles: the line being spoken is
highlighted as it plays and follows the player when you seek, and clicking a line
jumps to it. Bracketed cues and tags are left out. Results made before this
feature have no line timings and show no subtitles.

### Takes

Every generated line is saved as a take, keyed by its text, speaker, voice clip
and transcript, and sampling settings. Repeated identical lines have separate
takes, distinguished by their occurrence order. **Generate speech** reuses saved takes, so
after editing a script only changed lines are generated; when nothing changed, the
mix is rebuilt without starting SGLang-Omni. After a result, **Lines** appears under
the studio: a table of every line with its length and whether it was reused. Click
a row (or choose a line) to hear that line's take, before staging and effects. Click
**Regenerate line** for a new take of just that line, or **Remix** to rebuild the
mix from saved takes after changing sounds, staging or vocal effects (which are
applied at mix time). Takes live in the private workspace (`drama3_takes`) and
are removed after 30 days unused.

Drama 3 runs on SGLang-Omni, which requires Linux and a CUDA GPU.
**Speech settings** contains parallel requests, retries and custom sampling controls.
Auto respects the skip-existing voice and sound options; clear them when you want
new clips. Check **Save generated speakers to the selected group** (off by default)
to have Auto save the cast after the voices are generated, as **Save all speakers**
does. Interrupt controls stop their corresponding jobs.

## Script writer and cast

To use a chat model outside the app, open **Use another chat model** under
**Rewrite as a Drama 3 script**, click **Show full writer prompt**, and copy it.
It contains the system prompt, the current speaker slots, and the speech text.
Paste the model's JSON reply (code fences and surrounding chat text are fine)
and click **Apply pasted reply**; it is validated like the built-in writer
before the script and speaker slots change.

The script writer returns a Drama 3 script plus an indexed cast. It separates
narration and dialogue, resolves character aliases, preserves supplied dialogue
and story order, and fills the corresponding speaker slots with names, voice
descriptions, and preview speech. Each slot keeps the exact number used in
`<|speaker:N|>`; existing references and sparse IDs are preserved. Invalid or
inconsistent writer responses leave the current script and slots in place.

In **Speaker voices**, each speaker has a collapsible panel titled with its
index, name, and whether a voice is ready (for example `Speaker 2 · Mia · voice
ready`). Upload and transcribe a reference clip, or edit the voice description
and preview speech and click **Generate voice**. Its synthesis text becomes the
reference transcript automatically, and each panel shows its own progress and
errors. The **Speaker voices** title counts the shown speakers and how many
have a voice. **Generate all speaker voices** designs every shown speaker in one
job; it skips saved speakers and speakers without a description and preview
speech. **Skip speakers with a voice** (on by default) also skips speakers that
already have a clip; uncheck it to regenerate them, for example after changing
the voice generator. With Drama 3, the SGLang-Omni server is prepared once and up
to **Parallel requests** voices generate at the same time; a failed voice does not
stop the others. Qwen runs the voices one after another. **Voice generator** picks the
engine:

- **Qwen3-TTS VoiceDesign** (default) uses
  [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
  and also takes a preview language. Browsers that saved the previous default
  (Drama 3) switch to it once; choosing Drama 3 again is remembered.
- **Drama 3** generates one take with no reference on SGLang-Omni, with the
  sampling settings under **Speech settings**. The description is
  inserted as a delivery cue after the preview's opening phrase, for example
  `<|speaker:0|> Well now, [gravelly old sailor, slow and amused] the tide is turning.`,
  so it must not contain brackets. The cue is not part of the transcript.

Preview speech is separate from the current clip's transcript: editing a future
preview does not change the words paired with an existing clip. Uploading a new
clip clears the previous transcript. **Save speaker** stores either kind of clip
with its exact transcript, name, and description; older presets still load.

Use **Speaker preset group** inside **Speaker voices** to select or create a story/series
group and reuse it across chapters. Rewrites and newly named speaker slots select
matching saved voices from that group first, then from **Global**. Existing global
presets remain available, including a shared narrator; a group's narrator overrides
the global match. The writer uses the canonical name `narrator`, and matching also
recognizes legacy names such as `旁白` and `story teller`. **Save all speakers to
selected group** saves every shown speaker with a clip and exact transcript in one
action, including copies of selected presets from other groups. Saving the same
name again replaces that group's preset; other groups are unaffected.

**Manage saved speakers** opens the library inside **Speaker voices**.
Its list follows the selected group and updates as soon as a speaker is saved
or deleted from a slot or with **Save all speakers to selected group**, keeping
the current selection when it still exists. Search the current group's speakers, listen to a saved clip, edit its name,
description or exact transcript, copy it into another group, or delete it.
Renaming or copying refuses to overwrite a different saved speaker. Renames and
deletions also update any currently selected voice slots.
**Delete group and all its speakers** removes the selected custom group together
with its saved clips and presets, then switches back to Global. Copies saved in
other groups are unaffected. Global itself cannot be deleted.

VoiceDesign and Stable Audio 3 Medium sound generation require a CUDA GPU. They
share one separate Python 3.12 environment with `qwen-tts==0.1.1` and its pinned
Transformers stack, a pinned `stable-audio-tools`, `hf_xet`, and a prebuilt
FlashAttention 2.8.3 wheel; ComfyUI's packages are unchanged. Local installs
prepare it on first use. Set `H3_AUDIO_TOOLS_VENV` to override the default
`h3/audio-tools-venv` under the app directory. Modal bakes that environment into
the image, so existing deployments need rebuilding.

Weights download once over hf_xet: VoiceDesign snapshots cache under
`ComfyUI/models/qwen3_tts`, and Stable Audio 3 Medium (gated; set a Hugging Face
token) with its bundled T5Gemma encoder lands in `ComfyUI/models/stable_audio3/medium`
(both on Modal's models volume). One worker serves both models with
`HF_HUB_OFFLINE=1`; each model loads on its first request and stays in VRAM
beside Drama 3. ComfyUI and SGLang are released only when the GPU lacks room for a
model's first load. Other engines' jobs, **Unload**, and an SGLang start or wake
without enough free VRAM stop the worker. A slot's
**Interrupt voice generation** button cancels the work and stops the worker. Clips
are written under `ComfyUI/output/drama3_voices` and `ComfyUI/output/drama3_sounds`;
worker and setup logs under `ComfyUI/output/audio_tools`.

Drama 3 generates parallel, speaker-based chunks on SGLang-Omni. It has no
per-request seed. SGLang uses an isolated environment, stays loaded
between TTS jobs, and sleeps or stops when another engine needs the GPU. The app defaults
to 8 parallel chunks and no automatic retries; the server allows 32 concurrent
requests. Chunk quality checks and advisories are skipped when retries is zero;
enabling retries also enables those checks. Separately, a dropped connection or
HTTP 5xx from the server is retried twice per request. Missing `<sound>` clips
are generated before any speech, so a sound failure stops the job early. A
failed chunk or **Interrupt** returns at once without waiting for requests still
running on the server. Completed speech is saved as MP3.

Each take is trimmed at its edges and long silences inside it are shortened,
because the model sometimes generates spurious gaps; use `<#N#>` pause tags for
intended pauses. **Balance** uses uncapped top-k; **Strict** and **Expressive**
use top-k 30. Saved custom sampling settings remain respected. Zero top-k is sent
as SGLang's whole-vocabulary value. Dependency revisions live in `h3_sources.py`
and `h3_requirements.py` rather than in this guide.

SGLang configuration uses these environment variables:

| Variable | Default |
| --- | --- |
| `H3_DRAMA3_SGLANG_ENGINE_MEMORY` | `0.55` of GPU memory |
| `H3_DRAMA3_SGLANG_CODEC_MEMORY` | `0.08` of GPU memory |
| `H3_DRAMA3_SGLANG_MAX_SEQS` | `32` |
| `H3_DRAMA3_SGLANG_PORT` | `8096` |
| `H3_DRAMA3_SGLANG_VENV` | `h3/drama3-sglang-venv` under the app directory |
| `H3_DRAMA3_SGLANG_DATA` | `drama3-serving` beside ComfyUI |
| `H3_DRAMA3_SGLANG_MODELS` | `ComfyUI/models/drama3_serving` |
| `H3_DRAMA3_SGLANG_SLEEP` | `auto`; use `off` for stop/restart only |

Previously downloaded copies of the pinned checkpoint and existing SGLang
kernel caches in sibling serving directories are reused when available.
To reduce graph capture work on startup, the launcher disables prefill CUDA
graphs and streaming vocoder window graphs, and caps first-frame graph coverage
at 8 requests. Decode CUDA graphs remain enabled; the server still allows 32
concurrent requests by default. Prefill runs eagerly, which may increase time
to first audio.

When another engine needs the GPU, SGLang sleeps instead of stopping: NVIDIA
CUDA process checkpointing moves every Omni worker's GPU state to host RAM while
the processes stay alive, and the next speech job unloads ComfyUI and restores
them. On Modal's RTX PRO 6000 this freed 73 GiB in 8.5s and woke in 5.8s,
versus a 156s cold start. Sleeping state does not survive app/container exit,
and **Unload all models** discards it.

Sleep requires **Linux x86_64 and NVIDIA driver 610+** (for Omni's cross-process
CUDA IPC). The driver belongs to the host, so where it is older (as on Colab
at the time of writing) the app logs the detected version and uses
stop/restart. `run_h3.sh` (also used by Colab) prints an `SGLang sleep:` line
at startup saying which mode applies. The app downloads a SHA-256-verified, commit-pinned NVIDIA
[`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint/tree/00d5cce84c628088d6caa203fc4af40c1538b6f7)
helper into the persistent cache and launches Omni in a fresh checkpoint job.
Workers are found through GPU device handles in their `/proc` process group,
since Modal's gVisor attributes all GPU usage to PID 1 in NVML. Before sleeping
the app requires free host/container RAM (cgroup limits included) of the
server's GPU usage plus the greater of 10% or 2 GiB; afterwards it requires
every worker to report `checkpointed` and device memory to return near the
pre-launch baseline. Wake requires every worker to report `running` and the
server to pass a health check.

Otherwise the app stops SGLang as before and logs the reason. Missing RAM
headroom is retried on the next launch; a failed driver transition disables
sleep for the rest of the app session. Set `H3_DRAMA3_SGLANG_SLEEP=off` to opt
out. `modal run tests/modal_drama3_checkpoint.py` is a manual model-free probe
that checks three sleep/wake cycles with CUDA graph replay and IPC buffers on a
temporary Modal GPU. In the server log, `asleep after` / `awake after` confirm
a sleep/wake cycle; `Sleep unavailable` means an ordinary restart.

