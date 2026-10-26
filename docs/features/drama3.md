# Soundstage

Soundstage is Prism Studio's audio production workspace: write and direct
scripts, cast voices, stage sound effects and music, review individual takes,
and remix the result. Speech generation is powered by Fish Audio Drama 3.
Open **Create → Soundstage** to begin.

## Compose and generate

Soundstage works in four steps, laid out like Song and Dub & Translate: **Script**,
**Cast & sounds**, **Speak** and **Review lines**. Each step pairs a working card
with an action card that holds its next move; the action card stays in view while
you scroll. Moving between steps happens in the browser and sends no request.

1. **Script.** Enter a draft or a script in **Speech text**. Start each line with a
   speaker's name from the cast, such as `Mia: Hello.` (a full-width `：` works too),
   and add cues such as `[laughs]` for delivery. `Speaker 2:` means cast slot 2
   whatever its name. Text before the first named line, or prose that names nobody,
   is read by the first speaker with a voice. In a script that names its speakers, a
   line starting with a name not in the cast is an error, so a typo is not read aloud.
   Older scripts with `<|speaker:N|>` tags still work. Choose **Basic** for dialogue
   or **Advanced** (the default) for sound effects, music and vocal effects; Basic
   keeps those tags in the script but ignores them. **Rewrite script** turns any text
   into a Drama 3 script and fills the cast; review the result before moving on.
2. **Cast & sounds.** Use **Speaker voices** to choose a saved group, manage presets,
   upload clips, or generate a cast. Every speaker in the script needs a voice. The
   panel title shows the selected group and how many speakers it holds. In Advanced
   mode, review **Sound effects and music** too. **Generate all speaker voices** makes
   every missing voice in one job.
3. **Speak.** Choose a **Speech preset** (**Expressive** by default). **Generate
   speech** renders the current script and references without rewriting them.
4. **Review lines.** Hear, regenerate or remix single lines (see **Takes** below).

**Auto: rewrite, voice, sound and speak** does steps 1–3 in one go. It lives on
Speak, and the Script step has a shortcut that opens Speak and starts it. When Auto
stops, its status names the step whose status says why.

Under the player, the result's lines scroll as subtitles: the line being spoken is
highlighted as it plays and follows the player when you seek, and clicking a line
jumps to it. Bracketed cues and tags are left out. Consecutive lines of one speaker
are spoken as one take, so each line's time inside the take is placed at the pauses
between them, or estimated from its length when the take has too few pauses. Results
made before this feature have no line timings and show no subtitles; Remix gives older
results per-line timings without generating anything.

### Takes

Every generated line is saved as a take, keyed by its text, speaker name, voice clip
and transcript, and sampling settings; renaming a speaker generates its lines again.
Drama 3 receives only each line's text with its speaker's clip and transcript, never
a speaker number or name. Repeated identical lines have separate
takes, distinguished by their occurrence order. **Generate speech** reuses saved takes, so
after editing a script only changed lines are generated; when nothing changed, the
mix is rebuilt without starting SGLang-Omni. After a result, the **Review lines** step
shows a table of every line with its length and whether it was reused. Click
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

While the writer rewrites the script, Auto loads the models it will need on the
GPU, in order: the voice design engine (Qwen VoiceDesign, or SGLang-Omni for
Drama 3 voice design), Stable Audio 3 Medium in Advanced mode, then SGLang-Omni.
Models that are already loaded are kept. The preload never unloads one model to
make room for another; a model that does not fit or fails to load is left to the
step that needs it, and the line under the Auto status says which.

## Script writer and cast

To use a chat model outside the app, open **Use another chat model** on the
Script step and click **Copy full writer prompt**; a running
rewrite keeps going. The prompt contains the system prompt, the current speaker slots, and the speech text.
Paste the model's JSON reply (code fences and surrounding chat text are fine)
and click **Apply pasted reply**; it is validated like the built-in writer
before the script and speaker slots change.

The script writer returns a Drama 3 script plus an indexed cast. It separates
narration and dialogue, resolves character aliases, preserves supplied dialogue
and story order, and fills the corresponding speaker slots with names, voice
descriptions, and preview speech. The script names its speakers as the cast
does; each cast entry keeps its slot, and existing references and sparse slots
are preserved. Invalid or
inconsistent writer responses leave the current script and slots in place.

The writer directs most lines, narration included, with Fish Audio cues and the
seven scoped tags, briefing each cue like a note to a voice actor (the feeling plus the
line's intent or who it is aimed at). It keeps a speaker's continuous passage in one
staging block, since consecutive lines of one speaker are generated as one take, and
pins a sound to a sentence mid-passage only when it must land on those words. Each new
voice's preview is written as a calm reference: two to four sentences, about 10–20
seconds, in the character's everyday mood. It also says whether it preserved supplied text or wrote from a
brief; a preserved source is always checked line by line, and missing phrases are
reported in the status. The app then repairs placements Drama 3 performs poorly: a
cue that opens a line moves after its opening words, ellipses become commas or
periods, pause and silence cues are removed, and cues beyond one per sentence and
two per line are dropped. The status reports each repair, lines that repeat the
previous line's cue, and a script that leaves most lines undirected.

When fewer than 60% of a script's lines (of three or more) carry direction,
**Rewrite** and **Auto** make one more call to the same chat model, a performance
director (`prompt_drama3_direction.txt`) that may only insert cues and scoped tags.
Each of its lines is accepted only when the speaker, vocal effects and every word
and punctuation mark are unchanged; other lines keep the writer's version, and a
reply that changes lines, sounds or staging is ignored. Pasted replies from another
chat model are not sent to the director.

A long turn is sent to Drama 3 in several requests, and each request starts
afresh, so a request without its own delivery cue repeats the turn's last one
(never an audible event such as [laughs]) after its opening words.

The writer's script is formatted for reading: each turn, sound, and
`<parallel>`/`<sequential>` tag sits on its own line, indented two spaces per
level of nesting. Indentation and blank lines do not change the performance.
**Speech text** colors the script in VS Code's Light+ or Dark+ palette. The
narrator's name has its own slate color; characters take eight further colors in
order of their first line, each on a faint tint so a name never reads as a tag, and
none of them is a tag color. [cues], `<parallel>`, `<sequential>`, sounds, effect
tags, and `<#seconds#>` pauses each have theirs.

The writer starts every voice description by naming the voice's sex ("A female
voice: ..."), since voice design otherwise guesses from roles such as "mother" and
sometimes guesses wrong; the status names any description that still does not say.

Chinese characters with several readings can carry a pronunciation note: tone-number
pinyin in braces right after the characters, one syllable per character, such as
`把书还{huan2}给他` or `尉迟{yu4 chi2}恭` (tone 5 is neutral; write ü as `v`). The
writer adds notes only where a reading is easy to get wrong: a character away from its
everyday reading with little context to settle it, names and titles with special
readings, and literary readings. It leaves ordinary words such as 银行 or 还是 alone. The
script, subtitles, takes and the writer's checks keep the characters; only the request
to Drama 3 replaces each noted character with Drama 3's phoneme tag
(`<|phoneme_start|>huan2<|phoneme_end|>`). A note must follow exactly as many Chinese
characters as it has syllables; a malformed note is an error, as is a raw `<|phoneme_…|>`
tag in **Speech text**. The director keeps the writer's notes, and **Speech text**
colors them like attribute values. Notes change a line's text, so a line takes a new take
when a note is added or changed.

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
  `Well now, [gravelly old sailor, slow and amused] the tide is turning.`,
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

**Manage saved speakers** shows or hides the library inside **Speaker voices**.
Its list follows the selected group and updates as soon as a speaker is saved
or deleted from a slot or with **Save all speakers to selected group**, keeping
the current selection when it still exists. Search the current group's speakers, listen to a saved clip, edit its name,
description or exact transcript, copy it into another group, or delete it.
Renaming or copying refuses to overwrite a different saved speaker. Renames and
deletions also update any currently selected voice slots.
**Delete group and all its speakers** removes the selected custom group together
with its saved clips and presets, then switches back to Global. Copies saved in
other groups are unaffected. Global itself cannot be deleted.

Scripted clients reach the same library through API-only endpoints, which have no
controls on the page:

| Endpoint | Parameters | Returns |
| --- | --- | --- |
| `/drama3_library_groups` | | every group, Global first |
| `/drama3_library_speakers` | `group` | the group's own speakers: name, group, key, description, exact transcript, clip SHA-256 |
| `/drama3_library_save_speaker` | `group`, `name`, `audio`, `transcript`, `description` | the saved entry; creates the group, replaces a same-named speaker |
| `/drama3_library_speaker_clip` | `group`, `name` | a copy of the clip and its entry |
| `/drama3_library_sounds` | | every saved sound: name, description, duration, clip SHA-256 |
| `/drama3_library_save_sound` | `name`, `audio`, `description`, `duration` | the saved entry |
| `/drama3_library_sound_clip` | `name` | a copy of the clip and its entry |
| `/drama3_result_details` | `result` (the file name `/generate_drama3` returned) | the result's settings: every take with its key, check (`ok` false for a take kept after every attempt failed) and line timings |
| `/drama3_result_audio` | `result` (a file name `/generate_drama3_batch` listed) | a copy of that result's audio |

Clips are saved byte for byte, so a client can compare digests to keep its own copy
in step. A take key from `/drama3_result_details` is the `line` argument of
`/regenerate_drama3_line`. Saved sounds have no groups.

Batch endpoints let a script keep the GPU busy with one queue job per stage instead
of one per item. Each is a single GPU-queue job; an invalid or failed item fails
alone:

| Endpoint | Parameters | Returns |
| --- | --- | --- |
| `/generate_drama3_voices_batch` | `briefs` (JSON list of `{name, description, preview_text[, language]}`), `language` | the clips and a listing `{name, ok, file, transcript, error}` per brief (Qwen VoiceDesign, one worker request) |
| `/generate_drama3_sounds_batch` | `sounds` (JSON list of `{name, description, duration}`) | the clips and a listing `{name, ok, file, error}` per sound (Stable Audio 3 Medium, one worker request) |
| `/generate_drama3_batch` | `jobs` (JSON list), `fix_rounds` (0-3, default 1) | yields `(listing, status)`; one listing entry `{id, ok, result, status, failed, rounds, error}` per finished job |
| `/preload_drama3_models` | `voice_engine`, `script_mode` | the preload status |

A `/generate_drama3_batch` job is `{id, text, title, script_mode, temperature,
top_p, top_k, max_frames, parallel, retries, speakers, sounds}`. Each speaker is
`{slot, name, group, voice}`: the saved speaker `voice` in `group` gives the clip
and exact transcript, and `name` is the Cast name the script's lines use. Each
sound is `{name}` of a saved sound. Nothing is uploaded, so a client saves its
voices and sounds once and renders a whole book from them. The jobs run back to
back; after each one, takes that still fail their check are generated again up to
`fix_rounds` times before the next job starts. Every finished job is listed as soon
as it is done, so a client can fetch its audio (`/drama3_result_audio`) and details
while the next job renders. An interrupt stops the batch.

**Try new voices** designs one to four fresh clips for the selected speaker from
its voice description and exact clip transcript, using the Cast's **Voice
generator** and the Speak sampling settings. The saved clip is untouched until you
pick one of the new voices and press **Use selected voice**, which saves it with the
description and transcript it was made from. Keep the saved voice or generate
another batch as often as you like; selecting another speaker discards the tries.

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
to 16 parallel chunks and zero retries; the server allows 32 concurrent
requests. More parallel chunks make each line slower but a long script faster,
since the GPU generates more frames per second in total. Browsers that saved
the previous default of 8 switch to 16 once. Every new take is checked, and one that clearly failed (silent, holding a
fifth or less of a long line, or stopped by its length cap) is generated once more
even with zero retries; more retries give such takes more attempts. Uncertain
duration and pause advisories are saved with the result and shown in the status only
when retries are enabled. Separately, a dropped connection or
HTTP 5xx from the server is retried twice per request. Missing `<sound>` clips
are generated before any speech, so a sound failure stops the job early. A
failed chunk or **Interrupt** returns at once without waiting for requests still
running on the server. Completed speech is saved as MP3.

The mix follows the script. Gaps between speakers follow the lines: a line cut off
with a dash is barged in on, a question or exclamation is answered quickly, and a
line opening slow, sad or hushed gets a longer breath. In Advanced mode, a block with
an ambience or music loop opens with it alone for 0.8 s; loops fade in, dip under
speech and fade out over a second under whatever follows, and one-shot effects and
stingers always play to their end. Each speaker's takes are evened out toward the
cast's typical level (by at most 6 dB, so whispers and shouts keep their contrast).
Speaker vocal effects use Spotify's `pedalboard`, which setup installs; without it
they fall back to simpler SciPy versions.

Results are stereo. The narrator stands in the centre; when a script has two or more
other characters, they stand apart in order of their first line (left 30%, right 30%,
then nearer the centre and further out), so a conversation is heard across the scene.
A narrator with a single character stays centred. Sound clips keep Stable Audio's
stereo image (clips generated before this were mono and play in the centre). In
Advanced mode a sound can be placed with `pan="-0.6"` (-1 left to 1 right), moved
during the clip with `pan="-0.8,0"`, and set `distance="far"` (muffled, quieter, with
room reflections) or `distance="near"`; the writer adds these where the story implies
a position. When a scene's voices share one place effect (`<reverb_small_room>`,
`<reverb_large_hall>`, `<underwater>` or `<dream_sequence>`), its one-shot effects
get the same room, so a knock and a voice sound like one space; ambience and music
loops do not. A loop repeats from the point near its end that best matches its
start, so its seams are crossfaded where they are least audible. The finished mix is brought to -16 LUFS with a look-ahead limiter holding
peaks under -1.5 dBFS, linked across both channels. Remix applies all of this to older
results too.

Each take is trimmed at its edges and long silences inside it are shortened,
because the model sometimes generates spurious gaps; use `<#N#>` pause tags for
intended pauses. **Balance** uses uncapped top-k; **Strict** and **Expressive**
use top-k 30. Saved custom sampling settings remain respected. Zero top-k is sent
as SGLang's whole-vocabulary value. Dependency revisions live in `h3_sources.py`
and `h3_requirements.py` rather than in this guide.

SGLang configuration uses these environment variables:

| Variable | Default |
| --- | --- |
| `H3_DRAMA3_SGLANG_CHECKPOINT` | `fp8` (official `fishaudio/drama-3`); `int4` for official `fishaudio/drama-3-int4` |
| `H3_DRAMA3_SGLANG_ENGINE_MEMORY` | the checkpoint's weights plus 12 GiB, as a share of GPU memory |
| `H3_DRAMA3_SGLANG_CODEC_MEMORY` | `0.08` of GPU memory |
| `H3_DRAMA3_SGLANG_MAX_SEQS` | `32` |
| `H3_DRAMA3_SGLANG_PORT` | `8096` |
| `H3_DRAMA3_SGLANG_VENV` | `h3/drama3-sglang-venv` under the app directory |
| `H3_DRAMA3_SGLANG_DATA` | `drama3-serving` beside ComfyUI |
| `H3_DRAMA3_SGLANG_MODELS` | `ComfyUI/models/drama3_serving` |
| `H3_DRAMA3_SGLANG_SLEEP` | `auto`; use `off` for stop/restart only |

The default checkpoint is the official FP8 release: a 41.5 GB download with 37 GiB
of engine weights. `H3_DRAMA3_SGLANG_CHECKPOINT=int4` switches to Fish Audio's official
4-bit GPTQ release, which downloads 24 GB and holds 20.8 GiB of weights. In SGLang-Omni's
benchmark it was about 8% slower per request at the same WER. Each checkpoint has its
own pinned revision and folder, and the Storage console can fetch either.

The engine's memory share is sized to its checkpoint: the weights plus 12 GiB for the
KV cache (20 KiB per token), linear-attention state (about 64 MiB per request), CUDA
graphs and buffers. On a 95.6 GiB RTX PRO 6000 that is 0.513 for FP8 (previously
0.55) and 0.343 for int4. A smaller share also shrinks what sleep moves to host RAM
and what a Modal memory snapshot holds. Set `H3_DRAMA3_SGLANG_ENGINE_MEMORY` to
override it. The startup breakdown lists the KV and state pools SGLang allocated,
and any line where SGLang capped running requests below `H3_DRAMA3_SGLANG_MAX_SEQS`.

Previously downloaded copies of the pinned checkpoint and existing SGLang
kernel caches in sibling serving directories are reused when available.
The environment is installed with precompiled bytecode, so a fresh container does
not compile torch and sglang in every stage process at startup; environments
installed before this rebuild once. After each start the server log (and the app
log) lists a startup breakdown: when the server first printed output, and
SGLang's own weight-load and CUDA graph capture timings, each stamped with its
time since launch. The stage processes start concurrently, so their timings
overlap.
The launcher keeps upstream's engine CUDA graphs: decode, breakable prefill
graphs for batches of up to 2048 tokens, and first-frame graphs for up to 16
requests. They shorten each line at the cost of a longer cold start and some GPU
memory; sleep keeps them across engine switches. Streaming vocoder window graphs
are disabled because the app requests complete WAVs.

When another engine needs the GPU, SGLang sleeps instead of stopping: NVIDIA
CUDA process checkpointing moves every Omni worker's GPU state to host RAM while
the processes stay alive, and the next speech job unloads ComfyUI and restores
them. On Modal's RTX PRO 6000 this freed 73 GiB in 8.5s and woke in 5.8s,
versus a 156s cold start. Sleeping state does not survive app/container exit,
and **Unload all models** discards it.

Sleep requires **Linux x86_64 and NVIDIA driver 580+**. Omni's stages normally
exchange tensors over cross-process CUDA IPC, which only driver 610+ can
checkpoint. On 580-609 (Colab's driver at the time of writing) the app launches
Omni with an H3 platform (`h3_app/drama3/serving/omni_platform/`, loaded through
Omni's `SGLANG_OMNI_PLATFORM_SPEC`) that routes those tensors through host shared
memory instead, so no IPC is left to checkpoint. The driver belongs to the host;
on older drivers the app logs the detected version and uses stop/restart.
`run_h3.sh` (also used by Colab) prints an `SGLang sleep:` line at startup saying
which mode and stage transport apply. The app downloads a SHA-256-verified, commit-pinned NVIDIA
[`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint/tree/00d5cce84c628088d6caa203fc4af40c1538b6f7)
helper into the persistent cache; on 610+ it launches Omni in a fresh checkpoint
job, as NVIDIA's IPC support requires.
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
temporary Modal GPU. On Colab, stop the app and run
`python3 -m tests.colab_drama3_sleep` from the workspace: it runs the real
server through three sleep/wake cycles with speech after each, checks that every
GPU worker uses the SHM platform on 580-609, and with `--compare-ipc` also times
the same speech over stock CUDA IPC. In the server log, `asleep after` /
`awake after` confirm a sleep/wake cycle; `Sleep unavailable` means an ordinary
restart.

