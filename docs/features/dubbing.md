# Dub & Translate

**Dub & Translate** translates the speech in a video or recording and dubs it
with each speaker's own cloned voice. Open **Create** and choose the **Dub & Translate** task, after
Soundstage. It works in three steps, shown as a step strip: **Source**, **Review
lines** and **Dub**. A finished analysis opens Review lines.

## 1 · Source

Pick the source in one of three tabs: **Upload** a video or audio file, choose
one **From Media** (an upload and a Media pick replace each other), or open an
**Earlier session**. The card below shows the picked file's kind, length and
size. To use an online video, first download it in **Media → Download from a
URL** (see below). Choose the **Spoken language**, the language to **Dub into**
(any language can be typed), and any **Translation notes**.

**Earlier session** lists analyzed sessions; opening one restores its edited
lines and dub language, for example after reloading the page, and opens Review
lines.

**More options** holds the settings below; its title sums them up while it is
closed.

**Separate vocals from the background** (on by default) splits the soundtrack
with audio-separator. The music and effects then stay under the dub, and voice
references are clean. Models: Fast (UVR-MDX-NET), Balanced (BS-Roformer, the
default) and Quality (Mel-Band-Roformer). With separation off, the original
soundtrack is ducked under the dub, so the original voices remain faintly
audible.

**How to find the lines** offers three transcriptions as cards:

- **Qwen3-ASR on this server** runs entirely on this server. OmniVAD
  finds speech, and pauses shorter than **Join pauses shorter than** become one
  span. Qwen3-ASR 1.7B transcribes each span. Qwen3-ForcedAligner 0.6B times
  every word, so long speech is cut into subtitle-sized lines at sentence ends.
  Without word timing (aligner off, or a language it does not support), long
  spans are split at sentence ends and timed by their share of the text.
  Hy-MT2 1.8B then translates the lines. Local transcription does not separate
  speakers: every line is `speaker1`, and each line is voiced from its own audio.
- **MOSS on this server** tells speakers apart on this
  server, so each speaker keeps their own voice and gets the per-speaker
  settings below. MOSS-Transcribe-Diarize transcribes the whole recording (up to
  90 minutes) in one pass with speaker labels and timestamps; long turns are
  split into subtitle-sized lines at sentence ends, and Hy-MT2 translates them.
  It runs on SGLang-Omni, so it needs Linux, like Soundstage (see
  [Runtime](#runtime)).
- **Nemotron on this server** is for complex conversations where people talk
  over each other. Nemotron 3.5 Transcribe (NVFP4, Blackwell GPUs such as the RTX
  PRO 6000) transcribes up to 60 minutes in one pass and tags each word with its
  speaker and time. Lines are cut from one speaker's consecutive words, so two
  people talking at once stay two lines at the same time. The words and when
  each speaker talks are kept with the session; they set the Nearest clean voice
  default below and word subtitles for unedited lines. Needs Linux.
- **Gemini (cloud)** uploads the vocals (or the soundtrack) once. Gemini
  returns timed lines with speaker labels and their translations. The key comes
  from `GEMINI_API_KEY`, or from **Gemini API key for this session**.

**Translator** chooses who translates lines Gemini left empty, and lines later
in **Translate again**. **Translation notes** pass names, terms and tone to the
translator. Turn off **Translate now** to transcribe only.

**Refine speech boundaries with Align (experimental)** adds
[Desert Ant Labs Align](https://desertant.com/models/align/) after local
Qwen transcription. It is on by default; turn it off to compare the same source
without refinement: **Qwen3-ASR → Qwen3-ForcedAligner → Align**. ASR and forced
alignment run on VAD speech chunks, not the full recording. Align receives each
chunk's word timings and a matching audio window with context on either side.
VAD chunks remain pause-based, with no hard duration cap on uninterrupted speech.
MOSS keeps its existing transcription and speaker timing pipeline.

Align adjusts acoustic word timestamps. Line cuts use the first word's start
and the last word's end, keeping the existing text and speaker labels. These
cuts feed both subtitles and voice-reference extraction; reference clips still
include their small safety padding. Refinement cannot guarantee exact boundaries
and does not correct a wrong transcript or speaker assignment.

The published Node native runtime supports Linux x64/arm64 and Apple silicon;
Windows requires Linux/WSL for this option. Install Node.js 22+ with npm on the
server. On first use the app installs the pinned Align 4.2.1 SDK under
`ComfyUI/models/dubbing/align-runtime`; model downloads use `align-models` beside
it. The SDK's license reporting remains enabled. Align supports English, Chinese,
Japanese, Korean, French, German, Spanish, Portuguese and Italian. Unsupported
languages, missing timing, runtime failures or invalid corrections keep the
original line cuts, with a status message. Numbers and mixed-language speech
need particular attention during listening tests.

The session's saved `transcript.refinement` contains the original segments,
original word timestamps, accepted output word timestamps and fallback reasons.
Refinement settings are part of the transcript cache key. Untick **Reuse earlier
results** to rerun a failed refinement. Align uses the
[Desert Ant Labs source-available license](https://license.desertant.com/1.0).
MOSS and Gemini are unaffected by this option.

The **Analyze** card on the right lists the steps the analysis will take, such
as separating vocals, transcribing and translating, and marks the one running.

The first local run installs the isolated tools environment (see
[Runtime](#runtime)) and downloads the models into
`ComfyUI/models/dubbing`.

## 2 · Review lines

The board lists every line with its time, speaker, original and translation.
Its summary counts the lines ready to dub, the speakers, the lines still
untranslated and those that may be too long to say in their time (Drama 3's
estimate of their spoken length needs more than 1.6x speed). Lines are flagged
*Not translated*, *May be too long* or *Original voice*; the filters show all
lines, those that need attention, the untranslated ones or one speaker's. Above
the lines, a timeline has a row per speaker: striped lines need attention and
faded ones keep their original voice. Click a line, on the board or the
timeline, to edit it in the inspector beside it.

The inspector edits the focused line: its Start, End, Speaker (pick one or type
a new name), Original, Translation and **Dub this line**. A text field saves
when you leave it, times when you leave them or press Enter; a line given a new
start moves to its place, and the inspector follows it. Under the fields, a note
says whether the translation fits the line's time. ‹ and › move to the previous
and next line; **Add a line after** and **Delete line** change the list. Each
line's switch on the board turns Dub on or off. Edits made while a job runs are
kept: jobs write only their own results. With separation, every line that is not
voiced (switched off, untranslated or failed) keeps its original voice; without
stems the original soundtrack plays under every line anyway.

**Edit as a table** shows every line at once for bulk edits: changes save as you
type, and when an edit changes the order (a new Start time) or removes a row, the
table is renumbered to match. A new row is saved once it has a start, an end and
text.

Tick lines on the board to pick them for the line actions; without picks they
use the focused line:

- **Listen to original** plays the first picked line (or the focused line) from
  the original soundtrack at its current Start and End, without padding or voice
  generation. Its player is separate from the dub preview for comparison.
- **Preview** voices the first picked line, fits it to its timing and plays it.
  It reuses the line's saved take.
- **New take** records the line again; the dub then uses the new take.
- **Translate again** retranslates the picked lines. With none picked, it fills
  lines that have no translation, or translates every line when **Dub into**
  has changed since the last translation.

## 3 · Dub

The **Dub it** card says what the dub will do: how many lines it voices, how
many keep their original voice or are untranslated, and which lines may sound
rushed.

**Speakers** appears for sessions transcribed with MOSS, Nemotron or Gemini, with a card
per speaker: give a speaker a saved Drama 3 voice instead of its cloned
original, or change its volume. Changes save at once.

- **Clone from**: *Each line's own voice* clones every line from its original audio,
  which keeps each line's delivery. *One voice per speaker* uses each speaker's
  best clip (the longest one between 3 and 15 seconds), which is steadier.
  Lines shorter than 2 seconds always use their speaker's clip.
  *Nearest clean voice* (the default for Nemotron sessions; the others start on
  *Each line's own voice*) clones a line from its own audio when nobody else
  talks during it and it lasts at least 2 seconds; otherwise from the same
  speaker's closest such line, before or after, anywhere in the recording, cut to
  15 seconds at a sentence when Nemotron's words allow. A speaker who never talks
  alone is cloned from the line's own audio, overlapping voices included. Who
  talks when comes from the lines as transcribed and as edited (and Nemotron's
  words and diarization), so a line moved to another speaker still counts as
  cross-talk. It needs accurate speaker labels, which is why other transcribers
  do not default to it.
- **Delivery**: Strict, Balance (default) or Expressive Drama 3 sampling.
- **Timing**:
  - *Smart timing* (default) keeps each take as Drama 3 spoke it whenever it
    fits, and otherwise tries, in order, until it does:
    1. playing into the silence after the line, up to 0.1 s before the next one;
    2. starting up to 0.3 s early, into silence the previous dubbed line left;
    3. shortening pauses inside the take longer than 0.25 s to 0.15 s;
    4. speeding up to 1.25x;
    5. pushing the next line up to 0.5 s later, out of time its own take does
       not need, then speeding up to 1.25x again;
    6. speeding up to **Maximum speed-up**. Such a line is marked **Too long**
       in Review lines and listed on the **Dub it** card. If it still does not
       fit, it overlaps the next line by what is left; raise Maximum speed-up
       to squeeze it in instead, or shorten its translation.

    Shorter takes stay as they are. Lines that keep their original voice are
    never moved, two speakers who overlap in the source may overlap in the dub,
    and no line starts more than 0.5 s late. Translated subtitles start and end
    with the dubbed line. Preview assumes the other lines fill their own time.
  - *Match the original* stretches every take with FFmpeg's
    `atempo`, pitch preserved, so it starts and ends exactly where the original
    line does, as index-tts-vllm's FFmpeg duration control does. Lines sped up
    more than 1.6x or slowed below 0.75x are listed under **Check**; a shorter or
    longer translation sounds more natural there.
  - *Natural pace* keeps each take's own pace. A take longer than its slot (until
    the next line starts) is sped up by at most **Maximum speed-up**. Takes that
    still overrun are listed under **Check**: they overlap the next line, and a
    last line runs past the end of the source rather than being cut. The audio
    gets longer; a re-encoded video (burned subtitles) holds its last frame.
- Translated subtitles stay on screen while a line is still being spoken, up to
  the next line. With Smart timing they also start when the dubbed line does.
- **Dub volume** and **Background volume** set the mix.
- **Word-by-word subtitles** (off by default) times every word of both subtitle
  tracks with Qwen3's forced aligner: translations against their fitted takes,
  sources against the original voice, cached per clip. The player's line list and
  captions light each word as it is said. To add it to a finished dub, turn it on
  and dub again: unchanged lines keep their takes. Lines the aligner cannot time
  keep plain subtitles.
- **Keep the original audio as a second track** adds the source soundtrack to
  the dubbed video. Generation copies the picture without re-encoding and adds
  selectable translated and original subtitle tracks.
- After generation, **Export video with burned-in translated subtitles** makes
  a new MP4 in Media using the completed dub's translated subtitles. It re-encodes
  the picture with x264 and copies all audio tracks, without generating voices
  again. The original dub and interactive player remain uncaptioned. The export
  holds the last frame when the dub runs longer than the source.
- **Advanced → Fresh takes for lines** generates new takes for those lines even
  when they are unchanged. **Parallel requests** sets how many lines Drama 3
  voices at once (16 by default).

### Failed takes

Each line gets up to two attempts. A take fails its check when it is silent,
holds far less speech than its text needs, or runs until the length cap; a
dropped connection or HTTP 5xx is retried twice before the attempt counts. When
both attempts fail, the line keeps the better of them. When Drama 3 rejects the
line itself (an HTTP 4xx, or audio it cannot return), the dub carries on, and that
line keeps its original voice. A server that is down still stops the dub.

Both kinds are marked **Take failed** in Review lines, with the reason on hover.
They count under **Needs attention** and get a **Failed takes** filter, and the
**Dub it** card lists their numbers. To fix them, use **New take** on a line, or
enter the numbers under **Fresh takes for lines**, then dub again: every other
line reuses its saved take. A mark clears when the line gets a take that passes,
and it no longer applies once the line's start or translation is edited.

The finished dub opens in a player beside the files. It plays the dub with
**Dub** or **Original** voice, **Background music** on or off, and subtitles in
the translation, the original, both or none. The lines below it follow playback;
click one to jump there. Translated subtitles use the dub's timing, original ones
the source's. Without vocal separation the music cannot be switched off: the
player switches between the finished mix and the original soundtrack. Its
separate tracks are kept in `gradio_outputs/dubbing/.dub-player`, which Media
does not list, and are removed once their dub is deleted from Media.

The player's tracks and the MP3 mix are constant-bitrate MP3s: browsers seek a
long variable-bitrate MP3 by estimate and land seconds away from the time they
report, which put the dub seconds off its subtitles after every jump. The
tracks load only their headers until played, the picture and music pause while
the voice buffers, and they are re-synced only when more than 0.45 s off, so a
slow connection is not flooded with fresh seeks of the video.

Every take is saved by its text, voice clip and preset. Dubbing again after a
few edits voices only the changed lines. Results go to Media
(`gradio_outputs/dubbing`): the MP4 when the source is a video, the MP3 mix, and
`.srt` files for the translation and the original. Each result records its
settings.

## Download from a URL

**Media → Download from a URL** downloads a video page (YouTube, Bilibili, X and
most sites yt-dlp supports) into `gradio_outputs/downloads`. The file appears in
the library when it is complete. **Check link** shows the title, length and
available heights. **Quality** caps the height; yt-dlp falls back to the next
available format.

Sign-in, members-only or age-restricted videos need cookies. In **Cookies for
sign-in downloads**, paste a request copied as cURL from your browser's
DevTools, or upload a Netscape `cookies.txt`. Cookies are saved per site in the
private workspace (`.h3-workspace/cookies`, never served). They are used only
for links to that site.

YouTube's signature challenge needs a JavaScript runtime. Setup installs Deno
from PyPI next to the app's Python. A Node, Deno or Bun on `PATH` also works,
and `YTDLP_JS_RUNTIMES` (for example `node` or `deno:/opt/deno`) overrides the
choice.

## Runtime

- Dubbing tools run in `h3/dub-tools-venv` (`H3_DUB_TOOLS_VENV`), separate from
  ComfyUI and from the VoiceDesign/Stable Audio environment. qwen-asr pins
  Transformers 4.57.6, and audio-separator needs NumPy 2. The environment is
  installed on first use with `uv`; Modal bakes it into the image. Logs go to
  `ComfyUI/output/dub_tools`.
- First-use setup reports whether it is creating the environment, installing
  packages or checking imports. Installer output appears in the backend console
  and in `setup_*.log` under that log directory; the UI shows its latest output.
  This shared Python setup precedes ASR and Align. A ready environment is reused
  without reinstalling packages.
- After setup, worker progress replaces the preparation status: loading audio,
  detecting speech, loading models, separating vocals and processing ASR or
  translation batches. These updates appear in the UI and backend console.
- The tools worker stays loaded between steps of a dubbing session. Any other
  GPU job stops it, and the dub's speech step stops it before Drama 3 starts.
  When VRAM is short, loading a tool first unloads the worker's other models,
  then frees ComfyUI, the VoiceDesign worker and a loaded Drama 3 server.
- Analyze, line actions (Preview, New take, Translate again) and Generate dub are
  separate job families: each **Interrupt** stops only its own step. A
  download's Interrupt stops yt-dlp at its next progress update.
- Speech uses Drama 3 on SGLang-Omni, so dubbing requires Linux, like Soundstage.
  Transcription with Qwen3-ASR or Gemini, translation and downloads have no such
  limit.
- MOSS runs as a second SGLang-Omni server from Drama 3's environment (the
  pinned SGLang-Omni release serves it), on port 8097 (`H3_MOSS_SGLANG_PORT`).
  The checkpoint downloads once into `ComfyUI/models/dubbing/moss`. It keeps
  about 19 GiB (`H3_MOSS_SGLANG_MEMORY_GIB`), index-tts-vllm's 0.20 of an empty
  RTX PRO 6000: room for one 90-minute request's full context, with one request
  at a time. SGLang's `--mem-fraction-static` is a share of the memory free when
  the server starts, not of the whole GPU, so each launch sets it from the free
  memory then (for example 0.41 beside an awake Drama 3). A launch needs about
  21 GiB free.
  After each analysis it sleeps, keeping its CUDA state in host RAM, and the next
  analysis wakes it in seconds; every other job sleeps it too, and unloading
  models stops it. `H3_MOSS_SGLANG_SLEEP=off` stops it instead of sleeping. When
  the GPU is short, the idle dubbing tools stop first, then Drama 3 sleeps.
- Nemotron runs on vLLM from NeMo Speech's locked environment (Python 3.13,
  pinned in `h3_requirements.py`), installed on first use into
  `h3/nemotron-venv` (`H3_NEMOTRON_VENV`) with its editable checkout beside it,
  or baked into the Modal image. It serves on port 8098 (`H3_NEMOTRON_PORT`)
  with NVIDIA's recipe minus speculative decoding, one request at a time, and
  about 40 GiB (`H3_NEMOTRON_MEMORY_GIB`). The ~20 GB checkpoint downloads once
  into `ComfyUI/models/dubbing/nemotron`. It starts for an analysis and stops as
  soon as the words are back. The first start compiles FlashInfer's NVFP4
  kernels with CUDA 13's nvcc. A reply without valid word timings, or cut off,
  fails the analysis so it can be retried.
- Sessions (source copy, soundtrack and stems, references, lines) live in
  `.h3-workspace/dubbing/<session>`. Sessions untouched for 14 days are removed
  when a new one starts.
