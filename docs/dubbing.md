# Dub & Translate

**Dub & Translate** translates the speech in a video or recording and dubs it
with each speaker's own cloned voice. Open **Create** and choose the **Dub & Translate** task, after
Soundstage. It works in four steps.

## 1 · Source

Upload a video or audio file, or pick one from **Or pick from Media**. To use an
online video, first download it in **Media → Download from a URL** (see below).

**Separate vocals from the background** (on by default) splits the soundtrack
with audio-separator. The music and effects then stay under the dub, and voice
references are clean. Models: Fast (UVR-MDX-NET), Balanced (BS-Roformer, the
default) and Quality (Mel-Band-Roformer). With separation off, the original
soundtrack is ducked under the dub, so the original voices remain faintly
audible.

## 2 · Transcribe and translate

- **Local: Qwen3-ASR + aligner + Hy-MT** runs entirely on this server. OmniVAD
  finds speech, and pauses shorter than **Join pauses shorter than** become one
  span. Qwen3-ASR 1.7B transcribes each span. Qwen3-ForcedAligner 0.6B times
  every word, so long speech is cut into subtitle-sized lines at sentence ends.
  Without word timing (aligner off, or a language it does not support), long
  spans are split at sentence ends and timed by their share of the text.
  Hy-MT2 1.8B then translates the lines. Local transcription does not separate
  speakers: every line is `speaker1`, and each line is voiced from its own audio.
- **Gemini (cloud)** uploads the vocals (or the soundtrack) once. Gemini
  returns timed lines with speaker labels and their translations. The key comes
  from `GEMINI_API_KEY`, or from **Gemini API key for this session**.

**Translator** chooses who translates lines later, in **Translate again**.
**Translation notes** pass names, terms and tone to the translator. Turn off
**Translate while transcribing** to transcribe only.

The first local run installs the isolated tools environment (see
[Runtime](#runtime)) and downloads the models into
`ComfyUI/models/dubbing`.

## 3 · Review lines

Every line has Start, End, Speaker, Original, Translation and **Dub**. Edits
save to the session as you type; when an edit changes the order (a new Start
time) or removes a row, the table is renumbered to match. Edits made while a
job runs are kept: jobs write only their own results. Untick **Dub** to keep a
line's original voice. With separation, every line that is not voiced (unticked,
untranslated or failed) keeps its original voice; without stems the original
soundtrack plays under every line anyway.

- **Translate again** retranslates the listed lines (for example `3, 5-7`).
  With the box empty, it fills only lines that have no translation.
- **Preview line** voices the first listed line, fits it to its timing and plays
  it.

## 4 · Dub

- **Voice**: *Each line's own voice* clones every line from its original audio,
  which keeps each line's delivery. *One voice per speaker* uses each speaker's
  best clip (the longest one between 3 and 15 seconds), which is steadier.
  Lines shorter than 2 seconds always use their speaker's clip.
- **Speech preset**: Strict, Balance (default) or Expressive Drama 3 sampling.
- **Maximum speed-up**: a take longer than its slot (until the next line starts)
  is sped up, pitch preserved, by at most this factor. Takes that still overrun
  are listed under **Check** in the status.
- **Dub volume** and **Background volume** set the mix.
- **Keep the original audio as a second track** and **Burn translated subtitles
  into the video** shape the video. Burning re-encodes it with x264; selectable
  translated and original subtitle tracks are always added.
- **Fresh takes for lines** generates new takes for those lines even when they
  are unchanged.

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
- The tools worker stays loaded between steps of a dubbing session. Any other
  GPU job stops it, and the dub's speech step stops it before Drama 3 starts.
  When VRAM is short, loading a tool first unloads the worker's other models,
  then frees ComfyUI, the VoiceDesign worker and a loaded Drama 3 server.
- Analyze, line actions (Translate again, Preview line) and Generate dub are
  separate job families: each **Interrupt** stops only its own step. A
  download's Interrupt stops yt-dlp at its next progress update.
- Speech uses Drama 3 on SGLang-Omni, so dubbing requires Linux, like Soundstage.
  Transcription, translation and downloads have no such limit.
- Sessions (source copy, soundtrack and stems, references, lines) live in
  `.h3-workspace/dubbing/<session>`. Sessions untouched for 14 days are removed
  when a new one starts.
