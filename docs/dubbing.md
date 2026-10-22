# Dub & Translate

**Dub & Translate** translates the speech in a video or recording and dubs it
with each speaker's own cloned voice. Open **Create** and choose the **Dub & Translate** task, after
Soundstage. It works in three steps, shown as a step strip: **Source**, **Review
lines** and **Dub**. A finished analysis opens Review lines.

## 1 · Source

Upload a video or audio file, or pick one from **Or pick from Media**; choosing
one clears the other. To use an online video, first download it in **Media →
Download from a URL** (see below). Choose the **Spoken language**, the language
to **Dub into**, and any **Translation notes**.

**Continue an earlier session** reopens an analyzed session, with its edited
lines and dub language, for example after reloading the page.

**Separate vocals from the background** (on by default) splits the soundtrack
with audio-separator. The music and effects then stay under the dub, and voice
references are clean. Models: Fast (UVR-MDX-NET), Balanced (BS-Roformer, the
default) and Quality (Mel-Band-Roformer). With separation off, the original
soundtrack is ducked under the dub, so the original voices remain faintly
audible.

The action card on the right, **Find and translate the lines**, chooses how:

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

**Translator** chooses who translates lines Gemini left empty, and lines later
in **Translate again**. **Translation notes** pass names, terms and tone to the
translator. Turn off **Translate now** to transcribe only.

The first local run installs the isolated tools environment (see
[Runtime](#runtime)) and downloads the models into
`ComfyUI/models/dubbing`.

## 2 · Review lines

Every line has Start, End, Speaker, Original, Translation and **Dub**. The
summary above the table counts the lines ready to dub, the speakers and the
lines still untranslated. Edits save to the session as you type; when an edit
changes the order (a new Start time) or removes a row, the table is renumbered
to match. A new row is saved once it has a start, an end and text. Edits made
while a job runs are kept: jobs write only their own results. Untick **Dub** to keep a
line's original voice. With separation, every line that is not voiced (unticked,
untranslated or failed) keeps its original voice; without stems the original
soundtrack plays under every line anyway.

Click a line to pick it for the line actions, or type lines such as `3, 5-7`:

- **Preview** voices the first picked line, fits it to its timing and plays it.
  It reuses the line's saved take.
- **New take** records the line again; the dub then uses the new take.
- **Translate again** retranslates the picked lines. With none picked, it fills
  lines that have no translation, or translates every line when **Dub into**
  has changed since the last translation.

## 3 · Dub

- **Voice**: *Each line's own voice* clones every line from its original audio,
  which keeps each line's delivery. *One voice per speaker* uses each speaker's
  best clip (the longest one between 3 and 15 seconds), which is steadier.
  Lines shorter than 2 seconds always use their speaker's clip.
- **Delivery**: Strict, Balance (default) or Expressive Drama 3 sampling.
- **Timing**:
  - *Match the original timing* (default) stretches every take with FFmpeg's
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
  the next line.
- **Dub volume** and **Background volume** set the mix.
- **Keep the original audio as a second track** and **Burn translated subtitles
  into the video** shape the video. Burning re-encodes it with x264; selectable
  translated and original subtitle tracks are always added.
- **Advanced → Fresh takes for lines** generates new takes for those lines even
  when they are unchanged. **Parallel requests** sets how many lines Drama 3
  voices at once.

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
- Analyze, line actions (Preview, New take, Translate again) and Generate dub are
  separate job families: each **Interrupt** stops only its own step. A
  download's Interrupt stops yt-dlp at its next progress update.
- Speech uses Drama 3 on SGLang-Omni, so dubbing requires Linux, like Soundstage.
  Transcription, translation and downloads have no such limit.
- Sessions (source copy, soundtrack and stems, references, lines) live in
  `.h3-workspace/dubbing/<session>`. Sessions untouched for 14 days are removed
  when a new one starts.
