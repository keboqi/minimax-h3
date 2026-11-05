# Remake

The **Remake** task edits an existing video with
[Viggle Remake](https://huggingface.co/Viggle/Viggle-Remake). You give it a source clip, one to
four frames repainted to show the target, and a six-section prompt. The performance, camera and
soundtrack carry over, and the frames decide the new look. It sits in Create between Song and
Storyboard. The agreed design is the
[Remake Tab canvas](https://claude.ai/artifact/5wfjYbuMrwK1D2JW7VheHw).

## Using it

1. Choose **Swap** to replace a character or subject, or **Edit** to change objects, background,
   weather or light. Each has its own LoRA and six-step schedule.
2. Add the source and set **Start** and **End**. One render comes in steps of 17 frames, from 90 to 345
   at 24 fps (3.75 to 14.4 seconds), and rounds down to the nearest allowed length. Longer selections
   render in windows.
3. Scrub to a frame and press **Extract frame**. It fills the first empty slot with its frame number.
4. Press **Edit in Image**, which sends the frame to Qwen Image 2.1, and edit it there. Then bring the
   result back from Media with **Set as** › **Remake · Picture N**, or from the library picker on the
   slot, which keeps the frame number. Generate stays off while a slot holds the frame exactly as it was extracted.
5. Keep **Pin every picture to its frame** on for clips with cuts, or when the composition matters.
   Frame numbers count from Start; moving Start shifts them so each stays on its source frame.
6. If the editor moved the frame, use **Check alignment** and **Align to source**. It corrects a whole-frame
   shift or zoom from unchanged areas. It cannot fix a changed pose.
7. Write an **Edit request**, press **Rewrite / enhance**, and accept the suggestion. Gemini reads the
   clip and the frames. OpenAI-compatible services only see the frames.
8. Pick a **Short edge** and generate. 480 gives 832×480 and 736 gives 1280×736.

The canvas is always 16:9-class, in the source's orientation, so other shapes are center-cropped.
The soundtrack needs no extraction: it stays clean through sampling and comes out in the result, and it
guides mouth movement. A silent source stays silent.

## Long sources

One render holds a limited length, set by **Longest window** (9.4 s by default; lower it if long clips run out of
memory). A selection that fits stays a single run. A longer one renders in windows, and a **Windows** panel appears:

- Windows end on the source's cuts where they can. A shot too long for one window continues into the next: its first
  17 frames are the last 17 frames kept from the render before it, held unchanged through sampling, so the two meet in
  a plain cut with nothing blended. Those frames come from the render, so a window can't start until the one before
  it has rendered.
- **Smart** (on by default) needs a picture on window 1, plus either an edit request or a prompt for that window. Every
  later window takes the last frame of the render before it, to pass on the look, and gets its prompt from your
  request or, without one, from the prompt before it. Window 1's picture joins any window with no picture of its own.
- **Write a prompt for each window** is on by default. Turn it off to reuse window 1's prompt, which often works for
  a single scene. A prompt you typed for a window stays; prompts written earlier are replaced when that window renders.
- Pick a window to change it. The Reference frames and Remake prompt editors follow your choice, and pins count from
  the start of that window. **Copy carried frame to a slot** turns the carried frame into one you can edit.
- **Regenerate this window** renders only that window and any others you changed. The window after it shows
  *Earlier window changed*: its take still joins, and you decide whether to regenerate it. If it continues the same
  shot, regenerate it to remove the jump at that join, then do the same for any window that continues from it.
- **Combine final video** joins the takes and adds the source's soundtrack, so the audio has no seams. A run does this
  itself when every window is rendered.

A selection keeps its windows beside the renders in `h3/.remake`, with its own copy of the source. **Projects** at the
top of the tab lists them: **Load project** brings back the source, selection, windows, frames and prompts with
nothing to upload, and you can rename or delete them there. A project split before held frames keeps its crossfades;
upload the source again to split it the new way. Deleting removes the windows' renders and the copy of the
source; combined videos stay in Media. Only the joined video reaches Media. Windows skip post-processing, so that they join as sampled; finish the joined
video from Media.

## What follows MiniMax H3

Remake keeps no second copy of the H3 settings. It reads them when you open the task, so changes made in
MiniMax H3 apply as soon as you come back.

| Follows MiniMax H3 | Set by Remake | Not used |
|---|---|---|
| Base model, text encoder, video encoder and decoder, seed, attention, memory, acceleration, latent upscale, finishing | Swap or Edit LoRA, six-step Euler schedule, size, length, reference conditioning, pins, clean source audio | Turbo LoRAs, steps and scheduler, Semantic Bridge, refinement LoRA choice, Variations |

The run uses the base model's FL2VA checkpoint. Viggle documents stock FL2VA, so Next run marks every
other base as an untested pairing without blocking it. The Singularity profiles point at Ref2VA-derived
files.

Latent upscale is on by default in H3, so a default run samples at the canvas, doubles it (832×480 becomes
1664×960) and refines the last two steps of the schedule with the same Remake LoRA and audio held clean. The
reference stays at canvas size: rebuilding it at the doubled size made a 14 s clip about three times the largest
regular H3 job. This pairing is ours, not Viggle's, and Next run says so. Turn latent upscale off in MiniMax H3
for Viggle's exact recipe.

## How it runs

A run goes through the regular H3 `generate()` with a `RemakeGraph`, like a Storyboard take, so model
downloads, jobs, status, finishing and snapshots are shared. `build_remake_graph` replaces only the
conditioning and sampling around the normal model stack:

- `LoraLoaderBypassModelOnly` applies the LoRA, because merging into BF16 weights loses part of the update.
- `RemakeConditioning`, `RemakeKeyframeUntil` and `RemakeNoise` come from Viggle, vendored in
  `custom_nodes/H3Remake`. `UPSTREAM.md` there records the revision and the one local change, which lets a
  carried frame be pinned while other pictures stay unpinned.
- `RemakeHoldFrames` is local. It encodes the held frames and marks them as finished in the target, leaving the audio
  held as before. With latent upscale it runs before each pass, so the frames are held at both sizes.
- `request_overrides` selects FL2VA through First / last frame mode. The picture passed there is never
  used as a first frame.
- The request asks for 10 steps to satisfy Normal generation's minimum. The real six-step schedule
  replaces that once the run is prepared, so status and snapshots report six.

The LoRAs (about 3.8 GB each) download on first use, pinned by revision and SHA-256. They are a model
derivative of MiniMax H3 under the MiniMax H3 Community License. The tab carries Viggle's suggested credit.
