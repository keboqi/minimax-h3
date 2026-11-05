# Remake

The **Remake** task edits an existing video with
[Viggle Remake](https://huggingface.co/Viggle/Viggle-Remake). You give it a source clip, one to
four frames repainted to show the target, and a six-section prompt. The performance, camera and
soundtrack carry over, and the frames decide the new look. It sits in Create between Song and
Storyboard. The agreed design is the
[Remake UI Redesign canvas](https://claude.ai/artifact/Tp46iRwMtgYj8AwSuqAYft).

## Using it

The tab opens on a drop area and your **Projects**. Once a source is in, a viewer sits above a timeline, and the panel
on the right always shows the selection or the window under the playhead.

1. Pick **Swap** to replace a character or subject, or **Edit** to change objects, background, weather or light. Each
   has its own LoRA and six-step schedule.
2. Set **Start** and **End** on the timeline. One render comes in steps of 17 frames, from 90 to 345 at 24 fps (3.75 to
   14.4 seconds), and rounds down to the nearest allowed length. Longer selections render in windows.
3. Move the playhead to a frame and press **Extract frame**. It fills the first empty slot with its frame number, and a
   numbered marker appears under the timeline.
4. Type the change under the picture and press **Quick edit**. Qwen Image 2.1's Fast preset repaints the frame right
   there at its own size, and **Undo edit** brings the frame back. For more control, pick **Qwen Image 2.1** (the
   default) or **MiniMax H3** under **Edit with** and press **Edit in Image**, which opens that model in the Image task
   with the frame loaded. Qwen Image 2.1 opens in **Image edit** mode. **MiniMax H3** edits the frame as a first frame on
   the base model a run loads anyway, so no second diffusion model has to load, though its image LoRA, text encoder and
   VAE download on first use. It keeps the frame's size unless you change **Image edit size** there. Then bring the result
   back from Media with **Set as** › **Remake · Picture N**, or with **Choose the edited image** on the picture. Both keep
   the frame number. Generate stays off while a slot holds the frame exactly as it was extracted.
5. Keep **Pin to frames** on for clips with cuts, or when the composition matters. Frame numbers count from Start;
   moving Start shifts them so each stays on its source frame.
6. A picture that comes back is measured on its own and shows **Lined up** or **Shifted**. **Check alignment** opens the
   comparison in the viewer, and **Align to source** corrects a whole-frame shift or zoom from unchanged areas. It
   cannot fix a changed pose.
7. Write an **Edit request**, press **Rewrite / enhance**, and accept the suggestion. Gemini reads the clip and the
   frames. OpenAI-compatible services only see the frames.
8. Press the action at the top right, which says what it will do. The Source, Frames and Prompt steps beside it show
   what is still missing.

The viewer has **Source**, **Result** and **Compare**, where you drag a divider across the two.

**Size** starts on **Auto, match source**: the video keeps the source's shape and size, up to 1080p, whatever the
MiniMax H3 settings say. Sources up to 736 pixels on their short edge are sampled directly. Larger ones are sampled at
half size and doubled, so 1920×1080 samples 960×544 and finishes at 1920×1088, because both sides must be multiples of
32. Pick a short edge to use less memory: 480 samples 832×480 from a 16:9 source, and latent upscale then follows
MiniMax H3. Every size keeps the source's shape, up to what 1280×736 holds; anything left over is center-cropped.

The soundtrack needs no extraction: it stays clean through sampling and comes out in the result, and it
guides mouth movement. A silent source stays silent.

## Long sources

One render holds a limited length, set by **Longest window** (9.4 s by default; lower it if long clips run out of
memory). A selection that fits stays a single run. A longer one renders in windows, which appear as blocks on the
timeline, coloured and labelled by state:

- Windows end on the source's cuts where they can. A shot too long for one window continues into the next: its first
  17 frames are the last 17 frames kept from the render before it, held unchanged through sampling, so the two meet in
  a plain cut with nothing blended. These frames are hatched on the timeline. They come from the render, so a window
  can't start until the one before it has rendered.
- **Smart continue** (on by default) needs a picture on window 1, plus either an edit request or a prompt for that
  window. Every later window takes the last frame of the render before it, to pass on the look, and gets its prompt
  from your request or, without one, from the prompt before it. Window 1's picture joins any window with no picture of
  its own.
- **Prompt per window** is on by default. Turn it off to reuse window 1's prompt, which often works for a single scene.
  A prompt you typed for a window stays; prompts written earlier are replaced when that window renders. A chip above the
  prompt says which it is.
- Click a window, or scrub into it, to change it. Leaving a window saves it, and pins count from the start of that
  window. **Copy to a slot** turns the carried frame into one you can edit.
- **Regenerate this window** renders only that window and any others you changed. The window after it shows
  *Earlier window changed*: its take still joins, and you decide whether to regenerate it. If it continues the same
  shot, regenerate it to remove the jump at that join, then do the same for any window that continues from it.
- **Combine final video** joins the takes and adds the source's soundtrack, so the audio has no seams. A run does this
  itself when every window is rendered.

A selection keeps its windows beside the renders in `h3/.remake`, with its own copy of the source. The project name at
the top opens the list, where you can switch project, rename, delete or start from a new video. Opening one brings back
the source, selection, windows, frames and prompts with nothing to upload. A project split before held frames keeps its
crossfades; upload the source again to split it the new way. Deleting removes the windows' renders and the copy of the
source; combined videos stay in Media. Only the joined video reaches Media. Windows skip post-processing, so that they
join as sampled; finish the joined video from Media.

Sources, frames and pictures of a selection that never became a project sit in the same folder under `uploads` and are
cleared after a week.

## What follows MiniMax H3

Remake keeps no second copy of the H3 settings. It reads them when you open the task, so changes made in
MiniMax H3 apply as soon as you come back.

| Follows MiniMax H3 | Set by Remake | Not used |
|---|---|---|
| Base model, text encoder, video encoder and decoder, seed, attention, memory, acceleration; latent upscale and finishing with a chosen short edge | Swap or Edit LoRA, six-step Euler schedule, size, length, reference conditioning, pins, clean source audio | Turbo LoRAs, steps and scheduler, Semantic Bridge, refinement LoRA choice, Variations |

The run uses the base model's FL2VA checkpoint. Viggle documents stock FL2VA, so Next run marks every
other base as an untested pairing without blocking it. The Singularity profiles point at Ref2VA-derived
files.

A run that doubles samples at the canvas, doubles it (960×544 becomes 1920×1088) and refines the last two steps of the schedule with the same Remake LoRA and audio held clean. The
reference stays at canvas size: rebuilding it at the doubled size made a 14 s clip about three times the largest
regular H3 job. This pairing is ours, not Viggle's, and Next run says so. Pick a short edge and turn latent upscale off in
MiniMax H3 for Viggle's exact recipe.

## How it runs

The tab is a page of its own in a frame, like Storyboard: `h3_ui/remake_page.py` serves it and its JSON API from
`h3_ui/remake_static`, and `h3_ui/engines/remake.py` holds what that API calls. A hidden Gradio bridge beside the frame
keeps what still lives in Gradio: the `generate_remake` API, the H3 settings, and passing pictures to and from Media and
the Image task. The page holds the editor. Scrubbing asks the server for nothing unless the browser can't play the source.

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
