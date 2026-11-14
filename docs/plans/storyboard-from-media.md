# Storyboard: scenes from Media

Drafted 2026-11-14. Replaces the earlier *Remake Extend* draft, which planned a separate FL2VA
run with `RemakeHoldFrames` inside the Remake task.

## Status (implemented 2026-11-14, not yet run on a GPU)

Built as described below, with three changes from this draft:

- **No separate encode job.** Preparing a scene is CPU only: it fits the range and cuts a
  73-frame tail clip. The scene that continues it stages that clip and encodes it with
  `H3EncodeAVTail` inside its own graph, with the VAEs that run already loads, at each stage's
  size. There's no extra job, no model to resolve, and no saved latent for a Media scene.
  The Workshop can prepare too.
- **Use this video's size** became **Match the storyboard to this video's shape**. It keeps
  about the storyboard's pixel count instead of copying the video's size, which could be far
  larger than H3 handles. It's offered only until a scene has been generated.
- **Media's Set as › Storyboard** isn't built. Videos are picked from the Media library inside
  the scene, or uploaded.

User docs: `docs/features/storyboard.md`, *Starting from a video*. Tests:
`tests/test_storyboard_media.py` and the encode tests in `tests/test_storyboard_continuation.py`.

## Goal

Extend any existing video by putting it in a storyboard as a scene and letting the next scene
**continue** it. The new scene can be text, First / last frame (last frame only) or Reference
media, as continuing scenes already allow, so you can describe what happens next and give an
edited frame to end on or pictures to keep a look. Remake results reach Media, so this is also
how a Remake gets extended.

## Why this fits

A continuing scene reads only one thing from the scene before it: that take's saved H3
audio-video latent (`latent`, plus `stage_latent` with latent upscale), passed as
`continuation_source` (`h3_ui/storyboard/handlers.py:219`). `H3ContinuationPrefix` copies its
phase-aligned tail into the new clip and holds video and audio fixed while H3 samples the rest
(`custom_nodes/H3Acceleration/__init__.py:2279`). It only requires the latent's size and VAE to
match the storyboard. It doesn't care whether the take was sampled.

So a scene made from a Media video only has to produce what `record_take` stores: a playable
video and those latent files. Everything after it (continuing, takes, out-of-date marking,
joining, export) works as it is.

## How it behaves

- **Add scene › From Media** (and **Set as › Storyboard · new scene** in Media) adds a scene that
  holds a video instead of a prompt. It shows the clip with **In** and **Out** points. By default
  it uses the whole clip.
- The scene has no mode, prompt or takes list. It is *Done* once its clip is prepared, and
  *Not generated* until then. **Prepare clip** (Studio only) runs the encode below. Generate all
  remaining scenes prepares it like any other pending scene.
- A Media scene always starts as a cut; it can't continue the scene before it. The scene after it
  defaults to **Continue**.
- Changing the clip or its In/Out points makes the clip need preparing again. Scenes continuing
  from it become *Out of date* through the existing `affected_by`.
- Its length isn't limited to 5–15 s; only its last few seconds feed the next scene. The total
  shown in the editor includes the whole clip.
- **Size:** the clip is fitted to the storyboard's canvas by the scene's **Fit** option:
  - **Resize** (default): the whole frame is scaled to the canvas. Nothing is lost, but a clip of
    another shape is stretched or squeezed.
  - **Crop**: the frame is scaled to cover the canvas and the overflow is cut off, centered.
    Proportions stay, edges are lost.

  Padding to fit (black bars) isn't offered: the bars would be held in the continued frames and
  the next scene would have to carry them on.

  If the storyboard has no takes yet, the scene also offers **Use this video's size**, which sets
  the storyboard's width and height from the clip, rounded to the H3 grid, so neither fit changes
  anything. A storyboard that already has takes keeps its size, because changing it would break
  their continuations.
- **Fit preview:** under the option, the scene shows the **first and last frame** of the In–Out
  range as they will come out: on a box in the storyboard's shape, labelled with its size. The
  last frame matters most, because it's where the next scene picks up. The two frames are
  extracted once, when the clip or its In/Out points change. Switching Fit only changes how the
  page draws them (stretched to the box for Resize, `object-fit: cover` for Crop), so the preview
  updates at once, with no server call and before anything is prepared. When the clip already
  matches the canvas's shape, the option is shown as having no effect.
- In Workshop the scene can be added, trimmed and saved; preparing it needs Studio, like
  generating.

## Preparing the clip

Two outputs, both stored with the scene's take under `h3/.storyboard/<storyboard>/`:

1. **Conformed video** (CPU, ffmpeg): the In–Out range at 24 fps, fitted by the scene's **Fit**
   (`scale=W:H` for Resize; `scale=W:H:force_original_aspect_ratio=increase,crop=W:H` for Crop) to
   the size the storyboard's takes come out at (doubled with latent upscale), with a silent stereo track
   added when the clip has none. `render()` maps `[i:a]` from every clip and concatenates, so
   every take must already share frame rate, size and an audio stream.
2. **Tail latent** (GPU, VAE only, no diffusion model): a small ComfyUI graph run as a job:
   `LoadVideo` → `GetVideoComponents` → a new **`H3EncodeAVTail`** node → `H3SaveAVLatent`.
   - It takes the last **73 frames** (17 × 4 + 5) of the conformed video, so the frames end on
     the clip's last frame and the latent has 22 steps. `h3_continuation_plan` needs the tail to
     end on a step where `end % 5 == 2`, which a 5 + 17k frame clip does, and 73 frames covers
     every overlap Storyboard offers (22, 39, 56).
   - The node reads the conformed video, which already has the chosen fit, and only scales it down
     to the sampling canvas (same shape, so no second crop). It encodes it with the storyboard's
     own video VAE (its `video_encoder` choice).
   - Audio is the same 73 frames' worth, stereo, resampled to the audio VAE's rate and zero-padded
     to whole 800-sample latent frames before encoding. That's the approach `RemakeConditioning`
     uses (`custom_nodes/H3Remake/nodes.py:138`), so the track isn't centre-cropped and shifted.
     Then it's cut to the video's length. A silent clip encodes silence.
   - With latent upscale, a second branch encodes the same frames at the stage-one canvas
     (half size) and saves it with the `_stage1` prefix, so `continuation_stage_source` exists.
   - The take records `prefix_frames: 0` and an `inputs` fingerprint of the Media file's digest,
     In/Out, Fit, canvas size and VAE, so a change to any of them marks it out of date.

## Where it goes

| Area | Change |
| --- | --- |
| `custom_nodes/H3Acceleration` | `H3EncodeAVTail`: images + audio + VAEs + canvas → H3 AV latent; registered in `catalog.py`'s `STORYBOARD_NODES` |
| `h3_app/workflows/h3.py` | `build_media_tail_graph(video, width, height, frames, stage_size, prefix, vae settings)` |
| `h3_app/storyboard/model.py` | Media scene kind: fields (`media`, `in`, `out`, `fit` = `resize`/`crop`, default `resize`), no mode or prompt; `scene_issues`, `scene_states`, `new_seconds`, `total_seconds`, `runs_as` and link rules for it |
| `h3_app/storyboard/service.py` | Copy the Media file in, conform it, record the take; `delete` and `_collect_media` also clean up its files; `render()` unchanged once clips are conformed |
| `h3_ui/storyboard/handlers.py` | `_prepare_media_scene` beside `_take`, running the encode graph through the same job path; Generate all remaining scenes includes it |
| `h3_ui/storyboard/api.py`, `payload.py`, page static files | Add-from-Media, picker, In/Out, **Fit** with the first/last frame preview (an endpoint that extracts the two frames, cached per clip and In/Out; drawing done in the page), **Use this video's size**, scene card without a prompt |
| Media actions | **Set as › Storyboard · new scene**, with a choice of storyboard |
| `docs/features/storyboard.md` | A **Starting from a video** section; the limits below |
| `tests/` | Encode graph shape; tail length and phase (end step `% 5 == 2`) for overlaps 22/39/56; stage-one latent with latent upscale; model rules for the scene kind; out-of-date marking when the clip changes; conform arguments for Resize and Crop incl. silent audio; Fit change marks the scene and its continuations out of date; preview frames re-extracted on In/Out change but not on Fit change; `render()` with a Media scene first; browser test for adding one |

## Risks to test on GPU before exposing it

- **Encoded versus sampled latents.** Continuation has only ever started from latents the sampler
  produced. A VAE-encoded real clip should be in the same space, but the join can still show a
  colour, sharpness or motion change. Compare it against a storyboard take continued the usual
  way, with the same overlap.
- **Real footage.** Camera shake, cuts close to the end, and compression noise in the tail all go
  into the held frames. A cut inside the last 73 frames will hand over a split context. Warn when
  scene detection finds one there.
- **Audio.** Real recordings at the join: speech cut mid-word, music, room tone. Check that H3
  carries them on and doesn't mute or restart them.
- **Look of a Remake result.** A continuing scene keeps the edited look only through the held tail
  and the prompt. Test Reference media with the Remake pictures to see whether that's needed.
- **VAE choice.** Encode with each video encoder the storyboard may use, and confirm a fast or
  alternative encoder matches the decoder the take is played with.

The existing Storyboard limit still applies: continuing scenes have not run on a GPU either
(`docs/features/storyboard.md`, *Not validated yet*), so the first GPU session covers both.

## Order of work

1. `H3EncodeAVTail` and the encode graph, with tests; a manual GPU run continuing from one Media
   clip, compared against a sampled-take continuation.
2. Scene kind in the model and service, conform step, take recording.
3. Handler and job, Generate all remaining scenes.
4. UI: Add from Media, In/Out, Fit with its first/last frame preview, size, Media's **Set as**
   action.
5. Docs, and a note in `docs/features/remake.md` pointing to Storyboard for extending a result.

## Prior art

Read before writing the encode node:
[ComfyUI_MiniMax_H3_Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender) (continues
from an existing video),
[ComfyUI-Minimax-H3-Continuation](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation)
(latent-tail continuation and append), and
[ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream), which
Storyboard's masking already follows.
