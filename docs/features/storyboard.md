# Storyboard

The **Storyboard** task plans a long video as scenes, generates them one by one with
MiniMax H3, and joins them into one video. It sits in Create between Song and
Soundstage. The design and its reasoning are in the
[plan](../plans/long-video-storyboard.md).

## Using it

1. **New storyboard** takes a name, an optional story and a usual scene length. Each
   paragraph of the story becomes a draft scene. The current **MiniMax H3** render
   settings are copied into the storyboard and stay fixed, so every scene joins
   cleanly. **Apply current MiniMax H3 settings** replaces them later and marks every
   scene outdated.
2. **Storyboard and cast** holds cast pictures (up to nine) that any scene can use as
   H3 reference images. A scene with cast runs in Reference media mode with a
   `Name is <Picture n>.` line before its prompt; without cast it runs text to video.
3. Each scene has a title, a prompt, a length of 5, 10 or 15 seconds, and a link to the
   scene before it. **Flow** continues from the last frames of the previous scene's
   take in use. **Cut** starts a new shot. Scene options set the overlap (22, 39 or 56
   frames), the seed and how many takes one run makes. Edits save when a field loses
   focus.
4. **Generate scene** runs one scene; **Generate remaining** runs every draft and
   outdated scene in order, letting a cut scene run while a flow scene waits for the
   one before it. Every take is kept. The first take of a scene is put in use; **Use
   this take** switches, which outdates the scenes that flow from it up to the next cut.
5. **Play all** joins the takes in use into a preview. **Export to Media** joins them
   into a new Media video and saves a version. Flow joins drop the repeated overlap;
   cut joins are hard cuts. Outdated scenes are included unless the box is cleared.
6. **Versions** keeps snapshots of the scene list. Branching one, or duplicating the
   storyboard, makes a new storyboard that shares the existing takes.

A flow scene repeats its overlap, so it adds less new footage than its length: a 10 s
flow scene with the default 39-frame overlap adds about 8.4 s, and a 5 s one about
3.4 s. Scene cards and the timeline show the new seconds.

## How flow works

Every take saves its final H3 audio-video latent next to its MP4 (and the stage-one
latent when latent upscale is on). A flow take loads the previous take's latent and
`H3ContinuationPrefix` copies its phase-aligned tail into the head of the new clip,
with per-stream noise masks that keep that head fixed during sampling. With latent
upscale the tail is pasted at both resolutions, and the full-resolution paste also
replaces the stage-one mask that would otherwise reach refinement. The masking follows
[ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream)
(MIT) and relies on native ComfyUI per-token AV masks (Comfy-Org/ComfyUI#15375), which
the pinned ComfyUI includes.

Takes are written under `h3/.storyboard/<storyboard>/` in the ComfyUI output folder.
They stay out of Media and out of **Scan historical media**; only exports reach Media.
Storyboards force H3 post-processing off so each take's video matches its latent;
finish an export from Media instead. The H3 Regenerate refiner can't be used for flow
scenes.

## Storage and jobs

Storyboards are owner-scoped rows in the workspace database (schema 2). Creating one
is the explicit choice to keep its prompts and cast pictures on the server; cast
pictures live under the workspace state's `storyboards/` folder and are included in
workspace backups. A generation run is one job in **Jobs**. A run that stops (error,
interrupt or restart) leaves the waiting scenes recorded on the storyboard; nothing
resumes on its own, and **Generate remaining** continues from there.

## Not validated yet

Nothing here has run on a GPU. Seam quality, audio continuity, Spectrum with masked
sampling, and texture drift across long unbroken flow chains still need the Modal
validation run described in the plan.
