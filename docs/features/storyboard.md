# Storyboard

The **Storyboard** task makes a long video from a list of scenes. Each scene is one
MiniMax H3 clip. A scene either **continues** the one before it, so motion and sound
carry on, or **starts fresh** as a new shot. The scenes are joined into one video at
the end. It sits in Create between Remake and Soundstage. Background and decisions are
in the [plan](../plans/long-video-storyboard.md); the agreed design is the
[Storyboard Redesign canvas](https://claude.ai/artifact/ETR9sQf3pvaVTFTX737mv9).

Storyboard is a page of its own at `/storyboard/`, outside Gradio; the tab shows it in a
frame. The browser holds the whole storyboard, so moving between scenes (the filmstrip,
the arrows, or the Left and Right keys) makes no server request. Each edit saves that
field in the background, and the header says when everything is saved. Videos show their
first frame and load only when played. In the tab, the frame grows to fit the page, so
only the window itself scrolls.

## Starting

**New storyboard** asks for a name, a story and how scenes connect, and previews the
scenes as you type. Separate scene
prompts with `---` on its own line. Blank lines inside a scene stay in its prompt; text
without a separator creates one scene. A scene is 5 to 15 whole seconds long. A
`---N---` line, such as `---6---`, also sets the length of the scene after it, so the
first scene can start with one. Scenes without one use **Usual scene length**; a length
outside 5–15 stops Create and names the scene. *Continue one into the next* makes every later
scene continue; *Every scene starts fresh* makes them separate shots joined by cuts.
Either can be changed per scene. In Studio, the storyboard copies your current
**MiniMax H3** settings, which the tab hands to the page's server side each time it opens,
and keeps them, so every scene matches;
**Settings › Use my current MiniMax H3 settings** replaces them later and
marks every scene out of date.

To turn a full story or short idea into scenes, choose a scene count (2–24), the usual
scene length and the connection style, then click **Rewrite / extend into scenes**. The
writer gives each scene the length its action and dialogue need, using the usual length
as a guide. It writes each length as a `---N---` line before the scene, which you can
edit before creating. Both Workshop and
Studio have hosted **Prompt writer** controls on this page and under **Settings** for
Gemini or an OpenAI-compatible service, kept for this browser tab. The rewrite follows the same `prompts/prompt.txt` guidelines
as the MiniMax H3 page, including visual action, dialogue, soundscape, and music for each
scene. Review the result before creating; it does not create a storyboard automatically,
and failures keep your input.

Workshop uses the same editor with generation controls hidden. Storyboards created
there start with H3 defaults. Prepare the scenes and cast, then open the storyboard in
Studio, apply your current H3 settings if needed, and generate. Storyboards and saved
pictures are shared across the endpoints; unsaved form inputs are separate. Stop one
endpoint before editing on the other, as both write the same workspace on the Modal volume.
The storyboard picker marks a project **Generated** as soon as it has a recorded scene take.

## A scene

The filmstrip along the top shows each scene's picture, number, title, length and a
status: Done, Out of date, Not generated or Needs input. A line between two scenes marks
a fresh start (a cut). Click a scene to edit it; **Add scene** adds one at the end.

The heading names the scene and the H3 mode it runs as, and the inspector on the right
says what it comes after. It then offers exactly what that scene can use:

| | Opening scene / Start fresh | Continue the previous scene |
|---|---|---|
| Text to video | Yes | Yes |
| First / last frame | Both frames, plus up to 3 voices | **Last frame only**: it opens on the previous scene's last frames; up to 3 voices |
| Reference media | Yes | Yes, marked *recommended* |
| Prompt and prompt writer | Yes | Yes |

A continuing scene shows "No first frame" with the reason and a pointer to **Start
fresh**, so a first frame can never clash with the hand-over. Reference media uses the
storyboard's cast (tick who appears) plus the scene's own images, videos and audio; the
editor lists which `<Picture n>` each one becomes. A First / last frame scene takes up to
three voices, like the MiniMax H3 tab's FL2VA voice references: they become `<Audio 1>`
to `<Audio 3>` for new dialogue with those frames. Both modes keep these clips in the
scene's audio list, so switching between them keeps the voices. **Improve with the prompt writer**
uses the writer and keys set in MiniMax H3 in Studio, with the Storyboard page's hosted
Prompt writer settings filling anything left blank. In Workshop it uses the page's settings,
also used for rewriting the full story, and works from their defaults until you change them.

The inspector lists anything missing: a prompt, the frame the mode needs, a
reference, or a previous scene that has no take yet. **Generate this scene** stays
disabled until the list is clear. A continuing scene adds less new footage than its
length, because its first ~1.6 s (39 frames by default) repeats the previous scene; the
editor shows how much.

## Takes and regenerating

Every take is kept. A scene's first take is put in use; later takes are added beside
it. **Use this take** switches. When later scenes continue from this one, the page asks
first and names them, and the first fresh start after them that stays as it is:
**Switch and regenerate them**, **Switch, regenerate later**, or **Cancel**. Fresh
starts stop changes from rippling further.

## The full video

**Generate
all remaining scenes** makes every scene without a current take, in order, and skips
scenes that still need input. **Play full video** and **Export to Media** join the takes
in use: continuing scenes drop their repeated opening, fresh starts join as cuts. Export
saves a version. **Versions** keeps scene lists to come back to or branch.

## Auto render

**Auto render** in the top bar (GPU endpoint only) renders several storyboards
unattended. Tick the storyboards, each listed with how many scenes it has left and any
scene that still needs input, then **Render and export**. They run in the order listed:
each storyboard generates its remaining scenes exactly as **Generate all remaining
scenes** would, and once every scene is done its full video is exported to Media (with a
saved version) before the next storyboard starts. A storyboard whose scenes are already
all done is only exported. One with a scene that still needs input gets the scenes it
can, is marked **Not exported**, and the queue moves on; a failed storyboard is marked
**Failed** and the queue moves on too.

The queue runs on the server, so the page can be closed. While it runs, the top bar shows
which storyboard it is on, and a page showing the storyboard the queue just finished moves
to the next one with it. **Stop auto render** interrupts the current storyboard, keeping
the scenes made so far, and the rest don't start. Each storyboard's generation is its own
job in **Jobs**. The queue lives in the server process: a restart ends it, and nothing
resumes on its own.

## How continuing works

Every take saves its final H3 audio-video latent (and the stage-one latent with latent
upscale). A continuing take loads the previous take's latent, and
`H3ContinuationPrefix` copies its phase-aligned tail into the start of the new clip with
per-stream noise masks that keep it fixed while H3 samples the rest. With latent upscale
the tail is pasted at both resolutions. The masking follows
[ComfyUI-MiniMaxH3-PrefixStream](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream)
(MIT) and uses native ComfyUI per-token AV masks (Comfy-Org/ComfyUI#15375), which the
pinned ComfyUI includes.

Takes and scene pictures live under `h3/.storyboard/<storyboard>/` in the ComfyUI output
folder, outside Media and **Scan historical media**; only exports reach Media.
Storyboards force H3 post-processing off so each take matches its latent, and continuing
scenes can't use the H3 Regenerate refiner.

## Storage and jobs

Storyboards are shared rows in the workspace database (schema 2); the owner records
the creator, while every browser on the deployment can open and edit them. Creating one is
the explicit choice to keep its prompts and pictures on the server. Cast pictures are in
the workspace state's `storyboards/` folder and in workspace backups. A generation run
is one job in **Jobs**, runs on the server under the same GPU lock as every other job, and
the page polls it, so closing or reloading the page leaves it running and it shows again
on return. A joined video is made the same way. A run that stops, or a server restart,
leaves the waiting scenes recorded; nothing resumes on its own, and **Generate all
remaining scenes** continues from there.

The page's JSON API (`h3_ui/storyboard/page.py`) is for the page alone. Scripted clients
use the API below.

## API for scripted clients

Both endpoints publish four API-only endpoints (`h3_ui/storyboard/api.py`), so a
client can plan a storyboard elsewhere and keep it in step on the Workshop:

- `/storyboard_import(spec, files)` creates a whole storyboard from a JSON spec: name,
  optional settings as H3 profile fields, cast, and scenes with an optional `key`,
  mode, frames, references and how each starts. Media fields name uploaded files by
  file name.
- `/storyboard_update(id, spec, files)` makes an existing storyboard match a revised
  spec. Scenes matched by `key` keep their takes and stay *Done* unless their inputs
  changed. Other scenes are added or removed, and the order follows the spec. Name,
  settings and cast change only when the spec has them. Uploads that match stored
  media by content are not stored again; the editor reuses media the same way.
- `/storyboard_list()` and `/storyboard_details(id)` report the storyboards, each
  scene's state and what it still needs.

Import and update return what changed. A spec with an unknown field or a missing file
is refused as a whole, and a failed update changes nothing. Generating stays on the
page.

## Not validated yet

Nothing here has run on a GPU. Seam quality, audio continuity, continuing toward a last
frame, Spectrum with masked sampling, and texture drift across long chains of continuing
scenes need a real run on Modal.
