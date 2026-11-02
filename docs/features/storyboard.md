# Storyboard

The **Storyboard** task makes a long video from a list of scenes. Each scene is one
MiniMax H3 clip. A scene either **continues** the one before it, so motion and sound
carry on, or **starts fresh** as a new shot. The scenes are joined into one video at
the end. It sits in Create between Song and Soundstage. Background and decisions are
in the [plan](../plans/long-video-storyboard.md); the agreed design is the
[Storyboard Redesign canvas](https://claude.ai/artifact/WAJ39BupkHrpvdoJXLQ4zK).

## Starting

**Start a storyboard** asks for a name, a story and how scenes connect. Separate scene
prompts with `---` on its own line. Blank lines inside a scene stay in its prompt; text
without a separator creates one scene. *Continue one into the next* makes every later
scene continue; *Every scene starts fresh* makes them separate shots joined by cuts.
Either can be changed per scene. In Studio, the storyboard copies your current
**MiniMax H3** settings and keeps them, so every scene matches;
**Storyboard settings › Use my current MiniMax H3 settings** replaces them later and
marks every scene out of date.

To turn a full story or short idea into scenes, choose a scene count (2–24), clip length,
and connection style, then click **Rewrite / extend into scenes**. Both Workshop and
Studio have hosted **Prompt writer** controls above the form for Gemini or an
OpenAI-compatible service. The rewrite follows the same `prompts/prompt.txt` guidelines
as the MiniMax H3 page, including visual action, dialogue, soundscape, and music for each
scene. Review the result before creating; it does not create a storyboard automatically,
and failures keep your input.

Workshop uses the same editor with generation controls hidden. Storyboards created
there start with H3 defaults. Prepare the scenes and cast, then open the storyboard in
Studio, apply your current H3 settings if needed, and generate. Storyboards and saved
pictures are shared across the endpoints; unsaved form inputs are separate. Stop one
endpoint before editing on the other, as both write the same workspace on the Modal volume.

## A scene

The scene list on the left shows each scene's number, title, length, how it starts
(Opening, Continues scene N, Fresh start), its mode, and a status: Done, Out of date,
Not generated or Needs input. Between scenes it says "↓ continues into" or
"✂ cut to a fresh start". Click a scene to edit it.

The editor says where the scene is ("Scene 2 of 4", what it comes after) and which H3
mode it runs as. It then offers exactly what that scene can use:

| | Opening scene / Start fresh | Continue the previous scene |
|---|---|---|
| Text to video | Yes | Yes |
| First / last frame | Both frames | **Last frame only**: it opens on the previous scene's last frames |
| Reference media | Yes | Yes, marked *recommended* |
| Prompt and prompt writer | Yes | Yes |

A continuing scene shows "No first frame" with the reason and a pointer to **Start
fresh**, so a first frame can never clash with the hand-over. Reference media uses the
storyboard's cast (tick who appears) plus the scene's own images, videos and audio; the
editor lists which `<Picture n>` each one becomes. **Improve with the prompt writer**
uses the writer and keys set in MiniMax H3 in Studio. In Workshop it uses the Storyboard
page's hosted Prompt writer settings, also used for rewriting the full story.

**Before generating** lists anything missing: a prompt, the frame the mode needs, a
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

The timeline shows every scene by status, with fresh starts marked as cuts. **Generate
all remaining scenes** makes every scene without a current take, in order, and skips
scenes that still need input. **Play full video** and **Export to Media** join the takes
in use: continuing scenes drop their repeated opening, fresh starts join as cuts. Export
saves a version. **Versions** keeps scene lists to come back to or branch.

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
is one job in **Jobs**. A run that stops leaves the waiting scenes recorded; nothing
resumes on its own, and **Generate all remaining scenes** continues from there.

## Not validated yet

Nothing here has run on a GPU. Seam quality, audio continuity, continuing toward a last
frame, Spectrum with masked sampling, and texture drift across long chains of continuing
scenes need a real run on Modal.
