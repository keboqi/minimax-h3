# Correcting a Sound Stage script

Plan illustrations in Workshop or Studio. Both save dialogue cue anchors and stable shot IDs;
timestamps are resolved against the current mix. Older timestamp plans are upgraded using their
original audio, including shots placed in pauses or sound-only tails.

For a wording correction, save the script and choose Speak, or Auto-produce to include illustrations:

- Unchanged dialogue lines reuse their saved speech takes. Consecutive lines by the same speaker
  have independent takes, so changing one does not regenerate its neighbors. Voice or sampling
  changes still require new takes.
- Auto keeps small corrections to previously spoken or planned dialogue instead of rewriting the
  script. Corrections must retain the dialogue order and speakers and remain textually similar.
- Illustration prompts and images stay in place. Their cue anchors follow the corrected words,
  and their playback times adapt to the new speech. Partial drawings and character portraits are
  carried forward too. Repairing word alignment updates playback without redrawing images.
- An exported video whose timing changed must be saved again to Media.

Older projects with merged speech takes use verified alignment to recover individual unchanged
lines. If that alignment or a saved take is unavailable, the missing lines are synthesized again.
Explicit Remix and Regenerate requests for an older result retain its original take boundaries.

Automatic adaptation preserves what each image depicts. If a wording edit changes a visible action,
object, character or setting, edit that shot's prompt and redraw it. Adding, deleting or reordering
dialogue, changing speakers, and substantial rewrites require reviewing or planning shots again.
