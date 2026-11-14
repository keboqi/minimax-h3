# Soundstage mix enhancements test story

**Fastest:** set the script mode to Advanced (sound enhanced), clear the script box, open the
writer's "Another chat model" tab, paste all of
[soundstage-mix-enhancements-test.json](soundstage-mix-enhancements-test.json), and click
Apply pasted reply. That is a finished reply with the cast and sounds already placed for the
checks below, so no chat model is needed.

To test a chat model's own script instead, paste the story below as the Soundstage source,
copy the full writer prompt into that model, and paste its JSON reply back. The story is
written so that any reasonable script puts each enhancement from commit dd7bc5f where you can
hear it.

## Story (paste this)

```text
The Lantern Street Night Market

The night market on Lantern Street was loud with music. A busker at the corner played a bright, jangling guitar tune that never stopped, and the crowd hummed and laughed around the food stalls. Under the strings of paper lanterns, three friends stood arguing about who had lost the van keys.

Ada stood on the left, arms crossed. Tomas stood on the right, patting every pocket he had. Wren leaned between them, trying not to laugh.

"You had them last," Ada said. "I watched you lock the—"

"I did not lock anything," Tomas cut in. "You drove, so you—"

"I drove there. You drove the—"

"Stop, both of you," Wren said. "Turn out your pockets. Right now."

Tomas sighed and emptied his jacket onto a crate: a bus ticket, two coins, a dumpling wrapper. No keys.

"See? Nothing," he said. "So it has to be—"

"Don't you dare say it's me," Ada snapped. "Last time you said it was me, they were in your—"

"In my shoe. Once. That was once."

The guitar kept jangling. Somewhere far down the street, a stall owner shouted, "Last call for noodles! Last call!"

Then a delivery truck rumbled past, slowly, from the left end of the street to the right, its engine growling and its loose tailgate rattling as it went by. The three of them stepped back onto the curb until it was gone.

Wren held up a hand. "Wait. Listen."

From across the street, faint and far away, a man was calling, "Did someone lose a set of keys? Blue keychain! Anyone?"

"Blue keychain," Tomas whispered. "That's—"

"That's ours," Ada said, already running. "Hey! Over here! Those are—"

"Mine," Tomas finished, grinning. "I'm the one who lost them. Fine. I admit it."

They collected the keys and ducked out of the noise into the old stone church at the end of the street, where it was cool and quiet and every footstep echoed off the high walls. Nobody else was inside. A soft organ note hung in the air from somewhere near the altar.

"Okay," Wren said quietly, her voice ringing under the arches. "Nobody touches the keys again except—"

"Except me," Ada said.

"Except you," Tomas agreed. "Although technically, if you think about who parked—"

"Tomas."

"Right. Sorry. Keys stay with Ada."

Outside, faintly, the busker's guitar started a new song.
```

## What to listen for

**Presence dip (beds under speech).** The busker's guitar and the crowd should loop as beds
under the market dialogue, and the organ as a bed in the church. While people talk, the bed
should drop but keep its brightness and air, with only the 1–4 kHz consonant band pulled down
hard (the full 12 dB). Consonants like the "t" and "k" in "Tomas" and "keys" should stay crisp
over the guitar. If the bed now sounds too forward, lower `BED_DUCK_DB` toward 9–10 in
`h3_app/drama3/serving/audio.py`.

**Interaural delay (placement).** Ada should sit left and Tomas right, with Wren in between,
and the narrator in the centre with no delay. On headphones each voice should feel anchored
to its side instead of just being louder in one ear. The truck should sweep left to right,
with its image moving smoothly and no zipper or flanging noise as it crosses the centre. The
distant stall owner and the man calling about the keys should sound far away. Also check
once on a single speaker or a phone: hard-panned voices should not sound hollow.

**Cut-off fade (barge-ins).** Every line ending in "—" is cut off by the next speaker:
"lock the—", "so you—", "drove the—", "has to be—", "in your—", "That's—", "Those are—",
"except—", "who parked—". On each one, the interrupted voice should give way under the
incoming one instead of both playing at full level, with no click or sudden drop. The
three-way pile-up at the start ("lock the—" / "so you—" / "drove the—") is the hardest case.
Narration never barges in, so "That's ours," Ada said should start cleanly after the line
before it.

**Church reverb.** In the church, an interrupted voice's reverb sinks along with it and then
rings on quietly. On "Nobody touches the keys again except—" and "who parked—" the overlap
should not sound doubled, and the hall should not cut off abruptly. A written `<#-0.3#>`
interruption, or a cut-off right before a sound block, should give way the same way.

## Checking the writer's JSON

Before you render, check that the reply kept these. If it didn't, ask the other model to
fix them or edit the JSON:

- The guitar, the crowd and the organ are looped beds (`loop="true"`), not one-shots.
- The truck has a moving placement (`moving_left_to_right`, or `pan="-0.8,0.8"`).
- The stall owner and the keys man are far voices (`<far>` or `distance="far"`).
- The cut-off lines still end in "—", with no extra pause after them.
- The church scene has a reverberant room (for example `large_hall`).
