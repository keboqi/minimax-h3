"""Comprehensive showcase story for Soundstage (Drama 3).

Demonstrates multi-scene spatial staging (`room`, `stage`), ambience and music loops,
panned and moving one-shot sound effects, musical stingers, character/device/place vocal
effects, `<near>` and `<far>` voice distance, simultaneous overlap, listener reactions,
barge-in pauses (`<#-0.3#>`), cut-off dashes (`—`), all seven scoped delivery tags,
audible events, and Arpabet pronunciation notes.
"""

from __future__ import annotations

import json

EXAMPLE_STORY_TITLE = "The Beacon of Blackthorn Reef"

EXAMPLE_STORY_SPEAKERS: tuple[dict, ...] = (
    {
        "speaker_id": 0,
        "name": "narrator",
        "description": (
            "A male voice: mature, late forties, warm baritone with resonant chest tone, "
            "crisp studio articulation, and a measured, atmospheric storytelling pace. "
            "Clean dry studio recording, no background noise."
        ),
        "preview_text": (
            "The lighthouse at Blackthorn Reef had stood empty for thirty years, its great "
            "lantern dark above the Atlantic swells. When the harbour master sent us out "
            "before the autumn equinox, nobody expected the machinery inside the tower to "
            "still be warm."
        ),
    },
    {
        "speaker_id": 1,
        "name": "Elena",
        "description": (
            "A female voice: early thirties, clear mezzo-soprano, focused and observant "
            "marine investigator with steady breath control, natural conversational pacing, "
            "and crisp consonants. Clean dry studio recording."
        ),
        "preview_text": (
            "I checked the tide charts twice before we left the mainland pier this morning. "
            "The cable trench runs straight from the breakwater into the base of the reef, "
            "and somebody has been drawing power from it all week."
        ),
    },
    {
        "speaker_id": 2,
        "name": "Marcus",
        "description": (
            "A male voice: mid-sixties, weathered gravelly bass-baritone, former lighthouse "
            "keeper with a rough coastal cadence, unhurried delivery, and grounded authority. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "You can read all the blueprints you want back in the harbour office, miss, "
            "but granite and salt water keep their own ledger. When the gale comes around "
            "the point, this tower hums like a tuning fork."
        ),
    },
    {
        "speaker_id": 3,
        "name": "Vance",
        "description": (
            "A male voice: early forties, brisk tenor, coastal dispatch officer with "
            "clipped professional diction, alert energy, and an urgent operational cadence. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "Harbour control to survey launch, we are tracking a steep barometer drop "
            "twenty miles west of the shoals. Keep your channel open on sixteen and report "
            "as soon as you reach the landing."
        ),
    },
    {
        "speaker_id": 4,
        "name": "Aegis",
        "description": (
            "An androgynous voice: calm, precision-engineered mid-register voice with even "
            "syllable weight, cool synthetic clarity, and unshakeable composure. "
            "Clean dry studio recording."
        ),
        "preview_text": (
            "Primary telemetry remains online across all acoustic arrays in the lower "
            "sector. Harmonic calibration will resume once the outer pressure seals are "
            "locked and verified."
        ),
    },
    {
        "speaker_id": 5,
        "name": "Siren",
        "description": (
            "A female voice: timeless, airy alto with velvet resonance, slow hypnotic "
            "phrasing, and delicate breathy overtones. Clean dry studio recording."
        ),
        "preview_text": (
            "Beneath the white foam and the iron chains, the water remembers every ship "
            "that ever crossed the bar. Listen closely to the tide, and you will hear the "
            "bells still ringing in the deep."
        ),
    },
)

EXAMPLE_STORY_SCRIPT = """\
narrator: The Beacon of Blackthorn Reef.
<#0.6#>
<parallel room="small_room" stage="Elena=-0.4, Marcus=0.4">
  <sound loop="true" volume="0.35" duration="25" name="coastal_storm">Heavy coastal rain and wind lashing against thick lighthouse windows, distant ocean surf</sound>
  <sound loop="true" volume="0.2" duration="36" name="noir_cello">Slow brooding minor-key solo cello and muted piano pulse, dark maritime mystery, no vocals</sound>
  <sequential>
    <parallel>
      <sound name="iron_door" duration="3" pan="-0.6">Heavy rusted iron door creaking open and clanging shut against stone</sound>
      narrator: Rain hammered the lantern glass [tense, low] as Elena forced the watchroom door shut.
    </parallel>
    <parallel>
      <sound name="approaching_boots" duration="4" pan="0.8,-0.2">Heavy wet rubber boots walking across wooden floorboards</sound>
      narrator: Marcus crossed the timber floor [slowly, heavy] and set his lantern beside the brass console.
    </parallel>
    Elena: Look at the voltmeter, [quietly amazed] <emphasis>Marcus</emphasis>.
    Elena: Siobhan{SH IH0 V AO1 N} swore this station was decommissioned, [focused, analytical] yet the main bus is <stress>fully</stress> charged.
    <parallel>
      Marcus: We shut the diesel generators down [gravelly, uneasy] thirty years ago, <slow>girl</slow>. Whatever is feeding these coils isn't coming from the mainland grid.
      <sequential>
        <#1.4#>
        Elena: Wait. [gasps]
      </sequential>
    </parallel>
    Elena: <near><intimate_whisper> Listen to the casing, [hushed, intense] it's <whisper>breathing</whisper> under the floor.
    <parallel>
      <sound name="radio_static" duration="3" pan="0.4">Harsh marine VHF radio squelch burst and crackling static dropout</sound>
      narrator: On the shelf, the marine transceiver [sudden, sharp] crackled to life.
    </parallel>
    Vance: <walkie_talkie> Mainland dispatch to Blackthorn team, [urgent, brisk] come in! The storm front is <fast>closing the channel</fast> ahead of schedule.
    Elena: Copy, Vance, [calling out clearly] we're inside the lower watchroom, but we found an active <emphasis>signal array</emphasis>.
    Vance: <bad_signal> Get out of the tower, [panicked, breaking up] the sonar buoys just registered a <forceful>massive spike</forceful> right under the reef! [coughs]
  </sequential>
</parallel>
<#1.0#>
<parallel room="outdoors" stage="Elena=-0.5, Marcus=0.5">
  <sound loop="true" volume="0.4" duration="25" name="coastal_storm">Heavy coastal rain and wind lashing against thick lighthouse windows, distant ocean surf</sound>
  <sequential>
    narrator: They burst onto the outer stone gallery, [sweeping, dramatic] where gale-force spray tore across the <slow>catwalk</slow>.
    Marcus: <far> Hold the iron rail, [shouting over wind] <forceful>Elena</forceful>, the surge is breaking over the shoal!
    <#-0.3#>
    Elena: Look at the water, [breathless, stunned] the waves are <emphasis>standing still</emphasis> in a circle!
    <parallel>
      <sound name="deep_bell" duration="5" distance="far">Enormous bronze fog bell tolling once in the distance over stormy water</sound>
      narrator: Far out beyond the white foam, [solemn, eerie] a submerged bell tolled through the <soft>darkness</soft>.
    </parallel>
    <parallel>
      Elena: Above us!
      Marcus: The lantern!
    </parallel>
  </sequential>
</parallel>
<#1.0#>
<parallel room="large_hall" stage="Elena=-0.35, Marcus=0.35, Aegis=0.0">
  <sound loop="true" volume="0.35" duration="25" name="chamber_hum">Deep low-frequency mechanical hum and resonant stone room tone with slow water drips</sound>
  <sound loop="true" volume="0.2" duration="40" name="tension_ostinato">Tense pulsing synth bass and bowed contrabass ostinato, slow suspenseful build, no vocals</sound>
  <sequential>
    <parallel>
      <sequential>
        <sound name="iron_door" duration="3" pan="-0.6">Heavy rusted iron door creaking open and clanging shut against stone</sound>
        <sound name="glass_shatter" duration="3" pan="-0.5" distance="near">Lantern glass shattering sharply onto a stone floor</sound>
      </sequential>
      narrator: Inside the vaulted lantern chamber, [tense, echoing] a cracked prism fell shattering across the stone floor.
    </parallel>
    Aegis: <robotic_voice> Acoustic lock confirmed, [icy calm, steady] <emphasis>keeper</emphasis> recognized.
    Marcus: That voice, [shaken, disbelieving] we sealed the <stress>Aegis core</stress> behind lead bulkheads —
    <parallel>
      Marcus: when the old ship went down! [shaky breath]
      <sequential>
        <#0.4#>
        Elena: Wait, [turning on him, accusing] you <emphasis>knew</emphasis> it was down here?
      </sequential>
    </parallel>
    Marcus: We had no choice, [heavy with regret] the beacon wasn't built to guide ships, <slow>Elena</slow>. [sighs]
    <parallel>
      Aegis: <robotic_voice> Deep trench receiver opening now, [measured, relentless] relaying <stress>hydrophone</stress> channel zero.
      <sequential>
        <#1.1#>
        <sound name="radio_static" duration="3" pan="0.4">Harsh marine VHF radio squelch burst and crackling static dropout</sound>
      </sequential>
    </parallel>
  </sequential>
</parallel>
<#1.2#>
<parallel room="underwater">
  <sound loop="true" volume="0.35" duration="25" name="abyssal_current">Deep underwater current with muffled low-frequency pressure groan and rising air bubbles</sound>
  <sound loop="true" volume="0.2" duration="40" name="ethereal_choir_pad">Shimmering glass harmonica and warm ambient synth pad, solemn and transcendent, no vocals</sound>
  <sequential>
    narrator: Through the acoustic lens, [hushed, spellbound] the chamber filled with the sound of the <slow>flooded vault</slow> far below the reef.
    Siren: <ghost_spirit><far> Three decades of silence, [distant, hypnotic] and still the <soft>keepers</soft> return to the light.
    Elena: <near> Who is speaking, [whispering, awed] is that coming from <whisper>beneath the rock</whisper>?
    Siren: <ghost_spirit><near> Do not fear the tide, [warm, otherworldly] we only held the <emphasis>memory</emphasis> Siobhan{SH IH0 V AO1 N} left in the deep.
  </sequential>
</parallel>
<#0.8#>
Aegis: <robotic_voice><reverb_large_hall> Ignition sequence complete, [solemn, resonant] the <emphasis>beacon</emphasis> is online.
<parallel>
  <sound name="revelation_sting" volume="0.4" duration="6">Dramatic brass and timpani swell with high shimmering strings, single cinematic hit, no vocals</sound>
  narrator: A blinding beam of white light [warm, triumphant] swept out across the storm, <slow>parting the dark sea</slow>.
</parallel>
Marcus: Look at that horizon, [quiet wonder, relieved] <soft>still</soft> burning after all these years. [chuckles]"""


def example_story_reply_json() -> str:
    """The example story packaged in the writer's JSON reply format."""
    return json.dumps(
        {
            "title": EXAMPLE_STORY_TITLE,
            "source_mode": "brief",
            "script": EXAMPLE_STORY_SCRIPT,
            "speakers": list(EXAMPLE_STORY_SPEAKERS),
        },
        ensure_ascii=False,
    )
