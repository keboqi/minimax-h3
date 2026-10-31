"""The Guided writer's code stages: source segmentation and compiling stage answers into a script."""

import unittest

from h3_app.drama3.guided.compile import compile_script
from h3_app.drama3.guided.model import (
    AMBIENCE, DURING, EFFECT, INTERRUPT, MUSIC, REACTION, STINGER, WITH,
    Direction, Draft, Scene, Sound, SoundEvent, Staging, Utterance,
)
from h3_app.drama3.guided.segment import BRIEF, PROSE, SCRIPT, attribute_units, input_kind, segment_source
from h3_app.drama3.lint import coverage_note, lint_drama3_script
from h3_app.drama3.serving.tree import iter_sound_nodes, iter_speech_nodes, parse_drama3_tree
from h3_app.errors import H3Error

KETTLE = ("The kettle had just begun to whistle when the doorbell rang. Nora froze.\n"
          "“Are you expecting anyone?” Tom asked from the sofa.\n"
          "“No,” she said. “Nobody knows we moved here.”\n"
          "The bell rang again, longer this time. Tom stood up and switched off the lamp.\n"
          "“Stay behind me,” he whispered.")
NOODLES = ("第一卷 归途\n第一章 面馆\n"
           "傍晚，老街上的灯一盏盏亮了起来。\n"
           "“爸，你还记得这家面馆吗？”小雨停下脚步。\n"
           "父亲眯着眼看了很久，才慢慢点头：“你小时候，最爱吃他们家的牛肉面。”\n"
           "小雨笑了，眼睛却有点红。他把这家店叫作“老地方”。")
TEMPLE = ('预言之声: <old_gramophone>"当星辰排列成十字，诸神将重返人间。"\n'
          "Narration: 林夕关掉了留声机，神色凝重。她的手机突然响起。\n"
          '林夕(surprised): "喂？"\n'
          '陈教授: <phone_call>"林夕，是我。快离开那里！"')


def checked(test, script, source=""):
    """The script after the existing lint and parser, which must accept it unchanged."""
    test.assertEqual(lint_drama3_script(script), (script, []), script)
    tree = parse_drama3_tree(script)
    if source:
        test.assertEqual(coverage_note(source, script, source_mode="preserve"), "", script)
    return tree


class SegmentTests(unittest.TestCase):
    def test_input_kind_tells_scripts_prose_and_briefs_apart(self):
        self.assertEqual(input_kind(TEMPLE), SCRIPT)
        self.assertEqual(input_kind("<|speaker:0|> Hello there.\n<|speaker:1|> General."), SCRIPT)
        self.assertEqual(input_kind(KETTLE), PROSE)
        self.assertEqual(input_kind(NOODLES), PROSE)
        self.assertEqual(input_kind("Two friends arguing about pizza, 30 seconds."), BRIEF)
        self.assertEqual(input_kind('Write a scene where a detective says "Stop!" to a fleeing thief.'), BRIEF)
        self.assertEqual(input_kind(
            "Create a 45-second audio drama. Two astronauts hear a knock on the airlock. "
            'One whispers "Nobody else is up here."'
        ), BRIEF)
        # 她问：“…” is prose with an attribution, not a script label.
        self.assertEqual(input_kind("父亲慢慢点头：“你小时候最爱吃。”\n小雨笑了：“是啊。”"), PROSE)

    def test_quotes_split_from_their_attributions_in_reading_order(self):
        units = segment_source("“Chen,” the teacher said, waving, “come out.”").units
        self.assertEqual([(unit.kind, unit.text) for unit in units],
                         [("quote", "Chen,"), ("narration", "the teacher said, waving,"), ("quote", "come out.")])
        units = segment_source(KETTLE).units
        self.assertEqual([unit.text for unit in units if unit.kind == "quote"],
                         ["Are you expecting anyone?", "No,", "Nobody knows we moved here.", "Stay behind me,"])
        self.assertIn("Tom asked from the sofa.", [unit.text for unit in units])

    def test_long_narration_is_cut_at_most_two_sentences_a_unit(self):
        units = segment_source("One. Two. Three. Four. Five.\nSix.").units
        self.assertEqual([unit.text for unit in units], ["One. Two.", "Three. Four.", "Five.", "Six."])

    def test_leading_headings_become_one_heading_unit(self):
        units = segment_source(NOODLES).units
        self.assertEqual((units[0].kind, units[0].text), ("heading", "第一卷 归途 第一章 面馆"))
        self.assertEqual(units[1].text, "傍晚，老街上的灯一盏盏亮了起来。")

    def test_script_lines_keep_label_cue_hint_and_effects(self):
        units = segment_source(TEMPLE).units
        self.assertEqual([(unit.label, unit.text) for unit in units][:3], [
            ("预言之声", "当星辰排列成十字，诸神将重返人间。"),
            ("Narration", "林夕关掉了留声机，神色凝重。她的手机突然响起。"),
            ("林夕", "喂？")])
        self.assertEqual(units[0].effects, ("old_gramophone",))
        self.assertEqual(units[2].hint, "surprised")
        self.assertEqual(units[3].effects, ("phone_call",))
        # A multi-sentence quoted script line splits into <=2-sentence units with the hint on the first.
        multi = segment_source('Mia( breathless ): "One. Two. Three." \nTom: "Four."').units
        self.assertEqual([(u.label, u.text, u.hint) for u in multi], [
            ("Mia", "One. Two.", "breathless"),
            ("Mia", "Three.", ""),
            ("Tom", "Four.", ""),
        ])

    def test_attribution_needs_every_quote_and_reads_narrated_quotes_in_place(self):
        source = segment_source(NOODLES)
        with self.assertRaises(H3Error):
            attribute_units(source, {"s3": "小雨"})
        utterances = attribute_units(source, {"s3": "小雨", "s6": "父亲", "s8": "narrator"})
        self.assertEqual([(u.speaker, u.text) for u in utterances][-3:], [
            ("narrator", "父亲眯着眼看了很久，才慢慢点头："),
            ("父亲", "你小时候，最爱吃他们家的牛肉面。"),
            ("narrator", "小雨笑了，眼睛却有点红。他把这家店叫作“老地方”。")])
        self.assertEqual(utterances[0].kind, "heading")
        script = "\n".join(f"{u.speaker}: {u.text}" for u in utterances)
        self.assertEqual(coverage_note(NOODLES, script, source_mode="preserve"), "")


class CompileBasicTests(unittest.TestCase):
    def draft(self):
        utterances = attribute_units(segment_source(KETTLE), {"s2": "Tom", "s4": "Nora", "s6": "Nora", "s8": "Tom"})
        return Draft(utterances, [Scene("c1", tuple(u.id for u in utterances))], directions={
            "u1": Direction(cue="tense, low"),
            "u2": Direction(cue="tense, low"),  # repeats the line before: dropped
            "u3": Direction(cue="low, watching them"),  # "Tom asked from the sofa." after Tom: plain
            "u4": Direction(cue="too quickly", event="shaky breath"),
            "u5": Direction(cue="neutral, observing", phrase="said"),  # "she said.": plain, phrase kept
            "u6": Direction(cue="almost pleading", phrase="Nobody", tag="stress"),  # opens the line: not marked
            "u7": Direction(cue="低声", phrase="longer this time"),  # not English: dropped
            "u8": Direction(cue="hushed, urgent", phrase="behind", tag="slow"),
        })

    def test_cues_phrases_and_events_are_placed_by_code(self):
        notes = []
        script = compile_script(self.draft(), advanced=False, notes=notes)
        lines = script.split("\n")
        # insert_cue places a cue after the opening phrase, as the Single-pass rules ask.
        self.assertEqual(lines[0], "narrator: The kettle had just begun to whistle when the doorbell rang. [tense, low] Nora froze.")
        self.assertEqual(lines[1], "Tom: Are you expecting anyone?")
        self.assertIn("Nora: No, [shaky breath]", script)  # one word: no place for a cue, the event still closes it
        self.assertIn("The bell rang again, <emphasis>longer this time</emphasis>.", script)
        self.assertIn("Tom: Stay [hushed, urgent] <slow>behind</slow> me,", script)
        self.assertIn("narrator: Tom asked from the sofa.", lines)
        self.assertIn("narrator: she <emphasis>said</emphasis>.", lines)
        # Stressing the line's first word is skipped without a note; the others are reported.
        self.assertEqual(len(notes), 3, notes)
        checked(self, script, KETTLE)

    def test_basic_ignores_staging(self):
        draft = self.draft()
        draft.sounds = {"bell": Sound("bell", EFFECT, "Doorbell")}
        draft.staging = {"c1": Staging(events=(SoundEvent("bell", "u7"),))}
        self.assertNotIn("<", compile_script(draft, advanced=False).replace("<emphasis>", "").replace(
            "</emphasis>", "").replace("<slow>", "").replace("</slow>", ""))


class CompileAdvancedTests(unittest.TestCase):
    def test_joint_quote_plays_in_parallel_and_keeps_member_sound_anchors(self):
        from h3_app.drama3.guided.schema import scenes_from_units
        from h3_app.drama3.serving.tree import ParallelNode, SequentialNode

        source = segment_source('“We agree,” they said together. “Let us go,” they added.')
        utterances = attribute_units(source, {"s1": ("Mia", "Tom", "Nora"), "s3": ("Mia", "Tom")})
        draft = Draft(utterances, scenes_from_units(utterances, []))
        script = compile_script(draft, advanced=True)
        tree = checked(self, script)
        self.assertIsInstance(tree, SequentialNode)
        groups = [node for node in tree.children if isinstance(node, ParallelNode)]
        self.assertEqual(len(groups), 2)
        self.assertEqual([len(group.children) for group in groups], [3, 2])
        for group, names in zip(groups, [("Mia", "Tom", "Nora"), ("Mia", "Tom")]):
            self.assertEqual([iter_speech_nodes(branch)[0].turns[0].speaker
                              for branch in group.children], list(names))
        self.assertNotIn("<parallel>", compile_script(draft, advanced=False))

        draft.sounds = {"bell": Sound("bell", EFFECT, "A bell rings")}
        draft.staging = {"c1": Staging(events=(SoundEvent("bell", "u2", WITH),))}
        tree = checked(self, compile_script(draft, advanced=True))
        joint = next(node for node in tree.children if isinstance(node, ParallelNode))
        self.assertEqual(len(list(iter_sound_nodes(joint.children[0]))), 0)
        self.assertEqual(len(list(iter_sound_nodes(joint.children[1]))), 1)

    def museum(self):
        u = [
            Utterance("u1", "narrator", "The last visitor left an hour ago."),
            Utterance("u2", "Maya", "Did you hear that? Something moved behind the glass."),
            Utterance("u3", "Arthur", "Probably the vents.", kind=REACTION),
            Utterance("u4", "Arthur", "They do that at night, every night since the new heating went in."),
            Utterance("u5", "Arthur", "I checked it myself last winter."),
            Utterance("u6", "Maya", "Look at the sarcophagus.", kind=INTERRUPT),
            Utterance("u7", "narrator", "The stone lid shifted."),
            Utterance("u8", "Vance", "Dispatch to the museum, come in!", effects=("walkie_talkie",)),
            Utterance("u9", "Maya", "Arthur, we have to go.", effects=("near", "intimate_whisper")),
        ]
        scenes = [Scene("c1", ("u1", "u2", "u3", "u4", "u5", "u6")), Scene("c2", ("u7", "u8", "u9"))]
        sounds = {"hum": Sound("hum", AMBIENCE, "Low ventilation hum in an empty stone gallery"),
                  "drone": Sound("drone", MUSIC, "Slow low cello drone, no vocals"),
                  "scrape": Sound("scrape", EFFECT, "Heavy stone lid scraping across stone"),
                  "static": Sound("static", EFFECT, "Radio squelch and crackling static"),
                  "hit": Sound("hit", STINGER, "Single low orchestral hit, no vocals")}
        staging = {
            "c1": Staging(ambience="hum", positions=(("Maya", -0.4), ("Arthur", 0.4), ("narrator", 0.0)),
                          events=(SoundEvent("scrape", "u2", DURING, "far"),),
                          pauses=(("u5", "long"), ("u6", "short"))),
            "c2": Staging(ambience="hum", music="drone", room="large_hall",
                          events=(SoundEvent("hit", "u7", WITH), SoundEvent("static", "u8", "before", "right"))),
        }
        directions = {"u2": Direction(cue="frozen, barely breathing", phrase="behind the glass"),
                      "u3": Direction(cue="too casual"), "u8": Direction(cue="urgent, brisk")}
        return Draft(u, scenes, directions=directions, sounds=sounds, staging=staging)

    def test_scene_blocks_anchor_events_reactions_and_interruptions(self):
        notes = []
        script = compile_script(self.museum(), advanced=True, notes=notes)
        tree = checked(self, script)
        lines = script.split("\n")
        self.assertEqual(lines[0], '<parallel stage="Maya=-0.4, Arthur=0.4">')
        self.assertEqual(lines[1], '<sound name="hum" loop="true" volume="0.4" duration="25">'
                                   "Low ventilation hum in an empty stone gallery</sound>")
        # Maya's line carries the far scrape partway through and Arthur's reaction in a gap of it.
        block = script.split("Maya: Did you hear", 1)[0].rsplit("<parallel>", 1)[1]
        self.assertEqual(block, "\n")
        after = script.split("Maya: Did you hear", 1)[1].split("</parallel>", 1)[0]
        self.assertIn('<sound name="scrape" duration="4" distance="far">', after)
        self.assertIn("Arthur: Probably [too casual] the vents.", after)
        # Arthur's passage stays one take: the long pause before u5 is refused, u6 barges in.
        self.assertIn("night since the new heating went in.\nArthur: I checked", script)
        self.assertIn("<#0.6#>\n<#-0.3#>\nMaya: Look at the sarcophagus.", script)
        self.assertIn("u5: no pause inside one speaker's passage", notes)
        # Scene two: room, two loops, the stinger with its line, static before the radio voice.
        self.assertIn('<#0.8#>\n<parallel room="large_hall">', script)
        self.assertIn('<sound name="static" duration="4" pan="0.6">', script)
        self.assertIn("Vance: <walkie_talkie> Dispatch to the museum, [urgent, brisk] come in!", script)
        self.assertIn("Maya: <near><intimate_whisper> Arthur, we have to go.", script)
        self.assertEqual([sound.name for sound in iter_sound_nodes(tree)], ["hum", "scrape", "hum", "drone", "hit", "static"])
        self.assertEqual(len(notes), 1, notes)

    def test_invalid_staging_is_dropped_with_notes_and_the_script_stays_valid(self):
        draft = self.museum()
        draft.staging["c1"] = Staging(ambience="scrape", room="cathedral", events=(
            SoundEvent("missing", "u2"), SoundEvent("hum", "u2"), SoundEvent("scrape", "u8"),
            SoundEvent("scrape", "u1"), SoundEvent("scrape", "u2"), SoundEvent("scrape", "u4"),
            SoundEvent("scrape", "u5")))
        notes = []
        script = compile_script(draft, advanced=True, notes=notes)
        checked(self, script)
        self.assertTrue(any("kept the first 4" in note for note in notes), notes)
        self.assertTrue(any("dropped ambience 'scrape'" in note for note in notes), notes)
        self.assertTrue(any("unknown room 'cathedral'" in note for note in notes), notes)
        first_scene = script.split("<#0.8#>", 1)[0]
        self.assertNotIn("room=", first_scene)
        self.assertNotIn('loop="true"', first_scene)

    def test_a_preserved_chinese_script_keeps_its_effects_and_every_word(self):
        source = segment_source(TEMPLE)
        utterances = attribute_units(source, {"s1": "预言之声", "s2": "narrator", "s3": "林夕", "s4": "陈教授"})
        draft = Draft(utterances, [Scene("c1", tuple(u.id for u in utterances), place="ancient temple")],
                      directions={"u1": Direction(cue="ominous, slow"), "u3": Direction(cue="surprised")},
                      sounds={"ring": Sound("ring", EFFECT, "Smartphone ringtone, bright marimba, close indoors"),
                              "rumble": Sound("rumble", AMBIENCE, "Deep earth rumble inside a vast stone temple")},
                      staging={"c1": Staging(ambience="rumble", room="large_hall",
                                             events=(SoundEvent("ring", "u2", DURING),))})
        script = compile_script(draft, advanced=True)
        tree = checked(self, script, TEMPLE)
        self.assertIn("预言之声: <old_gramophone> 当星辰排列成十字，[ominous, slow] 诸神将重返人间。", script)
        self.assertIn("陈教授: <phone_call> 林夕，是我。快离开那里！", script)
        effects = {turn.speaker_effects for node in iter_speech_nodes(tree) for turn in node.turns}
        self.assertIn(("phone_call",), effects)


def _json(data):
    import json

    return json.dumps(data, ensure_ascii=False)


VOICES = {"voices": [
    {"name": "narrator", "description": "A male voice: calm, warm baritone. Clean studio recording.",
     "preview_text": "The evening was quiet. Nothing seemed out of place."},
    {"name": "Tom", "description": "A male voice: thirties, steady. Clean studio recording.",
     "preview_text": "I like a quiet flat. I like knowing who is at the door."},
    {"name": "Nora", "description": "A female voice: late twenties, bright. Clean studio recording.",
     "preview_text": "We only just moved in. I still get lost looking for the light switches."},
]}
PRESERVE = {"title": "The Doorbell", "cast": [{"name": "Tom", "aliases": ["he"]}, {"name": "Nora", "aliases": ["she"]}],
            "speakers": {"s2": "Tom", "s4": "Nora", "s6": "Nora", "s8": "Tom"},
            "scenes": [{"from": "s1", "place": "small flat at night", "change": "calm turns to fear"}]}
DIRECTION = {"lines": [{"id": "u1", "cue": "tense, low"}, {"id": "u8", "cue": "hushed, urgent", "stress": "behind"}]}
SOUND = {"sounds": [{"name": "rain", "kind": "ambience", "prompt": "Light rain on a window"},
                    {"name": "doorbell", "kind": "effect", "prompt": "Old doorbell ringing twice, close indoors"}],
         "scenes": [{"scene": "c1", "ambience": "rain", "music": None, "room": None,
                     "events": [{"sound": "doorbell", "at": "u7", "when": "with"}],
                     "voices": [{"id": "u8", "effects": ["intimate_whisper"]}]}]}


class GuidedWriterTests(unittest.TestCase):
    def test_preview_length_check_ignores_non_cjk_unicode(self):
        import json
        from h3_app.drama3.guided.schema import StageError, parse_voices

        previews = [
            "I’ll take the morning train and meet you near the station. We can walk to the village "
            "together, then find a quiet place for lunch. I will bring the map so we can choose our route.",
            "Je prépare le café pour notre promenade. Nous pourrons discuter tranquillement du chemin "
            "à suivre et regarder les jardins près de la rivière. Je prendrai aussi une carte pour le retour.",
        ]
        def reply(preview):
            return json.dumps({"voices": [{"name": "Tom", "description": "A male voice: calm. Clean studio recording.",
                                          "preview_text": preview}]})

        for preview in previews:
            with self.subTest(preview=preview):
                self.assertEqual(parse_voices(reply(preview), ["Tom"])["Tom"][1], preview)
        with self.assertRaises(StageError):
            parse_voices(reply("风雪交加。"), ["Tom"])

    """The orchestrator end to end, with each stage's model reply mocked by its prompt file."""

    def run_writer(self, text, replies, **kwargs):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from h3_app.prompt_writers import drama3 as writer

        root = Path(__file__).resolve().parents[1]
        runtime = SimpleNamespace(prompt_systems={
            "Drama 3": root / "prompts" / "prompt_drama3.txt",
            "Drama 3 direction": root / "prompts" / "prompt_drama3_direction.txt"})
        self.requests = []

        def fake(**call):
            stage = call["system_path"].stem.removeprefix("prompt_drama3_guided_")
            self.requests.append((stage, call["request_text"]))
            answers = replies[stage]
            answer = answers.pop(0) if isinstance(answers, list) else answers
            if answer is None:
                return "", "Prompt enhancement failed: HTTP 500"
            return (answer if isinstance(answer, str) else _json(answer)), "ok"

        options = dict(script_mode=writer.DRAMA3_SCRIPT_ADVANCED, return_sounds=True, return_title=True)
        options.update(kwargs)
        with patch.object(writer.common, "write_prompt", side_effect=fake):
            return writer.enhance_drama3_text(
                text, "gemini-3.5-flash-lite", "key", options.pop("voices", ()), runtime=runtime,
                workflow=writer.DRAMA3_WORKFLOW_GUIDED, **options)

    def test_preserved_prose_is_cast_directed_scored_and_validated_like_single_pass(self):
        script, status, cast, sounds, title = self.run_writer(KETTLE, {
            "preserve": PRESERVE, "voices": VOICES, "direction": DIRECTION, "sound": SOUND})
        self.assertIn("in 4 Guided steps", status, status)
        self.assertEqual(title, "The Doorbell")
        self.assertEqual([(entry["speaker_id"], entry["name"]) for entry in cast], [(0, "narrator"), (1, "Tom"), (2, "Nora")])
        self.assertIn('<sound name="rain" loop="true"', script)
        self.assertIn("<|speaker:1|> <intimate_whisper> Stay [hushed, urgent] <emphasis>behind</emphasis> me,", script)
        self.assertEqual({sound["name"] for sound in sounds}, {"rain", "doorbell"})
        self.assertNotIn("missing or reworded", status)
        self.assertEqual(sorted(stage for stage, _ in self.requests), ["direction", "preserve", "sound", "voices"])
        units = dict(self.requests)["preserve"]
        self.assertIn("s2 [quote] Are you expecting anyone?", units)
        self.assertIn("Language: the user text is English.", units)

    def test_a_bad_answer_is_retried_once_with_its_problems(self):
        broken = dict(PRESERVE, speakers={"s2": "Tom", "s6": "Nora", "s8": "Tom"})
        script, status, *_ = self.run_writer(KETTLE, {
            "preserve": [broken, PRESERVE], "voices": VOICES, "direction": DIRECTION, "sound": SOUND})
        self.assertNotIn("failed", status)
        retry = [request for stage, request in self.requests if stage == "preserve"][1]
        self.assertIn("- no speaker for quote unit(s) s4", retry)
        self.assertIn("in 5 Guided steps", status)

    def test_a_required_step_that_fails_twice_keeps_the_text(self):
        unsexed = {"voices": [dict(voice, description="Warm and calm.") for voice in VOICES["voices"]]}
        script, status, cast, *_ = self.run_writer(KETTLE, {
            "preserve": PRESERVE, "voices": unsexed, "direction": DIRECTION, "sound": SOUND})
        self.assertTrue(status.startswith("Prompt enhancement failed:"), status)
        self.assertIn("voice design", status)
        self.assertEqual((script, cast), (KETTLE, []))

    def test_optional_steps_that_fail_leave_a_plain_valid_script(self):
        script, status, cast, sounds, _title = self.run_writer(KETTLE, {
            "preserve": PRESERVE, "voices": VOICES, "direction": "not json", "sound": None})
        self.assertNotIn("<sound", script)
        self.assertNotIn("[", script)
        self.assertIn("Guided steps left out", status)
        self.assertEqual(len(cast), 3)

    def test_a_brief_is_written_then_compiled_for_basic_mode(self):
        from h3_app.prompt_writers import drama3 as writer

        brief = {"title": "Night Shift", "cast": [{"name": "Maya", "role": "curator"}, {"name": "Arthur", "role": "guard"}],
                 "scenes": [{"place": "museum", "change": "dread", "lines": [
                     {"speaker": "narrator", "text": "The last visitor left an hour ago."},
                     {"speaker": "Maya", "text": "Did you hear that?"},
                     {"speaker": "Arthur", "text": "Probably the vents."},
                     {"speaker": "Maya", "text": "Wait.", "type": "reaction"}]}]}
        voices = {"voices": [
            {"name": "narrator", "description": "A male voice: calm. Clean studio recording.", "preview_text": "It was late."},
            {"name": "Maya", "description": "A female voice: young. Clean studio recording.", "preview_text": "I work here."},
            {"name": "Arthur", "description": "A male voice: older. Clean studio recording.", "preview_text": "I guard it."}]}
        script, status, cast, sounds, title = self.run_writer(
            "Two people alone in a museum hear something.", {"brief": brief, "voices": voices,
                                                            "direction": {"lines": [{"id": "u2", "cue": "frozen"}]}},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertEqual(title, "Night Shift")
        self.assertEqual(script.split("\n"), ["<|speaker:0|> The last visitor left an hour ago.",
                                              "<|speaker:1|> Did [frozen] you hear that?",
                                              "<|speaker:2|> Probably the vents.", "<|speaker:1|> Wait."])
        self.assertEqual(sounds, [])
        self.assertNotIn("sound", [stage for stage, _ in self.requests])

    def test_adapt_to_voices_casts_only_the_voices_and_gives_them_the_narration(self):
        from h3_app.prompt_writers import drama3 as writer

        voices = {"voices": [dict(VOICES["voices"][1], name="Tom"), dict(VOICES["voices"][2], name="Nora")]}
        script, status, cast, *_ = self.run_writer(KETTLE, {
            "preserve": PRESERVE, "voices": voices, "direction": DIRECTION, "sound": SOUND},
            voices=((3, "Tom"), (5, "Nora")), speaker_mode=writer.DRAMA3_SPEAKER_ADAPT)
        self.assertNotIn("failed", status, status)
        self.assertEqual(sorted((entry["speaker_id"], entry["name"]) for entry in cast), [(3, "Tom"), (5, "Nora")])
        self.assertIn("Available voices (cast only these names): Tom, Nora", dict(self.requests)["preserve"])
        self.assertTrue(script.startswith("<parallel>") or "<|speaker:3|> The kettle" in script, script)

    def test_quoted_brief_and_preserve_brief_handoff_write_scenes_instead_of_preserving_instructions(self):
        from h3_app.prompt_writers import drama3 as writer

        brief = {"title": "Alley Chase", "cast": [{"name": "Cole", "role": "detective"}, {"name": "Vex", "role": "thief"}],
                 "scenes": [{"place": "rainy alley", "change": "cornered", "lines": [
                     {"speaker": "Cole", "text": "Stop! Drop the bag."},
                     {"speaker": "Vex", "text": "Make me."}]}]}
        voices = {"voices": [
            {"name": "Cole", "description": "A male voice: gritty baritone. Clean studio recording.", "preview_text": "Hold it right there."},
            {"name": "Vex", "description": "A female voice: sharp alto. Clean studio recording.", "preview_text": "Catch me if you can."}]}
        script, status, _cast, _sounds, title = self.run_writer(
            'Write a scene where a detective says "Stop!" to a fleeing thief.',
            {"brief": brief, "voices": voices, "direction": {"lines": []}},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        self.assertEqual(title, "Alley Chase")
        self.assertNotIn("preserve", [stage for stage, _ in self.requests])
        self.assertIn("<|speaker:0|> Stop! Drop the bag.", script)

        # A multi-sentence synopsis without brief keywords can still be confirmed as "kind": "brief" by S1.
        synopsis = ("The elevator stops between floors during a citywide blackout. "
                    "Two strangers inside realize both of them are carrying the same locked silver case. "
                    "Neither wants to admit who hired them.")
        script, status, *_ = self.run_writer(
            synopsis,
            {"preserve": {"kind": "brief"}, "brief": brief, "voices": voices, "direction": {"lines": []}},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        self.assertEqual([stage for stage, _ in self.requests][:2], ["preserve", "brief"])
        self.assertIn("<|speaker:0|> Stop! Drop the bag.", script)

    def test_narration_only_story_accepts_empty_cast(self):
        from h3_app.prompt_writers import drama3 as writer

        prose = ("The lighthouse stood dark against the rising tide. "
                 "Wave after wave struck the granite base. "
                 "Inside the lantern room, the brass pendulum swung without a sound.")
        script, status, cast, *_ = self.run_writer(
            prose,
            {"preserve": {"title": "Low Tide", "cast": [], "speakers": {},
                          "scenes": [{"from": "s1", "place": "lighthouse", "change": "storm rises"}]},
             "voices": {"voices": [VOICES["voices"][0]]},
             "direction": {"lines": [{"id": "u1", "cue": "hushed, steady"}]}},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        self.assertEqual([(entry["speaker_id"], entry["name"]) for entry in cast], [(0, "narrator")])
        self.assertIn("<|speaker:0|>", script)

    def test_null_cast_in_brief_triggers_retry_instead_of_crashing(self):
        from h3_app.prompt_writers import drama3 as writer

        bad_brief = {"title": "Night", "cast": None, "scenes": []}
        good_brief = {"title": "Night", "cast": [{"name": "Maya", "role": "curator"}],
                      "scenes": [{"place": "hall", "change": "quiet", "lines": [{"speaker": "Maya", "text": "Hello?"}]}]}
        voices = {"voices": [{"name": "Maya", "description": "A female voice: young. Clean studio recording.",
                              "preview_text": "I work in the gallery."}]}
        script, status, *_ = self.run_writer(
            "Maya is alone in the hall.",
            {"brief": [bad_brief, good_brief], "voices": voices, "direction": {"lines": []}},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        self.assertIn("in 4 Guided steps", status)
        self.assertIn("<|speaker:0|> Hello?", script)

    def test_single_speaker_mode_works_on_briefs_and_respects_named_slot(self):
        import json
        from h3_app.prompt_writers import drama3 as writer

        brief = {"title": "Solo", "cast": [{"name": "Maya", "role": "curator"}, {"name": "Arthur", "role": "guard"}],
                 "scenes": [{"place": "museum", "change": "dread", "lines": [
                     {"speaker": "narrator", "text": "The gallery was dark."},
                     {"speaker": "Maya", "text": "Who is there?"}]}]}
        voices = {"voices": [{"name": "Storyteller", "description": "A female voice: warm. Clean studio recording.",
                              "preview_text": "I tell the old stories every evening."}]}
        cast_context = ("Existing slot identities:\n"
                        + json.dumps([{"speaker_id": 0, "name": "Storyteller", "description": "", "has_reference": False}]))
        script, status, cast, *_ = self.run_writer(
            "A short museum ghost story.",
            {"brief": brief, "voices": voices, "direction": {"lines": []}},
            speaker_mode=writer.DRAMA3_SPEAKER_SINGLE,
            cast_context=cast_context,
            speaker_names={0: "Storyteller"},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        self.assertEqual([(entry["speaker_id"], entry["name"]) for entry in cast], [(0, "Storyteller")])
        self.assertEqual(script.split("\n"), ["<|speaker:0|> The gallery was dark.", "<|speaker:0|> Who is there?"])

    def test_adapt_to_voices_gives_narration_to_unused_voice_and_retries_when_a_voice_is_unused(self):
        from h3_app.prompt_writers import drama3 as writer

        one_character_prose = 'The corridor was dark.\n"Stay back," Tom warned.\nThe door clicked shut.'
        preserve_one = {"title": "Hall", "cast": [{"name": "Tom"}], "speakers": {"s2": "Tom"},
                        "scenes": [{"from": "s1", "place": "hall", "change": "door shuts"}]}
        voices = {"voices": [dict(VOICES["voices"][1], name="Tom"), dict(VOICES["voices"][2], name="Nora")]}
        script, status, cast, *_ = self.run_writer(
            one_character_prose,
            {"preserve": preserve_one, "voices": voices, "direction": {"lines": []}},
            voices=((3, "Tom"), (5, "Nora")),
            speaker_mode=writer.DRAMA3_SPEAKER_ADAPT,
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        # Tom speaks s2; Nora (unused in dialogue) gets the narration so both reference voices are used.
        self.assertEqual(sorted((entry["speaker_id"], entry["name"]) for entry in cast), [(3, "Tom"), (5, "Nora")])
        self.assertIn("The narration is read by Nora", status)
        self.assertIn("<|speaker:5|> The corridor was dark.", script)
        self.assertIn("<|speaker:3|> Stay back,", script)

    def test_tagged_script_and_cue_hints_round_trip_through_guided_writer(self):
        import json
        from h3_app.prompt_writers import drama3 as writer

        tagged = '<|speaker:2|> (whispering): "Hold still."\n<|speaker:4|>: "I hear them."'
        preserve = {"title": "Hideout", "cast": [{"name": "Maya", "aliases": ["Speaker 2"]},
                                                 {"name": "Arthur", "aliases": ["Speaker 4"]}],
                    "speakers": {}, "scenes": [{"from": "s1", "place": "cellar", "change": "steps pass"}]}
        voices = {"voices": [
            {"name": "Maya", "description": "A female voice: young. Clean studio recording.", "preview_text": "Quiet now."},
            {"name": "Arthur", "description": "A male voice: older. Clean studio recording.", "preview_text": "I am listening."}]}
        cast_context = (
            "Existing slot identities:\n"
            + json.dumps([{"speaker_id": 2, "name": "Maya", "description": "", "has_reference": False},
                          {"speaker_id": 4, "name": "Arthur", "description": "", "has_reference": False}])
            + "\nReusable speaker voices:\n"
            + json.dumps([{"name": "Maya", "description": "A female voice: custom preset description. Clean studio recording."}])
        )
        script, status, cast, *_ = self.run_writer(
            tagged,
            {"preserve": preserve, "voices": voices, "direction": {"lines": []}},
            cast_context=cast_context,
            speaker_names={2: "Maya", 4: "Arthur"},
            script_mode=writer.DRAMA3_SCRIPT_BASIC)
        self.assertNotIn("failed", status, status)
        self.assertEqual([(entry["speaker_id"], entry["name"]) for entry in cast], [(2, "Maya"), (4, "Arthur")])
        self.assertIn("custom preset description", cast[0]["description"])
        # The parenthesized hint (whispering) is applied even when the direction stage returns no cues.
        self.assertIn("<|speaker:2|> Hold [whispering] still.", script)

    def test_cjk_optimizations_joint_speakers_fuzzy_cast_attribution_and_preview_length(self):
        from h3_app.prompt_writers import drama3 as writer

        prose = (
            "“马上撤离！”寒骊上人神色一凝的说道。\n"
            "“是，宫主！”白瑶怡和灰发老者同时恭声应道。\n"
            "寒骊上人立刻激发法阵，仿佛受到了什么刺|激。"
        )
        # "寒骊0" resolves to "寒骊上人"; s3 is spoken jointly by 白瑶怡 and 灰发老者.
        preserve = {
            "title": "群妖压境",
            "cast": [{"name": "寒骊上人"}, {"name": "白瑶怡"}, {"name": "灰发老者"}],
            "speakers": {"s1": "寒骊0", "s3": ["白瑶怡", "老者"]},
            "scenes": [{"from": "s1", "place": "stone hall", "change": "orders given"}],
        }
        short_voices = {"voices": [
            {"name": "narrator", "description": "A male voice: calm. Clean studio recording.", "preview_text": "风雪交加。"},
            {"name": "寒骊上人", "description": "A male voice: deep. Clean studio recording.", "preview_text": "马上撤离。"},
            {"name": "白瑶怡", "description": "A female voice: clear. Clean studio recording.", "preview_text": "是，宫主。"},
            {"name": "灰发老者", "description": "A male voice: old. Clean studio recording.", "preview_text": "是，宫主。"},
        ]}
        full_preview = "北冥岛外寒风呼啸，漫天飞雪将整座冰原笼罩其中，所有修士都在各自做着最后的准备。"
        good_voices = {"voices": [dict(v, preview_text=full_preview) for v in short_voices["voices"]]}
        direction = {"lines": [
            {"id": "u1", "cue": "commanding, urgent"},
            {"id": "u2", "cue": "neutral narrator, low voice"},  # 11-char CJK attribution: cue ignored
            {"id": "u3", "cue": "respectful, crisp"},
            {"id": "u4", "cue": "solemn, firm"},
            {"id": "u6", "cue": "narrative, tense and low", "stress": "受到"},  # meta "narrative," stripped; 寒骊上人 not split
        ]}
        script, status, cast, *_ = self.run_writer(
            prose,
            {"preserve": preserve, "voices": [short_voices, good_voices], "direction": direction},
            script_mode=writer.DRAMA3_SCRIPT_BASIC,
        )
        self.assertNotIn("failed", status, status)
        self.assertEqual([entry["name"] for entry in cast], ["寒骊上人", "narrator", "白瑶怡", "灰发老者"])
        # Both 白瑶怡 and 灰发老者 speak the joint quote s3.
        self.assertIn("<|speaker:2|> 是，[respectful, crisp] 宫主！", script)
        self.assertIn("<|speaker:3|> 是，[solemn, firm] 宫主！", script)
        # 11-char attribution u2 stays plain; u6 cleans "narrative," and keeps "寒骊上人" intact; "刺|激" is cleaned to "刺激".
        self.assertIn("<|speaker:1|> 寒骊上人神色一凝的说道。", script)
        self.assertNotIn("寒骊 [", script)
        self.assertIn("刺激", script)
        self.assertNotIn("刺|激", script)

    def test_max_batch_lines_merges_scenes_into_fewer_requests(self):
        from h3_app.drama3.guided.model import Scene
        from h3_app.prompt_writers.drama3_guided import _batches

        scenes = [
            Scene("c1", tuple(f"u{i}" for i in range(1, 31)), "hall", "tense"),
            Scene("c2", tuple(f"u{i}" for i in range(31, 61)), "roof", "battle"),
            Scene("c3", tuple(f"u{i}" for i in range(61, 86)), "sky", "escape"),
        ]
        # Default 40 splits 30 + 30 + 25 = 85 utterances across 3 batches.
        self.assertEqual(len(_batches(scenes)), 3)
        self.assertEqual(len(_batches(scenes, 40)), 3)
        # Increasing max_batch_lines to 60 merges c1+c2 into 1 batch (2 batches total).
        self.assertEqual(len(_batches(scenes, 60)), 2)
        # Increasing max_batch_lines to 100 merges all 3 scenes (85 utterances) into a single batch!
        self.assertEqual(len(_batches(scenes, 100)), 1)


if __name__ == "__main__":
    unittest.main()
