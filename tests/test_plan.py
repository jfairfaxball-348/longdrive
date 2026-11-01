import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from longdrive.plan import (
    EVOLUTIONS,
    MAX_PROMPT_CHARS,
    PHRASE_BARS,
    Plan,
    Timing,
    journey_text,
    make_plan,
    parse_bpm,
    parse_journey,
    parse_key,
    parse_wordlist,
    related_keys,
)

WORDLISTS = Path(__file__).resolve().parents[1] / "wordlists"
STARTER_LISTS = ["funky", "ambient", "guitar-spa"]
EVERY_3_5M = replace(EVOLUTIONS["gentle"], section_s=210.0)

GROUPED_LIST = """
# A comment.
bpm: 110

## genre (pick 1)
micro-funk
disco house

## drums (pick 1)
soft hi-hats
rimshot

## bass (pick 1)
rubbery bass
rolling bassline
sub bass

## instruments (pick 2)
electric piano
clavinet
chord stabs
brass stabs

## texture (pick 1)
tape hiss
spring reverb

## mood (pick 2)
hypnotic
warm
clean
"""


def test_segments_cover_the_duration_exactly():
    plan = make_plan("pads", 3600, seed=1)
    assert sum(s.new_s for s in plan.segments) == pytest.approx(3600)
    for before, after in zip(plan.segments, plan.segments[1:], strict=False):
        assert after.start_s == pytest.approx(before.start_s + before.new_s)


def test_last_segment_is_trimmed_to_fit():
    plan = make_plan("pads", 200, seed=1, bpm=60, timing=Timing(hop_s=64))
    assert [s.new_s for s in plan.segments] == [64, 64, 64, 8]


def test_duration_shorter_than_one_hop_is_one_segment():
    plan = make_plan("pads", 45, seed=1)
    assert len(plan.segments) == 1
    assert plan.segments[0].new_s == 45


def test_same_seed_gives_the_same_plan():
    assert make_plan("pads", 900, seed=5) == make_plan("pads", 900, seed=5)


def test_different_seeds_give_different_music():
    assert make_plan("pads", 900, seed=5).segments != make_plan("pads", 900, seed=6).segments


def test_plan_survives_a_json_round_trip():
    plan = make_plan("pads", 900, seed=5, drums_db=-60, kick_db=-9)
    assert Plan.from_json(plan.to_json()) == plan


def test_a_plan_saved_before_drum_swaps_still_loads():
    saved = json.loads(make_plan("pads", 900, seed=5).to_json())
    del saved["drums_db"], saved["kick_db"]
    assert not Plan.from_json(json.dumps(saved)).swaps_drums


def test_a_list_can_set_its_drum_levels():
    words = parse_wordlist("bpm: 110\ndrums: -60\nkick: -9.5\nwarm bass")
    plan = make_plan("", 60, seed=1, words=words)
    assert (plan.drums_db, plan.kick_db) == (-60, -9.5)
    assert plan.swaps_drums


def test_explicit_drum_levels_beat_the_list():
    words = parse_wordlist("bpm: 110\ndrums: -60\nkick: -9\nwarm bass")
    plan = make_plan("", 60, seed=1, words=words, drums_db=-12, kick_db=-20)
    assert (plan.drums_db, plan.kick_db) == (-12, -20)


def test_tempo_and_key_are_read_from_the_prompt():
    plan = make_plan("slow strings in F# minor at 54 BPM", 60, seed=1)
    assert plan.bpm == 54
    assert plan.key == "F# minor"


def test_explicit_tempo_and_key_win_over_the_prompt():
    plan = make_plan("strings in F# minor at 54 BPM", 60, seed=1, bpm=70, key="C major")
    assert (plan.bpm, plan.key) == (70, "C major")


def test_defaults_are_slow():
    assert all(40 <= make_plan("pads", 60, seed=s).bpm <= 50 for s in range(30))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("A minor drone", "A minor"), ("Bb-maj choir", "Bb major"), ("a minor change", None)],
)
def test_parse_key(text, expected):
    assert parse_key(text) == expected


def test_parse_bpm_ignores_other_numbers():
    assert parse_bpm("1970s tape pads") is None


def test_the_prompt_is_kept_exactly(prompt=" warm piano, 110 BPM. Calm and steady. "):
    plan = make_plan(prompt, 60, seed=1)
    assert plan.segments[0].prompt == prompt.strip()


def test_long_prompts_are_refused_rather_than_cut():
    assert make_plan("x" * MAX_PROMPT_CHARS, 60, seed=1)
    with pytest.raises(ValueError, match="cut off"):
        make_plan("x" * (MAX_PROMPT_CHARS + 1), 60)


def test_a_steady_plan_has_one_movement():
    plan = make_plan("pads", 3600, seed=1)
    assert plan.movements == 1
    assert {s.prompt for s in plan.segments} == {"pads"}
    assert {s.key for s in plan.segments} == {plan.key}


LONG_PROMPT = (
    "Instrumental deep house, 110 BPM. Tight kick drums, warm bass, soft hi-hats, chord stabs "
    "and melodic fragments. Warm analog textures, gentle filter modulation and spacious "
    "backgrounds. Smooth groove and relaxing energy."
)


def test_a_long_prompt_is_varied_in_its_own_words():
    plan = make_plan(LONG_PROMPT, 1800, seed=1, evolution=EVOLUTIONS["gentle"])
    prompts = {s.prompt for s in plan.segments}
    assert plan.segments[0].prompt == LONG_PROMPT
    assert len(prompts) == plan.movements
    for prompt in prompts:
        assert prompt.startswith("Instrumental deep house, 110 BPM.")
    allowed = LONG_PROMPT.replace(".", ",").replace(" and ", ",").split(",")
    allowed = {phrase.strip() for phrase in allowed}
    for prompt in prompts - {LONG_PROMPT}:
        rest = prompt.split("110 BPM.")[1].rstrip(".")
        assert {phrase.strip() for phrase in rest.split(",")} <= allowed


def test_variations_differ_substantially():
    plan = make_plan(LONG_PROMPT, 3600, seed=1, evolution=EVOLUTIONS["strong"])
    lengths = [len(s.prompt) for s in plan.segments if s.fresh]
    assert max(lengths) < 0.75 * len(LONG_PROMPT)


def test_a_short_prompt_gets_a_direction_instead():
    plan = make_plan("Warm pads.", 3600, seed=1, evolution=EVOLUTIONS["gentle"])
    assert plan.segments[0].prompt == "Warm pads."
    assert plan.segments[-1].prompt.startswith("Warm pads. ")
    assert plan.segments[-1].prompt != "Warm pads."


def test_an_hour_of_gentle_evolution_has_movements_in_several_keys():
    plan = make_plan("pads", 3600, seed=1, key="A minor", evolution=EVOLUTIONS["gentle"])
    assert plan.movements == 6
    assert len({s.key for s in plan.segments}) >= 2


def test_strong_evolution_moves_further_and_comes_home():
    gentle = make_plan("pads", 3600, seed=1, key="A minor", evolution=EVOLUTIONS["gentle"])
    plan = make_plan("pads", 3600, seed=1, key="A minor", evolution=EVOLUTIONS["strong"])
    assert plan.movements > gentle.movements
    assert len({s.key for s in plan.segments}) >= 3
    assert plan.segments[0].key == plan.segments[-1].key == "A minor"


def test_movements_start_fresh_and_blend_out():
    journey = parse_journey("0:00 A.\n2:08 B.\n4:16 C.")
    plan = make_plan("pads", 400, seed=1, bpm=60, journey=journey, timing=Timing(hop_s=64))
    fresh = [s.index for s in plan.segments if s.fresh]
    assert fresh == [2, 4]
    assert plan.movements == 3
    for index in fresh:
        assert plan.segments[index - 1].overlap_s == plan.timing.blend_s
    assert plan.segments[0].overlap_s == plan.timing.crossfade_s


def test_the_blend_must_fit_inside_the_tail():
    with pytest.raises(ValueError, match="tail_s"):
        Timing(tail_s=10, blend_s=20)


def test_journeys_are_reproducible():
    one = make_plan("pads", 3600, seed=5, evolution=EVOLUTIONS["strong"])
    assert one == make_plan("pads", 3600, seed=5, evolution=EVOLUTIONS["strong"])


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("A minor", ("C major", "E minor", "D minor")),
        ("C major", ("A minor", "G major", "F major")),
        ("F# minor", ("A major", "C# minor", "B minor")),
        ("Db major", ("Bb minor", "Ab major", "F# major")),
    ],
)
def test_related_keys(key, expected):
    assert related_keys(key) == expected


def test_a_journey_is_read_from_a_timeline():
    text = """
    # the first section keeps the home key
    0:00 Just the groove.
    12:00 [C major] Chords come forward.
    1:05:30 Everything falls away.
    """
    sections = parse_journey(text)
    assert [s.start_s for s in sections] == [0, 720, 3930]
    assert [s.key for s in sections] == [None, "C major", None]
    assert sections[2].prompt == "Everything falls away."


def test_a_key_lasts_until_the_next_one():
    journey = parse_journey("0:00 A.\n2:00 [C major] B.\n4:00 C.")
    plan = make_plan("pads", 360, seed=1, bpm=60, key="A minor", journey=journey)
    assert plan.segments[0].key == "A minor"
    assert plan.segments[-1].key == "C major"


def test_a_journey_line_that_cannot_be_read_is_an_error():
    with pytest.raises(ValueError, match="line 2"):
        parse_journey("0:00 Fine.\nnot a time")


def test_a_journey_needs_no_separate_prompt():
    plan = make_plan("", 120, seed=1, journey=parse_journey("0:00 Warm pads, 90 BPM."))
    assert plan.prompt == "Warm pads, 90 BPM."
    assert plan.bpm == 90


def test_each_movement_of_a_journey_has_its_own_prompt():
    journey = parse_journey("0:00 First idea.\n2:08 Second idea.")
    plan = make_plan("", 260, seed=1, bpm=60, journey=journey, timing=Timing(hop_s=64))
    assert plan.segments[0].prompt == "First idea."
    assert plan.segments[-1].prompt == "Second idea."


def test_a_section_with_only_a_key_keeps_the_prompt_it_had():
    journey = parse_journey("0:00 Same idea.\n2:08 [C major]")
    timing = Timing(hop_s=64)
    plan = make_plan("", 260, seed=1, bpm=60, key="A minor", journey=journey, timing=timing)
    assert plan.segments[-1].prompt == "Same idea."
    assert plan.segments[-1].key == "C major"
    assert plan.movements == 2


def test_the_first_section_sets_the_key_of_the_piece():
    plan = make_plan("", 60, seed=1, journey=parse_journey("0:00 [E minor] An idea."))
    assert plan.key == "E minor"


def test_no_movement_can_be_longer_than_the_limit():
    journey = parse_journey("0:00 Fine.\n1:00 " + "x" * (MAX_PROMPT_CHARS + 1))
    with pytest.raises(ValueError, match="cut off"):
        make_plan("", 120, seed=1, journey=journey)


def test_context_must_fit_inside_the_previous_hop():
    with pytest.raises(ValueError):
        Timing(hop_s=20, context_s=30)


def test_crossfade_must_fit_inside_the_tail():
    with pytest.raises(ValueError):
        Timing(tail_s=2, crossfade_s=3, blend_s=2)


@pytest.mark.parametrize("duration", [0, 5])
def test_rejects_durations_the_model_cannot_render(duration):
    with pytest.raises(ValueError):
        make_plan("pads", duration)


def test_rejects_empty_prompts():
    with pytest.raises(ValueError):
        make_plan("   ", 60)


@pytest.mark.parametrize(
    ("bpm", "signature"), [(110, "4"), (124, "4"), (60, "4"), (44, "4"), (72, "3")]
)
def test_joins_land_on_phrase_boundaries(bpm, signature):
    plan = make_plan("pads", 3600, seed=1, bpm=bpm, time_signature=signature)
    bar_s = 60 / bpm * int(signature)
    for segment in plan.segments:
        phrases = segment.start_s / (PHRASE_BARS * bar_s)
        assert phrases == pytest.approx(round(phrases))
    bars = plan.timing.crossfade_s / bar_s
    assert bars == pytest.approx(round(bars))


def test_snapping_keeps_the_exact_duration():
    plan = make_plan("pads", 3600, seed=1, bpm=58)
    assert sum(s.new_s for s in plan.segments) == pytest.approx(3600)


def section_starts(plan):
    return [s.start_s for s in plan.segments if s.index == 0 or s.fresh]


def section_prompts(plan):
    return [s.prompt for s in plan.segments if s.index == 0 or s.fresh]


def has_extras(prompt):
    """Whether a prompt from GROUPED_LIST names anything from its instruments or texture."""
    extras = ("electric piano", "clavinet", "chord stabs", "brass stabs", "tape hiss", "reverb")
    return any(extra in prompt.lower() for extra in extras)


def test_a_section_starts_on_the_nearest_window():
    journey = parse_journey("0:00 A.\n3:30 B.\n7:00 C.\n10:29 D.")
    plan = make_plan("", 840, seed=1, bpm=110, journey=journey)
    assert section_starts(plan) == pytest.approx([0, 209.45, 418.91, 628.36], abs=0.01)


def test_a_section_never_slips_on_a_rounding_error():
    plan = make_plan("", 600, seed=1, bpm=110, journey=parse_journey("0:00 A.\n3:29 B."))
    assert section_starts(plan)[1] == pytest.approx(209.45, abs=0.01)


def test_a_section_too_close_to_the_last_is_dropped():
    journey = parse_journey("0:00 A.\n3:29 B.\n4:20 C.\n7:00 D.")
    plan = make_plan("", 840, seed=1, bpm=110, journey=journey)
    assert section_prompts(plan) == ["A.", "B.", "D."]


def test_a_section_in_the_closing_fade_is_dropped():
    journey = parse_journey("0:00 A.\n29:15 B.\n59:21 C.")
    plan = make_plan("", 3600, seed=1, bpm=110, journey=journey)
    assert section_prompts(plan) == ["A.", "B."]


@pytest.mark.parametrize("bpm", [90, 100, 110, 118, 124, 128, 140])
def test_section_spacing_is_whole_windows_at_any_tempo(bpm):
    words = replace(parse_wordlist(GROUPED_LIST), bpm=bpm)
    plan = make_plan("", 3600, seed=1, words=words, evolution=EVERY_3_5M)
    starts = section_starts(plan)
    gaps = [after - before for before, after in zip(starts, starts[1:], strict=False)]
    assert gaps == pytest.approx([gaps[0]] * len(gaps))
    windows = gaps[0] / plan.timing.hop_s
    assert windows == pytest.approx(round(windows))


def test_spacing_leaves_every_section_time_to_play_alone():
    plan = make_plan("pads", 3600, seed=1, bpm=110, evolution=replace(EVERY_3_5M, section_s=10.0))
    starts = section_starts(plan)
    assert min(b - a for a, b in zip(starts, starts[1:], strict=False)) > plan.timing.blend_s


def test_an_hour_every_three_and_a_half_minutes_has_17_even_sections():
    plan = make_plan(
        "", 3600, seed=1, words=parse_wordlist(GROUPED_LIST), evolution=EVERY_3_5M
    )
    starts = section_starts(plan)
    assert len(starts) == plan.movements == 17
    assert starts == pytest.approx([index * 209.4545 for index in range(17)], abs=0.01)
    assert len(plan.segments) == 69


def test_evolve_follows_a_section_length_too():
    plan = make_plan(LONG_PROMPT, 3600, seed=1, evolution=EVERY_3_5M)
    assert plan.movements == 17


def test_a_word_list_is_read_by_group():
    words = parse_wordlist(GROUPED_LIST)
    assert words.bpm == 110
    assert [group.name for group in words.groups] == [
        "genre", "drums", "bass", "instruments", "texture", "mood",
    ]
    assert [group.count for group in words.groups] == [1, 1, 1, 2, 1, 2]
    assert words.groups[1].entries == ("soft hi-hats", "rimshot")


def test_a_list_with_no_headings_is_one_group():
    words = parse_wordlist("\n".join(f"word {index}" for index in range(50)))
    assert len(words.groups) == 1
    assert words.groups[0].count == 10
    assert len(words.groups[0].entries) == 50


def test_a_short_flat_list_draws_everything_it_has():
    assert parse_wordlist("one\ntwo\nthree").groups[0].count == 3


def test_a_heading_with_no_count_picks_one():
    assert parse_wordlist("## bass\nsub bass").groups[0].count == 1


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "no entries"),
        ("# just a comment", "no entries"),
        ("## bass (pick 3)\none\ntwo", "pick 3"),
        ("## bass (pick 0)\none", "pick 0"),
        ("## empty (pick 1)\n## bass (pick 1)\none", "empty"),
        ("## bass pick two\none", "line 1"),
        ("frame: no place for the draw\none", "entries"),
    ],
)
def test_a_list_that_cannot_be_used_is_an_error(text, match):
    with pytest.raises(ValueError, match=match):
        parse_wordlist(text)


def test_a_word_list_needs_a_tempo():
    words = parse_wordlist("one\ntwo")
    with pytest.raises(ValueError, match="tempo"):
        make_plan("", 600, seed=1, words=words)


def test_an_explicit_tempo_beats_the_list():
    plan = make_plan("", 600, seed=1, bpm=120, words=parse_wordlist(GROUPED_LIST))
    assert plan.bpm == 120
    assert "120 BPM" in plan.segments[0].prompt


def test_every_section_shares_the_tempo():
    plan = make_plan(
        "", 3600, seed=1, words=parse_wordlist(GROUPED_LIST), evolution=EVERY_3_5M
    )
    assert {parse_bpm(prompt) for prompt in section_prompts(plan)} == {110}


def test_the_same_seed_draws_the_same_prompts():
    words = parse_wordlist(GROUPED_LIST)
    one = make_plan("", 3600, seed=5, words=words, evolution=EVERY_3_5M)
    assert one == make_plan("", 3600, seed=5, words=words, evolution=EVERY_3_5M)
    assert one != make_plan("", 3600, seed=6, words=words, evolution=EVERY_3_5M)


def test_every_prompt_has_the_frame_around_it():
    plan = make_plan(
        "", 3600, seed=1, words=parse_wordlist(GROUPED_LIST), evolution=EVERY_3_5M
    )
    for prompt in section_prompts(plan):
        assert re.match(r"Instrumental (micro-funk|disco house), 110 BPM\. ", prompt)
        assert prompt.endswith(". Smooth uninterrupted groove.")


def test_a_list_can_set_its_own_frame():
    words = parse_wordlist("frame: {genre} at {bpm}: {entries}\nbpm: 90\n## genre\nfunk\n## a\nb")
    plan = make_plan("", 60, seed=1, words=words)
    assert plan.segments[0].prompt == "funk at 90: B"


def test_a_flat_list_needs_no_genre():
    plan = make_plan("", 60, seed=1, bpm=110, words=parse_wordlist("warm bass"))
    assert plan.segments[0].prompt.startswith("Instrumental music, 110 BPM.")


def test_every_entry_is_used_before_any_repeats():
    entries = [f"word {index}" for index in range(50)]
    plan = make_plan(
        "", 3600, seed=2, bpm=110, words=parse_wordlist("\n".join(entries)), evolution=EVERY_3_5M
    )
    # The closing section repeats the opening, so it says nothing about the shuffle.
    drawn = [
        word
        for prompt in section_prompts(plan)[:-1]
        for word in re.findall(r"word \d+", prompt.lower())
    ]
    assert len(set(drawn[:50])) == 50
    assert len(set(drawn[50:100])) == 50


def test_no_entry_is_drawn_twice_for_one_prompt():
    words = parse_wordlist("## mood (pick 3)\nwarm\nclean\nhazy\ndark")
    plan = make_plan("", 3600, seed=3, bpm=110, words=words, evolution=EVERY_3_5M)
    for prompt in section_prompts(plan):
        found = re.findall(r"warm|clean|hazy|dark", prompt.lower())
        assert len(found) == len(set(found)) == 3


def test_a_piece_starts_and_ends_stripped_back_on_the_same_draw():
    plan = make_plan(
        "", 3600, seed=1, words=parse_wordlist(GROUPED_LIST), evolution=EVERY_3_5M
    )
    prompts = section_prompts(plan)
    assert prompts[0] == prompts[-1]
    assert plan.segments[0].key == plan.segments[-1].key
    assert not has_extras(prompts[0])
    assert all(has_extras(prompt) for prompt in prompts[1:-1])
    assert len(set(prompts[:-1])) == 16


def test_a_short_piece_is_not_stripped_back():
    words = parse_wordlist(GROUPED_LIST)
    plan = make_plan("", 420, seed=1, words=words, evolution=EVERY_3_5M)
    assert plan.movements == 2
    prompts = section_prompts(plan)
    assert len(set(prompts)) == 2
    assert has_extras(prompts[0])


def test_evolve_off_draws_one_full_prompt():
    plan = make_plan("", 3600, seed=1, words=parse_wordlist(GROUPED_LIST))
    assert plan.movements == 1
    assert {s.prompt for s in plan.segments} == {plan.prompt}


def test_a_flat_list_can_still_be_bookended():
    words = parse_wordlist("\n".join(f"word {index}" for index in range(50)))
    plan = make_plan("", 3600, seed=1, bpm=110, words=words, evolution=EVERY_3_5M)
    prompts = section_prompts(plan)
    assert prompts[0] == prompts[-1]
    assert len(re.findall(r"word \d+", prompts[0].lower())) == 10


def test_keys_move_and_come_home():
    plan = make_plan(
        "", 3600, seed=1, key="A minor", words=parse_wordlist(GROUPED_LIST), evolution=EVERY_3_5M
    )
    assert len({s.key for s in plan.segments}) >= 3
    assert plan.segments[-1].key == "A minor"


def test_a_saved_timeline_gives_back_the_same_sections():
    plan = make_plan(
        "", 3600, seed=4, key="A minor", words=parse_wordlist(GROUPED_LIST), evolution=EVERY_3_5M
    )
    journey = parse_journey(journey_text(plan))
    assert len(journey) == 17
    again = make_plan("", 3600, seed=4, journey=journey)
    assert section_prompts(again) == section_prompts(plan)
    assert section_starts(again) == pytest.approx(section_starts(plan))
    assert [s.key for s in again.segments] == [s.key for s in plan.segments]
    assert again.bpm == plan.bpm


@pytest.mark.parametrize("name", STARTER_LISTS)
def test_starter_lists_hold_about_fifty_entries_and_draw_ten(name):
    words = parse_wordlist((WORDLISTS / f"{name}.txt").read_text(encoding="utf-8"))
    assert words.bpm
    assert 45 <= sum(len(group.entries) for group in words.groups) <= 60
    assert sum(group.count for group in words.groups if group.name != "genre") == 10
    names = [group.name for group in words.groups]
    assert {"genre", "bass", "instruments", "texture", "mood"} <= set(names)


@pytest.mark.parametrize("name", STARTER_LISTS)
def test_starter_lists_swap_the_drums(name):
    words = parse_wordlist((WORDLISTS / f"{name}.txt").read_text(encoding="utf-8"))
    assert words.drums_db is not None
    assert words.kick_db is not None


@pytest.mark.parametrize("name", STARTER_LISTS)
def test_starter_lists_keep_to_a_steady_simple_groove(name):
    words = parse_wordlist((WORDLISTS / f"{name}.txt").read_text(encoding="utf-8"))
    entries = " ".join(entry for group in words.groups for entry in group.entries).lower()
    structure = r"\b(breakdown|drop|intro|outro|solo|vocals?|verse|chorus)\b"
    busy_drums = r"\b(intricate|fills?|rolls?|shuffl\w+|shakers?|breakbeat|tribal|congas?)\b"
    assert not re.findall(structure, entries)
    assert not re.findall(busy_drums, entries)


@pytest.mark.parametrize("name", STARTER_LISTS)
def test_an_hour_from_a_starter_list_fits_the_prompt_limit(name):
    words = parse_wordlist((WORDLISTS / f"{name}.txt").read_text(encoding="utf-8"))
    for seed in range(20):
        plan = make_plan("", 3600, seed=seed, words=words, evolution=EVERY_3_5M)
        assert plan.movements == 17
        assert max(len(prompt) for prompt in section_prompts(plan)) < MAX_PROMPT_CHARS / 2
