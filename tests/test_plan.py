import pytest

from longdrive.plan import (
    EVOLUTIONS,
    MAX_PROMPT_CHARS,
    PHRASE_BARS,
    Plan,
    Timing,
    make_plan,
    parse_bpm,
    parse_journey,
    parse_key,
    related_keys,
)


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
    plan = make_plan("pads", 900, seed=5)
    assert Plan.from_json(plan.to_json()) == plan


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
