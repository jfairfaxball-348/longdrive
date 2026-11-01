import argparse

import pytest

from longdrive import cli


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("3600", 3600),
        ("90s", 90),
        ("45m", 2700),
        ("1.5h", 5400),
        ("1h30m", 5400),
        ("2h15m30s", 8130),
        (" 10M ", 600),
    ],
)
def test_parse_duration(text, seconds):
    assert cli.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "1h30", "-5m", "h"])
def test_parse_duration_rejects_nonsense(text):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_duration(text)


def test_slug_is_safe_for_a_directory_name():
    assert cli.slug('Warm "piano" & pads: 60 BPM!') == "warm-piano-pads-60-bpm"
    assert cli.slug("!!!") == "piece"
    assert len(cli.slug("x" * 200)) <= 40


def test_generate_and_master_with_the_synth(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    arguments = ["generate", "soft drone", "-d", "30", "--engine", "synth", "--name", "demo"]
    assert cli.main(arguments) == 0
    assert (tmp_path / "demo" / "demo.wav").exists()
    assert cli.main(["status", "demo"]) == 0
    assert "complete" in capsys.readouterr().out
    assert cli.main(["master", "demo"]) == 0


def test_unusual_characters_in_a_prompt_survive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    arguments = ["generate", "ambient ♪ 雨", "-d", "12", "--engine", "synth", "--name", "unicode"]
    assert cli.main(arguments) == 0
    assert (tmp_path / "unicode" / "unicode.wav").exists()


def test_dry_run_renders_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    assert cli.main(["generate", "soft drone", "-d", "10m", "--dry-run", "--seed", "1"]) == 0
    assert '"duration_s": 600.0' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_quality_sets_the_number_of_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    arguments = ["generate", "drone", "-d", "12", "--engine", "synth", "--name", "demo"]
    assert cli.main([*arguments, "-q", "draft"]) == 0
    assert cli.Run(tmp_path / "demo").state["candidates"] == cli.CANDIDATES["draft"]


def test_model_choice_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    arguments = ["generate", "drone", "-d", "12", "--engine", "synth", "--name", "demo"]
    assert cli.main([*arguments, "-m", "sft", "-q", "draft"]) == 0
    assert cli.Run(tmp_path / "demo").settings == cli.MODELS["sft"]


def test_a_finished_name_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    arguments = ["generate", "soft drone", "-d", "12", "--engine", "synth", "--name", "demo"]
    assert cli.main(arguments) == 0
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(arguments)


def test_a_journey_file_replaces_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    journey = tmp_path / "journey.txt"
    journey.write_text("0:00 Soft drone.\n1:04 [C major] Bright drone.\n", encoding="utf-8")
    arguments = ["generate", "--journey", str(journey), "-d", "130", "--bpm", "120"]
    assert cli.main([*arguments, "--engine", "synth", "--name", "demo", "-q", "draft"]) == 0
    run = cli.Run(tmp_path / "runs" / "demo")
    assert run.plan.prompt == "Soft drone."
    assert run.plan.movements == 2


def test_nothing_to_render_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="prompt"):
        cli.main(["generate"])


def test_a_word_list_renders_and_resumes_like_any_other_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    words = tmp_path / "words.txt"
    words.write_text("bpm: 120\nwarm bass\nsoft hats\nchord stabs\n", encoding="utf-8")
    arguments = ["generate", "--words", str(words), "-d", "300", "--section-every", "2m"]
    assert cli.main([*arguments, "--engine", "synth", "--name", "demo", "-q", "draft"]) == 0
    run = cli.Run(tmp_path / "runs" / "demo")
    assert run.plan.bpm == 120
    assert run.plan.movements > 1
    assert (tmp_path / "runs" / "demo" / "plan.json").exists()
    assert cli.main(["resume", "demo"]) == 0
    assert cli.main(["status", "demo"]) == 0
    assert "complete" in capsys.readouterr().out


def test_word_lists_are_found_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    assert cli.main(["generate", "--words", "funky", "--dry-run", "--seed", "1"]) == 0
    plan = cli.Plan.from_json(capsys.readouterr().out)
    assert plan.bpm == 110
    assert plan.movements == 17


def test_dry_run_of_a_word_list_repeats_with_the_seed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    arguments = ["generate", "--words", "ambient", "--dry-run", "--seed", "9"]
    assert cli.main(arguments) == 0
    first = capsys.readouterr().out
    assert cli.main(arguments) == 0
    assert capsys.readouterr().out == first


def test_an_unknown_word_list_names_the_ones_there_are():
    with pytest.raises(SystemExit, match="funky"):
        cli.main(["generate", "--words", "nothing-called-this", "--dry-run"])


def test_a_broken_word_list_says_which_file(tmp_path):
    words = tmp_path / "broken.txt"
    words.write_text("## bass (pick 3)\none\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="broken.txt"):
        cli.main(["generate", "--words", str(words), "--dry-run"])


def test_a_word_list_with_no_tempo_is_explained(tmp_path):
    words = tmp_path / "words.txt"
    words.write_text("warm bass\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="tempo"):
        cli.main(["generate", "--words", str(words), "--dry-run"])
    assert cli.main(["generate", "--words", str(words), "--bpm", "110", "--dry-run"]) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["generate", "a prompt", "--words", "funky"],
        ["generate", "--journey", "x.txt", "--words", "funky"],
    ],
)
def test_words_replace_any_other_prompt(arguments):
    with pytest.raises(SystemExit, match="--words"):
        cli.main(arguments)


def test_section_spacing_needs_sections_to_space(tmp_path):
    journey = tmp_path / "journey.txt"
    journey.write_text("0:00 Soft drone.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="section-every"):
        cli.main(["generate", "--journey", str(journey), "--section-every", "3.5m"])
    with pytest.raises(SystemExit, match="section-every"):
        cli.main(["generate", "drone", "--evolve", "off", "--section-every", "3.5m"])


def test_section_spacing_works_with_evolve(capsys):
    arguments = ["generate", "drone, 110 BPM", "--section-every", "3.5m", "--dry-run"]
    assert cli.main(arguments) == 0
    assert cli.Plan.from_json(capsys.readouterr().out).movements == 17


def test_a_draw_can_be_saved_edited_and_rendered(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    saved = tmp_path / "draw.txt"
    arguments = ["generate", "--words", "funky", "--seed", "7", "--save-journey", str(saved)]
    assert cli.main(arguments) == 0
    assert "17 sections" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()

    text = saved.read_text(encoding="utf-8")
    assert text.startswith("#")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert len(lines) == 17
    assert lines[1].startswith("3:29 [")

    edited = text.replace(lines[1], "3:29 Instrumental dub techno, 110 BPM. Rewritten by hand.")
    saved.write_text(edited, encoding="utf-8")
    assert cli.main(["generate", "--journey", str(saved), "--dry-run", "--seed", "7"]) == 0
    plan = cli.Plan.from_json(capsys.readouterr().out)
    assert plan.movements == 17
    assert "Rewritten by hand." in {s.prompt.split("BPM. ")[1] for s in plan.segments}


def test_drum_levels_are_recorded_in_the_plan(capsys):
    assert cli.main(["generate", "drone", "--dry-run"]) == 0
    assert not cli.Plan.from_json(capsys.readouterr().out).swaps_drums
    assert cli.main(["generate", "drone", "--drums", "-60", "--kick", "-9", "--dry-run"]) == 0
    plan = cli.Plan.from_json(capsys.readouterr().out)
    assert (plan.drums_db, plan.kick_db) == (-60, -9)


def test_a_saved_draw_says_how_to_keep_its_drum_levels(tmp_path):
    saved = tmp_path / "draw.txt"
    arguments = ["generate", "--words", "funky", "--drums", "-60", "--kick", "-9"]
    assert cli.main([*arguments, "--save-journey", str(saved)]) == 0
    assert "--drums -60 --kick -9" in saved.read_text(encoding="utf-8")


def test_a_master_can_change_the_drum_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    plans = []
    monkeypatch.setattr(cli, "finish", lambda run, formats: plans.append(run.plan) or [])
    arguments = ["generate", "drone", "-d", "12", "--engine", "synth", "--name", "demo"]
    assert cli.main(arguments) == 0
    assert cli.main(["master", "demo", "--drums", "-18", "--kick", "-12"]) == 0
    assert (plans[0].drums_db, plans[0].kick_db) == (-18, -12)


def test_a_saved_draw_is_never_overwritten(tmp_path):
    saved = tmp_path / "draw.txt"
    saved.write_text("my edits", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(["generate", "--words", "funky", "--save-journey", str(saved)])
    assert saved.read_text(encoding="utf-8") == "my edits"


def test_a_dropped_section_is_mentioned(tmp_path, capsys):
    journey = tmp_path / "journey.txt"
    journey.write_text("0:00 A, 110 BPM.\n3:29 B.\n3:50 C.\n", encoding="utf-8")
    assert cli.main(["generate", "--journey", str(journey), "-d", "10m", "--dry-run"]) == 0
    assert "Dropped 1 sections" in capsys.readouterr().err


def test_status_without_runs_explains_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no runs"):
        cli.main(["status"])
