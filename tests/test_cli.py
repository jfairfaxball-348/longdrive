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


def test_status_without_runs_explains_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no runs"):
        cli.main(["status"])
