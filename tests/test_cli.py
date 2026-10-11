import sys

from imaging_server_kit import cli


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["serverkit", *argv])
    cli.main()


def test_install_skills(tmp_path, monkeypatch):
    target = tmp_path / "skills"

    run_cli(monkeypatch, "install-skills", "--dir", str(target))

    installed = sorted(p.name for p in target.iterdir())
    assert installed == ["isk-napari-qupath-gui", "isk-write-algorithm"]
    for name in installed:
        assert (target / name / "SKILL.md").is_file()


def test_install_skills_does_not_overwrite_without_force(tmp_path, monkeypatch):
    target = tmp_path / "skills"
    run_cli(monkeypatch, "install-skills", "--dir", str(target))

    skill_file = target / "isk-write-algorithm" / "SKILL.md"
    skill_file.write_text("edited")

    run_cli(monkeypatch, "install-skills", "--dir", str(target))
    assert skill_file.read_text() == "edited"

    run_cli(monkeypatch, "install-skills", "--dir", str(target), "--force")
    assert skill_file.read_text() != "edited"
