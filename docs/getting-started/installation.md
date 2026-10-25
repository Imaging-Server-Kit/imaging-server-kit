# Installation

The Imaging Server Kit supports Python 3.11 to 3.13.

## Installing with pip

Install the `imaging-server-kit` package with `pip`:

```sh
pip install imaging-server-kit
```

## Optional dependencies

The Napari and QuPath integrations rely on extra packages that are not installed by default. Install them through the corresponding extras:

| Command | Adds |
|---|---|
| `pip install "imaging-server-kit[napari]"` | [Napari](https://github.com/napari/napari) and [napari-toolkit](https://github.com/MIC-DKFZ/napari_toolkit), for dock widgets and the Napari plugin |
| `pip install "imaging-server-kit[qupath]"` | [QuBaLab](https://pypi.org/project/qubalab/) (`>=0.2.0`) and the Napari extra, for the QuPath integration |
| `pip install "imaging-server-kit[all]"` | All the above, plus the documentation and test dependencies |

!!! note
    The `qupath` extra requires Python 3.11 or later, because of QuBaLab.

## Agent skills

The package includes two skills for AI coding agents such as [Claude Code](https://claude.com/claude-code). They help the agent write algorithms with `@sk.algorithm` and build Napari or QuPath interfaces for them. Agents do not pick up skills from installed Python packages, so install them with:

```sh
serverkit install-skills
```

This copies the skills to `.claude/skills/` in the current directory, which makes them available in that project. To make them available in all projects, use `serverkit install-skills --user` (installs to `~/.claude/skills/`).

| Skill | Helps with |
|---|---|
| `isk-write-algorithm` | Turning a function into an algorithm: parameters, outputs, samples, tiling, and servers |
| `isk-napari-qupath-gui` | Running algorithms in Napari or QuPath, and troubleshooting the widgets |

Existing skills are not overwritten unless you pass `--force`, so after upgrading the package, run `serverkit install-skills --force` to get the updated skills. See [Command line](../reference/cli.md) for all options.

## Development version

Clone the repository and install the package in editable mode:

```sh
git clone https://github.com/Imaging-Server-Kit/imaging-server-kit.git
cd imaging-server-kit
pip install -e ".[all]"
```

## Next steps

[Try the demos](demos.md) to get a first look at what the package can do.
