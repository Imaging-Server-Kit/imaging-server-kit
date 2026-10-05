# Command line

Installing the package adds a `serverkit` command.

| Command | Description |
|---|---|
| `serverkit demo napari` | Open Napari with the demo algorithms. Requires the `napari` extra. |
| `serverkit demo serve` | Serve the demo algorithms at http://localhost:8000. |
| `serverkit tools napari` | Open Napari with the [built-in algorithms](../how-to/combine.md#built-in-algorithms). Requires the `napari` extra. |
| `serverkit tools serve` | Serve the built-in algorithms at http://localhost:8000. |
| `serverkit qupath` | Open the [QuPath](../how-to/qupath.md) connection panel. Requires the `qupath` extra. |
| `serverkit qupath --with-napari` | Same as `serverkit qupath`, with a Napari viewer to collect results that cannot be displayed in QuPath. |

Run `serverkit --help` (or `serverkit <command> --help`) to list the commands and options.
