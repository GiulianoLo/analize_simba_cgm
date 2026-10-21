# Simulation config

`simulations.json` here lists where each simulation's snapshots and caesar catalogues live, one
entry per simulation name (`cis100`, `cis25`, `cis50`, ...). It is a plain JSON file: edit it by hand,
or call `simbanator.add_simulation(...)`, which writes to it.

- Start from `simulations.example.json`: copy it to `simulations.json` and fill in your paths.
- `simulations.json` is not committed (the paths are machine-specific). The example is.
- To use another file, set `SIMBANATOR_CONFIG=/path/to/file.json`.
- The old hidden `~/.simbanator/config.json` is still read when this file does not exist, and,
  while it exists, every write is mirrored into it, because some notebooks still read it directly.
