# HGS Sediment-Nutrient Module

HGS Sediment-Nutrient Module is a Python/Tkinter graphical interface and post-processing workflow for HGS surface-water output, sediment routing, and nutrient-related analysis. The module reads an HGS OLF output file and channel file, runs sediment and nutrient post-processing, and writes station/table outputs for calibration or analysis.

## Package Contents

- `Source/` contains the Python source code.
- `Documentation/` contains the user manual.
- `PEST Template/` contains starter files for PEST calibration.
- `Example/` contains the example configuration notes. The large example HGS output files are distributed with the release package, not committed directly to the repository.
- `Model_Config.txt` is the default editable model configuration file.
- `Requirements.txt` lists the Python dependencies for running from source.

## Running The Model

For users without Python, use the executable distributed in the GitHub release package.

For users with Python installed:

```bat
Run Source With Python.bat
```

or run:

```bat
python "Source\HGS Sediment-Nutrient Module.py"
```

## Notes For PEST Calibration

Use the files in `PEST Template/` as the starting point for coupling this module with PEST. In PEST mode, the model writes text/table outputs only. Map generation is disabled in PEST mode.

## Large Files

The executable, full package archive, and large HGS example output files are provided as GitHub release assets instead of being stored directly in the repository.

