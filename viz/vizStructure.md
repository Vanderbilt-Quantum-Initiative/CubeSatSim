# Viz Output Structure

## Rule

Every viz file targets exactly one physics module. Its output goes to a subfolder of `viz/out/` named after that module.

```
viz/
├── vizStructure.md          ← this file
├── atmosphere_plots.py      ← visualises physics/atmosphere.py
├── link_loss_plots.py       ← visualises physics/link_loss.py
├── <module>_plots.py        ← future modules follow the same pattern
└── out/
    ├── atmosphere/          ← output of atmosphere_plots.py
    │   ├── cn2_profiles.png
    │   ├── hv_sensitivity.png
    │   └── attenuation.png
    ├── link_loss/           ← output of link_loss_plots.py
    │   ├── diffraction.png
    │   ├── pointing.png
    │   ├── loss_budget_45deg.png
    │   └── budget_vs_elev.png
    └── <module>/            ← future modules
```

`viz/out/` is not committed to git (add to `.gitignore`). Re-run the relevant `*_plots.py` file to regenerate.

## Conventions for new viz files

**Naming:** `<module>_plots.py` where `<module>` matches the physics filename without `.py`.

**Output directory:** Set a module-level constant at the bottom of the file:
```python
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "<module>")
```

**`plot_all()` signature:**
```python
def plot_all(save_dir: str | None = None) -> list[plt.Figure]:
    out = save_dir or _OUT_DIR
    os.makedirs(out, exist_ok=True)
    # render figures, save as out/<name>.png (no module prefix in filename—the folder provides it)
    ...
```

**Filename:** Use the plot's own descriptive name (e.g. `rytov_vs_elevation.png`), not a prefixed one (not `turbulence_rytov_vs_elevation.png`). The subfolder already identifies the module.

**`__main__` block:** Always include one so individual files can be run directly during development:
```python
if __name__ == "__main__":
    plot_all()
    plt.show()
```

## Planned modules (Build Sequence §8)

| Physics file          | Viz file                  | Output folder          |
|-----------------------|---------------------------|------------------------|
| atmosphere.py         | atmosphere_plots.py       | out/atmosphere/        |
| link_loss.py          | link_loss_plots.py        | out/link_loss/         |
| turbulence.py         | turbulence_plots.py       | out/turbulence/        |
| source.py             | source_plots.py           | out/source/            |
| detector.py           | detector_plots.py         | out/detector/          |
| detection.py          | detection_plots.py        | out/detection/         |
| decoy.py              | decoy_plots.py            | out/decoy/             |
| keyrate.py            | keyrate_plots.py          | out/keyrate/           |
| post_processing.py    | post_processing_plots.py  | out/post_processing/   |
| pass_sim.py           | pass_plots.py             | out/pass/              |
