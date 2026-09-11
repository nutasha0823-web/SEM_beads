# SEM Bead Detection

Detects and classifies particles (beads) in the circular wells of a microfluidic
chip, from scanning electron microscope images. Wells are located with a Hough
circle transform, then each well is classified by the mean brightness and
standard deviation of its interior.

Each well is sorted into one of four classes:

| Class | Meaning | Marker color |
|---|---|---|
| `particle` | A bead is present | green |
| `empty` | Nothing in the well (too dark) | red |
| `debris` | Something present, but not a clean bead (too uneven) | orange |
| `outside` | Glare, artifact, or background — not a well interior (too bright) | magenta |

## Input images

- **Expected resolution: 2048×2048.** SEM output is slightly larger because of
  the instrument's info bar — crop it off before analysis.
- Grayscale or color; color is converted to grayscale internally.
- **Shoot at a fixed magnification.** The Hough parameters `min_radius` /
  `max_radius` are in pixels, so changing magnification between images means
  re-tuning the config every time.

A sample image and its cropped JPEG are in `for_sharing/`.

## Install

Requires Python 3.13 (see `.python-version`; 3.9+ should work).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

**For the GUI you also need Tk**, which is a system package, not a pip one:

```bash
sudo apt install python3-tk        # Debian/Ubuntu
```

macOS and the python.org Windows installers ship Tk already. Without it the GUI
fails with `ModuleNotFoundError: No module named 'tkinter'`.

## Usage

### GUI (recommended)

```bash
python chip_ore_brightout_simple_gui.py
```

Browse to an image, adjust the sliders and spinboxes, press **Analyze Image**.
**Show Results** opens the diagnostic figure; **Save Config** writes the current
settings to `detection_config.json`.

This is currently the only entry point that reads and writes the config file.

### Command line

```bash
python chip_ore_brightout.py
```

> **The CLI scripts do not work as shipped.** They crash on a return-value
> mismatch before writing any output, they take no arguments (the image path is
> hardcoded to a Windows path near the bottom of the file), and they ignore
> `detection_config.json` entirely — their thresholds are hardcoded inside the
> function. See [REVIEW.md](REVIEW.md) for the details and
> [FIXING_GUIDE.md](FIXING_GUIDE.md) steps 1-3 for the fix.

## Configuration

`detection_config.json` holds all tunable parameters. **Read by the GUI only** —
see the warning above.

```json
{
    "brightness_threshold": 56,
    "brightness_upper_threshold": 160,
    "uniformity_threshold": 41,
    "hough_params": {
        "min_dist": 8,
        "param1": 50,
        "param2": 28,
        "min_radius": 7,
        "max_radius": 12
    }
}
```

### Classification thresholds

| Key | Meaning |
|---|---|
| `brightness_threshold` | Lower brightness bound. Below this the well is `empty`. |
| `brightness_upper_threshold` | Upper brightness bound. Above this it is glare or background, classified `outside`. |
| `uniformity_threshold` | Standard-deviation bound. Above this the contents are uneven — `debris` rather than a clean bead. |

### Hough circle parameters

These locate the wells themselves. Tune them once per magnification.

| Key | Meaning |
|---|---|
| `min_dist` | Minimum distance between detected well centers |
| `param1` | Upper threshold for the Canny edge detector |
| `param2` | Accumulator threshold for center detection — lower means more false positives |
| `min_radius` | Minimum well radius, in pixels |
| `max_radius` | Maximum well radius, in pixels |

Only the wells are found geometrically; everything after that is brightness
statistics, so the thresholds need re-tuning whenever imaging conditions change.

## Output

- **Console** — one line per well with its brightness, standard deviation, and
  class, followed by summary counts.
- **Diagnostic figure** (when `debug=True`) — six panels: the original image,
  the adaptive threshold, the detected-well mask, the annotated result, a
  statistics box, and a histogram of per-well brightness.
- **`wells_analysis_result.jpg`** — the annotated image, colored per the table
  at the top. The thin yellow inner circle on each well shows the 80%-radius
  region actually measured; the rim is excluded to avoid halo/glare artifacts.

## Repository layout

| Path | What it is |
|---|---|
| `chip_ore_brightout_simple_gui.py` | **Main entry point.** Tk GUI, four-class detection, reads/writes the config file. |
| `chip_ore_brightout.py` | CLI version of the same detector. Thresholds hardcoded; currently broken. |
| `chip_1.py` | Earlier variant: two classes only, threshold 70. Kept for reference. |
| `chip_2.py` | Earlier variant: three classes, thresholds 65 / 37. Kept for reference. |
| `detection_config.json` | Tunable parameters (GUI only). |
| `for_sharing/` | Hand-copied distribution bundle: a stale copy of the two scripts, a Russian user manual, and sample SEM images. Not kept in sync automatically. |
| `part_from_StereoVision/` | Unrelated dual-camera capture scripts adapted from the StereoVision project. Not part of the detection pipeline. |
| `REVIEW.md` | Code review findings. |
| `FIXING_GUIDE.md` | Step-by-step remediation plan. |

`chip_1.py`, `chip_2.py` and `chip_ore_brightout.py` are near-identical copies of
one ~300-line function that differ only in thresholds and input image. Consolidating
them is [FIXING_GUIDE.md](FIXING_GUIDE.md) step 7.

## Known issues

The CLI is broken, the GUI and CLI use different preprocessing (so the same
config gives different well counts in each), and `for_sharing/` has drifted from
the root copies. Full list in [REVIEW.md](REVIEW.md); ordered fixes in
[FIXING_GUIDE.md](FIXING_GUIDE.md).
