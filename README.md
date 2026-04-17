# Particle Detection in Wells - Optimized Version

This repository contains an optimized and user-friendly version of the particle detection code that finds particles in circular wells using Hough Circles transformation.

## Features

- **Configurable thresholds**: All brightness thresholds can be adjusted via a configuration file
- **Hough Circle parameters**: All Hough Circle detection parameters are configurable
- **Multiple interfaces**: Command-line interface and GUI option
- **Preserved core functionality**: Uses the same Hough Circle detection algorithm as the original
- **Easy to use**: No need to edit code to adjust thresholds

## Files

- `chip_ore_brightout_user_friendly.py`: Main command-line interface version
- `chip_ore_brightout_simple_gui.py`: GUI version with slider controls
- `detection_config.json`: Configuration file with default threshold values
- `chip_ore_brightout.py`: Original file (untouched as requested)


### Configuration File

The `detection_config.json` file contains all adjustable parameters:

```json
{
    "brightness_threshold": 56,           # Lower brightness threshold for particle detection
    "brightness_upper_threshold": 160,    # Upper brightness threshold  
    "uniformity_threshold": 41,           # Threshold for uniformity/std deviation
    "hough_params": {
        "min_dist": 8,                    # Minimum distance between circle centers
        "param1": 50,                     # Upper threshold for Canny edge detector
        "param2": 28,                     # Accumulator threshold for center detection
        "min_radius": 7,                  # Minimum circle radius to detect
        "max_radius": 12                  # Maximum circle radius to detect
    }
}
```

### GUI Version

To use the GUI version:

```bash
python chip_ore_brightout_simple_gui.py
```

The GUI provides sliders for all threshold values and spinboxes for Hough Circle parameters.

## Key Improvements

1. **No code editing required**: All thresholds are loaded from configuration file
2. **Hough Circle optimization**: Parameters are exposed for fine-tuning
3. **User-friendly interface**: Both command-line and GUI options
4. **Preserved original functionality**: Same detection algorithm as original code
5. **Better output**: Clear statistics and result visualization
6. **Fixed numeric stability**: Addressed potential overflow warnings in coordinate calculations

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- Pillow (`pip install pillow`) - for GUI version

## How It Works

The algorithm follows the same approach as the original:

1. Loads and preprocesses the image
2. Uses Hough Circles transformation to detect circular wells
3. Analyzes each well for particles based on brightness thresholds
4. Classifies wells as containing particles, debris, outside artifacts, or being empty
5. Displays results with visual annotations and statistics

All threshold values are now configurable without modifying the source code.