# OWLv2 Detect

`owlv2-detect` is a small CLI wrapper for zero-shot object detection with OWLv2. It ships with two pipelines:

- `general` for object detection on single images or the first frame of a video
- `black-marks` for pooled video sampling and mark consolidation ([why?](#black-marks-pipeline))

The CLI prints JSON results to stdout, and can optionally save annotated debug images when `--log-level debug` is enabled.

## Installation

You will need [uv](https://docs.astral.sh/uv/) for environment management.

Then, use the typical install flow:

1. Clone this repository

2. Set up a virtual environment and install dependencies:

    ```bash
    uv sync
    ```

    If you want GPU acceleration, pick exactly one PyTorch backend extra:

    ```bash
    # NVIDIA / CUDA
    uv sync --extra cuda

    # AMD / ROCm
    uv sync --extra rocm

    # Intel / XPU
    uv sync --extra xpu
    ```

    Do not combine the GPU extras. They are mutually exclusive.

## Quick Start

Run the default `general` pipeline on one image:

```bash
uv run main.py -i path/to/image.png -t "person" "dog"
```

Run the same pipeline explicitly as a subcommand:

```bash
uv run main.py general -i path/to/image.png -t "person" "dog"
```

Run the pooled video workflow ([wth???](#black-marks-pipeline)):

```bash
uv run main.py black-marks -i path/to/video.mp4
```

The subcommands print different JSON shapes:

- `general`: `{"file": "path/to/input.png", "objects": [{"label": "person", "bbox": [x1, y1, x2, y2], "score": 0.98}]}`
- `black-marks`: `{"file": "path/to/input.mp4", "markers": [[123.45, 67.89]]}`

Use the multi-pass filter syntax by repeating `--filter TYPE VALUE` pairs in the order you want them applied:

```bash
uv run main.py general \
  -i path/to/image.png \
  -t "object" \
  -f brightness 0.2 \
  -f contrast 0.4 \
  -f highlight 0.1 \
  -f shadow -0.1
```

## Models and Compatibility

Any OWLv2-backed model on Hugging Face should work with this CLI. The default model is `google/owlv2-base-patch16-ensemble` (`~650MB`), but you can specify any other model ID with the `--model` option.

The first time you run the CLI with a new model, it will download the model weights from Hugging Face and cache them in your local environment, next to the `./.venv` folder. Subsequent runs will use the cached weights.


## Documentation

### Commands

- `general`: runs OWLv2 on a single image, or on the first frame of a video.
- `black-marks`: samples multiple frames from a video, detects marks per frame, then pools the results.

### Help

```bash
uv run main.py --help
uv run main.py general --help
uv run main.py black-marks --help
```

### Common Options

- `-i, --input`: one or more input paths.
- `-c, --crop`: crop fractions in the order top, right, bottom, left.
- `-f, --filter`: repeated `TYPE VALUE` adjustment pairs.
- `-m, --model`: Hugging Face model ID to load.
- `-b, --batch-size`: number of images per model forward pass.
- `--detection-threshold`: minimum score kept after post-processing.
- `--log-level`: `quiet`, `warning`, `info`, or `debug`.

### Filter Types

All filter values should stay in the range `-1` to `1`, with `0` meaning no change. The current implementation applies the adjustments sequentially in the order provided.

- `brightness`: adjusts image brightness using a `2^value` multiplier.
- `contrast`: adjusts contrast using a `2^value` multiplier.
- `highlight`: targets brighter areas more strongly than darker ones.
- `shadow`: targets darker areas more strongly than brighter ones.

As a rule of thumb, `1` is a strong pass, `-1` is the inverse of that pass, and smaller magnitudes give gentler changes.\
If `1` or `-1` is not enough for you, pass that same filter as many times as you want to compound the effect.

> [!TIP]
> Run your command with `--log-level debug` to also save the preprocessed image next to the input file. This is useful for adjusting and verifying your filter and crop settings.


### Black-Marks Options

- `-r, --duration-range`: fractional video range to sample from.
- `-s, --sample-frame-count`: number of unique frames to sample.
- `-n, --top-n-marks`: maximum pooled marks to return.

### Output

Each run prints a JSON array. The exact object shape depends on the subcommand:

`general`

```jsonc
[
	{
		"file": "path/to/input.png",
        "objects": [
            {
                "label": "person",
                "bbox": [123.45, 67.89, 234.56, 278.9],
                "score": 0.98
            }
        ]
	},
    ...
]
```

`black-marks`

```jsonc
[
  {
    "file": "path/to/input.mp4",
    // centered x,y coordinates of the top N marks after pooling
    "markers": [[123.45, 67.89], ...]
  }
]
```

When `--log-level debug` is enabled, the CLI also logs the parsed arguments and saves a preprocessed debug image next to the input file.

In all `--log-level`, the output JSON is **always** the last line printed to stdout. If you want to use the STDOUT output in a script, simply collect the last line before parsing it as JSON.

To just get the JSON output without any debug images or logs, run with `--log-level quiet`.


## Black-Marks Pipeline

This was the original motivation for this project. I had some videos where the camera is fixed and focused on a mostly blank area containing a single region of interest (ROI). This ROI's boundary is always marked by four small black marks in the corners.

Even though the camera is fixed, the exact position of the arena and the ROI changes between videos. At the same time, any shifting within a single video is negligible.

Basic video filters might work, but they are not very reliable (I tried), and training my own neural network is not worth the time and effort.

Looking around, there are already some good general-purpose object detection models that only need to be tweaked slightly with text prompts and pre- or post-processing. I chose OWLv2 because it was one of the best zero-shot object detection models available when I created this project. As long as I preprocess the input images enough to bias the model's confidence toward the black marks within the frames, I can achieve reliable detection of the marks in a single pass. Then, simply by randomly sampling a few frames from the video and pooling the results, I can get a highly reliable estimate of the ROI's position.

The general process for this pipeline is as follows:

- Limit the video range to where the ROI is expected and its location remains stable.
- Sample a few frames from that range.
- Preprocess the frames by cropping them and applying basic image filters.
- Run OWLv2 detection on each frame using a text prompt for the black marks.
- Pool the results by removing duplicates, matching the marks across frames, filtering by confidence, and taking the top N marks that best match the ROI's corners.
- Return the centered coordinates of the pooled marks as the final result.
