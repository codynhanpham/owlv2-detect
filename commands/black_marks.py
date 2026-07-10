import argparse
import os

import tqdm
from PIL import Image, ImageDraw

from commands.common import (
    BlackMarksConfig,
    add_common_arguments,
    SortedHelpFormatter,
    build_common_config,
    get_debug_annotation_style,
    load_font,
    load_image_frame,
    load_sampled_frames,
    pool_detection_annotations,
    save_preprocessed_debug_image,
    detect_frames_batch,
    is_video_input,
    sample_frame_indices,
)

DEFAULT_TEXT_QUERIES = ["black mark"]


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "black-marks",
        help="Run the current pooled black-marks pipeline.",
        description="Run the existing black-marks pipeline with video sampling, pooling, and mark consolidation.",
        formatter_class=SortedHelpFormatter,
    )
    
    add_common_arguments(parser, "+",
        # Crop 3% from each side of the frame by default, to avoid edge artifacts and black borders in videos
        crop_default=[0.03, 0.03, 0.03, 0.03],
        
        # Default filters to make black marks more visible
        filter_default=[
            ["brightness", "0.5"],
            ["contrast", "0.7"],
            ["highlight", "1"],
            ["shadow", "-0.2"],
        ]
    )

    parser.add_argument(
        "-r",
        "--duration-range",
        nargs=2,
        type=float,
        default=[0.2, 0.8],
        metavar=("START", "END"),
        help="Video duration range to sample from as fractions of the total video length.",
    )
    parser.add_argument(
        "-s",
        "--sample-frame-count",
        type=int,
        default=30,
        help="Number of unique video frames to sample from the selected duration range.",
    )
    parser.add_argument(
        "-n",
        "--top-n-marks",
        type=int,
        default=4,
        help="Maximum number of pooled marks to return.",
    )
    parser.add_argument(
        "-t",
        "--text",
        nargs="+",
        default=DEFAULT_TEXT_QUERIES,
        help="Text query string(s) to send to OWLv2.",
    )
    return parser


def build_black_marks_config(namespace: argparse.Namespace) -> BlackMarksConfig:
    base_config = build_common_config(namespace)
    duration_range: tuple[float, float] = (
        float(namespace.duration_range[0]),
        float(namespace.duration_range[1]),
    )

    return BlackMarksConfig(
        input_paths=base_config.input_paths,
        model_id=base_config.model_id,
        batch_size=base_config.batch_size,
        detection_threshold=base_config.detection_threshold,
        crop=base_config.crop,
        adjustments=base_config.adjustments,
        log_level=base_config.log_level,
        text_queries=[str(value) for value in namespace.text],
        sample_frame_count=int(namespace.sample_frame_count),
        duration_range=duration_range,
        top_n_marks=int(namespace.top_n_marks),
    )


build_config = build_black_marks_config


def run_black_marks(config: BlackMarksConfig, show_progress: bool, save_preprocessed_debug: bool) -> list[dict[str, object]]:
    total_frames = 0
    frames_by_input: list[tuple[str, list[tuple[int, Image.Image]]]] = []
    for input_path in config.input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")

        if is_video_input(input_path):
            frame_indices = sample_frame_indices(input_path, config.sample_frame_count, config.duration_range)
            sampled_frames = load_sampled_frames(input_path, frame_indices)
        else:
            sampled_frames = [load_image_frame(input_path)]

        frames_by_input.append((input_path, sampled_frames))
        total_frames += len(sampled_frames)

    progress_bar = tqdm.tqdm(
        total=total_frames,
        desc="Detection",
        unit="frame",
        disable=not show_progress,
    )
    results: list[dict[str, object]] = []
    try:
        total_inputs = len(frames_by_input)
        for input_index, (input_path, sampled_frames) in enumerate(frames_by_input, start=1):
            detections_by_frame = detect_frames_batch(
                sampled_frames,
                config.crop,
                config.adjustments,
                config.top_n_marks,
                config.batch_size,
                config.text_queries,
                config.model_id,
                config.detection_threshold,
                progress_bar,
                input_index,
                total_inputs,
            )

            pooled_annotations = pool_detection_annotations(detections_by_frame, config.top_n_marks)
            if not pooled_annotations:
                results.append({"file": input_path, "markers": []})
                continue

            if show_progress:
                display_image = sampled_frames[0][1].copy()
                draw = ImageDraw.Draw(display_image)
                annotation_style = get_debug_annotation_style(display_image.size)
                font = load_font(annotation_style.font_size)

                for index, ((center_x, center_y), label_index, score_value) in enumerate(pooled_annotations, start=1):
                    box = [
                        round(center_x - annotation_style.marker_radius),
                        round(center_y - annotation_style.marker_radius),
                        round(center_x + annotation_style.marker_radius),
                        round(center_y + annotation_style.marker_radius),
                    ]
                    draw.rectangle(box, outline="red", width=annotation_style.line_width)
                    if 0 <= label_index < len(config.text_queries):
                        label_value = config.text_queries[label_index]
                    else:
                        label_value = str(label_index)
                    draw.text(
                        (
                            box[0] + annotation_style.label_gap,
                            max(0, box[1] - annotation_style.font_size - annotation_style.label_gap),
                        ),
                        f"{index} - {label_value} ({score_value:.2f})",
                        fill="red",
                        font=font,
                        stroke_width=annotation_style.stroke_width,
                        stroke_fill="black",
                    )

                base_name, _ = os.path.splitext(os.path.basename(input_path))
                output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_pooled_median_detection.png")
                display_image.save(output_path)

                if save_preprocessed_debug:
                    save_preprocessed_debug_image(input_path, sampled_frames, config)

            json_coords = [[round(center_x, 2), round(center_y, 2)] for (center_x, center_y), _, _ in pooled_annotations]
            results.append({"file": input_path, "markers": json_coords})
    finally:
        progress_bar.close()

    return results


run = run_black_marks