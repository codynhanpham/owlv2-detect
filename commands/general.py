import argparse
import logging
import os

from PIL import ImageDraw

from commands.common import (
    GeneralConfig,
    add_common_arguments,
    build_common_config,
    detect_single_frame,
    get_debug_annotation_style,
    SortedHelpFormatter,
    load_font,
    load_primary_frame,
    save_preprocessed_debug_image,
)


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "general",
        help="Run single-frame general-purpose detection.",
        description="Run OWLv2 on one image, or on the first frame of a video, without pooling.",
        formatter_class=SortedHelpFormatter,
    )
    add_common_arguments(parser, "+", filter_default=[], detection_threshold_default=0.05)
    parser.add_argument(
        "-t",
        "--text",
        nargs="+",
        required=True,
        help="Text query string(s) to send to OWLv2.",
    )
    return parser


def build_general_config(namespace: argparse.Namespace) -> GeneralConfig:
    base_config = build_common_config(namespace)
    return GeneralConfig(
        input_paths=base_config.input_paths,
        model_id=base_config.model_id,
        batch_size=base_config.batch_size,
        detection_threshold=base_config.detection_threshold,
        crop=base_config.crop,
        adjustments=base_config.adjustments,
        log_level=base_config.log_level,
        text_queries=[str(value) for value in namespace.text],
    )


build_config = build_general_config


def run_general(config: GeneralConfig, show_progress: bool, save_preprocessed_debug: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for input_path in config.input_paths:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")

        input_frame = load_primary_frame(input_path)
        detections = detect_single_frame(
            input_frame,
            config.crop,
            config.adjustments,
            config.text_queries,
            config.model_id,
            config.detection_threshold,
        )

        if show_progress:
            annotated_image = input_frame[1].copy()
            draw = ImageDraw.Draw(annotated_image)
            annotation_style = get_debug_annotation_style(annotated_image.size)
            font = load_font(annotation_style.font_size)

            for box, score_value, label_index in detections:
                rect = [round(box[0]), round(box[1]), round(box[2]), round(box[3])]
                draw.rectangle(rect, outline="red", width=annotation_style.line_width)
                if 0 <= label_index < len(config.text_queries):
                    label_value = config.text_queries[label_index]
                else:
                    label_value = str(label_index)
                draw.text(
                    (
                        rect[0] + annotation_style.label_gap,
                        max(0, rect[1] - annotation_style.font_size - annotation_style.label_gap),
                    ),
                    f"{label_value} ({score_value:.2f})",
                    fill="red",
                    font=font,
                    stroke_width=annotation_style.stroke_width,
                    stroke_fill="black",
                )

            base_name, _ = os.path.splitext(os.path.basename(input_path))
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_general_detection.png")
            annotated_image.save(output_path)
            logging.info(f"Saved detection image: {output_path}")

            if save_preprocessed_debug:
                save_preprocessed_debug_image(input_path, [input_frame], config)

        objects: list[dict[str, object]] = []
        for box, score_value, label_index in detections:
            if 0 <= label_index < len(config.text_queries):
                label_value: str = config.text_queries[label_index]
            else:
                label_value = str(label_index)

            objects.append(
                {
                    "label": label_value,
                    "bbox": [round(box[0], 2), round(box[1], 2), round(box[2], 2), round(box[3], 2)],
                    "score": round(score_value, 4),
                }
            )

        results.append({"file": input_path, "objects": objects})

    return results


run = run_general