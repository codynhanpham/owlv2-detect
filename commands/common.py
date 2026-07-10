import argparse
import contextlib
import logging
import os
import random
import warnings
from dataclasses import dataclass
from functools import lru_cache
from statistics import median

import cv2
import torch
import tqdm
from PIL import Image, ImageEnhance, ImageFont, ImageOps
from huggingface_hub import snapshot_download
from scipy.optimize import linear_sum_assignment
from transformers import Owlv2ForObjectDetection, Owlv2Processor

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
QUIET_CONSOLE = False
DEFAULT_MODEL_BATCH_SIZE = 8
DEFAULT_MODEL_ID = "google/owlv2-base-patch16-ensemble"
IMAGE_ADJUSTMENT_NAMES = {"brightness", "contrast", "highlight", "shadow"}
ImageAdjustment = tuple[str, float]


class SortedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    def add_arguments(self, actions):
        sorted_actions = sorted(
            actions,
            key=lambda action: (action.dest == "help", not action.required),
        )
        for action in sorted_actions:
            if action.required and action.help:
                required_marker = "(required)"
                if required_marker not in action.help.lower():
                    action.help = f"{action.help} {required_marker}"
        super().add_arguments(sorted_actions)


HelpFormatter = SortedHelpFormatter


@dataclass(frozen=True)
class BaseConfig:
    input_paths: list[str]
    model_id: str
    batch_size: int
    detection_threshold: float
    crop: tuple[float, float, float, float]
    adjustments: list[ImageAdjustment]
    log_level: str


@dataclass(frozen=True)
class GeneralConfig(BaseConfig):
    text_queries: list[str]


@dataclass(frozen=True)
class BlackMarksConfig(GeneralConfig):
    sample_frame_count: int
    duration_range: tuple[float, float]
    top_n_marks: int


@dataclass(frozen=True)
class DebugAnnotationStyle:
    line_width: int
    font_size: int
    label_gap: int
    stroke_width: int
    marker_radius: int


def build_common_config(namespace: argparse.Namespace) -> BaseConfig:
    crop: tuple[float, float, float, float] = (
        float(namespace.crop[0]),
        float(namespace.crop[1]),
        float(namespace.crop[2]),
        float(namespace.crop[3]),
    )

    return BaseConfig(
        input_paths=[str(value) for value in namespace.input],
        model_id=str(namespace.model),
        batch_size=int(namespace.batch_size),
        detection_threshold=float(namespace.detection_threshold),
        crop=crop,
        adjustments=[(str(adjustment[0]), float(adjustment[1])) for adjustment in namespace.filter],
        log_level=str(namespace.log_level),
    )


def get_model_cache_dir(model_id: str) -> str:
    safe_model_id = model_id.replace(":", "__").replace("/", "__")
    return os.path.join(os.path.dirname(__file__), "..", ".cache", "hf-models", safe_model_id)


@contextlib.contextmanager
def suppress_console_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def resolve_local_model_dir(model_id: str) -> str:
    model_cache_dir = get_model_cache_dir(model_id)
    config_path = os.path.join(model_cache_dir, "config.json")
    if os.path.exists(config_path):
        return model_cache_dir

    os.makedirs(model_cache_dir, exist_ok=True)
    return snapshot_download(
        repo_id=model_id,
        local_dir=model_cache_dir,
        local_dir_use_symlinks=False,
    )  # type: ignore


@lru_cache(maxsize=4)
def get_processor_and_model(model_id: str) -> tuple[Owlv2Processor, Owlv2ForObjectDetection]:
    if QUIET_CONSOLE:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")

    model_dir = resolve_local_model_dir(model_id)
    load_context = suppress_console_output() if QUIET_CONSOLE else contextlib.nullcontext()
    with load_context:
        processor = Owlv2Processor.from_pretrained(model_dir, local_files_only=True)
        model = Owlv2ForObjectDetection.from_pretrained(
            model_dir,
            device_map="auto",
            local_files_only=True,
        )
    return processor, model


def add_common_arguments(
    parser: argparse.ArgumentParser,
    input_nargs: int | str,
    *,
    crop_default: list[float] | None = None,
    filter_default: list[list[str]] | None = None,
    detection_threshold_default: float = 0.012,
) -> None:
    if crop_default is None:
        crop_default = [0.0, 0.0, 0.0, 0.0]
    if filter_default is None:
        filter_default = []

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        nargs=input_nargs,
        default=argparse.SUPPRESS,
        help="Path to the input image or video.",
    )
    parser.add_argument(
        "-c",
        "--crop",
        nargs=4,
        type=float,
        default=crop_default,
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        help="Cropping fractions in the order top, right, bottom, left.",
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="append",
        nargs=2,
        default=filter_default,
        metavar=("TYPE", "VALUE"),
        help=(
            "Repeated TYPE VALUE adjustments to apply in order. Supported types: brightness, contrast, "
            "highlight, and shadow. Values should be between -1 and 1, where 0 is neutral."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID to load.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_MODEL_BATCH_SIZE,
        help="Number of images to process per model forward pass.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=detection_threshold_default,
        help="Minimum detection score to keep after post-processing.",
    )
    parser.add_argument(
        "--log-level",
        choices=("quiet", "warning", "info", "debug"),
        default="quiet",
        help="Controls warnings, progress display, and image export.",
    )


def configure_runtime(log_level: str) -> tuple[bool, bool]:
    global QUIET_CONSOLE
    show_progress = log_level in {"info", "debug"}
    save_preprocessed_debug = log_level == "debug"
    QUIET_CONSOLE = log_level == "quiet"

    level_map = {
        "quiet": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    root_level = level_map[log_level]

    warnings.filterwarnings("default" if log_level == "warning" else "ignore")
    logging.basicConfig(level=root_level, format="%(levelname)s: %(message)s", force=True)

    logging.getLogger().setLevel(root_level)

    logger_level = logging.WARNING if log_level in {"warning", "debug"} else logging.ERROR
    for logger_name in ("transformers", "huggingface_hub", "PIL"):
        logging.getLogger(logger_name).setLevel(logger_level)

    return show_progress, save_preprocessed_debug


def is_video_input(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def sample_frame_indices(path: str, count: int, duration_range: tuple[float, float]) -> list[int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if total_frames <= 0:
        raise RuntimeError(f"Video has no readable frames: {path}")

    start_fraction, end_fraction = duration_range
    start_frame = max(0, int(total_frames * start_fraction))
    end_frame = min(total_frames - 1, int(total_frames * end_fraction) - 1)
    if end_frame < start_frame:
        raise RuntimeError("Video is too short to sample from the selected duration range.")

    candidates = list(range(start_frame, end_frame + 1))
    if not candidates:
        raise RuntimeError("No frame candidates available in the sampling range.")

    count = min(count, len(candidates))
    return sorted(random.sample(candidates, count))


def load_sampled_frames(path: str, frame_indices: list[int]) -> list[tuple[int, Image.Image]]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    sampled_frames: list[tuple[int, Image.Image]] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read frame {frame_index} from {path}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append((frame_index, Image.fromarray(frame_rgb)))
    finally:
        capture.release()

    return sampled_frames


def load_image_frame(path: str) -> tuple[int, Image.Image]:
    image = Image.open(path).convert("RGB")
    return 0, image


def load_primary_frame(path: str) -> tuple[int, Image.Image]:
    if is_video_input(path):
        return load_sampled_frames(path, [0])[0]
    return load_image_frame(path)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


def get_debug_annotation_style(image_size: tuple[int, int]) -> DebugAnnotationStyle:
    width, height = image_size
    shortest_side = max(1, min(width, height))
    font_size = max(12, round(shortest_side / 50))
    line_width = max(1, round(shortest_side / 500))
    label_gap = max(4, round(font_size * 0.2))
    stroke_width = max(1, round(font_size / 10))
    marker_radius = max(2, round(shortest_side / 360))

    return DebugAnnotationStyle(
        line_width=line_width,
        font_size=font_size,
        label_gap=label_gap,
        stroke_width=stroke_width,
        marker_radius=marker_radius,
    )


def parse_adjustments(raw_adjustments: list[list[str]]) -> list[ImageAdjustment]:
    parsed_adjustments: list[ImageAdjustment] = []
    for raw_adjustment in raw_adjustments:
        if len(raw_adjustment) != 2:
            raise ValueError("Each filter adjustment must provide exactly a type and a value.")

        adjustment_name = str(raw_adjustment[0]).strip().lower()
        if adjustment_name not in IMAGE_ADJUSTMENT_NAMES:
            supported = ", ".join(sorted(IMAGE_ADJUSTMENT_NAMES))
            raise ValueError(f"Unsupported filter type: {raw_adjustment[0]!r}. Supported types: {supported}.")

        adjustment_value = float(raw_adjustment[1])
        if not -1.0 <= adjustment_value <= 1.0:
            raise ValueError(f"Filter value for {adjustment_name} must be between -1 and 1.")

        parsed_adjustments.append((adjustment_name, adjustment_value))

    return parsed_adjustments


def _tone_mask(image: Image.Image, start: int, end: int, *, invert: bool = False) -> Image.Image:
    grayscale_image = ImageOps.grayscale(image)

    if end <= start:
        return grayscale_image.point(lambda _: 255 if invert else 0)

    scale = 255.0 / float(end - start)

    def map_value(value: int) -> int:
        if value <= start:
            mapped_value = 0
        elif value >= end:
            mapped_value = 255
        else:
            mapped_value = int((value - start) * scale)

        if invert:
            mapped_value = 255 - mapped_value
        return mapped_value

    return grayscale_image.point(map_value)


def _blend_brightness(image: Image.Image, value: float, mask: Image.Image | None = None) -> Image.Image:
    if value == 0:
        return image

    factor = 2.0**value
    adjusted_image = ImageEnhance.Brightness(image).enhance(factor)
    if mask is None:
        return adjusted_image

    blend_weight = abs(value)
    scaled_mask = mask.point(lambda pixel_value, weight=blend_weight: int(pixel_value * weight))
    return Image.composite(adjusted_image, image, scaled_mask)


def apply_adjustments(image: Image.Image, adjustments: list[ImageAdjustment]) -> Image.Image:
    if not adjustments:
        return image

    adjusted_image = image
    for adjustment_name, adjustment_value in adjustments:
        if adjustment_name == "brightness":
            adjusted_image = _blend_brightness(adjusted_image, adjustment_value)
        elif adjustment_name == "contrast":
            adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(2.0**adjustment_value)
        elif adjustment_name == "highlight":
            highlight_mask = _tone_mask(adjusted_image, 96, 255)
            adjusted_image = _blend_brightness(adjusted_image, adjustment_value, highlight_mask)
        elif adjustment_name == "shadow":
            shadow_mask = _tone_mask(adjusted_image, 0, 159, invert=True)
            adjusted_image = _blend_brightness(adjusted_image, adjustment_value, shadow_mask)

    return adjusted_image


def preprocess_image(
    image: Image.Image,
    crop: tuple[float, float, float, float],
    adjustments: list[ImageAdjustment],
) -> tuple[Image.Image, float, float]:
    top_fraction, right_fraction, bottom_fraction, left_fraction = crop
    width, height = image.size
    left = width * left_fraction
    top = height * top_fraction
    right = width - (width * right_fraction)
    bottom = height - (height * bottom_fraction)

    cropped = image.crop((left, top, right, bottom))
    cropped = apply_adjustments(cropped, adjustments)
    return cropped, left, top


def save_preprocessed_debug_image(
    input_path: str,
    sampled_frames: list[tuple[int, Image.Image]],
    config: BaseConfig,
) -> str:
    preprocessed_image, _, _ = preprocess_image(sampled_frames[0][1], config.crop, config.adjustments)
    base_name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_preprocessed.png")
    preprocessed_image.save(output_path)
    return output_path


def box_center(box: list[float] | torch.Tensor) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def box_distance(center_a: tuple[float, float], center_b: tuple[float, float]) -> float:
    return (center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--"

    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{remaining_minutes:02d}:{remaining_seconds:02d}"
    return f"{remaining_minutes:02d}:{remaining_seconds:02d}"


def inference_autocast():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def detect_frames_batch(
    sampled_frames: list[tuple[int, Image.Image]],
    crop: tuple[float, float, float, float],
    adjustments: list[ImageAdjustment],
    top_n_marks: int,
    batch_size: int,
    text_queries: list[str],
    model_id: str,
    detection_threshold: float,
    progress_bar: tqdm.tqdm | None,
    current_file_index: int | None = None,
    total_file_count: int | None = None,
) -> list[tuple[int, list[tuple[list[float], float, int]]]]:
    processor, model = get_processor_and_model(model_id)

    detections_by_frame: list[tuple[int, list[tuple[list[float], float, int]]]] = []
    total_frames = len(sampled_frames)
    for batch_start in range(0, total_frames, batch_size):
        batch_frames = sampled_frames[batch_start : batch_start + batch_size]

        preprocessed_images: list[Image.Image] = []
        offsets: list[tuple[float, float]] = []
        for _, frame_image in batch_frames:
            preprocessed_image, left, top = preprocess_image(frame_image, crop, adjustments)
            preprocessed_images.append(preprocessed_image)
            offsets.append((left, top))

        texts = [text_queries for _ in preprocessed_images]
        inputs = processor(
            text=texts,
            images=preprocessed_images,
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
        ).to(model.device)  # type: ignore

        with torch.inference_mode(), inference_autocast():
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1] for image in preprocessed_images])
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=detection_threshold,
                target_sizes=target_sizes,  # type: ignore
            )

        for (frame_index, _), (left, top), result in zip(batch_frames, offsets, results):
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]

            mapped_boxes = [
                torch.tensor([float(box[0]) + left, float(box[1]) + top, float(box[2]) + left, float(box[3]) + top])
                for box in boxes
            ]

            filtered_boxes: list[tuple[torch.Tensor, float, int]] = []
            for index, box in enumerate(mapped_boxes):
                score_value = float(scores[index].item())
                label_value = int(labels[index].item())
                if score_value < detection_threshold:
                    continue

                too_close = False
                for existing_index, (existing_box, existing_score, _) in enumerate(filtered_boxes):
                    if box_distance(box_center(box), box_center(existing_box)) < 100:
                        too_close = True
                        if score_value > existing_score:
                            filtered_boxes[existing_index] = (box, score_value, label_value)
                        break

                if not too_close:
                    filtered_boxes.append((box, score_value, label_value))

            filtered_boxes.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
            filtered_boxes = filtered_boxes[:top_n_marks]

            detections: list[tuple[list[float], float, int]] = []
            for box, score_value, label_value in filtered_boxes:
                detections.append((box.detach().cpu().tolist(), score_value, label_value))

            detections_by_frame.append((frame_index, detections))

        if progress_bar is not None:
            progress_bar.update(len(batch_frames))
            if current_file_index is not None and total_file_count is not None:
                rate = progress_bar.format_dict.get("rate")
                elapsed_seconds = progress_bar.format_dict.get("elapsed")
                elapsed_text = format_duration(elapsed_seconds)
                remaining_text = "--:--"
                if rate and progress_bar.total:
                    remaining_seconds = max(0.0, (progress_bar.total - progress_bar.n) / rate)
                    remaining_text = format_duration(remaining_seconds)
                progress_bar.set_postfix_str(
                    f"file {current_file_index}/{total_file_count}, {elapsed_text}<{remaining_text}"
                )

    return detections_by_frame


def detect_single_frame(
    frame: tuple[int, Image.Image],
    crop: tuple[float, float, float, float],
    adjustments: list[ImageAdjustment],
    text_queries: list[str],
    model_id: str,
    detection_threshold: float,
) -> list[tuple[list[float], float, int]]:
    processor, model = get_processor_and_model(model_id)

    _, frame_image = frame
    preprocessed_image, left, top = preprocess_image(frame_image, crop, adjustments)
    inputs = processor(
        text=[text_queries],
        images=[preprocessed_image],
        return_tensors="pt",  # type: ignore
        padding=True,  # type: ignore
    ).to(model.device)  # type: ignore

    with torch.inference_mode(), inference_autocast():
        outputs = model(**inputs)
        target_sizes = torch.tensor([preprocessed_image.size[::-1]])
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=detection_threshold,
            target_sizes=target_sizes,  # type: ignore
        )

    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    detections: list[tuple[list[float], float, int]] = []
    for index, box in enumerate(boxes):
        score_value = float(scores[index].item())
        label_value = int(labels[index].item())
        mapped_box = [
            float(box[0]) + left,
            float(box[1]) + top,
            float(box[2]) + left,
            float(box[3]) + top,
        ]
        detections.append((mapped_box, score_value, label_value))

    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def match_frame_detections(
    reference_centers: list[tuple[float, float]],
    frame_centers: list[tuple[float, float]],
) -> list[tuple[float, float] | None]:
    reference_count = len(reference_centers)
    frame_count = len(frame_centers)
    if reference_count == 0 or frame_count == 0:
        return [None] * reference_count

    cost_matrix = [[box_distance(reference_center, frame_center) for frame_center in frame_centers] for reference_center in reference_centers]
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched: list[tuple[float, float] | None] = [None] * reference_count
    for row_index, col_index in zip(row_indices, col_indices):
        matched[row_index] = frame_centers[col_index]
    return matched


def pool_detection_centers(
    detections_by_frame: list[tuple[int, list[tuple[list[float], float, int]]]],
    top_n_marks: int,
) -> list[tuple[float, float]]:
    return [center for center, _, _ in pool_detection_annotations(detections_by_frame, top_n_marks)]


def pool_detection_annotations(
    detections_by_frame: list[tuple[int, list[tuple[list[float], float, int]]]],
    top_n_marks: int,
) -> list[tuple[tuple[float, float], int, float]]:
    usable_frames = [(frame_index, detections) for frame_index, detections in detections_by_frame if detections]
    if not usable_frames:
        return []

    reference_frame_index, reference_detections = max(
        usable_frames,
        key=lambda item: (len(item[1]), sum(score for _, score, _ in item[1])),
    )
    reference_detections = sorted(reference_detections, key=lambda item: box_center(item[0]))
    reference_centers = [box_center(detection[0]) for detection in reference_detections]
    reference_labels = [int(detection[2]) for detection in reference_detections]
    reference_scores = [float(detection[1]) for detection in reference_detections]

    pooled_xs = [[center_x] for center_x, _ in reference_centers]
    pooled_ys = [[center_y] for _, center_y in reference_centers]

    for frame_index, frame_detections in detections_by_frame:
        if frame_index == reference_frame_index or not frame_detections:
            continue

        frame_detections = sorted(frame_detections, key=lambda item: box_center(item[0]))
        frame_centers = [box_center(detection[0]) for detection in frame_detections]
        matched_centers = match_frame_detections(reference_centers, frame_centers)

        for pooled_index, matched_center in enumerate(matched_centers):
            if matched_center is None:
                continue
            pooled_xs[pooled_index].append(matched_center[0])
            pooled_ys[pooled_index].append(matched_center[1])

    pooled_centers = [(median(xs), median(ys)) for xs, ys in zip(pooled_xs, pooled_ys)]
    pooled_annotations = [
        (center, reference_labels[index], reference_scores[index])
        for index, center in enumerate(pooled_centers)
    ]
    pooled_annotations.sort(key=lambda item: (item[0][0], item[0][1]))
    return pooled_annotations[:top_n_marks]