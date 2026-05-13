import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate masked and YOLO-annotated camera images.")
    parser.add_argument("--data-root", default="./Data/Multi_Modal")
    parser.add_argument("--scenarios", nargs="+", default=["scenario32", "scenario33", "scenario34"])
    parser.add_argument("--source-subdir", default="camera_data")
    parser.add_argument("--mask-subdir", default="camera_mask")
    parser.add_argument("--mask-name", default=None, help="Optional fixed mask filename inside mask-subdir.")
    parser.add_argument("--masked-subdir", default="camera_data_mask")
    parser.add_argument("--masked-yolo-subdir", default="camera_data_mask_yolo")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--box-width", type=int, default=3)
    return parser.parse_args()


def find_mask_path(unit1_dir: Path, scenario_name: str, mask_subdir: str, mask_name: str | None) -> Path:
    mask_dir = unit1_dir / mask_subdir
    candidates = []
    if mask_name:
        candidates.append(mask_dir / mask_name)
    candidates.extend(
        [
            mask_dir / f"{scenario_name}_mask.png",
            mask_dir / f"{scenario_name}_mask.jpg",
            mask_dir / "scenario_mask.png",
            mask_dir / "scenario_mask.jpg",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find fixed mask for {scenario_name} under {mask_dir}")


def load_binary_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("RGB").resize(target_size, Image.NEAREST)
    mask_arr = np.asarray(mask_img, dtype=np.uint8)

    blue_dominant = (mask_arr[..., 2] > 180) & (mask_arr[..., 2] > mask_arr[..., 1] + 40) & (
        mask_arr[..., 2] > mask_arr[..., 0] + 40
    )
    keep_mask = ~blue_dominant

    if keep_mask.mean() < 0.01:
        gray = np.asarray(mask_img.convert("L"), dtype=np.uint8)
        keep_mask = gray > 127
    return keep_mask.astype(np.uint8)


def apply_mask(image: Image.Image, keep_mask: np.ndarray) -> Image.Image:
    image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    masked_arr = image_arr.copy()
    masked_arr[keep_mask == 0] = 0
    return Image.fromarray(masked_arr)


def build_yolo(model_name: str):
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Please install it on AutoDL before running YOLO preprocessing.")
    return YOLO(model_name)


def draw_vehicle_boxes(image: Image.Image, results, box_width: int) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = results.names.get(cls_id, str(cls_id)).lower()
        if cls_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        for offset in range(box_width):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(255, 0, 0))
    return annotated


def process_scenario(args, scenario_name: str, yolo_model):
    unit1_dir = Path(args.data_root) / scenario_name / "unit1"
    source_dir = unit1_dir / args.source_subdir
    masked_dir = unit1_dir / args.masked_subdir
    masked_yolo_dir = unit1_dir / args.masked_yolo_subdir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {source_dir}")

    masked_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_yolo:
        masked_yolo_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    mask_path = find_mask_path(unit1_dir, scenario_name, args.mask_subdir, args.mask_name)
    print(f"[{scenario_name}] source={source_dir}")
    print(f"[{scenario_name}] mask={mask_path}")

    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        keep_mask = load_binary_mask(mask_path, image.size)
        masked_image = apply_mask(image, keep_mask)

        masked_output_path = masked_dir / image_path.name
        masked_image.save(masked_output_path)

        if not args.skip_yolo:
            results = yolo_model.predict(
                source=np.asarray(masked_image),
                imgsz=args.img_size,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            annotated = draw_vehicle_boxes(masked_image, results, args.box_width)
            annotated.save(masked_yolo_dir / image_path.name)

        if index % 200 == 0 or index == len(image_paths):
            print(f"[{scenario_name}] processed {index}/{len(image_paths)}")


def main():
    args = parse_args()
    yolo_model = None if args.skip_yolo else build_yolo(args.yolo_model)

    for scenario_name in args.scenarios:
        process_scenario(args, scenario_name, yolo_model)


if __name__ == "__main__":
    main()
