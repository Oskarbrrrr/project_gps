import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

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
    parser.add_argument("--masked-enhance-subdir", default="camera_data_mask_enhance")
    parser.add_argument("--masked-yolo-subdir", default="camera_data_mask_yolo")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--box-width", type=int, default=3)
    parser.add_argument("--background-dim-factor", type=float, default=0.35)
    parser.add_argument("--background-saturation-factor", type=float, default=0.55)
    parser.add_argument("--vehicle-brightness-factor", type=float, default=1.18)
    parser.add_argument("--vehicle-contrast-factor", type=float, default=1.22)
    parser.add_argument("--vehicle-sharpness-factor", type=float, default=1.15)
    parser.add_argument("--vehicle-expand", type=int, default=10)
    parser.add_argument("--vehicle-halo-width", type=int, default=4)
    parser.add_argument("--mask-fill-small-holes", type=int, default=12000)
    parser.add_argument("--mask-remove-small-islands", type=int, default=12000)
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


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = []

    for x in range(height):
        for y in range(width):
            if (not mask[x, y]) or visited[x, y]:
                continue

            stack = [(x, y)]
            visited[x, y] = True
            coords = []

            while stack:
                cx, cy = stack.pop()
                coords.append((cx, cy))

                for nx in range(max(0, cx - 1), min(height, cx + 2)):
                    for ny in range(max(0, cy - 1), min(width, cy + 2)):
                        if mask[nx, ny] and (not visited[nx, ny]):
                            visited[nx, ny] = True
                            stack.append((nx, ny))

            components.append(coords)

    return components


def clean_fixed_mask(keep_mask: np.ndarray, fill_small_holes: int, remove_small_islands: int) -> np.ndarray:
    keep_mask = keep_mask.astype(bool)

    if fill_small_holes > 0:
        masked_area = ~keep_mask
        for coords in connected_components(masked_area):
            touches_border = any(
                x == 0 or y == 0 or x == keep_mask.shape[0] - 1 or y == keep_mask.shape[1] - 1
                for x, y in coords
            )
            if (not touches_border) and len(coords) <= fill_small_holes:
                for x, y in coords:
                    keep_mask[x, y] = True

    if remove_small_islands > 0:
        for coords in connected_components(keep_mask):
            if len(coords) <= remove_small_islands:
                for x, y in coords:
                    keep_mask[x, y] = False

    return keep_mask.astype(np.uint8)


def load_binary_mask(
    mask_path: Path,
    target_size: tuple[int, int],
    fill_small_holes: int,
    remove_small_islands: int,
) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("RGB").resize(target_size, Image.NEAREST)
    mask_arr = np.asarray(mask_img, dtype=np.uint8)

    blue_dominant = (mask_arr[..., 2] > 180) & (mask_arr[..., 2] > mask_arr[..., 1] + 40) & (
        mask_arr[..., 2] > mask_arr[..., 0] + 40
    )
    keep_mask = ~blue_dominant

    if keep_mask.mean() < 0.01:
        gray = np.asarray(mask_img.convert("L"), dtype=np.uint8)
        keep_mask = gray > 127
    return clean_fixed_mask(keep_mask, fill_small_holes, remove_small_islands)


def apply_mask(image: Image.Image, keep_mask: np.ndarray) -> Image.Image:
    image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    masked_arr = image_arr.copy()
    masked_arr[keep_mask == 0] = 0
    return Image.fromarray(masked_arr)


def apply_soft_mask(
    image: Image.Image,
    keep_mask: np.ndarray,
    brightness_factor: float,
    saturation_factor: float,
) -> Image.Image:
    rgb_image = image.convert("RGB")
    dimmed = ImageEnhance.Brightness(rgb_image).enhance(brightness_factor)
    dimmed = ImageEnhance.Color(dimmed).enhance(saturation_factor)

    original_arr = np.asarray(rgb_image, dtype=np.uint8)
    dimmed_arr = np.asarray(dimmed, dtype=np.uint8)
    keep_mask_3d = keep_mask.astype(bool)[..., None]
    mixed_arr = np.where(keep_mask_3d, original_arr, dimmed_arr)
    return Image.fromarray(mixed_arr)


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


def enhance_vehicle_regions(
    image: Image.Image,
    results,
    brightness_factor: float,
    contrast_factor: float,
    sharpness_factor: float,
    expand: int,
    halo_width: int,
) -> Image.Image:
    enhanced = image.copy().convert("RGB")
    draw = ImageDraw.Draw(enhanced)
    image_w, image_h = enhanced.size

    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = results.names.get(cls_id, str(cls_id)).lower()
        if cls_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(image_w, x2 + expand)
        y2 = min(image_h, y2 + expand)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = enhanced.crop((x1, y1, x2, y2))
        crop = ImageEnhance.Brightness(crop).enhance(brightness_factor)
        crop = ImageEnhance.Contrast(crop).enhance(contrast_factor)
        crop = ImageEnhance.Sharpness(crop).enhance(sharpness_factor)
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        enhanced.paste(crop, (x1, y1, x2, y2))

        for offset in range(max(1, halo_width)):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=(255, 64, 64),
            )

    return enhanced


def process_scenario(args, scenario_name: str, yolo_model):
    unit1_dir = Path(args.data_root) / scenario_name / "unit1"
    source_dir = unit1_dir / args.source_subdir
    masked_dir = unit1_dir / args.masked_subdir
    masked_enhance_dir = unit1_dir / args.masked_enhance_subdir
    masked_yolo_dir = unit1_dir / args.masked_yolo_subdir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {source_dir}")

    masked_dir.mkdir(parents=True, exist_ok=True)
    masked_enhance_dir.mkdir(parents=True, exist_ok=True)
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
        keep_mask = load_binary_mask(
            mask_path,
            image.size,
            fill_small_holes=args.mask_fill_small_holes,
            remove_small_islands=args.mask_remove_small_islands,
        )
        masked_image = apply_mask(image, keep_mask)
        soft_masked_image = apply_soft_mask(
            image,
            keep_mask,
            brightness_factor=args.background_dim_factor,
            saturation_factor=args.background_saturation_factor,
        )

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
            enhanced = enhance_vehicle_regions(
                soft_masked_image,
                results,
                brightness_factor=args.vehicle_brightness_factor,
                contrast_factor=args.vehicle_contrast_factor,
                sharpness_factor=args.vehicle_sharpness_factor,
                expand=args.vehicle_expand,
                halo_width=args.vehicle_halo_width,
            )
            enhanced.save(masked_enhance_dir / image_path.name)
        else:
            soft_masked_image.save(masked_enhance_dir / image_path.name)

        if index % 200 == 0 or index == len(image_paths):
            print(f"[{scenario_name}] processed {index}/{len(image_paths)}")


def main():
    args = parse_args()
    yolo_model = None if args.skip_yolo else build_yolo(args.yolo_model)

    for scenario_name in args.scenarios:
        process_scenario(args, scenario_name, yolo_model)


if __name__ == "__main__":
    main()
