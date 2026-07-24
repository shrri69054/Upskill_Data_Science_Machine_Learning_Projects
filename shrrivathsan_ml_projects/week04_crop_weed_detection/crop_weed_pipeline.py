"""End-to-end YOLOv8 crop/weed detection pipeline."""
import argparse
import random
import shutil
from pathlib import Path
import cv2
import albumentations as A


def preprocess_images(input_dir, output_dir, target_size=(512, 512)):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in supported:
            continue
        img = cv2.imread(str(img_path))
        if img is not None:
            cv2.imwrite(str(output_dir / img_path.name), cv2.resize(img, target_size, interpolation=cv2.INTER_AREA))


def augment_dataset_albumentations(image_path, label_path, out_img_dir, out_lbl_dir, n_augments=2):
    out_img_dir, out_lbl_dir = Path(out_img_dir), Path(out_lbl_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.4),
        A.Rotate(limit=15, p=0.5), A.GaussNoise(p=0.3), A.CLAHE(p=0.3)
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1))
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    bboxes, class_labels = [], []
    if Path(label_path).exists():
        for line in Path(label_path).read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(v) for v in parts[1:]])
    stem = Path(image_path).stem
    for i in range(n_augments):
        result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        out_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        name = f"{stem}_aug{i}"
        cv2.imwrite(str(out_img_dir / f"{name}.jpg"), out_img)
        with open(out_lbl_dir / f"{name}.txt", "w") as f:
            for cls, bbox in zip(result["class_labels"], result["bboxes"]):
                f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bbox)}\n")


def split_dataset(images_dir, labels_dir, output_dir, ratios=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    images = sorted(Path(images_dir).glob("*.jpg"))
    random.shuffle(images)
    n_train, n_val = int(len(images) * ratios[0]), int(len(images) * ratios[1])
    splits = {"train": images[:n_train], "val": images[n_train:n_train+n_val], "test": images[n_train+n_val:]}
    output_dir = Path(output_dir)
    for split, items in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy2(img, img_out / img.name)
            label = Path(labels_dir) / f"{img.stem}.txt"
            if label.exists(): shutil.copy2(label, lbl_out / label.name)
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(f"path: {output_dir.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: crop\n  1: weed\n")
    return yaml_path


def train_yolov8(yaml_path, model_variant="yolov8n.pt", epochs=50, img_size=512, batch=16, project="runs/train"):
    from ultralytics import YOLO
    model = YOLO(model_variant)
    return model.train(data=str(yaml_path), epochs=epochs, imgsz=img_size, batch=batch,
                       project=project, name="crop_weed", patience=15, lr0=0.01,
                       lrf=0.0001, weight_decay=0.0005, mosaic=1.0, degrees=10.0,
                       fliplr=0.5, device=0)


def evaluate_model(weights_path, yaml_path, split="test"):
    from ultralytics import YOLO
    metrics = YOLO(weights_path).val(data=str(yaml_path), split=split)
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    return metrics


def run_inference(weights_path, source, conf_threshold=0.4, output_dir="results"):
    from ultralytics import YOLO
    return YOLO(weights_path).predict(source=source, conf=conf_threshold, save=True, project=output_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-images", default="data/clean_images")
    p.add_argument("--labels", default="data/labels_clean")
    p.add_argument("--workdir", default="data")
    p.add_argument("--weights", default="runs/train/crop_weed/weights/best.pt")
    p.add_argument("--skip-training", action="store_true")
    args = p.parse_args()
    root = Path(args.workdir)
    resized = root / "resized_512"; aug_img = root / "augmented/images"; aug_lbl = root / "augmented/labels"; dataset = root / "dataset"
    preprocess_images(args.clean_images, resized)
    for img in resized.glob("*.jpg"):
        augment_dataset_albumentations(img, Path(args.labels) / f"{img.stem}.txt", aug_img, aug_lbl, 2)
    yaml_path = split_dataset(aug_img, aug_lbl, dataset)
    if not args.skip_training:
        train_yolov8(yaml_path)
        evaluate_model(args.weights, yaml_path)

if __name__ == "__main__":
    main()
