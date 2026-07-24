# Crop and Weed Detection using YOLOv8

Dataset format: images plus YOLO `.txt` labels, where class `0 = crop` and `1 = weed`.

```bash
pip install ultralytics opencv-python albumentations
python crop_weed_pipeline.py
```

For preprocessing/splitting only:

```bash
python crop_weed_pipeline.py --skip-training
```
