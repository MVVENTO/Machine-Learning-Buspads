![](https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png)

# Machine Learning Using Custom Dataset (Buspads)

## Project Overview
This project focuses on training a machine learning model using a custom dataset (Buspads) with the YOLOv5 framework. The goal is to detect and classify objects effectively within the dataset.

## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.7+
- PyTorch
- YOLOv5 repository
- OpenCV
- Pandas
- Matplotlib

You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
1. Collect and label images in the **Buspads** dataset using tools like LabelImg or Roboflow.
2. Structure the dataset as follows:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   ├── test/
   ├── labels/
   │   ├── train/
   │   ├── val/
   │   ├── test/
   ```
3. Ensure that label files are in YOLO format (i.e., `class x_center y_center width height`).
4. Update the dataset configuration file (`dataset.yaml`) to include:
   ```yaml
   train: path/to/train/images
   val: path/to/val/images
   test: path/to/test/images
   nc: <number_of_classes>
   names: ['class1', 'class2', ...]
   ```

## Training the Model
Run the following command to start training:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --device 0
```
- `--img`: Input image size
- `--batch`: Batch size
- `--epochs`: Number of training epochs
- `--data`: Path to dataset configuration
- `--weights`: Pre-trained model weights (use `yolov5s.pt`, `yolov5m.pt`, etc.)
- `--device`: GPU/CPU selection (`0` for GPU, `cpu` for CPU)

## Evaluating the Model
To evaluate model performance:
```bash
python val.py --data dataset.yaml --weights runs/train/exp/weights/best.pt --img 640
```

## Running Inference
To test the model on new images:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path/to/images
```

## Results and Visualization
- Training logs and results are stored in `runs/train/exp/`
- Use `tensorboard` to visualize logs:
  ```bash
  tensorboard --logdir runs/train
  ```
  
![](https://github.com/MVVENTO/Machine-Learning-Buspads/blob/main/MLresults-buspads.png)

## Conclusion
This project demonstrates how to train a YOLOv5 model using a custom dataset (Buspads). Further improvements can be made by optimizing hyperparameters and augmenting the dataset for better accuracy.

---
For any issues or contributions, feel free to open a discussion or pull request!

