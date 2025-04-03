![](https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png)

# Machine Learning Using Custom Dataset (Buspads)

## Overview
This project applies machine learning techniques to a custom dataset focused on bus pad detection. The goal is to develop a model that accurately identifies and classifies bus pads in urban environments using deep learning frameworks.

## Features
- Custom dataset tailored for bus pad detection
- Data preprocessing and augmentation techniques
- Model training using YOLOv5
- Evaluation and validation of model performance
- Deployment-ready inference script

## Dataset
The dataset consists of annotated images focusing on bus pads in various urban conditions. Each image is labeled in the YOLO format (`class x_center y_center width height`) to assist in model training and evaluation.

### Dataset Structure
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
The dataset configuration file (`dataset.yaml`) must be updated as follows:
```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
nc: <number_of_classes>
names: ['class1', 'class2', ...]
```

### Example Images
#### Sample Annotated Image
![Sample Annotated Image](path/to/annotated_image.jpg)

#### Training Process Visualization
![Training Process](path/to/training_process.jpg)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/machine-learning-buspads.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset and place it in the `data/` directory.

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

## Results
The model's performance is assessed using accuracy, precision, recall, and F1-score. The following table summarizes the results:

![](https://github.com/MVVENTO/Machine-Learning-Buspads/blob/main/MLresults-buspads.png)
# ![](https://github.com/MVVENTO/Machine-Learning-Buspads/blob/main/MLresults-buspads.png)

| Metric  | Score  |
|---------|--------|
| Accuracy| 90.5%  |
| Precision | 88.7%  |
| Recall  | 91.2%  |
| F1-score | 89.9%  |

## Future Improvements
- Increase dataset size for better generalization
- Improve annotation quality
- Optimize hyperparameters for better performance
- Implement real-time detection for deployment

## License
This project is licensed under the MIT License.

For any inquiries, please contact [your-email@example.com](mailto:your-email@example.com).


## Conclusion
This project demonstrates how to train a YOLOv5 model using a custom dataset (Buspads). Further improvements can be made by optimizing hyperparameters and augmenting the dataset for better accuracy.

---
For any issues or contributions, feel free to open a discussion or pull request!

