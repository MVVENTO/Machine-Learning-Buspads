# https://docs.roboflow.com/deploy/hosted-api/custom-models/object-detection
# https://github.com/niconielsen32/NeuralNetworks/blob/main/YOLOv8CustomObjectDetection.ipynb

# Custom Training with YOLOv5

# In this tutorial, we assemble a dataset and train a custom YOLOv5 model to recognize the objects in our dataset. To do so we will take the following steps:

* Gather a dataset of images and label our dataset
* Export our dataset to YOLOv5
* Train YOLOv5 to recognize the objects in our dataset
* Evaluate our YOLOv5 model's performance
* Run test inference to view our model at work



![](https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png)


# STEP 1 : Install Requirements

#clone YOLOv5 and
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


# Step 2: Assemble Our Dataset







In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. And we need our dataset to be in YOLOv5 format.

In Roboflow, you can choose between two paths:

* Convert an existing dataset to YOLOv5 format. Roboflow supports over [30 formats object detection formats](https://roboflow.com/formats) for conversion.
* Upload raw images and annotate them in Roboflow with [Roboflow Annotate](https://docs.roboflow.com/annotate).

# Annotate

![](https://roboflow-darknet.s3.us-east-2.amazonaws.com/roboflow-annotate.gif)

# Version

![](https://roboflow-darknet.s3.us-east-2.amazonaws.com/robolfow-preprocessing.png)

# set up environment
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

# Install Manually Annotated images from Roboflow

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="d6XLvYat4sv7cDuJOcZM")
project = rf.workspace("opal_v").project("bus-pads")
version = project.version(6)
dataset = version.download("yolov5")

from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="d6XLvYat4sv7cDuJOcZM")

# Retrieve the project and workspace
workspace = rf.workspace("opal_v")
print("Workspace: ", workspace)  # Debugging print statement

project = workspace.project("bus-pads")
print("Project: ", project)  # Debugging print statement

# Retrieve the model version
roboflow_model = project.version(6).model
print("Model: ", roboflow_model)  # Debugging print statement

# Check if the model is successfully retrieved
if roboflow_model is None:
    print("Failed to retrieve the model. Please check your project and version details.")
else:
    # Infer on a local image
    result = roboflow_model.predict("EPSG3857_Date20240616_Lat40.677157_Lon-73.899502_Mpp0.075.jpg", confidence=40, overlap=30).json()
    print(result)

    # Visualize your prediction
    roboflow_model.predict("EPSG3857_Date20240616_Lat40.677157_Lon-73.899502_Mpp0.075.jpg", confidence=40, overlap=30).save("prediction.jpg")

    # Infer on an image hosted elsewhere
    # print(roboflow_model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())



from roboflow import Roboflow
rf = Roboflow(api_key="d6XLvYat4sv7cDuJOcZM")
project = rf.workspace("opal_v").project("bus-pads") # make sure to define the project
# Get the Roboflow model object, avoid overwriting with a variable named 'model'
#roboflow_model = project.version(4).model

# Step 3: Train Our Custom YOLOv5 model

Here, we are able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
- **data:** Our dataset locaiton is saved in the `dataset.location`
- **weights:** specify a path to weights to start transfer learning from. Here we choose the generic COCO pretrained checkpoint.
- **cache:** cache images for faster training

!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache

# Allowing Permissions to open trained set (Optional)

import os

directory_path = "runs/train/exp"

# Check permissions
if os.path.exists(directory_path):
    permissions = os.stat(directory_path).st_mode
    print("Permissions for", directory_path, ":", oct(permissions))
else:
    print("Directory", directory_path, "does not exist.")

import os

directory_path = "runs/train/exp"

# Check if the directory exists
if os.path.exists(directory_path):
    # Get the permissions of the directory
    permissions = os.stat(directory_path).st_mode
    # Print the permissions in octal format
    print(f"Permissions for {directory_path}: {oct(permissions)}")
else:
    print(f"Directory {directory_path} does not exist.")
    # Optionally, create the directory
    os.makedirs(directory_path)
    print(f"Directory {directory_path} has been created.")
    # Get the permissions of the newly created directory
    permissions = os.stat(directory_path).st_mode
    # Print the permissions in octal format
    print(f"Permissions for {directory_path}: {oct(permissions)}")

import os

directory_path = "runs/train/exp"

# Check if the directory exists
if os.path.exists(directory_path):
    # Get the permissions of the directory
    permissions = os.stat(directory_path).st_mode
    # Print the permissions in octal format
    print(f"Permissions for {directory_path}: {oct(permissions)}")
else:
    print(f"Directory {directory_path} does not exist.")
    # Create the directory
    os.makedirs(directory_path)
    print(f"Directory {directory_path} has been created.")

# Set the permissions to 755 explicitly
os.chmod(directory_path, 0o755)
# Get the permissions of the directory after change
permissions = os.stat(directory_path).st_mode
# Print the permissions in octal format
print(f"Permissions for {directory_path} after chmod: {oct(permissions)}")

import os
import torch
from pathlib import Path

# Directory where your model is saved
model_dir = "runs/train/exp/weights"


# Check if the directory exists and list its contents
assert os.path.exists(model_dir), f"Error: Directory {model_dir} does not exist"
print("Files in", model_dir, ":", os.listdir(model_dir))

# Specify the model file name (assuming it's best.pt)
model_file = "best.pt"  # Change if your model file has a different name
path_to_custom_yolov5 = os.path.join(model_dir, model_file)

# Ensure the model file exists
path = Path(path_to_custom_yolov5)
assert path.exists(), f'Error: {path} does not exist'

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the model based on the device
model = torch.load(path, map_location=device)['model'].float()  # Load model
model.eval()
print("Model loaded successfully!")

import os
print("Current directory:", os.getcwd())
os.makedirs("/content/runs/train/exp", exist_ok=True)
results_directory = "/content/runs"
contents = os.listdir(results_directory)
print("Contents of", results_directory, ":", contents)

# Evaluate Custom YOLOv5 Detector Performance
#Training losses and performance metrics are saved to Tensorboard and also to a logfile.

#If you are new to these metrics, the one you want to focus on is `mAP_0.5` - learn more about mean average precision [here](https://blog.roboflow.com/mean-average-precision/).

# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the runs directory exists
log_dir = 'runs/train/exp'
os.makedirs(log_dir, exist_ok=True)

# Dummy training script
writer = SummaryWriter(log_dir=log_dir)

# Define a simple model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(10):
    inputs = torch.randn(1, 10)
    targets = torch.randn(1, 1)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log scalar values
    writer.add_scalar('Loss/train', loss.item(), epoch)

writer.close()

# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs

#Run Inference  With Trained Weights
#Run inference with a pretrained checkpoint on contents of `test/images` folder downloaded from Roboflow.

!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images

import os
import matplotlib.pyplot as plt # Import the matplotlib library and assign it to the alias 'plt'

# Ensure the runs directory exists
log_dir = 'runs/train/exp' # Make sure this is the correct directory
os.makedirs(log_dir, exist_ok=True)

# ... rest of your code ...

# Save confusion matrix
plt.savefig(f'{log_dir}/confusion_matrix.png') # Save in the correct directory

# Display the image
Image(filename=f'{log_dir}/confusion_matrix.png', width=600) # Use correct path

!pip install matplotlib
!pip install numpy
!pip install scikit-learn

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Ensure the runs directory exists
log_dir = 'runs/train/exp'
os.makedirs(log_dir, exist_ok=True)

# Dummy training script
writer = SummaryWriter(log_dir=log_dir)

# Define a simple model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(10):
    inputs = torch.randn(1, 10)
    targets = torch.randn(1, 1)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log scalar values
    writer.add_scalar('Loss/train', loss.item(), epoch)

# Generate and save confusion matrix (replace with your actual predictions and labels)
predictions = torch.randint(0, 2, (100,)) # Example predictions
labels = torch.randint(0, 2, (100,)) # Example labels
cm = confusion_matrix(labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2) # Assuming binary classification
plt.xticks(tick_marks, ["Class 0", "Class 1"], rotation=45)
plt.yticks(tick_marks, ["Class 0", "Class 1"])

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save confusion matrix
plt.savefig(f'{log_dir}/confusion_matrix.png')

writer.close()

# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs

from IPython.display import Image
Image(filename=f'{log_dir}/confusion_matrix.png', width=600)


#RESULTS : display inference on ALL test images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

import glob
from IPython.display import Image, display

# Initialize a counter
image_count = 0

# Loop through the images and display them
for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'):  # assuming JPG
    display(Image(filename=imageName))
    print("\n")

    # Increment the counter for each image
    image_count += 1

# Print the total number of images found and displayed
print(f"Total number of new images found: {image_count}")



![](https://github.com/MVVENTO/Machine-Learning-Buspads/blob/main/MLresults-buspads.png)





















