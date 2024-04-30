# Object Detection
####  Using the COCO pre-trained model h5 and  the MRCNN algorithm
### Overview
This repository contains the implementation of a Mask R-CNN model trained for detecting drones in images. The model is built using the Matterport Mask R-CNN implementation and trained on a custom drone dataset.

### Model Architecture
The Mask R-CNN architecture consists of a backbone network (e.g., ResNet) followed by two subnetworks: a Region Proposal Network (RPN) and a Mask Head. The RPN proposes candidate object bounding boxes, while the Mask Head refines these boxes and predicts binary masks for each object.

### Dataset
The model was trained on a custom drone dataset consisting of 207 images for training and 70 images for validation. The dataset includes images captured from various angles and distances to ensure robust detection performance.

### Training
The model was trained for 40 epochs using the Adam optimizer with a learning rate of 0.001. The training process involved optimizing the following losses:

RPN Loss
Classification Loss
Mask Loss

### Usage
To use the trained model for inference, follow these steps:

Download the trained weights (drone_mask_rcnn.h5) from the releases section of Huggingface https://huggingface.co/bilalmashooq/Drone_Detection_MRCNN 
Install the required dependencies by running pip install -r requirements.txt.
Use the provided script (prediction.py) to perform inference on your images. 
