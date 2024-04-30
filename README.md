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

##### Model Training
Set Up Root Directory:

Establish a root directory (e.g., Object Detection).
Copy Necessary Files:

Within the root directory, duplicate the 'mrcnn' directory.
Download Pre-trained Weights:

Retrieve the pre-trained weights and place them in the root directory.
Access the weights via this [link](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
Create Object Detection Script:

Craft a script dedicated to object detection.
Save this script within the root directory. An exemplary script can be found at: mask-rcnn-prediction.py.
Refer to the subsequent section for the script's code.
Execute the Script:

Run the script to initiate the object detection process.
##### Use Pretrained Model
To use the trained model for inference, follow these steps:

Download the trained weights (drone_mask_rcnn.h5) from the releases section of Huggingface [Trained Model](https://huggingface.co/bilalmashooq/Drone_Detection_MRCNN)

Install the required dependencies by running pip install -r requirements.txt.
Use the provided script (prediction.py) to perform inference on your images. 
