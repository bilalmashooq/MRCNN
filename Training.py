import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros, asarray
from keras.callbacks import Callback
import mrcnn.utils
import mrcnn.config
import mrcnn.model
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.class_losses = []
        self.rpn_losses = []
        self.box_losses = []
        self.class_val_losses = []
        self.rpn_val_losses = []
        self.box_val_losses = []
        self.val_accs = []
        self.accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.class_losses.append(logs.get('class_loss'))
        self.rpn_losses.append(logs.get('rpn_class_loss'))
        self.box_losses.append(logs.get('mrcnn_bbox_loss'))
        # Add validation losses for class, rpn, and box
        self.class_val_losses.append(logs.get('val_class_loss'))
        self.rpn_val_losses.append(logs.get('val_rpn_class_loss'))
        self.box_val_losses.append(logs.get('val_mrcnn_bbox_loss'))
        self.val_accs.append(logs.get('val_mrcnn_class_accuracy'))
        self.accs.append(logs.get('mrcnn_class_accuracy'))


class droneDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "drone")
        images_dir = os.path.join(dataset_dir, 'Train', 'Images')
        annotations_dir = os.path.join(dataset_dir, 'Train', 'Annotation', 'annotation.json')

        if not is_train:
            images_dir = os.path.join(dataset_dir, 'Val', 'Images')
            annotations_dir = os.path.join(dataset_dir, 'Val', 'Annotation', 'Val_json.json')

        with open(annotations_dir) as f:
            data = json.load(f)

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            img_path = os.path.join(images_dir, filename)
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=data)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        data = info['annotation']
        boxes, w, h = self.extract_boxes(data)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('drone'))
        return masks, asarray(class_ids, dtype='int32')

    def extract_boxes(self, data):
        boxes = list()
        for key in data.keys():
            regions = data[key]['regions']
            for region in regions:
                shape_attributes = region['shape_attributes']
                if shape_attributes['name'] == 'rect':
                    x, y, w, h = shape_attributes['x'], shape_attributes['y'], shape_attributes['width'], shape_attributes['height']
                    coors = [x, y, x + w, y + h]
                    boxes.append(coors)

        width = max(coors[2] for coors in boxes)
        height = max(coors[3] for coors in boxes)
        return boxes, width, height


class droneConfig(mrcnn.config.Config):
    NAME = "drone_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001


# Custom Data Sequence Generator
class DataSequence(Sequence):
    def __init__(self, dataset, config, batch_size=1):
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dataset.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_ids = self.dataset.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_masks = []
        for image_id in batch_ids:
            image, image_meta, class_ids, bbox, mask = mrcnn.model.load_image_gt(self.dataset, self.config, image_id,
                                                                                  augment=False, use_mini_mask=False)
            batch_images.append(image)
            batch_masks.append(mask)
        return np.array(batch_images), [np.array(batch_masks), np.array(class_ids), np.array(bbox), np.array(image_meta)]

# Train dataset
train_dataset = droneDataset()
train_dataset.load_dataset(dataset_dir=r'C:\Users\muham\PycharmProjects\mrcnn\Dataset', is_train=True)
train_dataset.prepare()

# Validation dataset
validation_dataset = droneDataset()
validation_dataset.load_dataset(dataset_dir=r'C:\Users\muham\PycharmProjects\mrcnn\Dataset', is_train=False)
validation_dataset.prepare()

# Model Configuration
drone_config = droneConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training',
                             model_dir='./',
                             config=drone_config)

model.load_weights(filepath='mask_mrcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model and retrieve the training history using the custom callback
history_callback = LossHistory()
model.train(train_dataset=train_dataset,
            val_dataset=validation_dataset,
            learning_rate=drone_config.LEARNING_RATE,
            epochs=40,  # Choose the number of epochs you want to train for
            layers='heads',
            custom_callbacks=[history_callback])

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history_callback.losses) + 1), history_callback.losses, label='Loss')
plt.plot(np.arange(1, len(history_callback.val_losses) + 1), history_callback.val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the Class loss
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history_callback.class_losses) + 1), history_callback.class_losses, label='Class Loss')
plt.plot(np.arange(1, len(history_callback.class_val_losses) + 1), history_callback.class_val_losses, label='Val Class Loss')
plt.title('Training and Validation Class Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'Class', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

# Plot the RPN loss
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history_callback.rpn_losses) + 1), history_callback.rpn_losses, label='RPN Loss')
plt.plot(np.arange(1, len(history_callback.rpn_val_losses) + 1), history_callback.rpn_val_losses, label='Val RPN Loss')
plt.title('Training and Validation RPN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'RPN', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

# Plot the Box loss
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history_callback.box_losses) + 1), history_callback.box_losses, label='Box Loss')
plt.plot(np.arange(1, len(history_callback.box_val_losses) + 1), history_callback.box_val_losses, label='Val Box Loss')
plt.title('Training and Validation Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'Box', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

# Plot validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history_callback.val_accs) + 1), history_callback.val_accs, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Plot training accuracy
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(history_callback.accs) + 1), history_callback.accs, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Save the trained model
model_path = 'drone_mask_rcnn_coco_datasplit40.h5'
model.keras_model.save_weights(model_path)