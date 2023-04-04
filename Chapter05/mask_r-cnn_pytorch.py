import numpy as np

print("Mask R-CNN instance segmentation")


def draw_segmentation_mask(img: np.array, segm_objects: dict):
    """Draw bounding boxes and segmentation masks"""

    # iterate over the network output for all segmentation masks and boxes
    for mask, box, score in zip(segm_objects[0]['masks'].detach().numpy(),
                                segm_objects[0]['boxes'].detach().numpy().astype(int),
                                segm_objects[0]['scores'].detach().numpy()):

        # filter the boxes by objectness score
        if score > 0.5:
            # transform bounding box format
            box = [(box[0], box[1]), (box[2], box[3])]

            # overlay the segmentation mask on the image with random color
            img[(mask > 0.5).squeeze(), :] = np.random.uniform(0, 255, size=3)

            # draw the bounding box
            cv2.rectangle(img=img,
                          pt1=box[0],
                          pt2=box[1],
                          color=(255, 255, 255),
                          thickness=2)


# Download object detection image
import os.path
import requests

image_file = 'source_2.png'
if not os.path.isfile(image_file):
    url = 'https://github.com/ivan-vasilev/Python-Deep-Learning-3rd-Edition/blob/main/Chapter05/source_2.png'
    r = requests.get(url)
    with open(image_file, 'wb') as f:
        f.write(r.content)

# load the pytorch model
from torchvision.models.detection import \
    maskrcnn_resnet50_fpn_v2, \
    MaskRCNN_ResNet50_FPN_V2_Weights

model = maskrcnn_resnet50_fpn_v2(
    weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# set the model in evaluation mode
model.eval()

# read the image file
import cv2

image = cv2.imread(image_file)

# transform the input to tensor
import torchvision.transforms as transforms

transform = transforms.ToTensor()

nn_input = transform(image)

# run the model
segmented_objects = model([nn_input])

draw_segmentation_mask(image, segmented_objects)

cv2.imshow("Object detection", image)
cv2.waitKey()
