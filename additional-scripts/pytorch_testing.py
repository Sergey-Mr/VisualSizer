import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os
# To make grid of images.
from torchvision.utils import make_grid
from torchvision.io import read_image
from PIL import Image

image_1_path = os.path.join ('mecd.jpg')
image_2_path = os.path.join('input', 'image_2.jpg')
image_3_path = os.path.join('input', 'image_3.jpg')

image_1 = read_image(image_1_path)
image_2 = read_image(image_2_path)
image_3 = read_image(image_3_path)

image_1 = F.resize(image_1, (450, 450))
image_2 = F.resize(image_2, (450, 450))
image_3 = F.resize(image_3, (450, 450))

grid = make_grid([image_1, image_2, image_3])

def show(image):
    #plt.figure(figsize=(12, 9))
    plt.imshow(np.transpose(image, [1, 2, 0]))
    plt.show()


#show(grid)

from torchvision.utils import draw_bounding_boxes

boxes = torch.tensor([
    [135, 50, 210, 365], 
    [210, 59, 280, 370],
    [300, 240, 375, 380]
])
colors = ['red', 'red', 'green']
result = draw_bounding_boxes(
    image=image_1, 
    boxes=boxes, 
    colors=colors, 
    width=3
)
#show(result)

colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0)]
result = draw_bounding_boxes(
    image_1, boxes=boxes, 
    colors=colors, width=3
)
#show(result)

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms as transforms

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

detection_threshold = 0.8
transform = transforms.Compose([
        transforms.ToTensor(),
])

def read_transform_return(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image_transposed = np.transpose(image, [2, 0, 1])
    # Convert to uint8 tensor.
    int_input = torch.tensor(image_transposed)
    # Convert to float32 tensor.
    tensor_input = transform(image)
    tensor_input = torch.unsqueeze(tensor_input, 0)
    return int_input, tensor_input


int_input, tensor_input = read_transform_return(image_1_path)
model = fasterrcnn_resnet50_fpn(pretrained=True, min_size=800)
model.eval()
outputs = model(tensor_input)

pred_scores = outputs[0]['scores'].detach().cpu().numpy()
pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels'].cpu().numpy()]
pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
pred_classes = pred_classes[:len(boxes)]

colors=np.random.randint(0, 255, size=(len(boxes), 3))
colors = [tuple(color) for color in colors]

result_with_boxes = draw_bounding_boxes(
    image=int_input, 
    boxes=torch.tensor(boxes), width=4, 
    colors=colors,
    labels=pred_classes,
)
#show(result_with_boxes)

from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import draw_segmentation_masks

VOC_SEG_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
label_color_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus 
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

int_input, tensor_input = read_transform_return(image_1_path)
model = fcn_resnet50(pretrained=True)
model.eval()
outputs = model(tensor_input)
labels = torch.argmax(outputs['out'][0].squeeze(), dim=0).detach().cpu().numpy()
boolean_mask = torch.tensor(labels, dtype=torch.bool)
seg_result = draw_segmentation_masks(
    image=int_input, 
    masks=boolean_mask,
    alpha=0.5
)
#show(seg_result)

num_classes = outputs['out'].shape[1]
masks = outputs['out'][0]
class_dim = 0 # 0 as it is a single image and not a batch.
all_masks = masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]

seg_result = draw_segmentation_masks(
    int_input, 
    all_masks,
    colors=label_color_map,
    alpha=0.5
)
show(seg_result)