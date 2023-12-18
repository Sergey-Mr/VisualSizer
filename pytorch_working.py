import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os
# To make grid of images.
from torchvision.utils import make_grid
from torchvision.io import read_image
from PIL import Image

def show(image):
    #plt.figure(figsize=(12, 9))
    plt.imshow(np.transpose(image, [1, 2, 0]))
    #plt.subplots_adjust(bottom = 0)
    #plt.subplots_adjust(top = 1)
    #plt.subplots_adjust(right = 1)
    #plt.subplots_adjust(left = 0)
    plt.axis('off')
    plt.savefig('output-2.png', bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()


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


def mask_create(input_image):
    image_1_path = os.path.join (input_image)
    #image_1_path = os.path.join ('images-archive/fallowfield-1.jpg')

    image_1 = read_image(image_1_path)

    #show(image_1)

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision import transforms as transforms

    detection_threshold = 0.8
    global transform
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])

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
                   #(192, 128, 128), # person
                   (255, 255, 255), # person
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
        alpha=1
    )

    show(seg_result)


#mask_create('mecd.jpg')