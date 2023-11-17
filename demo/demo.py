import sys
import os

from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
torch.set_grad_enabled(False)
import numpy as np
import fire
import os.path as osp
from detectron2.config import get_cfg
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from tools.train_net import Trainer, DetectionCheckpointer
from glob import glob

import torchvision as tv
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

import matplotlib.colors
import seaborn as sns
import torchvision.ops as ops
from torchvision.ops import box_area, box_iou
import random

import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont
from copy import copy


def filter_boxes(instances, threshold=0.0):
    indexes = instances.scores >= threshold
    # assert indexes.sum() > 0
    boxes = instances.pred_boxes.tensor[indexes, :]
    pred_classes = instances.pred_classes[indexes]
    return boxes, pred_classes, instances.scores[indexes]


def assign_colors(pred_classes, label_names, seed=1):
    all_classes = torch.unique(pred_classes).tolist()
    all_classes = [label_names[ci] for ci in all_classes]
    colors = list(sns.color_palette("hls", len(all_classes)).as_hex())
    random.seed(seed)
    random.shuffle(colors)
    # class2color = {}
    # for cname, hx in zip(all_classes, colors):
    #     class2color[cname] = hx
    # colors = [class2color[label_names[cid]] for cid in pred_classes.tolist()]
    colors = [colors[cid] for cid in pred_classes.tolist()]
    return colors

def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(draw_bounding_boxes)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    _, H, W = image.shape

    # Determine the scaling factor to make the shortest side 500 pixels
    min_side = min(H, W)
    scale_factor = 500 / min_side
    resize_transform = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((int(H * scale_factor), int(W * scale_factor))),
                                           transforms.ToTensor(),
                                           transforms.ConvertImageDtype(torch.uint8)])
    image = resize_transform(image)
    boxes = torch.tensor([[x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor] for x1, y1, x2, y2 in boxes])

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            label_pos = (bbox[0] + margin, bbox[1] + margin)
            textbox = draw.textbbox(label_pos, label, font=txt_font)
            draw.rectangle(textbox, fill=color)
            draw.text(label_pos, label, font=txt_font, fill="black")

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

def list_replace(lst, old=1, new=10):
    """replace list elements (inplace)"""
    i = -1
    lst = copy(lst)
    try:
        while True:
            i = lst.index(old, i + 1)
            lst[i] = new
    except ValueError:
        pass
    return lst


def main(
        # config_file="configs/open-vocabulary/lvis/vitl.yaml",
        config_file="configs/few-shot/vitl_shot5.yaml",
        # rpn_config_file="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
        rpn_config_file="configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml",
        # model_path="weights/trained/open-vocabulary/lvis/vitl_0069999.pth",
        model_path="weights/trained/few-shot/vitl_0089999.pth",

        image_dir='demo/data/images/rd-crcl',
        output_dir='demo/output',
        # category_space="demo/data/images/rd-crcl/prototypes.pth",
        category_space="demo/rd-crcl_prototypes.pth",
        device='cuda:0',
        overlapping_mode=False,
        topk=1,
        output_pth=False,
        threshold=0.35,
        should_print=False
    ):
    assert osp.abspath(image_dir) != osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
    config.DE.TOPK = topk
    config.MODEL.MASK_ON = False

    config.freeze()

    augs = utils.build_augmentation(config, False)
    augmentations = T.AugmentationList(augs)

    # building models
    model = Trainer.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()
    model = model.to(device)

    if category_space is not None:
        category_space = torch.load(category_space)

        if len(category_space["label_names"]) < 2:
            category_space["label_names"].append("_blank_")
            real_prototypes = category_space["prototypes"]
            blank_prototypes = torch.zeros(1, 1024)
            category_space["prototypes"] = torch.cat((real_prototypes, blank_prototypes), 0)
        model.label_names = category_space["label_names"]
        model.test_class_weight = category_space["prototypes"].to(device)

    label_names =  model.label_names
    if 'mini soccer' in label_names: # for YCB
        label_names = list_replace(label_names, old='mini soccer', new='ball')

    for img_file in tqdm(glob(osp.join(image_dir, '*.png'))):
        if 'mask' in img_file:
            continue
        base_filename = osp.splitext(osp.basename(img_file))[0]

        dataset_dict = {}
        image = utils.read_image(img_file, format="RGB")
        dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]

        aug_input = T.AugInput(image)
        augmentations(aug_input)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(device)

        batched_inputs = [dataset_dict]

        output = model(batched_inputs)[0]
        output['label_names'] = model.label_names
        if output_pth:
            torch.save(output, osp.join(output_dir, base_filename + '.pth'))

        # visualize output
        instances = output['instances']
        boxes, pred_classes, scores = filter_boxes(instances, threshold=threshold)

        if overlapping_mode:
            # remove some highly overlapped predictions
            mask = box_area(boxes) >= 400
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            scores = scores[mask]
            mask = ops.nms(boxes, scores, 0.3)
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            scores = scores[mask]
            areas = box_area(boxes)
            indexes = list(range(len(pred_classes)))
            for c in torch.unique(pred_classes).tolist():
                box_id_indexes = (pred_classes == c).nonzero().flatten().tolist()
                for i in range(len(box_id_indexes)):
                    for j in range(i+1, len(box_id_indexes)):
                        bid1 = box_id_indexes[i]
                        bid2 = box_id_indexes[j]
                        arr1 = boxes[bid1].cpu().numpy()
                        arr2 = boxes[bid2].cpu().numpy()
                        a1 = np.prod(arr1[2:] - arr1[:2])
                        a2 = np.prod(arr2[2:] - arr2[:2])
                        top_left = np.maximum(arr1[:2], arr2[:2]) # [[x, y]]
                        bottom_right = np.minimum(arr1[2:], arr2[2:]) # [[x, y]]
                        wh = bottom_right - top_left
                        ia = wh[0].clip(0) * wh[1].clip(0)
                        if ia >= 0.9 * min(a1, a2): # same class overlapping case, and larger one is much larger than small
                            if a1 >= a2:
                                if bid2 in indexes:
                                    indexes.remove(bid2)
                            else:
                                if bid1 in indexes:
                                    indexes.remove(bid1)

            boxes = boxes[indexes]
            pred_classes = pred_classes[indexes]
        colors = assign_colors(pred_classes, label_names, seed=4)
        output = to_pil_image(draw_bounding_boxes(torch.as_tensor(image).permute(2, 0, 1), boxes, labels=[f'{label_names[cid]}_{score:.2f}' for cid, score in zip(pred_classes.tolist(), scores.tolist())], colors=colors,
                                                  font='arial.ttf',
                                                  font_size=30
                                                  ))
        # output.save(osp.join(output_dir, base_filename + '.out.jpg'))
        plt.imshow(output)
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)