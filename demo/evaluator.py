import json
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tabulate import tabulate
from torch.cuda import OutOfMemoryError

from demo import *

TRAIN_TEST_SPLIT = 0.8

IMAGES_DIR = 'demo/data/images'
SPLIT_PATH = 'demo/data/train_test_split.json'

# SOLAR_PANEL_BASE_PATH = '../solar_panel'
# IMAGES_DIR = os.path.join(SOLAR_PANEL_BASE_PATH, 'Maxar_HD_and_Native_Solar_Panel_Image_Chips/image_chips')
# SPLIT_PATH = f'{SOLAR_PANEL_BASE_PATH}/train_test_split.json'
#
# IMAGES_DIR = 'datasets/person'
# SPLIT_PATH = f'datasets/person/train_test_split.json'

# IMAGES_DIR = 'datasets/solar_panel'
# SPLIT_PATH = f'datasets/solar_panel/train_test_split.json'
#
REDO_ALL = False


class Evaluator:

    def __init__(self):
        self._data_split()

    def _data_split(self):
        if not REDO_ALL and os.path.exists(SPLIT_PATH):
            with open(SPLIT_PATH, 'r') as f:
                self.class_to_split = json.load(f)
                return

        object_classes = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]
        self.class_to_split = dict()

        total = 0
        for object_class in object_classes:
            images = glob(os.path.join(IMAGES_DIR, object_class, '*.mask.png'))
            if len(images) < 5:
                continue

            total += len(images)

            random.shuffle(images)
            split_index = min(int(len(images) * TRAIN_TEST_SPLIT), 100)
            train_data = images[:split_index]
            test_data = images[split_index:]

            if 'person' in images[0]:
                images = glob(os.path.join(IMAGES_DIR, object_class, '*.mask.png'))
                train_data = images[:len(images) - 1]
                test_data = [images[-1]]

            self.class_to_split[object_class] = {
                'train': train_data,
                'test': test_data
            }

        with open(SPLIT_PATH, 'w') as f:
            json.dump(self.class_to_split, f, indent=4)

    def build_prototypes(self):
        import torch
        import os
        import os.path as osp
        import torchvision as tv
        from glob import glob
        from detectron2.data import transforms as T
        from torchvision.transforms import functional as tvF
        torch.set_grad_enabled(False)
        to_pil = tv.transforms.functional.to_pil_image
        from collections import defaultdict
        import torchvision.ops as ops
        import torch.nn.functional as F
        RGB = tv.io.ImageReadMode.RGB

        pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
        normalize_image = lambda x: (x - pixel_mean) / pixel_std

        def iround(x): return int(round(x))

        def resize_to_closest_14x(img):
            h, w = img.shape[1:]
            h, w = max(iround(h / 14), 1) * 14, max(iround(w / 14), 1) * 14
            return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)

        device = 0

        resize_op = T.ResizeShortestEdge(
            short_edge_length=800,
            max_size=1333,
        )

        for object_class in tqdm(self.class_to_split, desc='Building prototypes'):
            prototypes_path = os.path.join(IMAGES_DIR, object_class, 'prototypes.pth')
            if not REDO_ALL and os.path.exists(prototypes_path):
                continue

            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

            # reading metas
            class2images = {}
            for mask_file in self.class_to_split[object_class]['train']:
                image_file = sorted(glob(f'{os.path.splitext(mask_file)[0][:-4]}*'), key=len)[0]
                if object_class not in class2images:
                    class2images[object_class] = []
                class2images[object_class.strip().lower()].append((image_file, mask_file))

            classes = sorted(class2images.keys())

            model = model.to(device)
            class2tokens = {}
            for cls, images in tqdm(class2images.items()):
                class2tokens[cls] = []
                for image_file, mask_file in images:
                    image = tv.io.read_image(image_file, RGB).permute(1, 2, 0)
                    resize = resize_op.get_transform(image)
                    mask = tv.io.read_image(mask_file).permute(1, 2, 0)

                    mask = torch.as_tensor(resize.apply_segmentation(mask.numpy())).permute(2, 0, 1) != 0
                    image = torch.as_tensor(resize.apply_image(image.numpy())).permute(2, 0, 1)

                    image14 = resize_to_closest_14x(image)
                    mask_h, mask_w = image14.shape[1] // 14, image14.shape[2] // 14
                    nimage14 = normalize_image(image14)[None, ...]
                    r = model.get_intermediate_layers(nimage14.to(device),
                                                      return_class_token=True, reshape=True)
                    patch_tokens = r[0][0][0].cpu()
                    mask14 = tvF.resize(mask, (mask_h, mask_w))
                    if mask14.sum() <= 0.5:
                        continue
                    avg_patch_token = (mask14 * patch_tokens).flatten(1).sum(1) / mask14.sum()
                    class2tokens[cls].append(avg_patch_token)

            for cls in class2tokens:
                class2tokens[cls] = torch.stack(class2tokens[cls]).mean(dim=0)

            prototypes = F.normalize(torch.stack([class2tokens[c] for c in classes]), dim=1)

            category_dict = {
                'prototypes': prototypes,
                'label_names': classes
            }

            torch.save(category_dict, prototypes_path)

    @staticmethod
    def _save_predictions(object_class, results, prototypes_class):
        new_results = dict()

        for filename, results_dict in results.items():
            new_results[filename] = dict()
            for key, val in results_dict.items():
                new_results[filename][key] = results_dict[key].cpu().detach().tolist()

        with open(os.path.join(IMAGES_DIR, object_class, f'{prototypes_class}_results.json'), 'w') as f:
            json.dump(new_results, f, indent=4)

    @staticmethod
    def _load_predictions(object_class, prototypes_class, threshold=0.0, overlapping_mode=False):
        results_path = os.path.join(IMAGES_DIR, object_class, f'{prototypes_class}_results.json')
        if not os.path.exists(results_path):
            return dict()

        with open(results_path, 'r') as f:
            results = json.load(f)

        for results_dict in results.values():
            boxes = np.array(results_dict['boxes'])
            pred_classes = np.array(results_dict['pred_classes'], dtype=np.int32)
            scores = np.array(results_dict['scores'])

            indices_to_keep = scores >= threshold

            results_dict['boxes'] = boxes[indices_to_keep]
            results_dict['pred_classes'] = pred_classes[indices_to_keep]
            results_dict['scores'] = scores[indices_to_keep]

        for results_dict in results.values():
            for key, val in results_dict.items():
                if key == 'pred_classes':
                    results_dict[key] = torch.tensor(results_dict[key], dtype=torch.int)
                else:
                    results_dict[key] = torch.tensor(results_dict[key])

            if overlapping_mode:
                # remove some highly overlapped predictions
                mask = box_area(results_dict['boxes']) >= 10  # TODO 400
                boxes = results_dict['boxes'][mask]
                pred_classes = results_dict['pred_classes'][mask]
                scores = results_dict['scores'][mask]
                mask = ops.nms(boxes, scores, 0.3)
                boxes = boxes[mask]
                pred_classes = pred_classes[mask]
                scores = scores[mask]
                areas = box_area(boxes)
                indexes = list(range(len(pred_classes)))
                for c in torch.unique(pred_classes).tolist():
                    box_id_indexes = (pred_classes == c).nonzero().flatten().tolist()
                    for i in range(len(box_id_indexes)):
                        for j in range(i + 1, len(box_id_indexes)):
                            bid1 = box_id_indexes[i]
                            bid2 = box_id_indexes[j]
                            arr1 = boxes[bid1].cpu().numpy()
                            arr2 = boxes[bid2].cpu().numpy()
                            a1 = np.prod(arr1[2:] - arr1[:2])
                            a2 = np.prod(arr2[2:] - arr2[:2])
                            top_left = np.maximum(arr1[:2], arr2[:2])  # [[x, y]]
                            bottom_right = np.minimum(arr1[2:], arr2[2:])  # [[x, y]]
                            wh = bottom_right - top_left
                            ia = wh[0].clip(0) * wh[1].clip(0)
                            if ia >= 0.9 * min(a1,
                                               a2):  # same class overlapping case, and larger one is much larger than small
                                if a1 <= a2:  # TODO swap
                                    if bid2 in indexes:
                                        indexes.remove(bid2)
                                else:
                                    if bid1 in indexes:
                                        indexes.remove(bid1)

                results_dict['boxes'] = boxes[indexes]
                results_dict['pred_classes'] = pred_classes[indexes]
                results_dict['scores'] = scores[indexes]

        return results

    def _make_predictions_for_class(self, object_class, object_classes):
        for prototypes_class in tqdm(object_classes, desc=f'Predicting for {object_class}', position=1):
            results = self._load_predictions(object_class, prototypes_class) if not REDO_ALL else dict()

            with torch.no_grad():
                results = self._make_predictions_for_class_helper(
                    images=self.class_to_split[object_class]['test'],
                    category_space=os.path.join(IMAGES_DIR, prototypes_class, 'prototypes.pth'),
                    output_dir=os.path.join(IMAGES_DIR, object_class),
                    results=results,
                    object_class=object_class
                )

            self._save_predictions(object_class, results, prototypes_class)

    @staticmethod
    def _make_predictions_for_class_helper(
            # config_file="configs/open-vocabulary/lvis/vitl.yaml",
            config_file="configs/few-shot/vitl_shot5.yaml",
            # rpn_config_file="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
            rpn_config_file="configs/RPN/mask_rcnn_R_50_C4_1x_fewshot_14.yaml",
            # model_path="weights/trained/open-vocabulary/lvis/vitl_0069999.pth",
            model_path="weights/trained/few-shot/vitl_0089999.pth",
            images=[],
            # image_dir='demo/input/input_boat',
            output_dir='demo/output',
            category_space="demo/boat_prototypes.pth",
            device='cuda:0',
            overlapping_mode=True,
            topk=1,
            output_pth=False,
            threshold=0.0,
            results=dict(),
            object_class=None
    ):
        # category_space='datasets/person\\person\\prototypes.pth'
        for img_file in images:
            img_file = img_file.replace('.mask', '')
            if img_file not in results:
                break
        else:
            return results
        # assert osp.abspath(image_dir) != osp.abspath(output_dir)
        # os.makedirs(output_dir, exist_ok=True)

        config = get_cfg()
        config.merge_from_file(config_file)
        config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
        config.DE.TOPK = topk
        config.MODEL.MASK_ON = False

        config.freeze()

        augs = utils.build_augmentation(config, False)
        augmentations = T.AugmentationList(augs)

        # building models
        def get_model():
            model = Trainer.build_model(config).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device)['model'])
            model.eval()
            model = model.to(device)
            return model
        model = get_model()

        category_space_name = category_space
        if category_space is not None:
            category_space = torch.load(category_space)

            if len(category_space["label_names"]) < 2:
                category_space["label_names"].append("_blank_")
                real_prototypes = category_space["prototypes"]
                blank_prototypes = torch.zeros(1, 1024)
                category_space["prototypes"] = torch.cat((real_prototypes, blank_prototypes), 0)
            model.label_names = category_space["label_names"]
            model.test_class_weight = category_space["prototypes"].to(device)

        label_names = model.label_names
        if 'mini soccer' in label_names:  # for YCB
            label_names = list_replace(label_names, old='mini soccer', new='ball')

        for img_file in tqdm(images, desc=f'Making predictions for {object_class} using {category_space_name}', position=2):
            retry = True
            while retry:
                try:
                    img_file = sorted(glob(f'{os.path.splitext(img_file)[0][:-4]}*'), key=len)[0]
                    if img_file in results:
                        retry = False
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

                    results[img_file] = {
                        'boxes': boxes.cpu().detach(),
                        'pred_classes': pred_classes.cpu().detach(),
                        'scores': scores.cpu().detach()
                    }

                    retry = False
                    del boxes
                    del pred_classes
                    del scores
                    del batched_inputs
                    del dataset_dict
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                except OutOfMemoryError as e:
                    if retry:
                        print('Out of memory...retrying with new model')
                        return results
                        model = get_model()
                        retry = False
                    else:
                        raise e
        return results

    def make_predictions(self):
        object_classes = self.class_to_split.keys()
        for object_class in tqdm(object_classes, desc='Making predictions for all classes', position=0):
            self._make_predictions_for_class(object_class, object_classes)

    @staticmethod
    def _load_ground_truth(filepath):
        mask_filepath = os.path.splitext(filepath)[0] + '.mask.png'
        mask = Image.open(mask_filepath)
        mask = np.array(mask)

        bboxes = list()
        for object_id in np.unique(mask):
            if object_id == 0:
                continue

            object_coordinates = np.where(mask == object_id)

            min_y, min_x = np.min(object_coordinates, axis=1)
            max_y, max_x = np.max(object_coordinates, axis=1)

            bbox = (min_x, min_y, max_x, max_y)

            bboxes.append(bbox)

        return torch.Tensor(bboxes)

    def evaluate(self):
        # thresholds = [0.0, 0.45, 0.60, 0.75]
        thresholds = [0.3]
        ious_to_test = [0.5]#, 0.7]
        columns = []
        for threshold in thresholds:
            for iou in ious_to_test:
                columns.extend([f'r@t={threshold}_iou={iou}', f'p_self@t={threshold}_iou={iou}', f'p_overall@t={threshold}_iou={iou}'])

        evaluation_results = pd.DataFrame(columns=['class'] + columns)

        # d[threshold][iou]
        total_tp = defaultdict(lambda: defaultdict(int))
        total_p = defaultdict(lambda: defaultdict(int))
        total_pred = defaultdict(lambda: defaultdict(int))

        for object_class in self.class_to_split.keys():
            object_class_results = [object_class]

            for threshold in thresholds:
                for iou in ious_to_test:

                    object_class_overall_pred = 0
                    object_class_tp = 0
                    object_class_self_pred = 0
                    object_class_p = 0

                    for prototypes_class in self.class_to_split.keys():
                        show = True

                        results = self._load_predictions(object_class, prototypes_class=object_class, threshold=threshold, overlapping_mode=True)

                        for filepath, results_dict in results.items():
                            ground_truth = self._load_ground_truth(filepath)
                            object_class_p += len(ground_truth) if object_class == prototypes_class else 0
                            total_p[threshold][iou] += len(ground_truth) if object_class == prototypes_class else 0
                            if len(results_dict['boxes']) == 0:
                                continue

                            ious = np.array(box_iou(ground_truth, results_dict['boxes']))
                            max_iou_idxs = ious.argmax(axis=1)
                            # max_ious1 = ious[:, max_iou_idxs][0]
                            max_ious = np.array(ious).max(axis=1)

                            max_iou_idxs_keep_mask = max_ious > iou
                            results_to_keep = max_iou_idxs[max_iou_idxs_keep_mask]

                            tp_count = np.sum(max_iou_idxs_keep_mask) if object_class == prototypes_class else 0

                            object_class_tp += tp_count

                            if not object_class == prototypes_class:
                                assert tp_count == 0

                            total_tp[threshold][iou] += tp_count
                            object_class_overall_pred += len(results_dict['boxes'])
                            total_pred[threshold][iou] += len(results_dict['boxes'])

                            if object_class == prototypes_class:
                                object_class_self_pred += len(results_dict['boxes'])

                            include_gt = True

                            if show and object_class == prototypes_class:
                                image = utils.read_image(filepath, format="RGB")
                                results_to_keep = np.array(range(len(results_dict['boxes'])))  # TODO remove for only best results
                                label_names = [object_class] * len(results_dict['boxes'][results_to_keep])
                                pred_classes = results_dict['pred_classes'][results_to_keep]
                                boxes = results_dict['boxes'][results_to_keep]

                                if include_gt:
                                    label_names = [object_class] * len(ground_truth) + label_names
                                    pred_classes = torch.cat((torch.tensor([(1 if len(results_to_keep) > 0 else 0)] * len(ground_truth)), pred_classes))
                                    boxes = torch.cat((ground_truth, boxes))

                                colors = assign_colors(pred_classes, label_names, seed=4)
                                labels = ([] if not include_gt else [''] * len(ground_truth)) + [f'{score:.2f}' for cid, score in
                                          zip(results_dict['pred_classes'][results_to_keep].tolist(),
                                              results_dict['scores'][results_to_keep].tolist())]

                                output = to_pil_image(draw_bounding_boxes(torch.as_tensor(image).permute(2, 0, 1),
                                                                          boxes=boxes,
                                                                          labels=labels,
                                                                          colors=colors,
                                                                          font='arial.ttf',
                                                                          font_size=15))

                                plt.figure(figsize=(20, 10))
                                plt.subplot(1, 2, 1)
                                plt.imshow(image, cmap='gray')  # Adjust the colormap ('cmap') if needed
                                plt.subplot(1, 2, 2)
                                plt.imshow(output)
                                plt.imshow(output)
                                plt.suptitle(object_class)
                                plt.tight_layout()
                                plt.show()
                                show = False
                                x=1

                        if object_class == prototypes_class:
                            pass
                            # print(f'{object_class_tp / object_class_p:.3f}: {object_class}')

                        x=1

                    recall = object_class_tp / object_class_p
                    precision_self = object_class_tp / object_class_self_pred if object_class_self_pred > 0 else None
                    precision_overall = object_class_tp / object_class_overall_pred if object_class_overall_pred > 0 else None
                    object_class_results.extend([recall, precision_self, precision_overall])
                    # evaluation_results = pd.concat([
                    #     evaluation_results, pd.DataFrame([[object_class, recall, precision_self, precision_overall]],
                    #                                      columns=evaluation_results.columns),
                    # ],
                    #     ignore_index=True,
                    # )
            evaluation_results = pd.concat([
                evaluation_results, pd.DataFrame([object_class_results],
                                                 columns=evaluation_results.columns),
            ],
                ignore_index=True,
            )


        results = ['all']
        for threshold in thresholds:
            for iou in ious_to_test:
                recall = total_tp[threshold][iou] / total_p[threshold][iou]
                precision_self = None
                precision_overall = total_tp[threshold][iou] / total_pred[threshold][iou]
                results.extend([recall, precision_self, precision_overall])
        evaluation_results = pd.concat(
            [
                evaluation_results, pd.DataFrame([results],
                                                 columns=evaluation_results.columns),
            ],

            ignore_index=True,
        )
        # print(f'{total_tp / total_p:.3f}')
        print(tabulate(evaluation_results.replace(np.nan, None).replace([pd.NaT], [None]), missingval='-',
                       headers='keys', tablefmt='psql', floatfmt='.2f'))


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.build_prototypes()
    evaluator.make_predictions()
    evaluator.evaluate()
