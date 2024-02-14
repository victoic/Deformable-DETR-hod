# Home Object Dangerousness Dataset
# by AVA Lundgren & Richard

"""
Module providing access to the class for the House Object Dangerousness
detection and classification dataset.
"""

import torch
from torchvision.io import read_image
import datasets.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from pycocotools.coco import COCO

import os
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
from util.misc import get_local_rank, get_local_size

class HODataset(Dataset):
    def __init__(self, annotations_file, img_dir, coco=False, transform=None, target_transform=None, cache_mode=False):
        self.images_paths = []
        self.labels = []
        self.coco = COCO(annotations_file)
        for img in self.coco.loadImgs(self.coco.getImgIds()):
            annotations = []
            for ann in self.coco.loadAnns(self.coco.getAnnIds(img['id'])):
                annotation = {'category_id': ann['category_id'], 'area':ann['area'], 'bbox':ann['bbox']}
                annotations.append(annotation)
            target = {'image_id': img['id'], 'annotations': annotations}
            self.images_paths.append(img['file_name'])
            self.labels.append(target)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.cache_mode = cache_mode

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        path = self.coco.loadImgs(img_id)[0]['file_name']
        print(os.path.join(self.root, path))
        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_hod_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.494, 0.523, 0.538], [0.072, 0.089, 0.099])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root, root / args.train_anns),
        "val": (root, root / args.val_anns),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = HODataset(ann_file, img_folder, transform=make_hod_transforms(image_set), cache_mode=args.cache_mode)
    return dataset