# Home Object Dangerousness Dataset
# by AVA Lundgren & Richard

"""
Module providing access to the class for the House Object Dangerousness
detection and classification dataset.
"""

from torchvision.io import read_image
import datasets.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

from pycocotools.coco import COCO

import os
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
from util.misc import get_local_rank, get_local_size

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

class HODataset(VisionDataset):
    def __init__(self, img_dir, ann_file, transform=None, target_transform=None, transforms=None,
        cache_mode=False, local_rank=0, local_size=1):
        super(HODataset, self).__init__(img_dir, transforms, transform, target_transform)
        self.coco = COCO(ann_file)
        #self.images_paths = []
        #self.labels = []
        # for img in self.coco.loadImgs(self.coco.getImgIds()):
        #     annotations = []
        #     for ann in self.coco.loadAnns(self.coco.getAnnIds(img['id'])):
        #         annotation = {'category_id': ann['category_id'], 'area':ann['area'], 'bbox':ann['bbox']}
        #         annotations.append(annotation)
        #     target = {'image_id': img['id'], 'annotations': annotations}
        #     self.images_paths.append(img['file_name'])
        #     self.labels.append(target)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        path = self.coco.loadImgs(img_id)[0]['file_name']
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

    # def read_xml(self, file, idx):
    #     with open(file,'r') as fp:
    #       string_xml = fp.read()
    #       xml = ET.ElementTree(ET.fromstring(string_xml))
    #       img_path = xml.find('path').text
    #       rois = xml.findall('object')
    #       target = {'image_id': idx, 'annotations': []}
    #       for roi in rois:
    #         label = 1 if roi.find('name').text == 'unsafe' else 0
    #         xmin = int(roi.find('bndbox').find('xmin').text)
    #         ymin = int(roi.find('bndbox').find('ymin').text)
    #         xmax = int(roi.find('bndbox').find('xmax').text)
    #         ymax = int(roi.find('bndbox').find('ymax').text)
    #         width = xmax - xmin
    #         height = ymax - ymin
    #         area = width * height
    #         annotation = {'category_id': label, 'area': area, 'bbox': [xmin, ymin, width, height]}
    #         target['annotations'].append(annotation)
    #       return img_path, target

    # def read_annotations(self, annotations_file):
    #     imgs = []
    #     labels = []
    #     idx = 0
    #     with open(annotations_file, 'r') as f:
    #       for line in f:
    #         xml_path = line.replace('\n', '').replace('\\','/')
    #         img, label = self.read_xml(xml_path, idx)
    #         imgs.append(img)
    #         labels.append(label)
    #         idx+=1
    #     return imgs, labels

    # def to_coco(self, image, target):
    #     w, h = image.size

    #     image_id = target["image_id"]
    #     image_id = torch.tensor([image_id])

    #     anno = target["annotations"]

    #     boxes = [obj["bbox"] for obj in anno]
    #     # guard against no boxes via resizing
    #     boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    #     boxes[:, 2:] += boxes[:, :2]
    #     boxes[:, 0::2].clamp_(min=0, max=w)
    #     boxes[:, 1::2].clamp_(min=0, max=h)

    #     classes = [obj["category_id"] for obj in anno]
    #     classes = torch.tensor(classes, dtype=torch.int64)

    #     keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    #     boxes = boxes[keep]
    #     classes = classes[keep]

    #     target = {}
    #     target["boxes"] = boxes
    #     target["labels"] = classes
    #     target["image_id"] = image_id

    #     area = torch.tensor([obj["area"] for obj in anno])
    #     iscrowd = torch.tensor([0 for obj in anno])
    #     target["area"] = area[keep]
    #     target["iscrowd"] = iscrowd[keep]

    #     target["orig_size"] = torch.as_tensor([int(h), int(w)])
    #     target["size"] = torch.as_tensor([int(h), int(w)])

    #     return image, target

def build(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided HOD path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root, root / f'hod_anns_coco_train.json'),
        "val": (root, root / f'hod_anns_coco_test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = HODataset(img_folder, ann_file, transforms=make_hod_transforms(image_set),
        cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset