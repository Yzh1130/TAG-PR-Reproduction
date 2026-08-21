import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_train_dataset(Dataset):
    def __init__(
        self,
        ann_root,
        image_root,
        transform,
        split,
        max_words=30,
        use_bridge=False,
    ):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file))
        self.transform = transform
        self.use_bridge = use_bridge
        self.person2text = defaultdict(list)
        self.ground_images_by_person = defaultdict(list)

        person_id2idx = {}
        n = 0
        self.pairs = []
        ground_images_seen = defaultdict(set)

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            # for single view datasets, we set cam_id to 1
            cam_id = ann.get('cam_id', 1)
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            if (
                self.use_bridge
                and cam_id == 1
                and image_path not in ground_images_seen[person_idx]
            ):
                self.ground_images_by_person[person_idx].append(image_path)
                ground_images_seen[person_idx].add(image_path)
            for caption in ann['captions']:
                caption = pre_caption(caption, max_words)
                self.pairs.append((image_path, caption, cam_id, person_idx))
                self.person2text[person_idx].append(caption)

        if self.use_bridge and self._is_rank_zero():
            self._print_bridge_index_summary()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, cam_id, person = self.pairs[index]
        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        sample = {
            'image': image,
            'caption': caption,
            'cam_id': cam_id,
            'id': person,
        }
        if not self.use_bridge:
            return sample

        bridge_candidates = self.ground_images_by_person.get(person, ())
        if cam_id == 0 and bridge_candidates:
            candidate_index = torch.randint(len(bridge_candidates), (1,)).item()
            bridge_image_path = bridge_candidates[candidate_index]
            bridge_image_pil = Image.open(bridge_image_path)
            bridge_image = self.transform(bridge_image_pil.convert('RGB'))
            has_bridge = 1
        else:
            bridge_image = torch.zeros_like(image)
            has_bridge = 0
        sample['bridge_image'] = bridge_image
        sample['has_bridge'] = has_bridge
        return sample

    @staticmethod
    def _is_rank_zero():
        return (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

    def _print_bridge_index_summary(self):
        total_pairs = len(self.pairs)
        aerial_pairs = sum(cam_id == 0 for _, _, cam_id, _ in self.pairs)
        ground_pairs = sum(cam_id == 1 for _, _, cam_id, _ in self.pairs)
        bridgeable_persons = {
            person
            for _, _, cam_id, person in self.pairs
            if cam_id == 0 and self.ground_images_by_person.get(person)
        }
        bridgeable_pairs = sum(
            cam_id == 0 and person in bridgeable_persons
            for _, _, cam_id, person in self.pairs
        )
        bridgeable_percent = (
            100.0 * bridgeable_pairs / total_pairs if total_pairs else 0.0
        )
        average_candidates = (
            sum(
                len(self.ground_images_by_person[person])
                for person in bridgeable_persons
            )
            / len(bridgeable_persons)
            if bridgeable_persons
            else 0.0
        )

        print("BRIDGE INDEX SUMMARY")
        print(f"  total pairs: {total_pairs}")
        print(f"  aerial pairs (cam_id==0): {aerial_pairs}")
        print(f"  ground pairs (cam_id==1): {ground_pairs}")
        print(
            "  identities with ground images: "
            f"{len(self.ground_images_by_person)}"
        )
        print(
            "  bridgeable pairs: "
            f"{bridgeable_pairs} ({bridgeable_percent:.2f}% of total pairs)"
        )
        print(
            "  average candidate ground images per bridgeable identity: "
            f"{average_candidates:.2f}"
        )
        print("END BRIDGE INDEX SUMMARY")


class ps_eval_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            self.image.append(image_path)

            person_id = ann['id']
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, max_words))
                self.txt2person.append(person_id)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption