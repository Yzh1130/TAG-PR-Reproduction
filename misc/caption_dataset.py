import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _initialize_text_length_stats(
    dataset,
    max_words,
    text_length,
    collect_token_stats=False,
):
    if max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    if text_length <= 1:
        raise ValueError("text_length must leave room for SOS and EOS tokens")
    dataset.max_words = max_words
    dataset.text_length = text_length
    dataset.collect_token_stats = collect_token_stats
    dataset.text_sample_count = 0
    dataset.words_after_truncation = 0
    dataset.truncated_sample_count = 0
    dataset.token_overflow_count = 0 if collect_token_stats else None


def _preprocess_caption(dataset, caption):
    caption, word_count, was_truncated = pre_caption(
        caption,
        dataset.max_words,
        return_stats=True,
    )
    dataset.text_sample_count += 1
    dataset.words_after_truncation += min(word_count, dataset.max_words)
    dataset.truncated_sample_count += int(was_truncated)
    if dataset.collect_token_stats:
        from text_utils.tokenizer import _tokenizer

        token_count = len(_tokenizer.encode(caption)) + 2  # SOS and EOS
        dataset.token_overflow_count += int(token_count > dataset.text_length)
    return caption


def _print_text_length_summary(dataset):
    count = dataset.text_sample_count
    mean_words = dataset.words_after_truncation / count if count else 0.0
    truncated_ratio = 100.0 * dataset.truncated_sample_count / count if count else 0.0
    print("TEXT LENGTH SUMMARY")
    print(f"  max_words: {dataset.max_words}")
    print(f"  mean words after truncation: {mean_words:.2f}")
    print(
        "  truncated samples: "
        f"{dataset.truncated_sample_count} ({truncated_ratio:.2f}%)"
    )
    print(
        f"  samples exceeding text_length={dataset.text_length} after tokenization: "
        f"{dataset.token_overflow_count}"
    )
    if dataset.token_overflow_count:
        print("  WARNING: CLIP tokenize will truncate these samples to text_length")
    print("END TEXT LENGTH SUMMARY")


class ps_train_dataset(Dataset):
    def __init__(
        self,
        ann_root,
        image_root,
        transform,
        split,
        max_words=30,
        text_length=77,
        use_bridge=False,
        print_text_summary=False,
    ):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file))
        self.transform = transform
        self.use_bridge = use_bridge
        print_text_summary = print_text_summary and self._is_rank_zero()
        _initialize_text_length_stats(
            self,
            max_words,
            text_length,
            collect_token_stats=print_text_summary,
        )
        self.person2text = defaultdict(list)
        self.ground_images_by_person = defaultdict(list)

        domain_names = sorted({
            ann['file_path'].split('/')[0]
            for ann in anns
        })
        self.domain_to_id = {
            domain_name: domain_id
            for domain_id, domain_name in enumerate(domain_names)
        }
        self.domain_pair_counts = Counter()

        person_id2idx = {}
        n = 0
        self.pairs = []
        ground_images_seen = defaultdict(set)

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            domain_name = ann['file_path'].split('/')[0]
            domain_id = self.domain_to_id[domain_name]
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
                caption = _preprocess_caption(self, caption)
                self.pairs.append(
                    (image_path, caption, cam_id, person_idx, domain_id)
                )
                self.person2text[person_idx].append(caption)
                self.domain_pair_counts[domain_id] += 1

        if self._is_rank_zero():
            self._print_domain_summary()
        if self.use_bridge and self._is_rank_zero():
            self._print_bridge_index_summary()
        if print_text_summary:
            _print_text_length_summary(self)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, cam_id, person, domain_id = self.pairs[index]
        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        sample = {
            'image': image,
            'caption': caption,
            'cam_id': cam_id,
            'id': person,
            'domain_id': domain_id,
        }
        if not self.use_bridge:
            return sample

        bridge_candidates = self.ground_images_by_person.get(person, ())
        if cam_id == 0 and bridge_candidates:
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            bridge_seed = (torch.initial_seed() + index) % (2 ** 32)
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(bridge_seed)
                    np.random.seed(bridge_seed)
                    random.seed(bridge_seed)
                    candidate_index = torch.randint(
                        len(bridge_candidates),
                        (1,),
                    ).item()
                    bridge_image_path = bridge_candidates[candidate_index]
                    bridge_image_pil = Image.open(bridge_image_path)
                    bridge_image = self.transform(
                        bridge_image_pil.convert('RGB')
                    )
            finally:
                np.random.set_state(numpy_rng_state)
                random.setstate(python_rng_state)
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
        aerial_pairs = sum(cam_id == 0 for _, _, cam_id, _, _ in self.pairs)
        ground_pairs = sum(cam_id == 1 for _, _, cam_id, _, _ in self.pairs)
        bridgeable_persons = {
            person
            for _, _, cam_id, person, _ in self.pairs
            if cam_id == 0 and self.ground_images_by_person.get(person)
        }
        bridgeable_pairs = sum(
            cam_id == 0 and person in bridgeable_persons
            for _, _, cam_id, person, _ in self.pairs
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

    def _print_domain_summary(self):
        total_pairs = len(self.pairs)
        print("DOMAIN SUMMARY")
        for domain_name, domain_id in self.domain_to_id.items():
            pair_count = self.domain_pair_counts[domain_id]
            percentage = (
                100.0 * pair_count / total_pairs if total_pairs else 0.0
            )
            print(
                f"  {domain_name}: id={domain_id}, pairs={pair_count}, "
                f"percentage={percentage:.2f}%"
            )
        print("END DOMAIN SUMMARY")


class ps_eval_dataset(Dataset):
    def __init__(
        self,
        ann_root,
        image_root,
        transform,
        split,
        max_words=30,
        text_length=77,
    ):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []
        self.img_cam_ids = []
        _initialize_text_length_stats(self, max_words, text_length)

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            self.image.append(image_path)

            person_id = ann['id']
            self.img2person.append(person_id)
            self.img_cam_ids.append(ann.get('cam_id', 1))
            for caption in ann['captions']:
                self.text.append(_preprocess_caption(self, caption))
                self.txt2person.append(person_id)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long)
        self.img_cam_ids = torch.tensor(self.img_cam_ids, dtype=torch.long)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

def pre_caption(caption, max_words=50, return_stats=False):
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
    word_count = len(caption_words) if caption else 0
    was_truncated = word_count > max_words
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if return_stats:
        return caption, word_count, was_truncated
    return caption
