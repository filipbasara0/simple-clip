import torch
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict

from tqdm.auto import tqdm
from io import BytesIO
from base64 import b64decode

np.random.seed(42)

from PIL import Image

def get_image_tranforms(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.ToTensor()
    ])


def _grayscale_to_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


class COCODataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 tokenizer,
                 transforms,
                 image_key,
                 text_key,
                 shuffle_captions=True):

        captions = []
        for row in tqdm(data):
            captions.append(row[text_key])

        print("captions", len(captions), captions[0])
        encoded_captions = tokenizer(captions,
                                     padding=True,
                                     truncation=True,
                                     max_length=100)

        # group images, since we can have multiple captions for an image
        # we avoid having duplicate images in a single batch
        grouped_data = defaultdict(list)
        for idx, row in enumerate(tqdm(data)):
            grouped_data[row[image_key][0]].append({
                "input_ids":
                encoded_captions["input_ids"][idx],
                "attention_mask":
                encoded_captions["attention_mask"][idx]
            })

        self.data = list(grouped_data.items())
        print("data", len(self.data))
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        image_str, encoded_texts = self.data[idx]
        image = Image.open(BytesIO(b64decode(image_str)))
        image = self.transforms(image)

        if self.shuffle_captions:
            encoded_text = np.random.choice(encoded_texts)
        else:
            encoded_text = encoded_texts[0]
        instance = {
            key: torch.tensor(value)
            for key, value in encoded_text.items()
        }
        instance["image"] = image

        return instance

    def __len__(self):
        return len(self.data)


class SBUDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 tokenizer,
                 transforms,
                 image_key="image_path",
                 text_key="captions",
                 shuffle_captions=True):

        valid_indices = []
        captions = []
        for idx, row in enumerate(tqdm(data)):
            if row[image_key]:
                captions.append(row[text_key])
                valid_indices.append(idx)
        print("captions", len(captions), captions[0])
        self.encoded_captions = tokenizer(captions,
                                          padding=True,
                                          truncation=True,
                                          max_length=100)

        self.data = data.select(valid_indices)
        print("data", len(self.data),
              len(self.encoded_captions["input_ids"][0]))
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        image = self.transforms(image)

        instance = {
            key: torch.tensor(value[idx])
            for key, value in self.encoded_captions.items()
        }
        instance["image"] = image

        return instance

    def __len__(self):
        return len(self.data)
    

# coco + textcap + sbucaptions
class CombinedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 tokenizer,
                 transforms,
                 shuffle_captions=True):

        self.tokenizer = tokenizer

        self.data = data
        self.transforms = transforms
        self.shuffle_captions = shuffle_captions

    def __getitem__(self, idx):
        row = self.data[idx]
        image, captions = row["image"], row["caption"]

        if self.shuffle_captions:
            caption = np.random.choice(captions)
        else:
            caption = captions[0]
        encoded_caption = self.tokenizer(caption,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=100)
        instance = {
            key: torch.tensor(value)
            for key, value in encoded_caption.items()
        }
        instance["image"] = self.transforms(image)

        return instance

    def __len__(self):
        return len(self.data)
