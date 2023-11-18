import torch

from simple_clip.encoders import resnet18, resnet50, efficientnet_v2_s, TextEncoder
from simple_clip.custom_datasets.clip_datasets import COCODataset, SBUDataset
from transformers import DistilBertTokenizer
import datasets


def get_dataset(dataset_name,
                dataset_path,
                transforms,
                split="train",
                shuffle_captions=True):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    if dataset_name == "coco":
        # TODO: combine textcap and coco
        data = datasets.load_dataset("MMInstruction/M3IT", dataset_name)[split]
        return COCODataset(data,
                           tokenizer,
                           transforms=transforms,
                           image_key="image_base64_str",
                           text_key="outputs",
                           shuffle_captions=shuffle_captions)
    elif dataset_name == "textcap":
        data = datasets.load_dataset("MMInstruction/M3IT", dataset_name)[split]
        return COCODataset(data,
                           tokenizer,
                           transforms=transforms,
                           image_key="image_base64_str",
                           text_key="outputs",
                           shuffle_captions=shuffle_captions)
    elif dataset_name == "sbucaptions":
        data = datasets.load_from_disk("data/sbu_captions_images")["train"]
        return SBUDataset(data,
                          tokenizer,
                          transforms=transforms,
                          image_key="image",
                          text_key="caption")
    raise Exception(
        "Invalid dataset name - options are [flickr8k, coco, sbucaptions]")


def get_image_encoder(model_name):
    if model_name == "resnet18":
        return resnet18()
    elif model_name == "resnet50":
        return resnet50()
    elif model_name == "efficientnet":
        return efficientnet_v2_s()
    raise Exception(
        "Invalid model name - options are [resnet18, resnet50, efficientnet]")


def get_text_encoder(model_name="distilbert-base-uncased"):
    return TextEncoder(model_name)


def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.inference_mode
def get_feature_size(encoder):
    """Get the feature size from the encoder using a dummy input."""
    encoder.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = encoder(dummy_input)
    return output.shape[1]
