import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from simple_clip.utils import get_feature_size


def contrastive_loss(logits):
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)
    return (loss_images + loss_texts) / 2


def siglip_loss(logits):
    n = logits.size(0)
    # -1 for off-diagonals and 1 for diagonals
    labels = 2 * torch.eye(n, device=logits.device) - 1
    # pairwise sigmoid loss
    return -torch.sum(F.logsigmoid(labels * logits)) / n


class CLIP(torch.nn.Module):

    def __init__(self,
                 image_encoder,
                 text_encoder,
                 image_mlp_dim=False,
                 text_mlp_dim=768,
                 proj_dim=256,
                 init_tau=np.log(1.0),
                 init_b=0):
        super(CLIP, self).__init__()

        if not image_mlp_dim:
            image_mlp_dim = get_feature_size(image_encoder)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(image_mlp_dim, image_mlp_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(image_mlp_dim, proj_dim, bias=False))
        
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(text_mlp_dim, text_mlp_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(text_mlp_dim, proj_dim, bias=False))
        
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return image_features @ text_features.t() * self.t_prime.exp() + self.b

    def extract_image_features(self, images):
        image_features = self.image_encoder(images)
        return self.image_projection(image_features)

    def extract_text_features(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        return self.text_projection(text_features)
