import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 mlp_out_dim=768,
                 mlp_in_dim=False):
        super(CLIP, self).__init__()

        if not mlp_in_dim:
            mlp_in_dim = get_feature_size(image_encoder)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_dim, mlp_in_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_in_dim, mlp_out_dim, bias=False))
        
        self.t_prime = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, image, input_ids, attention_mask):
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(input_ids, attention_mask)
        return image_features @ text_features.t() * self.t_prime.exp() + self.b

    def extract_image_features(self, images):
        image_features = self.image_encoder(images)
        return self.projection_head(image_features)

    def extract_text_features(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)
