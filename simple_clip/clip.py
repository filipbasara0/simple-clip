import torch

import torch.nn.functional as F

from simple_clip.utils import get_feature_size


def clip_loss(logits):
    targets = torch.arange(logits.size(0)).to(logits.device)
    loss_images = F.cross_entropy(logits, targets)
    loss_texts = F.cross_entropy(logits.t(), targets)

    return loss_images, loss_texts


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

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)
        image_features = self.projection_head(image_features)
        text_features = self.text_encoder(input_ids, attention_mask)

        # logits per image
        logits = image_features @ text_features.t()
        return logits

    def extract_image_features(self, images):
        image_features = self.image_encoder(images)
        return self.projection_head(image_features)

    def extract_text_features(self, input_ids, attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        return text_features
