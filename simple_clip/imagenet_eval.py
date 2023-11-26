import datasets

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


class ImageNetValidation:

    def __init__(self, transform):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = datasets.load_dataset("imagenet-1k")
        dataset = ImageNetDataset(data["validation"], transform)
        self.dataloader = DataLoader(dataset,
                                batch_size=256,
                                num_workers=4)
        
        labels = data["validation"].features["label"].int2str(list(range(1000)))
        self.label_queries = [f"a photo of a {l}" for l in labels]

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.encoded_texts = tokenizer(self.label_queries, padding=True, truncation=True, max_length=100)

    @torch.inference_mode
    def evaluate(self, model):
        model.eval()
        with torch.no_grad():
            image_features, labels = self._get_image_embs_labels(model)
            text_features = self._get_text_embs(model)

            preds = image_features @ text_features.t()

            labels = labels.cpu().detach().tolist()
            
            preds = preds.argmax(dim=-1).cpu().detach().tolist()

            acc = accuracy_score(labels, preds)
            print("Accuracy ImagNet Val: ", acc)

    def _get_image_embs_labels(self, model):
        embs, labels = [], []
        for images, targets in tqdm(self.dataloader):
            with torch.no_grad():
                images = images.to(self.device)
                out = model.extract_image_features(images)
                features = out.cpu().detach().tolist()
                embs.extend(features)
                labels.extend(targets.cpu().detach().tolist())
        return torch.tensor(embs).to(self.device), torch.tensor(labels).to(self.device)
    
    def _get_text_embs(self, model):
        input_ids = torch.tensor(self.encoded_texts["input_ids"]).to(self.device)
        attention_mask = torch.tensor(self.encoded_texts["attention_mask"]).to(self.device)
        return model.extract_text_features(input_ids, attention_mask)
        

class ImageNetDataset(Dataset):

    def __init__(self,
                 data,
                 transforms):

        self.data = data
        self.transforms = transforms

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]
        image = self.transforms(image)
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.data)
