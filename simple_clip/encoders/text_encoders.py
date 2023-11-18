from torch import nn
from transformers import DistilBertModel


class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.model = DistilBertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state.mean(dim=1)
