from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pathlib import Path
import os


class StringModel:
    def __init__(self,
                 pretrained_model='string_store',
                 device=None,
                 ):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModel.from_pretrained(pretrained_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, device=self.device)
        self.model.eval()

    def __call__(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)













if __name__ == '__main__':
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']
    sModel = StringModel()
    print(sModel(sentences))
