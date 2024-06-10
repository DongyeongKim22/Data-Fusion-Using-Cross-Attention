import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel



class TextEmbedding(nn.Module):
    def __init__(self, pretrained_model):
        super(TextEmbedding, self).__init__()
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_heads, output_dim=512):
        super(TextEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, embeddings, src_key_padding_mask):
        device = embeddings.device
        src_key_padding_mask = src_key_padding_mask.to(device)
        embeddings = embeddings.transpose(0, 1)
        embeddings = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        # embeddings = embeddings[:, 0, :]
        # embeddings = self.fc(embeddings)
        return embeddings

class MMFTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_dim=2048, num_layers=6, num_heads=8):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)
        self.embedding_layer = TextEmbedding(self.bert_model)
        self.encoder_layer = None
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def initialize_encoder(self, embeddings):
        embedding_dim = embeddings.shape[-1]
        self.encoder_layer = TextEncoder(embedding_dim, self.hidden_dim, self.num_layers, self.num_heads).to(embeddings.device)

    def forward(self, texts):
        #text list
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.bert_model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        #text embedding
        embeddings = self.embedding_layer(input_ids, attention_mask)

        if self.encoder_layer is None:
            self.initialize_encoder(embeddings)

        src_key_padding_mask = attention_mask == 0
        # src_key_padding_mask = src_key_padding_mask.transpose(0, 1).to(embeddings.device) 
        
        encoded_output = self.encoder_layer(embeddings, src_key_padding_mask=src_key_padding_mask)
        return encoded_output