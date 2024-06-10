import torch
import torch.nn as nn

class MultiHeadDotProduct(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadDotProduct, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.linear = nn.Linear(embed_dim, embed_dim)
        
        # Sequential layers for additional processing
        self.head_layers = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1)
        )
    
    def forward(self, image_emb, text_emb):
        # Linear transformation
        image_emb = self.linear(image_emb)
        text_emb = self.linear(text_emb)
        
        # Split embeddings into multiple heads
        image_emb = image_emb.view(-1, self.num_heads, self.head_dim)
        text_emb = text_emb.view(-1, self.num_heads, self.head_dim)
        
        # Dot product for each head
        dot_products = torch.einsum("bhd,bhd->bh", image_emb, text_emb)
        
        # Pass through additional layers for each head
        dot_products = dot_products.view(-1, self.head_dim)
        dot_products = self.head_layers(dot_products)
        
        # Combine the results from all heads
        combined_result = dot_products.view(-1, self.num_heads).mean(dim=1)
        return combined_result