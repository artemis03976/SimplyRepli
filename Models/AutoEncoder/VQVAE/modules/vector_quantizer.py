import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(VectorQuantizer, self).__init__()

        self.embed_dim = embed_dim

        # discrete embedding codebook
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def quantize(self, z_e):
        # permute to fit the codebook shape
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flattened = z_e.view(-1, self.embed_dim)

        # calculate L2 distance for each point
        distance = torch.cdist(z_e_flattened, self.embedding.weight)
        # get the nearest embedding
        min_encoding_idxs = torch.argmin(distance, dim=1).view(z_e.shape[:-1])

        return min_encoding_idxs

    def forward(self, z_e):
        # quantize the encoded image
        min_encoding_idxs = self.quantize(z_e)
        z_q = self.embedding(min_encoding_idxs)

        # permute back to the original shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # compute loss
        embedding_loss = F.mse_loss(z_e.detach(), z_q)
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        return z_q, embedding_loss, commitment_loss
