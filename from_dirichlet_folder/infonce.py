import torch
import torch.nn.functional as F


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: Tensorer med form (batch_size, embedding_dim)
                  z_i = clean embeddings, z_j = noisy embeddings
        """
        batch_size = z_i.size(0)

        # Normalisera embeddings (cosine similarity)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Kombinera: (2*B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Likhetsmatris: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature

        # Maskera bort identitet (likhet med sig själv)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Skapa "positiv"-index för varje rad:
        positives = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
        ).to(z.device)

        # Loss = CrossEntropy där varje rad ska matcha sin positiva partner
        labels = positives
        loss = F.cross_entropy(sim, labels)

        return loss
