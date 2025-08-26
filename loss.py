import torch
import torch.nn.functional as F


def contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, device: torch.device, temperature: float = 0.07
):
    """
    Computes NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.
    """
    batch_size, _ = z1.shape

    # Normalisera inbäddningar
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Konkatenera de två vyerna
    features = torch.cat([z1, z2], dim=0)

    # Beräkna cosinus-likhetsmatrisen
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Skapa en mask för de positiva paren
    # Vi behöver en mask som matchar 1:a sample med 33:e, 2:a med 34:e, osv.
    pos_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    pos_mask = pos_mask.repeat(2, 2)

    # Skapa en mask som sätter de positiva paren till 1 och resten till 0
    # Positiva par är (0, 32), (1, 33), ... och (32, 0), (33, 1), ...
    positive_pairs_mask = torch.roll(pos_mask, shifts=batch_size, dims=1)

    # Skapa en mask för diagonalen (som vi vill ignorera)
    diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

    # Ta bort diagonalen från likhetsmatrisen och positiva par-masken
    similarity_matrix = similarity_matrix[~diag_mask].view(2 * batch_size, -1)
    positive_pairs_mask = positive_pairs_mask[~diag_mask].view(2 * batch_size, -1)

    # Extrahera positiva likheter med hjälp av masken
    positives = similarity_matrix[positive_pairs_mask].view(2 * batch_size, 1)

    # Beräkna log-förlusten
    loss = -torch.log(
        torch.exp(positives)
        / torch.sum(torch.exp(similarity_matrix), dim=1, keepdim=True)
    )

    return torch.mean(loss)
