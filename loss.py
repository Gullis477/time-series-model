import torch
import torch.nn.functional as F

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, device: torch.device, temperature: float = 0.07):
    """
    Computes NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    This loss function is a core component of contrastive learning methods like SimCLR.
    It works by bringing positive pairs (different augmentations of the same image)
    closer together while pushing negative pairs (different images) apart in the
    embedding space.

    Args:
        z1 (torch.Tensor): Tensor of normalized embeddings from one view, shape (batch_size, embedding_dim).
        z2 (torch.Tensor): Tensor of normalized embeddings from a different view, shape (batch_size, embedding_dim).
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to perform computations on.
        temperature (float): The temperature scaling parameter, which controls the
                             sharpness of the probability distribution over negative samples.

    Returns:
        torch.Tensor: The mean NT-Xent loss for the batch.
    """
    batch_size, _ = z1.shape

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    features = torch.cat([z1, z2], dim=0)

    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)

    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    similarity_matrix = torch.exp(similarity_matrix / temperature)

    pos_mask = torch.cat([
        F.one_hot(torch.arange(batch_size).to(device), num_classes=2 * batch_size),
        F.one_hot(torch.arange(batch_size).to(device), num_classes=2 * batch_size)
    ], dim=0)
    pos_mask = pos_mask[~mask].view(2 * batch_size, -1).bool()

    positives = similarity_matrix[pos_mask].view(2 * batch_size, -1)

    loss = -torch.log(positives / torch.sum(similarity_matrix, dim=1, keepdim=True))
    
    return torch.mean(loss)