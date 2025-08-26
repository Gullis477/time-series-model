import torch
import torch.nn.functional as F


def contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, device: torch.device, temperature: float = 0.07
):
    """
    Computes NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.
    """
    batch_size, _ = z1.shape

    # Konkatenera de två vyerna till en enda tensor
    features = torch.cat([z1, z2], dim=0)

    # Beräkna cosinuslikheten mellan alla par i batchen
    # Använder F.normalize inbyggt i torch.matmul för effektivitet
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Skapa en mask för de positiva paren (diagonalen och de motsatta paren)
    # Exempel: (0, 32), (1, 33), osv. samt de omvända paren (32, 0), (33, 1)
    positive_mask = torch.zeros(
        2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device
    )
    positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = (
        True
    )
    positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = (
        True
    )

    # Identifiera positiva logiter (de som modellen ska maximera)
    positives = similarity_matrix[positive_mask].view(2 * batch_size, 1)

    # Skapa en mask för alla logiter förutom de positiva paren
    # Vi exkluderar även diagonalen (där en token jämförs med sig själv)
    logits_mask = torch.ones_like(positive_mask, dtype=torch.bool, device=device)
    logits_mask[positive_mask] = False
    logits_mask[torch.eye(2 * batch_size, dtype=torch.bool, device=device)] = False

    # Extrahera de negativa logiterna
    negatives = similarity_matrix[logits_mask].view(2 * batch_size, -1)

    # Slå ihop de positiva och negativa logiterna till en enda tensor
    logits = torch.cat([positives, negatives], dim=1)

    # Förlusten är nu en cross-entropy förlust, där det första elementet (index 0) är det korrekta svaret
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)
