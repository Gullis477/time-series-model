import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any


def extract_representations_and_labels(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, List[Any]]:
    """
    Extraherar normaliserade representationer och motsvarande sanna etiketter
    från modellen med hjälp av en DataLoader.

    Args:
        model (torch.nn.Module): Den tränade modellen (t.ex. ContrastiveTransformer).
        dataloader (DataLoader): DataLoader som ger data och etiketter.
        device (torch.device): Enheten (CPU/GPU) att utföra beräkningarna på.

    Returns:
        Tuple[torch.Tensor, List[Any]]:
            - all_representations (torch.Tensor): Samtliga normaliserade representationer,
                                                form (N, embedding_dim).
            - all_true_labels (List[Any]): Motsvarande sanna etiketter för varje representation.
    """
    model.eval()
    representations_list = []
    true_labels_list = []

    with torch.no_grad():
        for x_clean, mask_clean, x_noise, mask_noise, emitter_id in dataloader:
            x_noise = x_noise.to(device)
            if mask_noise is not None:
                mask_noise = mask_noise.to(device)

            z = model(x_noise, padding_mask=mask_noise)
            z = F.normalize(z, dim=1)

            representations_list.append(z.cpu())
            true_labels_list.extend(emitter_id.cpu().numpy())

    all_representations = torch.cat(representations_list, dim=0)
    return all_representations, true_labels_list


def cluster_by_similarity(
    representations: torch.Tensor, true_labels: List[Any], threshold: float
) -> Dict[str, Any]:
    """
    Klustrar representationer baserat på likhet med en tröskel.

    Args:
        representations (torch.Tensor): De representationer som ska klustras.
        true_labels (List[Any]): De sanna etiketterna för varje representation.
        threshold (float): Tröskelvärde för klusterlikhet.

    Returns:
        Dict[str, Any]: En ordbok med klustermått och predikerade etiketter.
    """
    # Se till att representationerna är på CPU och i numpy-format
    Z = representations.cpu().numpy()

    # Förbered sanna etiketter
    unique_ids = sorted(list(set(true_labels)))
    label_to_int = {uid: i for i, uid in enumerate(unique_ids)}
    y_true = [label_to_int[lbl] for lbl in true_labels]

    y_pred = []
    clusters = []
    for z in Z:
        best_score = -1
        best_cluster_id = -1
        for j, cluster_center in enumerate(clusters):
            # Beräkna cosinuslikhet med befintliga kluster
            sim_score = np.dot(z, cluster_center)
            if sim_score > best_score:
                best_cluster_id = j
                best_score = sim_score

        if best_score == -1 or best_score < threshold:
            # Skapa ett nytt kluster
            y_pred.append(len(clusters))
            clusters.append(z)
        else:
            # Tilldela till bästa befintliga kluster
            y_pred.append(best_cluster_id)

    # Beräkna klustermått
    hom = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    return {
        "homogeneity": hom,
        "completeness": comp,
        "adjusted_rand_score": ari,
        "adjusted_mutual_info_score": ami,
        "num_clusters": len(set(y_pred)),
        "y_pred": y_pred,  # Lämpar sig för felsökning/visualisering
    }
