from copy import deepcopy
from create_data import create_data
from loss import contrastive_loss
from model import ContrastiveTransformer
from contrastive_data import ContrastiveSignalDataset
from emitter_data import generate_balanced_data, build_emitter
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from evaluation import extract_representations_and_labels, cluster_by_similarity

from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Använder enhet: {device}")

    num_features = 4
    input_length = 300
    model = ContrastiveTransformer(num_features=num_features, input_length=input_length)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    number_of_emitters = 9 * 5
    emitters = generate_balanced_data(number_of_emitters)
    dataset_train = ContrastiveSignalDataset(emitters)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=1)

    val_emitters = []
    for _ in range(5):
        emitter = build_emitter()
        for _ in range(5):
            deep_copy_emitter = deepcopy(emitter)
            val_emitters.append(deep_copy_emitter)

    val_dataset = ContrastiveSignalDataset(val_emitters)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )  # Använd shuffle=False för konsekvent evaluering

    num_epochs = 100
    print(f"Börjar träning för {num_epochs} epoker...")

    for epoch in range(num_epochs):
        total_loss = 0
        total_pos_distance = 0
        total_neg_distance = 0
        num_batches = len(train_loader)

        model.train()

        for x_clean, mask_clean, x_noise, mask_noise, idx in train_loader:
            x_clean = x_clean.to(device)
            x_noise = x_noise.to(device)

            if mask_clean is not None:
                mask_clean = mask_clean.to(device)
            if mask_noise is not None:
                mask_noise = mask_noise.to(device)

            optimizer.zero_grad()

            z_clean = model(x_clean, mask_clean)
            z_noise = model(x_noise, mask_noise)

            # Beräkna och logga distanserna (för insikt)
            batch_size = z_clean.size(0)
            pos_distances = torch.norm(z_clean - z_noise, p=2, dim=1)
            total_pos_distance += torch.mean(pos_distances).item()

            all_features = torch.cat([z_clean, z_noise], dim=0)
            dist_matrix = torch.cdist(all_features, all_features, p=2)

            pos_mask = torch.zeros_like(dist_matrix, dtype=torch.bool, device=device)
            pos_mask[
                torch.arange(batch_size), torch.arange(batch_size) + batch_size
            ] = True
            pos_mask[
                torch.arange(batch_size) + batch_size, torch.arange(batch_size)
            ] = True
            diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
            neg_mask = ~(pos_mask | diag_mask)
            neg_distances = dist_matrix[neg_mask]
            total_neg_distance += torch.mean(neg_distances).item()

            loss = contrastive_loss(z_clean, z_noise, device=device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        avg_pos_dist = total_pos_distance / num_batches
        avg_neg_dist = total_neg_distance / num_batches

        print(
            f"Epok [{epoch+1}/{num_epochs}], "
            f"Genomsnittlig förlust: {avg_loss:.4f}, "
            f"Positiv distans: {avg_pos_dist:.4f}, "
            f"Negativ distans: {avg_neg_dist:.4f}"
        )

        # --- Evalueringsfas var 10:e epok ---
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Utvärderar modell efter epok {epoch+1} ---")

            # Extrahera representationer och sanna etiketter
            representations, true_labels = extract_representations_and_labels(
                model, val_loader, device
            )

            # Använd din likhetsbaserade klustringsfunktion
            # Ett fast tröskelvärde är satt för demonstration
            threshold = (1 - avg_pos_dist) * 0.9
            metrics = cluster_by_similarity(representations, true_labels, threshold)

            print(
                f"  Homogenitet: {metrics['homogeneity']:.4f}, "
                f"Fullständighet: {metrics['completeness']:.4f}, "
                f"ARI: {metrics['adjusted_rand_score']:.4f}, "
                f"AMI: {metrics['adjusted_mutual_info_score']:.4f}, "
                f"Antal kluster: {metrics['num_clusters']}"
            )
            print("-------------------------------------------\n")

    print("Träning avslutad!")
