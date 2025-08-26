from create_data import create_data
from loss import contrastive_loss
from model import ContrastiveTransformer
from contrastive_data import ContrastiveSignalDataset
from emitter_data import generate_balanced_data
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

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

            # Maskerna skickas bara om de inte är None
            if mask_clean is not None:
                mask_clean = mask_clean.to(device)
            if mask_noise is not None:
                mask_noise = mask_noise.to(device)

            optimizer.zero_grad()

            z_clean = model(x_clean, mask_clean)
            z_noise = model(x_noise, mask_noise)

            # --- Korrigerad kod för att beräkna distanserna ---
            batch_size = z_clean.size(0)

            # Beräkna positiva distanser (mellane z_clean och z_noise)
            pos_distances = torch.norm(z_clean - z_noise, p=2, dim=1)
            total_pos_distance += torch.mean(pos_distances).item()

            # Beräkna negativa distanser
            # Slå ihop alla representationer till en enda tensor
            all_features = torch.cat([z_clean, z_noise], dim=0)

            # Beräkna en distansmatris (L2-norm) mellan alla par
            dist_matrix = torch.cdist(all_features, all_features, p=2)

            # Skapa en mask för de positiva paren
            pos_mask = torch.zeros_like(dist_matrix, dtype=torch.bool, device=device)
            pos_mask[
                torch.arange(batch_size), torch.arange(batch_size) + batch_size
            ] = True
            pos_mask[
                torch.arange(batch_size) + batch_size, torch.arange(batch_size)
            ] = True

            # Skapa en mask för diagonalen
            diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

            # Skapa en mask som inkluderar alla par UTOM positiva par och diagonalen
            neg_mask = ~(pos_mask | diag_mask)

            # Välj ut de negativa distanserna med masken
            neg_distances = dist_matrix[neg_mask]

            total_neg_distance += torch.mean(neg_distances).item()

            # --- Fortsättning av din befintliga träningskod ---
            loss = contrastive_loss(z_clean, z_noise, device=device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Skriv ut genomsnittlig förlust och distanser per epok
        avg_loss = total_loss / num_batches
        avg_pos_dist = total_pos_distance / num_batches
        avg_neg_dist = total_neg_distance / num_batches

        print(
            f"Epok [{epoch+1}/{num_epochs}], "
            f"Genomsnittlig förlust: {avg_loss:.4f}, "
            f"Positiv distans: {avg_pos_dist:.4f}, "
            f"Negativ distans: {avg_neg_dist:.4f}"
        )

    print("Träning avslutad!")
