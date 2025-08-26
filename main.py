from create_data import create_data
from loss import contrastive_loss
from model import ContrastiveTransformer
from contrastive_data import ContrastiveSignalDataset
from emitter_data import generate_balanced_data
import numpy as np
import torch
import torch.optim as optim  # Importerar optimeraren

from torch.utils.data import DataLoader

if __name__ == "__main__":
    # 1. Definiera enheten
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Använder enhet: {device}")

    # 2. Skapa modellen och flytta den till enheten
    num_features = 4
    input_length = 300
    model = ContrastiveTransformer(num_features=num_features, input_length=input_length)
    model.to(device)

    # 3. Definiera optimeraren
    # Adam är ett bra standardval för de flesta deep learning-modeller
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. Skapa datan
    seed = 42
    number_of_emitters = (
        9 * 5
    )  # To get balanced data, this number should be divisble with 9
    emitters = generate_balanced_data(number_of_emitters)
    dataset_train = ContrastiveSignalDataset(emitters)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=1)

    num_epochs = 100  # Antal träningsomgångar
    print(f"Börjar träning för {num_epochs} epoker...")

    # 5. Träningsloopen
    for epoch in range(num_epochs):
        total_loss = 0

        # Sätter modellen i träningstillstånd (viktigt för t.ex. Dropout)
        model.train()

        for x_clean, mask_clean, x_noise, mask_noise, idx in train_loader:
            # Flytta data till enheten
            x_clean = x_clean.to(device)
            x_noise = x_noise.to(device)
            if mask_clean is not None:
                mask_clean = mask_clean.to(device)
            if mask_noise is not None:
                mask_noise = mask_noise.to(device)

            # Nollställ gradienterna från förra batchen
            optimizer.zero_grad()

            # Forward pass: Beräkna representationerna
            z_clean = model(x_clean, mask_clean)
            z_noise = model(x_noise, mask_noise)

            # Beräkna förlusten
            loss = contrastive_loss(z_clean, z_noise, device=device)

            # Backward pass: Beräkna gradienterna
            loss.backward()

            # Uppdatera modellens vikter
            optimizer.step()

            total_loss += loss.item()

        # Skriv ut genomsnittlig förlust per epok
        avg_loss = total_loss / len(train_loader)
        print(f"Epok [{epoch+1}/{num_epochs}], Genomsnittlig förlust: {avg_loss:.4f}")

    print("Träning avslutad!")
