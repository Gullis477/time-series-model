from create_data import create_data
from loss import contrastive_loss
from model import ContrastiveTransformer
from contrastive_data import ContrastiveSignalDataset
from emitter_data import generate_balanced_data
import numpy as np
import torch

from torch.utils.data import DataLoader

if __name__ == "__main__":
    # 1. Definiera vilken enhet som ska användas (GPU om tillgänglig, annars CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Använder enhet: {device}")

    # 2. Skapa din modell och flytta den till den valda enheten
    num_features = 4
    input_length = 300
    model = ContrastiveTransformer(num_features=num_features, input_length=input_length)
    model.to(device)

    seed = 42
    number_of_emitters = 90
    number_of_signals = 5

    emitters = generate_balanced_data(9 * 5)
    dataset_train = ContrastiveSignalDataset(emitters)

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=1)
    print(len(train_loader))

    for x_clean, mask_clean, x_noise, mask_noise, idx in train_loader:
        # 3. Flytta dina data-tensors till samma enhet som modellen
        x_clean = x_clean.to(device)
        x_noise = x_noise.to(device)

        # Masker behöver oftast också flyttas, beroende på din implementation
        if mask_clean is not None:
            mask_clean = mask_clean.to(device)
        if mask_noise is not None:
            mask_noise = mask_noise.to(device)

        z_clean = model(x_clean, mask_clean)
        z_noise = model(x_noise, mask_noise)

        # 4. Nu ska loss-funktionen inte klaga
        loss = contrastive_loss(z_clean, z_noise, device=device)
        print(loss)
