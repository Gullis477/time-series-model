import numpy as np
from infonce import InfoNCELoss
from contrastive_data import ContrastiveSignalDataset
from contrastive_transformer import ContrastiveTransformer
from emitter_data import build_emitter  # type: ignore
from torch.utils.data import DataLoader
import torch
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
import torch.nn.functional as F
import datetime


def extract_representations_and_labels(model, dataloader, device):
    """
    Returnerar två listor:
      - representations: tensorer med normaliserade representationer
      - true_labels: motsvarande emitter_id för varje representation
    """
    model.eval()
    representations = []
    true_labels = []
    with torch.no_grad():
        for x_clean, mask_clean, x_noise, mask_noise, emitter_id in dataloader:
            x_noise, mask_noise = x_noise.to(device), mask_noise.to(device)
            z_j = model(x_noise, padding_mask=mask_noise)
            z_j = F.normalize(z_j, dim=1)
            representations.append(z_j)
            true_labels.append(emitter_id[0])
    return representations, true_labels


def cluster_by_similarity(Z, true_labels, threshold):
    """
    Klustrar representationer Z med likhetsbaserad tröskel.
    Returnerar y_pred (kluster-id för varje punkt).
    """
    true_labels_str = [str(lbl) for lbl in true_labels]
    unique_ids = sorted(set(true_labels_str))
    label_to_int = {uid: i for i, uid in enumerate(unique_ids)}
    y_true = [label_to_int[lbl] for lbl in true_labels_str]

    y_pred = []
    clusters = []
    for z in Z:
        best_score = -1
        best_cluster_id = -1
        for j, cluster in enumerate(clusters):
            sim_score = np.dot(z, cluster)
            if sim_score > best_score:
                best_cluster_id = j
                best_score = sim_score

        if best_score == -1 or best_score < threshold:
            y_pred.append(len(clusters))
            clusters.append(z)
        else:
            y_pred.append(best_cluster_id)
    return y_pred, y_true


# Modell- och träningsparametrar
model_params = {
    "num_features": 4,
    "input_length": 300,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 6,
    "d_rep": 64,
    "dim_feedforward": 512,
    "dropout": 0.1,
}
training_params = {
    "batch_size": 8,
    "num_epochs": 1,
    "learning_rate": 1e-4,
    "loss_temperature": 0.1,
}

# Valideringsdata-parametrar
num_emitters = 5
num_signals = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
norm_mode = "physical"  # "physical", "zscore", "minmax", "none"
train_emitters = [build_emitter() for _ in range(1024)]
train_dataset = ContrastiveSignalDataset(
    train_emitters, max_length=300, norm_mode=norm_mode
)
train_loader = DataLoader(
    train_dataset, batch_size=training_params["batch_size"], shuffle=True
)
val_emitters = []
for _ in range(num_emitters):
    emitter = build_emitter()
    for _ in range(num_signals):
        deep_copy_emitter = deepcopy(emitter)
        val_emitters.append(deep_copy_emitter)

val_dataset = ContrastiveSignalDataset(
    val_emitters, max_length=300, norm_mode=norm_mode
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

model = ContrastiveTransformer(**model_params).to(device)
criterion = InfoNCELoss(temperature=training_params["loss_temperature"])
optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])

start_time = datetime.datetime.now()
log_filename = f"training_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
with open(log_filename, "w", encoding="utf-8") as log_file:
    log_file.write(f"Start time: {start_time}\n")
    log_file.write(f"Model: {model.__class__.__name__}\n")
    log_file.write("Model parameters:\n")
    for k, v in model_params.items():
        log_file.write(f"  {k}: {v}\n")
    log_file.write("Training parameters:\n")
    for k, v in training_params.items():
        log_file.write(f"  {k}: {v}\n")
    log_file.write(f"Train dataset size: {len(train_dataset)}\n")
    log_file.write(f"Eval dataset size: {len(val_dataset)}\n")
    log_file.write(f"Validation num_emitters: {num_emitters}\n")
    log_file.write(f"Validation num_signals_per_emitter: {num_signals}\n")
    log_file.write("epoch,loss,avg_sim,homogeneity,completeness,ARI,AMI,num_clusters\n")

    for epoch in range(training_params["num_epochs"]):
        model.train()
        total_loss = 0
        for x_clean, mask_clean, x_noise, mask_noise, emitter_id in train_loader:
            x_clean = x_clean.to(device)
            x_noise = x_noise.to(device)

            z_i = model(x_clean, padding_mask=mask_clean.to(device))
            z_j = model(x_noise, padding_mask=mask_noise.to(device))

            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        z_i_norm = F.normalize(z_i, dim=1)
        z_j_norm = F.normalize(z_j, dim=1)
        sim = torch.sum(z_i_norm * z_j_norm, dim=1)
        avg_sim = sim.mean().item()

        representations, true_labels = extract_representations_and_labels(
            model, val_loader, device
        )
        Z = torch.cat(representations, dim=0).cpu().numpy()
        threshold = 0.9 * avg_sim
        y_pred, y_true = cluster_by_similarity(Z, true_labels, threshold)
        hom = homogeneity_score(y_true, y_pred)
        comp = completeness_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)

        log_file.write(
            f"{epoch+1},{avg_loss:.6f},{avg_sim:.6f},{hom:.6f},{comp:.6f},{ari:.6f},{ami:.6f},{len(set(y_pred))}\n"
        )
        log_file.flush()

    print(f"Training complete. Log saved to {log_filename}.")
