"""Modulo para transfer learning"""

import os
import time
import json
import psutil
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import torch
# from torch import nn  # pylint: disable=no-name-in-module
import torch.nn as nn  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt


# ----------
# CONFIG
# ----------
BATCH_SIZE = 32  # 16, 24, 32, 38
EPOCHS = 5
LEARNING_RATE = 1e-3
IMG_SIZE = 224
VAL_SPLIT = 0.2
SEED = 42
EXPERIMENT_NAME = "mobilenetv2-pets-transfer-learning"


# ----------
# DATASET
# ----------
transform = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

dataset = OxfordIIITPet(root="data", download=True, transform=transform)
num_classes = len(dataset.classes)

# lista de nombres de clases
class_names = dataset.classes  # ['Abyssinian', 'Bengal', ...]

# Train/valid split
num_val = int(len(dataset) * VAL_SPLIT)
num_train = len(dataset) - num_val
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(dataset, [num_train, num_val], generator=g)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g,
    worker_init_fn=lambda worker_id: torch.manual_seed(SEED),
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ----------
# MODEL
# ----------
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Freeze feature extractor
for p in model.features.parameters():
    p.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ----------
# MLflow setup
# ----------
mlflow.set_experiment(EXPERIMENT_NAME)
RUN_NAME = f"mobilenetv2_lr{LEARNING_RATE}_bs{BATCH_SIZE}"

with mlflow.start_run(run_name=RUN_NAME) as run:

    process = psutil.Process()

    # -------- log hyperparameters --------
    mlflow.log_params(
        {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "img_size": IMG_SIZE,
            "model": "MobileNetV2",
            "dataset": "Oxford-IIIT Pet",
            "data_split_seed": SEED,
        }
    )

    train_losses = []
    val_losses = []

    # ----------
    # TRAINING LOOP
    # ----------
    for epoch in range(EPOCHS):

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()

        model.train()
        running_train_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        train_epoch_time = time.time() - t0

        # ---- validation ----
        t1 = time.time()

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)

        val_epoch_time = time.time() - t1

        # Guardar como artifact
        with open("labels.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f)

        # ----- log losses and accuracy -----
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)
        mlflow.log_metric("epoch_train_time_sec", train_epoch_time, step=epoch)
        mlflow.log_metric("epoch_val_time_sec", val_epoch_time, step=epoch)
        mlflow.log_artifact("labels.json")
        if DEVICE == "cuda":
            gpu_mem = torch.cuda.memory_allocated() / (1024**2)  # en MB
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
            mlflow.log_metric("gpu_memory_MB", gpu_mem, step=epoch)
            mlflow.log_metric("gpu_memory_peak_MB", gpu_mem_peak, step=epoch)

        # ---- CPU Memory ----
        cpu_mem_mb = process.memory_info().rss / (1024**2)
        mlflow.log_metric("cpu_memory_MB", cpu_mem_mb, step=epoch)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

    # ----------
    # Save loss curves as artifact
    # ----------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    os.makedirs("plots", exist_ok=True)
    PLOT_PATH = "plots/loss_curve.png"
    plt.savefig(PLOT_PATH)
    plt.close()

    mlflow.log_artifact(PLOT_PATH)

    # ----------
    # log final metrics
    # ----------
    mlflow.log_metric("final_val_accuracy", epoch_val_acc)
    mlflow.log_metric("final_val_loss", epoch_val_loss)

    # ----------
    # Register model in MLflow
    # ----------

    # ejemplo de entrada: un batch de 1 imagen 3x224x224
    input_example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32).cpu()
    with torch.no_grad():
        output_example = model(input_example.to(DEVICE)).cpu()
    signature = infer_signature(input_example.numpy(), output_example.numpy())

    mlflow.pytorch.log_model(
        model,
        name="model",
        input_example=input_example.numpy(),
        signature=signature,
        registered_model_name="mobilenetv2_oxford_pets",
    )

    print("Model logged and registered in MLflow")
