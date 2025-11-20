import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
import json
from datasets import load_dataset
import evaluate
import numpy as np
import mlflow
import mlflow.pytorch
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
import psutil
import time
import GPUtil

# -----------------------------
# Cargar dataset
# -----------------------------
dataset = load_dataset("beans")


# Fusionar clases 0 y 1 en una sola etiqueta (1 = rust)
def merge_labels(example):
    if example["labels"] in [0, 1]:
        example["labels"] = 1  # rust
    else:
        example["labels"] = 0  # healthy
    return example


dataset["train"] = dataset["train"].map(merge_labels)
dataset["validation"] = dataset["validation"].map(merge_labels)
dataset["test"] = dataset["test"].map(merge_labels)

# Nuevo esquema de etiquetas
id2label = {0: "healthy", 1: "rust"}
label2id = {v: k for k, v in id2label.items()}

# Ver distribución (debería quedar desbalanceado)
print("Train label distribution:", Counter(dataset["train"]["labels"]))

# -----------------------------
# Preparar modelo y processor
# -----------------------------
"""
Descarga un modelo preentrenado de Hugging Face y lo exporta a formato ONNX.
"""
model_name = "google/mobilenet_v2_1.0_224"
print(f"Cargando modelo preentrenado: {model_name}")
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)


# Procesar imágenes
def preprocess(examples):
    return {
        "pixel_values": processor(
            [img.convert("RGB") for img in examples["image"]], return_tensors="pt"
        ).pixel_values
    }


dataset["train"] = dataset["train"].map(preprocess, batched=True)
dataset["validation"] = dataset["validation"].map(preprocess, batched=True)
dataset["test"] = dataset["test"].map(preprocess, batched=True)

# Ajustar el modelo para clasificación binaria
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# -----------------------------
# Métricas
# -----------------------------
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    return acc


def compute_metrics_extended(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    try:
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        roc_auc = auc(*roc_curve(labels, probs)[:2])
    except Exception:
        roc_auc = np.nan
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": roc_auc,
    }


# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    # save_strategy="no",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_strategy="epoch",
    report_to=["mlflow"],  # Reportar métricas a MLflow
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DefaultDataCollator(),
    # tokenizer=processor,
    compute_metrics=compute_metrics,
)
# Configuración MLflow
mlflow.set_experiment("beans_classification_experiment_extended_metrics")

with mlflow.start_run() as run:

    print(f"Tracking MLflow run_id={run.info.run_id}")

    # Log de recursos
    start_time = time.time()
    mlflow.log_param(
        "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )

    # Entrenamiento con callback manual
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(int(training_args.num_train_epochs)):
        print(f"Epoch {epoch+1}/{training_args.num_train_epochs}")
        train_metrics = trainer.train()
        eval_metrics = trainer.evaluate()

        # Log manual de métricas por época
        train_loss.append(train_metrics["train_loss"])
        val_loss.append(eval_metrics["eval_loss"])
        train_acc.append(train_metrics.get("train_accuracy", np.nan))
        val_acc.append(eval_metrics["eval_accuracy"])

        mlflow.log_metric("train_loss", train_loss[-1], step=epoch)
        mlflow.log_metric("val_loss", val_loss[-1], step=epoch)
        mlflow.log_metric("val_accuracy", val_acc[-1], step=epoch)

        # GPU tracking
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                mlflow.log_metric("gpu_util", gpu.load * 100, step=epoch)
                mlflow.log_metric("gpu_mem_used_MB", gpu.memoryUsed, step=epoch)

    # Log de métricas finales
    metrics = trainer.evaluate(dataset["test"])
    for k, v in metrics.items():
        mlflow.log_metric(f"final_{k}", v)

    # Curvas ROC (train + val)
    def plot_roc(split):
        ds = dataset[split]
        preds = trainer.predict(ds)
        labels = preds.label_ids
        probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1)[
            :, 1
        ].numpy()
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve ({split})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        path = f"./roc_{split}.png"
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path)

    plot_roc("train")
    plot_roc("validation")

    # Guardar curva de pérdida
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./loss_curve.png")
    mlflow.log_artifact("./loss_curve.png")

    # Guardar curva de accuracy
    plt.figure()
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("./accuracy_curve.png")
    mlflow.log_artifact("./accuracy_curve.png")

    # Exportar el modelo a ONNX
    # Poner el modelo en modo evaluación
    model.eval()
    model.cpu()

    # Determinar tamaño de entrada de manera robusta
    size = processor.size
    if isinstance(size, dict):
        height = size.get("height", size.get("shortest_edge", 224))
        width = size.get("width", height)
    elif isinstance(size, int):
        height = width = size
    else:
        height = width = 224  # valor por defecto seguro

    # print(f"Usando tamaño de entrada: {height}x{width}")

    # Crear una entrada dummy compatible con el modelo
    input_example = torch.randn(1, 3, height, width)

    # Exportar a ONNX
    output_path = "./mylib/model.onnx"
    print(f"Exportando modelo a {output_path}...")
    torch.onnx.export(
        model,
        input_example,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )

    # config = AutoConfig.from_pretrained(model_name, use_fast=False) # esto si quiereas labels del pretrained
    labels_file = "./mylib/labels.json"
    with open(labels_file, "w") as f:
        # json.dump(config.id2label, f)# son las pretrained
        json.dump(id2label, f)

    print(
        f"Modelo exportado correctamente a {output_path} y etiqeutas de clases a {labels_file}"
    )

    # Log ONNX model and PyTorch model in MLflow
    mlflow.pytorch.log_model(
        model, name="pytorch_model", input_example=input_example.numpy()
    )
    mlflow.log_artifact(output_path)
    mlflow.log_artifact(labels_file)

    # Log tiempo total
    elapsed = time.time() - start_time
    mlflow.log_metric("train_time_sec", elapsed)

    print(f"Modelo registrado en MLflow con run_id={run.info.run_id}")
