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
# freeze base model parameters
for param in model.base_model.parameters():
    param.requires_grad = False

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")
    print("Modelo movido a GPU:", next(model.parameters()).device)

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
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_strategy="epoch",
    report_to=["mlflow"],  # Reportar métricas a MLflow
    no_cuda=False,
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

print("Trainer usará dispositivo:", trainer.args.device)

# Configuración MLflow
mlflow.set_experiment("beans_classification_experiment_extended")

with mlflow.start_run() as run:

    # """
    # Descarga un modelo preentrenado de Hugging Face y lo exporta a formato ONNX.
    # """
    # model_name = "google/mobilenet_v2_1.0_224"
    # print(f"Cargando modelo preentrenado: {model_name}")
    # processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)

    # dataset["train"] = dataset["train"].map(
    #     lambda examples: {"pixel_values": processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt").pixel_values},
    #     batched=True
    # )
    # dataset["validation"] = dataset["validation"].map(
    #     lambda examples: {"pixel_values": processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt").pixel_values},
    #     batched=True
    # )
    # dataset["test"] = dataset["test"].map(
    #     lambda examples: {"pixel_values": processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt").pixel_values},
    #     batched=True
    # )

    # Cargar el modelo preentrenado
    # model = AutoModelForImageClassification.from_pretrained(model_name)
    # # Ajustar el modelo para clasificación binaria
    # model = AutoModelForImageClassification.from_pretrained(
    #     model_name,
    #     num_labels=2,
    #     id2label=id2label,
    #     label2id=label2id,
    #     ignore_mismatched_sizes=True,
    # )

    # accuracy = evaluate.load("accuracy")

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     acc = accuracy.compute(predictions=predictions, references=labels)
    #     # Log metric in MLflow
    #     mlflow.log_metric("accuracy", acc["accuracy"])
    #     return acc

    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     eval_strategy="epoch",
    #     save_strategy="no",
    #     learning_rate=5e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["validation"],
    #     data_collator=DefaultDataCollator(),
    #     # tokenizer=processor,
    #     compute_metrics=compute_metrics,
    # )

    # Entrenar el modelo
    trainer.train()

#     # Loguear métricas
#     eval_metrics = trainer.evaluate(dataset["validation"])
#     for key, value in eval_metrics.items():
#         mlflow.log_metric(key, value)

#     # Exportar el modelo a ONNX
#     # Poner el modelo en modo evaluación
#     model.eval()
#     model.cpu()

#     # Determinar tamaño de entrada de manera robusta
#     size = processor.size
#     if isinstance(size, dict):
#         height = size.get("height", size.get("shortest_edge", 224))
#         width = size.get("width", height)
#     elif isinstance(size, int):
#         height = width = size
#     else:
#         height = width = 224  # valor por defecto seguro

#     # print(f"Usando tamaño de entrada: {height}x{width}")

#     # Crear una entrada dummy compatible con el modelo
#     input_example = torch.randn(1, 3, height, width, dtype=torch.float32)

#     # Exportar a ONNX
#     output_path = "./mylib/model.onnx"
#     print(f"Exportando modelo a {output_path}...")
#     torch.onnx.export(
#         model,
#         input_example,
#         output_path,
#         input_names=["input"],
#         output_names=["output"],
#         dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#         opset_version=12,
#     )

#     # config = AutoConfig.from_pretrained(model_name, use_fast=False) # esto si quiereas labels del pretrained
#     labels_file = "./mylib/labels.json"
#     with open(labels_file, "w") as f:
#         # json.dump(config.id2label, f)# son las pretrained
#         json.dump(id2label, f)

#     print(f"Modelo exportado correctamente a {output_path} y etiqeutas de clases a {labels_file}")

#     # Log ONNX model and PyTorch model in MLflow
#     mlflow.pytorch.log_model(model, name="pytorch_model", input_example=input_example.numpy())
#     mlflow.log_artifact(output_path)
#     mlflow.log_artifact(labels_file)

#     print(f"Modelo registrado en MLflow con run_id={run.info.run_id}")
