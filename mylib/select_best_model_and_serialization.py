"""Modulo para seleccionar y serializar el mejor modelo"""
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import torch # pylint: disable=import-error,no-member

# ----------------------
# CONFIG
# ----------------------
MODEL_NAME = "mobilenetv2_oxford_pets"
OUTPUT_ONNX_PATH = "./mylib/model.onnx"
OUTPUT_JSON_PATH = "./mylib/labels.json"
IMG_SIZE = 224  # tamaño de imagen que usa MobileNetV2
NUM_CLASSES = 37  # Oxford-IIIT Pet

# ----------------------
# 1. Conectar con MLflow y obtener versiones activas
# ----------------------
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

# # Filtrar solo versiones activas (Production, Staging)
# active_versions = [v for v in versions if v.current_stage in ("Production", "Staging")]

# if not active_versions:
if not versions:
    raise ValueError(f"No hay versiones registradas para el modelo {MODEL_NAME}")

# ----------------------
# 2. Elegir la mejor versión por final_val_accuracy
# ----------------------
best_version = None
best_acc = -1.0

for v in versions:
    run_id = v.run_id
    metrics = client.get_run(run_id).data.metrics
    val_acc = metrics.get("final_val_accuracy", -1.0)
    if val_acc > best_acc:
        best_acc = val_acc
        best_version = v

print(f"Mejor versión: {best_version.version}, run_id: {best_version.run_id}, acc={best_acc:.4f}")

# ----------------------
# 3. Cargar el modelo PyTorch
# ----------------------
MODEL_URI = f"runs:/{best_version.run_id}/model"
print(f"Cargando modelo desde {MODEL_URI} ...")
model = mlflow.pytorch.load_model(MODEL_URI)
model = model.to("cpu")
model.eval()

# ----------------------
# 4. Exportar a ONNX
# ----------------------
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)  # CPU
onnx_path = Path(OUTPUT_ONNX_PATH)

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_ONNX_PATH,
    input_names=['input'],
    output_names=['output'],
    # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    # export_params=True,
    opset_version=18,
)

print(f"Modelo exportado a ONNX: {onnx_path.resolve()}")

# ----------------------
# 5. Obtener las etiquetas asociadas a esa versión
# ----------------------
class_names = {}
if best_version is not None:
    run_id = best_version.run_id
    # Descargar el artifact labels.json asociado al modelo
    ARTIFACT_PATH = "labels.json"  # ruta donde lo guardaste al logear
    local_path = client.download_artifacts(run_id, ARTIFACT_PATH)

    # Cargar las clases
    with open(local_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f)
