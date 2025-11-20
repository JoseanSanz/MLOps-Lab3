import mlflow
from mlflow.tracking import MlflowClient
import torch
import json
from pathlib import Path

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
model_uri = f"runs:/{best_version.run_id}/model"
print(f"Cargando modelo desde {model_uri} ...")
model = mlflow.pytorch.load_model(model_uri)
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
if best_version is not None:
    run_id = best_version.run_id
    # Descargar el artifact labels.json asociado al modelo
    artifact_path = "labels.json"  # ruta donde lo guardaste al logear
    local_path = client.download_artifacts(run_id, artifact_path)

    # Cargar las clases
    with open(local_path, "r") as f:
        class_names = json.load(f)

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(class_names, f)

