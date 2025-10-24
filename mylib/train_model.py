import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
import json


output_path = "./mylib/model.onnx"
model_name = "google/mobilenet_v2_1.0_224"

"""
Descarga un modelo preentrenado de Hugging Face y lo exporta a formato ONNX.
"""
print(f"Cargando modelo preentrenado: {model_name}")
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

# Determinar tamaño de entrada de manera robusta
size = processor.size
if isinstance(size, dict):
    height = size.get("height", size.get("shortest_edge", 224))
    width = size.get("width", height)
elif isinstance(size, int):
    height = width = size
else:
    height = width = 224  # valor por defecto seguro

print(f"Usando tamaño de entrada: {height}x{width}")

# Crear una entrada dummy compatible con el modelo
dummy_input = torch.randn(1, 3, height, width)

print(f"Exportando modelo a {output_path}...")
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12,
)

config = AutoConfig.from_pretrained(model_name, use_fast=False)
labels_file = "labels.json"
with open(labels_file, "w") as f:
    json.dump(config.id2label, f)

print(f"Modelo exportado correctamente a {output_path} y etiqeutas de clases a {labels_file}")