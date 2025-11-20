"""
Módulo que contiene funciones para predecir etiquetas de imágenes y redimensionarlas.
"""

from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


class ImageClassifier:
    def __init__(
        self, model_name="google/mobilenet_v2_1.0_224"
    ):  # "facebook/deit-tiny-patch16-224"
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, image_bytes: bytes) -> str:
        """Recibe bytes de imagen y devuelve la etiqueta predicha."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_class_idx = logits.argmax(-1).item()
            label = self.model.config.id2label[pred_class_idx]
        return label

    # Crea una función que reciba una imagen y la redimensione a un tamaño que se le pasa como parámetro
    def resize_image(self, image_bytes: bytes, width: int, height: int) -> Image.Image:
        """Redimensiona una imagen al tamaño especificado.

        Args:
            image (PIL.Image.Image): Imagen a redimensionar.
            width (int): Anchura deseada.
            height (int): Altura deseada.

        Returns:
            PIL.Image.Image: Imagen redimensionada.
        """

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_resized = image.resize((width, height))

        return image_resized


# Carga global del modelo (solo una vez)
model = ImageClassifier()

# # Crea una función que devuelva una etiqueta aleatoria de entre 'dog' y 'cat'
# def predict_label(_image: Image.Image):
#     """Devuelve una etiqueta aleatoria: 'dog' o 'cat'.
#     Simple función que simula la salida de un modelo.
#     """
#     return random.choice(["dog", "cat"])
