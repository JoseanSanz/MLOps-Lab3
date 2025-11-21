"""
Módulo que contiene funciones para predecir etiquetas de imágenes y redimensionarlas.
"""

# from transformers import AutoImageProcessor, AutoModelForImageClassification
import json
import io
import torch # pylint: disable=import-error,no-member
import numpy as np
import onnxruntime as ort
from PIL import Image


class ImageClassifier:
    """Class for the image classifier"""
    def __init__(
        self, model_path="./mylib/model.onnx", labels_file="./mylib/labels.json"
    ):  # "facebook/deit-tiny-patch16-224"
        """
        Inicializa la sesión ONNX Runtime.
        """
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=sess_options
        )

        self.input_name = self.session.get_inputs()[0].name

        # Cargar clases
        with open(labels_file, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        # self.id2label = {}
        # with open(labels_file, encoding="utf-8") as f:
        #     self.id2label = json.load(f)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Convierte bytes de imagen en un tensor normalizado listo para inferencia.
        """
        # Convertir bytes a imagen PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Redimensionar y normalizar (ajusta si tu modelo usa otro tamaño)
        image = image.resize((256, 256))
        img = np.array(image).astype(np.float32) / 255.0

        # Cambiar orden de canales: HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        # Añadir dimensión de batch: (1, 3, 224, 224)
        return np.expand_dims(img, axis=0)

    def predict(self, image_bytes: bytes) -> str:
        """Recibe bytes de imagen y devuelve la etiqueta predicha."""
        # image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = {self.input_name: self.preprocess(image_bytes)}

        with torch.no_grad(): # pylint: disable=no-member
            outputs = self.session.run(None, inputs)
            logits = outputs[0]
            pred_class_idx = int(logits.argmax(-1).item())
            label = self.class_names[pred_class_idx]
            # label = self.id2label.get(str(pred_class_idx), f"class_{pred_class_idx}")
        return label

    # Crea una función que reciba una imagen y la redimensione a un tamaño
        # que se le pasa como parámetro
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
