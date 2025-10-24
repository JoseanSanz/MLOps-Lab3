"""
Módulo de "modelo" mínimo: etiqueta aleatoriamente como 'dog' o 'cat'.
También incluye una función para redimensionar imágenes.
"""

import random
from PIL import Image


# Crea una función que devuelva una etiqueta aleatoria de entre 'dog' y 'cat'
def predict_label(_image: Image.Image):
    """Devuelve una etiqueta aleatoria: 'dog' o 'cat'.
    Simple función que simula la salida de un modelo.
    """
    return random.choice(["dog", "cat"])


# Crea una función que reciba una imagen y la redimensione a un tamaño que se le pasa como parámetro
def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Redimensiona una imagen al tamaño especificado.

    Args:
        image (PIL.Image.Image): Imagen a redimensionar.
        width (int): Anchura deseada.
        height (int): Altura deseada.

    Returns:
        PIL.Image.Image: Imagen redimensionada.
    """
    return image.resize((width, height))
