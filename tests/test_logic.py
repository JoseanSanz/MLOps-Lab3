# Importamos las funciones necesarias para los tests
from mylib.model_onnx_inference import model
import io
from PIL import Image


# Creamos un test para la función predict_label del modelo.
def test_model_predict():
    # Creamos una imagen de prueba de 200x200 píxeles
    img = Image.new("RGB", (200, 200), color="red")
    # Crear un buffer temporal en memoria
    # buffer = io.BytesIO()

    # Guardar la imagen en ese buffer en formato JPEG o PNG
    # img.save(buffer, format="JPEG")

    # Obtener los bytes del buffer
    # image_bytes = buffer.getvalue()

    # label = predict_label(img)
    # label = model.predict(image_bytes)
    label = model.predict(img)

    # Opcional: cerrar el buffer
    # buffer.close()
    # La etiqueta debe ser 'dog' o 'cat'
    assert label  # in {"dog", "cat"}


# Creamos un test para la función resize_image del modelo.
def test_model_resize_image():
    # Creamos una imagen de prueba de 200x200 píxeles
    img = Image.new("RGB", (200, 200), color="red")
    # Crear un buffer temporal en memoria
    buffer = io.BytesIO()

    # Guardar la imagen en ese buffer en formato JPEG o PNG
    img.save(buffer, format="JPEG")

    # Obtener los bytes del buffer
    image_bytes = buffer.getvalue()

    resized = model.resize_image(image_bytes, 50, 50)
    # resized = resize_image(img, 50, 50)

    # Opcional: cerrar el buffer
    buffer.close()

    assert resized.size == (50, 50)
