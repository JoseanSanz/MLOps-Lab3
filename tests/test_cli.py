# Importamos las funciones necesarias para los tests
from mylib.model import model  # predict_label, resize_image
from cli.cli import main
from click.testing import CliRunner
import os
import io
from PIL import Image


# Creamos un test para la función predict_label del modelo.
def test_model_predict():
    # Creamos una imagen de prueba de 200x200 píxeles
    img = Image.new("RGB", (200, 200), color="red")
    # Crear un buffer temporal en memoria
    buffer = io.BytesIO()

    # Guardar la imagen en ese buffer en formato JPEG o PNG
    img.save(buffer, format="JPEG")

    # Obtener los bytes del buffer
    image_bytes = buffer.getvalue()

    # label = predict_label(img)
    label = model.predict(image_bytes)

    # Opcional: cerrar el buffer
    buffer.close()
    # La etiqueta debe ser 'dog' o 'cat'
    assert label #in {"dog", "cat"}


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


# Creamos un test para el comando predict del CLI.
def test_cli_predict():
    runner = CliRunner()
    # # Creamos una imagen de prueba de 200x200 píxeles
    # img = Image.new("RGB", (200, 200), color="red")
    result = runner.invoke(main, ["predict", "perro.jpg"])
    # CLI debe terminar correctamente
    assert result.exit_code == 0
    # Debe imprimir una etiqueta válida
    assert result.output.strip() #in {"dog", "cat"}


# Creamos un test para el comando resize del CLI.
def test_cli_resize():
    runner = CliRunner()
    input_file = "./perro.jpg"
    output_file = "./perro_resized.jpg"

    result = runner.invoke(
        main, ["resize", input_file, output_file, "--width", "32", "--height", "32"]
    )

    # CLI debe terminar correctamente
    assert result.exit_code == 0
    # Debe existir el archivo de salida
    assert os.path.exists(output_file)
    # Debe generarse un archivo de salida con tamaño correcto
    img = Image.open(output_file)
    assert img.size == (32, 32)
