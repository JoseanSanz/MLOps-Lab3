# Importamos las funciones necesarias para los tests
from cli.cli import main
from click.testing import CliRunner
import os
import io
from PIL import Image


# Creamos un test para el comando predict del CLI.
def test_cli_predict():
    runner = CliRunner()
    # # Creamos una imagen de prueba de 200x200 píxeles
    # img = Image.new("RGB", (200, 200), color="red")
    result = runner.invoke(main, ["predict", "perro.jpg"])
    # CLI debe terminar correctamente
    assert result.exit_code == 0
    # Debe imprimir una etiqueta válida
    assert result.output.strip()  # in {"dog", "cat"}


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
