"""
Main cli or app entry point para redimensionar imágenes y predecir etiquetas aleatorias (dog/cat).
"""

# Importamos las librerías necesarias
from pathlib import Path
import sys
import io
import click
from PIL import Image
from mylib.model_onnx_inference import model  # predict_label, resize_image


# Creamos un grupo de comandos
@click.group()
def main():
    """CLI principal para redimensionar una imagen y predecir una etiqueta de forma aleatoria."""


# Creamos un comando para redimensionar imágenes llamado resize, que asociamos al grupo anterior, que recibe dos argumentos input_path y output_path de tipo Path, y dos opciones width y height de tipo int con valores por defecto 100 y que utiliza la función resize_image del módulo mylib.model
@main.command("resize")
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--width", type=int, default=100, help="Anchura de la imagen resultante.")
@click.option("--height", type=int, default=100, help="Altura de la imagen resultante.")
def resize(input_path: str, output_path: str, width: int, height: int):
    """
    Redimensiona una imagen y guarda el resultado.
        input_path: Ruta de la imagen de entrada.
        output_path: Ruta donde guardar la imagen redimensionada.
        width: Anchura de la imagen resultante.
        height: Altura de la imagen resultante.
    Examples:
            ./cli.py resize perro.jpg perro_resized.jpg
            ./cli.py resize perro.jpg perro_resized.jpg --width 500 --height 500
    """

    try:
        with Image.open(Path(input_path)) as img:
            # aquí trabajas con la imagen
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            image_bytes = img_bytes.getvalue()
            resized = model.resize_image(image_bytes, width, height)
            # img = img.convert("RGB")
            # resized = resize_image(img, width, height)
            resized.save(Path(output_path))
    except FileNotFoundError:
        click.echo(f"Error: el archivo '{input_path}' no existe.")
        sys.exit(1)


# Creamos un comando para predecir la etiqueta de una imagen llamado predict, que asociamos al grupo anterior, y recibe un argumento image_path de tipo Path, y utiliza la función predict_label del módulo mylib.model
@main.command("predict")
@click.argument(
    "image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def predict(image_path: str):
    """
    Predice una etiqueta aleatoria (dog/cat) para una imagen.
        image_path: Ruta de la imagen de entrada.
    Example:
        ./cli.py predict perro.jpg
    """

    try:
        with Image.open(Path(image_path)) as img:
            # aquí trabajas con la imagen
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            image_bytes = img_bytes.getvalue()
            label = model.predict(image_bytes)
            # label = predict_label(img)
            click.echo(click.style(label, fg="green"))
    except FileNotFoundError:
        click.echo(f"Error: el archivo '{image_path}' no existe.")
        sys.exit(1)


# Punto de entrada principal
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
