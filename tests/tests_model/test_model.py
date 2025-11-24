from pathlib import Path

# Creamos un test para ver si existe el ONNX
def test_ONNX_exists():
    file_path = Path("./mylib/model.onnx")
    assert file_path.exists(), f"El archivo {file_path} no existe" 

# Creamos un test para ver si existe el JSon con las etiquetas de las clases
def test_JSON_exists():
    file_path = Path("./mylib/labels.json")
    assert file_path.exists(), f"El archivo {file_path} no existe" 