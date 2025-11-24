# genera codigo para realizar el testing con pytest de fastapi_app.py similar a test_flask_main.py
import io
import pytest
from api.api import app
from PIL import Image
from fastapi.testclient import TestClient
from pathlib import Path


@pytest.fixture
def client():
    """Cliente de pruebas de FastAPI."""
    return TestClient(app)


def test_home_endpoint(client):
    """Verifica que el endpoint / devuelve el mensaje correcto."""
    response = client.get("/")
    assert response.status_code == 200


# # creamos un test para el endpoint /predict
# def test_predict_endpoint(client):
#     """Verifica que /predict devuelve etiqueta válida."""
#     input_path = Path("./perro.jpg")
#     input_path = Path("./imagen_0.png")
#     with Image.open(input_path, 'r') as img:
#         img = img.convert("RGB")
#         img_bytes = io.BytesIO()
#         img.save(img_bytes, format="JPEG")
#         img_bytes.seek(0)  # 🔹 muy importante: volver al inicio del buffer
#     # with open(input_path, "rb") as f:
#     #     img_bytes = io.BytesIO(f.read())  # lee directamente los bytes

#     response = client.post(
#         "/predict",
#         files={"file": ("perro.jpg", img_bytes, "image/jpeg")},
#     )
#     # ✅ Verificar respuesta HTTP
#     assert response.status_code == 200, response.text
#     data = response.json()
#     assert "label" in data
#     # assert data["label"] in {"dog", "cat"}


def test_predict_endpoint(client):
    input_path = Path("./perro.jpg")
    with open(input_path, "rb") as f:
        img_bytes = io.BytesIO(f.read())

    response = client.post(
        "/predict",
        files={"file": ("perro.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "label" in data


# genera un test para predict con una imagen no válida
def test_predict_endpoint_invalid_image(client):
    """Verifica que /predict maneja imagen no válida."""
    invalid_image_bytes = io.BytesIO(b"not an image content")
    response = client.post(
        "/predict",
        files={"file": ("main.py", invalid_image_bytes, "image/jpeg")},
    )
    # ✅ Verificar respuesta HTTP
    assert response.status_code == 400, response.text
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "No se ha subido una imagen válida"


# genera un test para resize con una imagen válida
def test_resize_endpoint(client):
    """Verifica que /resize redimensiona imagen correctamente."""
    input_path = Path("./perro.jpg")
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)  # 🔹 muy importante: volver al inicio del buffer

    data = {"width": "50", "height": "50"}
    response = client.post(
        "/resize",
        files={"file": ("perro.jpg", img_bytes, "image/jpeg")},
        data=data,
    )
    # ✅ Verificar respuesta HTTP
    assert response.status_code == 200, response.text
    data = response.json()
    assert "message" in data
    assert "Imagen redimensionada guardada en" in data["message"]
