# Importamos las librerías necesarias
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from mylib.model_onnx_inference import model  # predict_label, resize_image
from PIL import Image
import io
from pathlib import Path


# Crear instancia de FastAPI
app = FastAPI(
    title="API de Clasificación de Imágenes con FastAPI",
    description="API para redimensionar imágenes y predecir etiquetas de perros y gatos usando mylib.model",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

# Directorio donde se guardarán las imágenes redimensionadas
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Endpoint raíz
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


# Endpoint para redimensionar imágenes
@app.post("/resize")
async def resize(
    file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)
):
    """
    Endpoint REST para redimensionar una imagen enviada en un formulario.

    Espera un formulario con:
    - file: imagen (multipart/form-data)
    - width: anchura deseada (int)
    - height: altura deseada (int)
    """

    try:
        width = int(width)
        height = int(height)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Width y Height deben ser enteros"
        ) from exc

    try:
        # Abrir la imagen en memoria
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img.verify()  # verifica que sea una imagen válida

        # Reabrir la imagen para poder procesarla, ya que verify() deja la imagen cerrada
        # img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Redimensionar la imagen
        resized = model.resize_image(contents, width, height)

        # Guardar la imagen redimensionada en el directorio de uploads
        filename = Path(file.filename)
        output_path = UPLOAD_DIR / f"{filename.stem}_resized.jpg"
        resized.save(output_path)

        return {"message": f"Imagen redimensionada guardada en {output_path}"}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail="No se ha subido una imagen válida"
        ) from e


# Endpoint para predecir etiqueta de imagen
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint REST para predecir la etiqueta de una imagen enviada en un formulario.

    Espera un formulario con:
    - file: imagen (multipart/form-data)
    """

    try:
        # Abrir la imagen en memoria
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img.verify()  # verifica que sea una imagen válida

        # Reabrir la imagen para poder procesarla, ya que verify() deja la imagen cerrada
        # img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Predecir la etiqueta
        label = model.predict(contents)
        # label = predict_label(img)

        return {"label": label}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail="No se ha subido una imagen válida"
        ) from e


if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
