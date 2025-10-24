#!/usr/bin/env python

# genera una aplicacon gradio para redimensionar imágenes y predecir etiquetas que consuma la API FastAPI llamada fastapi_app.py
import gradio as gr
import requests
import io

# URL de tu API FastAPI
API_URL = "https://lab3-latest.onrender.com"


# Función que se ejecuta al subir una imagen
def predecir_etiqueta(image):
    try:
        # Convertir la imagen a bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        # Enviar la imagen a la API FastAPI
        files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("label")
    except requests.exceptions.RequestException as e:
        # Si hubo cualquier error en la conexión o timeout
        return f"Error en la conexión con la API: {str(e)}"
    except ValueError as e:
        # Si response.json() falla
        return f"Error: la API no devolvió JSON válido: {str(e)}"
    # except requests.exceptions.HTTPError as e:
    #     return f"Error: {response.json().get('detail', str(e))}"


# Crear interfaz Gradio
iface = gr.Interface(
    fn=predecir_etiqueta,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Etiqueta Predicha"),
    title="Clasificador de Imágenes con FastAPI y Gradio",
    description="Sube una imagen de un perro o un gato y obtén la etiqueta predicha usando la API FastAPI /predict",
)

# Lanzar interfaz
if __name__ == "__main__":
    iface.launch()
