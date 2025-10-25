from datasets import load_dataset

# Cargar dataset
dataset = load_dataset("beans")

id_imagen = 900  # Cambia este valor para ver diferentes imágenes

# Acceder a un ejemplo del train
example = dataset['train'][id_imagen]  # Primer ejemplo del set de entrenamiento

# crea un bucle que muestre el id y la etiqueta de cada ejemplo en el dataset de train
# for i in range(len(dataset['train'])):
#     example_loop = dataset['train'][i]
#     print("ID: {}, Etiqueta: {}".format(i, example_loop['labels']))

# Obtener la imagen y la etiqueta
image = example['image']
label = example['labels']  # o 'label' dependiendo del dataset

print("Etiqueta de la imagen con id {}: {}".format(id_imagen, label))

# genera código para guardar la imagen en un archivo llamado 'imagen_id_imagen.png' 
image.save(f'imagen_{id_imagen}.png')