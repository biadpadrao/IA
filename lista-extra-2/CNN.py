# CÓDIGO-FONTE: https://www.deeplearningbook.com.br/reconhecimento-de-imagens-com-redes-neurais-convolucionais-em-python-parte-4/

import tensorflow as tf
import keras as K

# Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image

# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Pré-processamento
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('test_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

def custom_generator(generator):
    while True:
        data = next(generator)
        yield data

repeating_training_set = custom_generator(training_set)

# Treinamento
classifier.fit_generator(repeating_training_set, 
                         steps_per_epoch=100, 
                         epochs=5, 
                         validation_data=validation_set, 
                         validation_steps=50)

# Primeira Imagem
import numpy as np
from keras.preprocessing import image
from IPython.display import display, Image

from PIL import Image as PilImage

# Open the image file
img = PilImage.open('test_set/bart/bart23.bmp')

# Convert the image to PNG
img.save('test_set/bart/bart23.png')

test_image = image.load_img('test_set/bart/bart23.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

display(Image(filename='test_set/bart/bart23.png'))
prediction

# Segunda Imagem
import numpy as np
from keras.preprocessing import image
from IPython.display import display, Image

from PIL import Image as PilImage

# Open the image file
img = PilImage.open('test_set/homer/homer1.bmp')
# Convert the image to PNG
img.save('test_set/homer/homer1.png')

test_image = image.load_img('test_set/homer/homer1.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

display(Image(filename='test_set/homer/homer1.png'))
prediction

# Terceira Imagem
import numpy as np
from keras.preprocessing import image
from IPython.display import display, Image

from PIL import Image as PilImage

# Open the image file
img = PilImage.open('test_set/homer/homer98.bmp')
# Convert the image to PNG
img.save('test_set/homer/homer98.png')

test_image = image.load_img('test_set/homer/homer98.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

display(Image(filename='test_set/homer/homer98.png'))
prediction

# Segunda Imagem
import numpy as np
from keras.preprocessing import image
from IPython.display import display, Image

from PIL import Image as PilImage

# Open the image file
img = PilImage.open('test_set/bart/bart60.bmp')

# Convert the image to PNG
img.save('test_set/bart/bart60.png')

test_image = image.load_img('test_set/bart/bart60.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Homer'
else:
    prediction = 'Bart'

display(Image(filename='test_set/bart/bart60.png'))
prediction

# Evaluate the model
score = classifier.evaluate(validation_set, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
