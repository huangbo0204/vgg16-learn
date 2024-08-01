import keras
from keras.api.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=True)
model.summary()

img_path = './data/elephant.jpg'
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction = model.predict(x)
names = [p[1]+",概率:"+str(p[2] * 100) + "%" for p in imagenet_utils.decode_predictions(prediction)[0]]
print(names)
