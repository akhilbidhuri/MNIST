import neuralnet as net
import random 
import tflearn.datasets.mnist as mnist
import numpy as np
from skimage import io
model = net.model
path_to_model = 'final-model.tflearn'
path_to_image = 'custom.jpg' # Change this to the file path/name of the image file you want to use
model.load(path_to_model)
# Load image (normalized)
x = io.imread(path_to_image).reshape((28, 28, 1)).astype(np.float) / 255
result = model.predict([x])[0] # Predict
prediction = result.index(max(result)) # The index represents the number predicted in this case
print("Prediction", prediction)