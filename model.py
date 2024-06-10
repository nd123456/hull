import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from io import BytesIO
from PIL import Image
import cv2
import io



# Load the pre-trained model
resnet_model = load_model('C:/Users/Namya/Documents/GSL HULL/model2.h5')

# Define class names
class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Preprocess input image
def preprocess_img(img_stream):
    try:
        Image._initialized = 1

        img = Image.open(img_stream)
        img = img.convert("RGB")
        print(img)
        img = img.resize((200, 200))  # Resize the image
        x = image.img_to_array(img)
        print(x) #problem
        #x = np.repeat(x,3, axis=-1)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = preprocess_input(x)  # Preprocess the input image
        return x
    except Exception as e:
        # Print detailed error message for debugging
        print("Error in preprocessing image:", str(e))
        return None


# Make prediction
def predict_result(img):
    '''print("Input image shape:", img.shape)
    print(resnet_model.summary())
    print("Preprocessed input image:", img)
    for layer in resnet_model.layers:
        print(layer.name, layer.output_shape)'''
    try:
        print('i am here')
        pred = resnet_model.predict(img)
        print(pred)
        predicted_class = class_names[np.argmax(pred)]
        print('this is also working')
        return predicted_class
    except Exception as e:
        print("Error during prediction:", str(e))
