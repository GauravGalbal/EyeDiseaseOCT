from scale_layer import Scale
from flask import Flask, request
import cv2
import numpy as np
import os
import tensorflow as tf
import keras

app = Flask(__name__)


def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    # img = crop_image_from_gray(img)    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    # img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def process_image(img):
    sigmaX = 10

    Save_Folder = 'processed'
    app.config['Save_Folder'] = Save_Folder
    os.makedirs(Save_Folder, exist_ok=True)
    
    image = circle_crop(img, sigmaX)
    
    output_image_path = os.path.join(app.config['Save_Folder'], f'{os.path.basename(img)}')
    cv2.imwrite(output_image_path, image)
    return output_image_path



# Specify the directory where you want to save uploaded images.
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the "uploads" directory exists.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/checkImage', methods=['GET'])
def get_image():
    # Getting Image
    image_file = request.files['image']

    # Saving to Backend
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(input_image_path)

    # Processing Image
    image_path = process_image(input_image_path)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))  # Use the same target size as during training
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = img / 255.0  # Normalize the image data (if you used rescaling during training)

    model = keras.models.load_model("eye_oct_Model.h5", custom_objects={'Scale': Scale})

    predictions = model.predict(img)
    print(predictions)

    if(predictions[0][0] > predictions[0][1]):
        return 'Normal Eye'
    else:
        return 'Infected Eye'


if __name__ == '__main__':
    app.run()

