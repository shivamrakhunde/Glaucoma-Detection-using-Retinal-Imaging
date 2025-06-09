import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = r'C:\\Users\\shiva\\OneDrive\\Desktop\\Glaucoma Detection\\model\\combine_cnn.h5'  # Path to your .h5 model file
# MODEL_PATH = r'C:\\Users\\shiva\\OneDrive\\Desktop\\Glaucoma Detection\\model\\glaucoma_detection_model.h5'  # Path to your .h5 model file
IMG_SIZE = (256, 256)

model = load_model(MODEL_PATH)

def predict_glaucoma(img_path):
    try:
        # Preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        confidence = round(float(prediction[0][0]) * 100, 2)
        diagnosis="Low" if confidence < 25 else "Moderate" if confidence<50 else "High" if confidence<75 else "Very High"
        diagnosis+=" Risk of Glaucoma"
        return prediction, diagnosis, confidence
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 0,"Error in analysis", 0.0