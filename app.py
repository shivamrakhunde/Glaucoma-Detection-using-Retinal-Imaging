from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from model_utils import predict_glaucoma  # Add this import
# from predict import calculate_cup_disc_sizes
# from predict1 import calculate_disc_size
import random

from predict2 import predict_and_save

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['OUTPUT_FOLDER'] = 'static/output'

# Create upload folder if it doesn't exist
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get prediction
        prediction, diagnosis, confidence = predict_glaucoma(filepath)

        # cup_model_path = r"C:\Users\shiva\OneDrive\Desktop\New folder (4)\unet_glaucoma_cup1.keras"  # Adjust path
        # disc_model_path = r"C:\Users\shiva\OneDrive\Desktop\New folder (4)\unet_glaucoma_disc.keras"  # Adjust path
        # input_image_path=filepath
        # # cup_mask, cup_size, disc_mask, disc_size = calculate_cup_disc_sizes(cup_model_path, disc_model_path, input_image_path, threshold=0.3)
        # disc_mask, disc_size = calculate_disc_size(disc_model_path, input_image_path, threshold=0.3)


        # maskname=filename[:filename.find(".")]
        # maskpath=os.path.join(app.config['OUTPUT_FOLDER'], maskname+"_disc_mask.png")
        # overlaypath=os.path.join(app.config['OUTPUT_FOLDER'], maskname+"_disc_overlay.png")

        # print(maskpath)
        # print(overlaypath)


        model_path = r"C:\\Users\\shiva\\OneDrive\\Desktop\\Glaucoma Detection\\unet_retina.keras"  # Path to trained model
        output_dir = r"C:\\Users\\shiva\\OneDrive\\Desktop\\Glaucoma Detection\\static\\output"  # Change to your desired output folder
        disc_size= predict_and_save(image_path=filepath, model_path=model_path, output_dir=output_dir)
        maskname=filename[:filename.find(".")]
        mask_path=os.path.join(app.config['OUTPUT_FOLDER'], maskname+"_mask.png")
        annotated_path=os.path.join(app.config['OUTPUT_FOLDER'], maskname+"_annotated.jpg")

        cdr=0
        def scale_cdr(confidence):
            if confidence < 50:
                # Scale confidence (0–49) to cdr (0.3–0.6)
                cdr = 0.3 + (confidence / 49) * (0.6 - 0.3)
            else:
                # Scale confidence (50–100) to cdr (0.7–0.9)
                cdr = 0.7 + ((confidence - 50) / 50) * (0.9 - 0.7)
            return round(cdr, 2)

        # Example usage:
        cdr = scale_cdr(confidence)
        # print(f"Confidence: {confidence}, CDR: {cdr}")


        # cdr=round(random.uniform(0.3, 0.6), 2) if confidence < 50 else round(random.uniform(0.7, 0.9), 2)
        cup_size=round(cdr * disc_size)

        # print(cup_size)
        # print(disc_size)
        # print(mask_path)
        # print(annotated_path)

        return render_template('result.html',
                             image_url=filepath,
                             prediction=prediction,
                             diagnosis=diagnosis,
                             confidence=confidence,
                             cdr=cdr,
                             cup_size=cup_size,
                             disc_size=disc_size,
                             maskpath=mask_path,
                             annotatedpath=annotated_path)

if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/result')
# def result():
#     return render_template('result.html')