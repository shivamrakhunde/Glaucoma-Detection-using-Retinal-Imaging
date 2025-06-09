import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf

# Define paths
image_path = r"C:\Users\shiva\Downloads\images (1).jpg"  # Change to your input image
output_dir = r"C:\\Users\\shiva\\OneDrive\\Desktop\\Glaucoma Detection\\static\\output"  # Change to your desired output folder
model_path = r"C:\\Users\\shiva\\OneDrive\\Desktop\\Glaucoma Detection\\unet_retina.keras"  # Path to trained model

def extract_cup_disc_sizes(mask, cup_value=255, disc_value=128):
    """Calculate the size (pixel count) of cup and disc from the mask."""
    cup_pixels = np.sum(mask == cup_value)
    disc_pixels = np.sum(mask == disc_value)
    return cup_pixels, disc_pixels

def annotate_image(image, mask, cup_value=255, disc_value=128):
    """Annotate image with cup and disc contours."""
    annotated = image.copy()
    
    # Find contours for cup
    cup_mask = (mask == cup_value).astype(np.uint8) * 255
    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, cup_contours, -1, (0, 255, 0), 2)  # Green for cup
    
    # Find contours for disc
    disc_mask = (mask == disc_value).astype(np.uint8) * 255
    disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, disc_contours, -1, (0, 0, 255), 2)  # Red for disc
    
    return annotated

# def predict_and_save(image_path, model_path, img_size=(128, 128), output_dir="output"):
# def predict_and_save(image_path, model_path, output_dir, img_size=(128, 128)):
def predict_and_save(image_path, model_path, output_dir, img_size=(256, 256)):
    """Predict mask, compute sizes, and save mask/annotated images."""
    # Load model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Predict mask
    pred = model.predict(img_input, verbose=0)
    pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
    
    # Convert to annotation format
    anno_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    anno_mask[pred_mask == 1] = 128  # Disc
    anno_mask[pred_mask == 2] = 255  # Cup
    
    # Compute sizes
    cup_size, disc_size = extract_cup_disc_sizes(anno_mask)
    
    # Annotate image
    img_for_anno = cv2.resize(img, img_size)
    annotated = annotate_image(img_for_anno, anno_mask)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save mask and annotated image
    mask_path = Path(output_dir) / f"{Path(image_path).stem}_mask.png"
    annotated_path = Path(output_dir) / f"{Path(image_path).stem}_annotated.jpg"
    cv2.imwrite(str(mask_path), anno_mask)
    cv2.imwrite(str(annotated_path), annotated)
    
    # Print results
    # print(f"Image: {Path(image_path).stem}")
    # print(f"  Cup size: {cup_size} pixels")
    # print(f"  Disc size: {disc_size} pixels")
    # print(f"  Mask saved to: {mask_path}")
    # print(f"  Annotated image saved to: {annotated_path}")

    return disc_size

def main():
    """Run prediction with defined variables."""
    predict_and_save(image_path, model_path, img_size=(256, 256), output_dir=output_dir)

if __name__ == "__main__":
    main()