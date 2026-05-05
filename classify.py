import cv2
import os

# returns (prediction, confidence) tuple
def classify_img_path(image_path, model):

    if not os.path.exists(image_path):
        print("File not found:", image_path)
        return None

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return None

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = model.single_image_classification(img_rgb)

    prediction = result.get("prediction", "Unknown") if result else "Unknown"
    confidence = result.get("confidence", 0.0) if result else 0.0

    # Return None if prediction no-species
    if prediction.lower() in ["no-species", "unknown"]:
        return None

    return prediction, confidence

def classify_img(img, model):
    if img is None:
        print("Invalid image.")
        return None

    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = model.single_image_classification(img_rgb)

    if result is None:
        return None

    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0.0)

    # Ignore no-species predictions
    if prediction.lower() in ["no-species", "unknown"]:
        return None

    return prediction, confidence