import torch
import torchvision.transforms as transforms
from PIL import Image
from loadPreTrain import XceptionPretrained  # Import your model

# Load the trained model
def load_model(model_path, device):
    model = XceptionPretrained()  # Initialize model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load trained weights
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to match Xception input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Make a prediction
def predict(image_path, model, device, class_names):
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        confidence, predicted_class = torch.max(probabilities, dim=1)  # Get highest probability

    predicted_label = class_names[predicted_class.item()]  # Get class name
    confidence_score = confidence.item() * 100  # Convert to percentage

    print(f"Predicted Class: {predicted_label}, Confidence: {confidence_score:.2f}%")
    return predicted_label, confidence_score

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "checkpoints/xception_shoplifting_epoch10.pth"  # Adjust as needed
    model = load_model(model_path, device)

    image_path = r"D:\Akses_ai\shoplift\Extracted_Frames\Shoplifting\Shoplifting048_x264\frame_00033.jpg"  # Path to test image
    class_names = ["Not Shoplifting", "Shoplifting"]  # Adjust based on your dataset labels

    predict(image_path, model, device, class_names)
