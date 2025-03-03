from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import io

# Load model
model = torch.load('Classifier.pth', weights_only=False)
model.eval()  # Set the model to evaluation mode

# Define image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

# Route to render the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))

    # Convert grayscale images to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = test_transform(img).unsqueeze(0)  # Apply transformations and add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted_class = torch.max(output, 1)

    return f'Predicted class: {predicted_class.item()}'
if __name__ == '__main__':
    app.run(debug=True)
