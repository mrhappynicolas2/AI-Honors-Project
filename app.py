import torch
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
from torchvision import models
from diffusers import StableDiffusionPipeline
import traceback
import os

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face LoRA filenames (hosted in the same model repo)
model_files = {
    "Benign cases": "BenginLast.safetensors",  # typo matches your uploaded filename
    "Malignant cases": "MalignantLast.safetensors",
    "Normal cases": "HealthyLast.safetensors"
}

# Prompt mapping for each case
prompt_map = {
    "Benign cases": "a lung CT scan showing benign tissue",
    "Malignant cases": "a lung CT scan showing malignant cancer",
    "Normal cases": "a healthy lung CT scan"
}

# Classification labels
class_names = ["Benign cases", "Malignant cases", "Normal cases"]

# Image transform for ResNet classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Load your ResNet model
resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 3)
resnet_model.load_state_dict(torch.load("model.pth", map_location=device))
resnet_model.to(device)
resnet_model.eval()

# Load base Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# Function to generate image using LoRA adapter
def generate_lung_image(case_type):
    try:
        if case_type not in model_files:
            return Image.new("RGB", (224, 224), color="red")

        weight_name = model_files[case_type]
        prompt = prompt_map[case_type]

        # Load LoRA adapter from Hugging Face model repo
        pipe.load_lora_weights("mrhappynicolas/BenginLast", weight_name=weight_name)

        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            image = pipe(prompt).images[0]

        return image

    except Exception as e:
        print("❌ Error generating image:", e)
        print(traceback.format_exc())
        return Image.new("RGB", (224, 224), color="red")

# Function to classify uploaded lung CT image
def classify_lung_image(img: Image.Image):
    try:
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = resnet_model(img_t)
            _, predicted_idx = torch.max(outputs, 1)
        return f"Prediction: {class_names[predicted_idx.item()]}"
    except Exception as e:
        print("❌ Error during classification:", e)
        print(traceback.format_exc())
        return "Error during classification."

# Gradio UI: Generator tab
image_generator_interface = gr.Interface(
    fn=generate_lung_image,
    inputs=gr.Radio(choices=list(model_files.keys()), label="Select Case Type"),
    outputs=gr.Image(type="pil"),
    title="Lung CT Image Generator",
    description="Generate a synthetic lung CT scan using LoRA-tuned Stable Diffusion."
)

# Gradio UI: Classifier tab
classification_interface = gr.Interface(
    fn=classify_lung_image,
    inputs=gr.Image(type="pil", label="Upload Lung CT Image"),
    outputs="text",
    title="Lung CT Classifier",
    description="Upload a lung CT scan and classify it as Benign, Malignant, or Normal."
)

# Combine both into tabs
interface = gr.TabbedInterface(
    [image_generator_interface, classification_interface],
    ["Image Generator", "Image Classifier"]
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)

