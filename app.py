import gradio as gr
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


# Define the captioning function
def generate_caption(image):
    try:
        # Convert NumPy array from Gradio to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # Process image for BLIP
        inputs = processor(images=image, return_tensors="pt")
        # Generate caption
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error: {str(e)}"


# giving a custom theme
custom_theme = gr.themes.Soft(
    primary_hue="blue",  # Clean blue for buttons and highlights
    secondary_hue="cyan",  # Subtle cyan accents
    neutral_hue="slate",  # Soft slate for backgrounds and borders
    radius_size="md",  # Medium rounded corners for a polished look
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],  # Modern, clean font
    font_mono=[
        gr.themes.GoogleFont("Source Code Pro"),
        "monospace",
    ],  # Code-friendly font
)

# Create Gradio interface with the theme
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning AI - BLTP",
    description="Upload an image to generate a caption.",
    examples=["test1.jpg", "test2.jpg", "test3.jpg"],
    flagging_mode="never",
    theme=custom_theme,
)

# Launch the interface
iface.launch(share=True)
