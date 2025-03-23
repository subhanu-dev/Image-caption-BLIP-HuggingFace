import gradio as gr
import torch
import numpy as np
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from PIL import Image

# Load models and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


# Function to generate caption and initialize history
def generate_caption(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = caption_model.generate(
            **inputs, num_beams=5, max_length=30, early_stopping=True, temperature=0.7
        )
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return (
            image,
            caption,
            "",
            [],
        )  # Return image, caption, clear answer, reset history
    except Exception as e:
        return None, f"Error: {str(e)}", "", []


# Function to answer question and update history
def answer_question(image, question, history):
    try:
        if image is None:
            return "Please upload an image first.", "", history
        if not question:
            return "Please enter a question.", "", history
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt")
        outputs = vqa_model.generate(
            **inputs, num_beams=5, max_length=100, early_stopping=True
        )
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        # Update history with new Q&A pair
        history.append((question, answer))
        history_text = "\n\n".join([f"**Q: {q}**\nA: {a}" for q, a in history])
        return answer, "", history_text  # Return answer, clear question, update history
    except Exception as e:
        return f"Error: {str(e)}", "", history


# Custom theme
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Source Code Pro"), "monospace"],
)

# Gradio interface
with gr.Blocks(theme=custom_theme, title="Image Captioning & VQA AI - BLIP") as iface:
    gr.Markdown("## Image Captioning & VQA AI - BLIP")
    gr.Markdown(
        "Upload an image to get a caption, then ask questions and see the history!"
    )

    # States for image and history
    image_state = gr.State()
    history_state = gr.State(value=[])

    with gr.Row():
        # Image input
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload an Image")

        # captions Question input and answer output
        with gr.Column():
            caption_btn = gr.Button("Generate Caption")
            caption_output = gr.Textbox(placeholder="Caption will appear here...")

            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a Question", placeholder="e.g., What color is the dog?"
                )
                question_btn = gr.Button("Get Answer")

                answer_output = gr.Textbox(placeholder="Answer will appear here...")

                # History display
                history_output = gr.Markdown(
                    label="Question & Answer History", value="No questions asked yet."
                )

    # Examples
    gr.Examples(
        examples=[
            ["test1.jpg"],
            ["test2.jpg"],
            ["test3.jpg"],
        ],
        inputs=[image_input],
    )

    # Event handlers
    caption_btn.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=[
            image_state,
            caption_output,
            answer_output,
            history_state,
        ],  # Store image, show caption, clear answer, reset history
    )
    question_btn.click(
        fn=answer_question,
        inputs=[image_state, question_input, history_state],
        outputs=[
            answer_output,
            question_input,
            history_output,
        ],  # Show answer, clear question, update history
    )

# Launch the interface
iface.launch(share=True)
