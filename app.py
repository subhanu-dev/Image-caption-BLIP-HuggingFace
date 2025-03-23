import gradio as gr
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
        if image is None:
            return None, "⚠️ Please upload an image first before asking questions!", []
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = caption_model.generate(**inputs, temperature=0.7)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return (
            image,
            caption,
            [],
        )
    except Exception as e:
        return None, f"Error: {str(e)}", []


# Function to answer question and update history
def answer_question(image, question, history):
    try:
        if image is None:
            error_message = "⚠️ Please upload an image first before asking questions!"
            history_text = f"<div style='color: #ef4444; padding: 0.5rem; border-radius: 0.375rem; background-color: #fee2e2'>{error_message}</div>"
            return (
                question,
                history_text,
            )  # Return original question (don't clear it) and error message

        if not question:
            return "", history

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        inputs = processor(images=image, text=question, return_tensors="pt")
        outputs = vqa_model.generate(
            **inputs, num_beams=5, max_length=20, early_stopping=True
        )
        answer = processor.decode(outputs[0], skip_special_tokens=True)

        # Update history with new Q&A pair and styled answer
        history.append((question, answer))
        history_text = "\n\n".join(
            [
                f"**Q: {q}**\n<div style='color: #2563eb; margin-left: 1rem; margin-top: 0.5rem'>Answer: {answer}</div>"
                for q, answer in history
            ]
        )
        return "", history_text  # Clear question input only on successful answer

    except Exception as e:
        error_message = f"⚠️ Error: {str(e)}"
        history_text = f"<div style='color: #ef4444; padding: 0.5rem; border-radius: 0.375rem; background-color: #fee2e2'>{error_message}</div>"
        return question, history_text  # Keep the question in case of error


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
    gr.Markdown(
        """
        <div style="text-align: center">
            <h2>Image Captioning & Question Answering AI - BLIP</h2>
            <br>
        </div>
        """
    )
    gr.Markdown(" Upload an image to get a caption then ask questions about the image.")

    # States for image and history
    image_state = gr.State()
    history_state = gr.State(value=[])

    with gr.Row():
        # Image input
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload an Image")
            caption_btn = gr.Button("Generate Caption")

        # captions Question input and answer output
        with gr.Column():
            caption_output = gr.Textbox(
                label="Image Description", placeholder="Caption will appear here..."
            )

            gr.Markdown("### Ask questions about the image")

            question_input = gr.Textbox(
                label="Ask a Question", placeholder="e.g., What color is the dog?"
            )
            question_btn = gr.Button("Get Answer")

            # answer_output = gr.Text(show_label=False)

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
            history_state,
        ],
    )
    question_btn.click(
        fn=answer_question,
        inputs=[image_state, question_input, history_state],
        outputs=[
            question_input,
            history_output,
        ],
    )

# Launch the interface
iface.launch()
