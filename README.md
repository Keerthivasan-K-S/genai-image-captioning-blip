# Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

## AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

## PROBLEM STATEMENT:
Generating accurate textual descriptions for images requires effective integration of computer vision and natural language processing techniques. This project aims to provide an easy-to-use image captioning system using the BLIP model and a Gradio interface for seamless user interaction.

## DESIGN STEPS:

### STEP 1:
Configure environment variables, load the Hugging Face API key, and set up the get_completion function to interact with the image-to-text (BLIP) model endpoint.

### STEP 2:
Define functions to convert uploaded images to base64 strings and retrieve captions from the API using the BLIP model.

### STEP 3:
Build an interactive Gradio interface for users to upload images and receive captions, with optional examples and shareable link.

## PROGRAM:
```py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
import torch

# Load BLIP model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Caption generation function
def generate_caption(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    output = model.generate(**inputs)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption

# Gradio UI
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning using BLIP",
    description="Upload an image to generate an automatic caption."
)

# Launch app
demo.launch()
```

## OUTPUT:

<img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/bb0ac4c2-3050-457a-b635-6fa3bde80f9c" />


## RESULT:

Thus, The designing and deploying of a prototype application for image captioning by utilizing the BLIP image-captioning model is executed successfully.
