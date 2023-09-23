import gradio as gr
import torch
from PIL import Image
import tempfile

import torch

# Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
path = "obj.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)


def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.LANCZOS) 
    results = model(im)  # inference

    # Save the annotated image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        annotated_image_path = temp_file.name
        results[0].save(annotated_image_path)  # Save the annotated image

    # Return the path to the annotated image file
    return annotated_image_path

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="filepath", label="Output Image")


title = "YOLOv5"
description = "YOLOv5 Gradio demo for object detection. Upload an image or click an example image to use."

article = "YOLOv5 Leprosy AI Demo" 

examples = [["lp1.jpg"],["lp2.jpg"],["nlp1.jpg"],["nlp2.jpg"]]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch(
    debug=True)
