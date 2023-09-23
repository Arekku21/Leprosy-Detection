import gradio as gr
import torch
from PIL import Image

import torch

# Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
path = "obj.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)


def yolo(im, size=640):

    g = (size / max(im.size))  # gain

    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference

    # results.render()  # updates results.imgs with boxes and labels

    # return Image.fromarray(results.imgs[0])

    # Retrieve the annotated image from the results (modify this based on your model's output structure)
    annotated_image = results.render()[0] 

    # Return the annotated image
    return Image.fromarray(annotated_image)

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="filepath", label="Output Image")


title = "YOLOv5"
description = "YOLOv5 Gradio demo for object detection. Upload an image or click an example image to use."

article = "YOLOv5 Leprosy AI Demo" 

examples = [["lp1.jpg"],["lp2.jpg"],["nlp1.jpg"],["nlp2.jpg"]]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch(
    debug=True)
