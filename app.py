

import gradio as gr
import cv2
import requests
import os

import torch
from PIL import Image
import tempfile

import numpy as np
import math

file_urls = [
    'https://www.dropbox.com/scl/fi/onrg1u9tqegh64nsfmxgr/lp2.jpg?rlkey=2vgw5n6abqmyismg16mdd1v3n&dl=1',
    'https://www.dropbox.com/scl/fi/lvqft5kibtl05e5dank0u/nlp1.jpg?rlkey=gj90a7nsbp0torl81t4ot8h0n&dl=1',
    'https://www.dropbox.com/scl/fi/adcykczwpvg5ipxl4dhqi/nlp2.jpg?rlkey=0npa4ttmfgld8anqr820yr5js&dl=1',
    'https://www.dropbox.com/scl/fi/xq103ic7ovuuei3l9e8jf/lp1.jpg?rlkey=g7d9khyyc6wplv0ljd4mcha60&dl=1'
]

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(
            file_urls[i],
            f"video.mp4"
        )
    else:
        download_file(
            file_urls[i],
            f"image_{i}.jpg"
        )

path = "obj.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
img_path  = [['image_0.jpg'], ['image_1.jpg'],['image_2.jpg'],['image_3.jpg']]
video_path = [['video.mp4']]

def show_preds_image(image_path):
    image = cv2.imread(image_path)

    results = model(image_path)

    pandas_result = results.pandas().xyxy[0]

    array_results = pandas_result.to_numpy()

    array_results = array_results.tolist()

    array_bounding_box= []

    array_model_result = []

    array_model_confidence = []

    for item in array_results:
        array_bounding_box.append([item[0],item[1],item[2],item[3]])
        array_model_result.append(item[6])
        array_model_confidence.append(str(round(item[4],1)*100))

    for numbers in range(len(array_model_result)):
      x1, y1 = int(math.floor(array_bounding_box[numbers][0])), int(math.floor(array_bounding_box[numbers][1]))  # top-left corner
      x2, y2 = int(math.floor(array_bounding_box[numbers][2])), int(math.floor(array_bounding_box[numbers][3]))  # bottom-right corner

      # draw a rectangle over the image using the bounding box coordinates
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0,255), 1)
      cv2.putText(image, array_model_result[numbers] + " " +array_model_confidence[numbers] + "%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Leprosy Detection",
    examples=img_path,
    cache_examples=False,
)



# def show_preds_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             frame_copy = frame.copy()
#             outputs = model.predict(source=frame)
#             results = outputs[0].cpu().numpy()
#             for i, det in enumerate(results.boxes.xyxy):
#                 cv2.rectangle(
#                     frame_copy,
#                     (int(det[0]), int(det[1])),
#                     (int(det[2]), int(det[3])),
#                     color=(0, 0, 255),
#                     thickness=2,
#                     lineType=cv2.LINE_AA
#                 )
#             yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

# inputs_video = [
#     gr.components.Video(type="filepath", label="Input Video"),

# ]
# outputs_video = [
#     gr.components.Image(type="numpy", label="Output Image"),
# ]
# interface_video = gr.Interface(
#     fn=show_preds_video,
#     inputs=inputs_video,
#     outputs=outputs_video,
#     title="Pothole detector",
#     examples=video_path,
#     cache_examples=False,
# )


gr.TabbedInterface(
    [interface_image],
    tab_names=['Image inference']
).queue().launch()