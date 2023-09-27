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
    'https://www.dropbox.com/scl/fi/xq103ic7ovuuei3l9e8jf/lp1.jpg?rlkey=g7d9khyyc6wplv0ljd4mcha60&dl=1',
    'https://www.dropbox.com/scl/fi/fagkh3gnio2pefdje7fb9/Non_Leprosy_210823_86_jpg.rf.5bb80a7704ecc6c8615574cad5d074c5.jpg?rlkey=ks8afue5gsx5jqvxj3u9mbjmg&dl=1',
    'https://www.dropbox.com/scl/fi/gh4zotrzic5y00ok3crje/Non_Leprosy_210823_46_jpg.rf.76c20cb340114a98618ade07c3e6b413.jpg?rlkey=pxdjlhxipmsd12gr4veyg691v&dl=1',
    'https://www.dropbox.com/scl/fi/r8vgo1xrledlsw7rxq4ar/Tropmed-91-216-g001.jpg?rlkey=6iajn3xoa6zsxtxh4exq4z3p5&dl=1',
    'https://www.dropbox.com/scl/fi/kxv0q49e92h3fr7ihvqbu/Non_Leprosy_210823_8_jpg.rf.e2d44b96e1bb9b5111b780adec5ba94a.jpg?rlkey=g25iq6vbwqs1glusyv1lgv5a2&dl=1'
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
img_path  = [['image_0.jpg'], ['image_1.jpg'],['image_2.jpg'],['image_3.jpg'],['image_4.jpg'],['image_5.jpg'],]
video_path = [['video.mp4']]

def show_preds_image_and_labels(image_path):
    image = cv2.imread(image_path)

    results = model(image_path)

    pandas_result = results.pandas().xyxy[0]

    array_results = pandas_result.to_numpy()

    array_results = array_results.tolist()

    #print raw results
    print("Raw results: ", array_results)

    array_bounding_box= []

    array_model_result = []

    array_model_confidence = []

    #for labelling bounding box
    for item in array_results:
        array_bounding_box.append([item[0],item[1],item[2],item[3]])
        array_model_result.append(item[6])
        array_model_confidence.append(str(round(item[4],1)*100))

    for numbers in range(len(array_model_result)):
      x1, y1 = int(math.floor(array_bounding_box[numbers][0])), int(math.floor(array_bounding_box[numbers][1]))  # top-left corner
      x2, y2 = int(math.floor(array_bounding_box[numbers][2])), int(math.floor(array_bounding_box[numbers][3]))  # bottom-right corner

      if array_model_result[numbers] == "Lep":
        # draw a rectangle over the image using the bounding box coordinates

        #if the value of leprosy conf is < 0.45 then label it as NLp to show the max voting value
        if  float(array_model_confidence[numbers]) > 45.0:

          cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0,255), 1)
          cv2.putText(image, array_model_result[numbers] + " " + array_model_confidence[numbers] + "%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        elif float(array_model_confidence[numbers]) < 45.0:

          cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255,0), 1)
          cv2.putText(image, "Non Lep" + " " + array_model_confidence[numbers] + "%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

      elif array_model_result[numbers] == "Non Lep":
        
        # draw a rectangle over the image using the bounding box coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255,0), 1)
        cv2.putText(image, array_model_result[numbers] + " " + array_model_confidence[numbers] + "%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    #labelling dictionary

    array_results_conf_large = []

    for yolo_results in array_results:
      if yolo_results[4] > 0.45:
        array_results_conf_large.append(yolo_results)

    print("Large results only: ",array_results_conf_large)

    num_lep = 0
    confidence_lep = 0

    num_non_lep = 0
    confidence_non_lep = 0

    for item in array_results_conf_large:
        if item[6] == "Lep":
          num_lep+=1
          confidence_lep += item[4]
        elif item[6] == "Non Lep":
          num_non_lep+=1
          confidence_non_lep += item[4]

    labels = {}

    #if num_lep is more than non lep
    if num_lep > num_non_lep:
        labels["Leprosy"] = round(confidence_lep/num_lep,2)
        labels["Non Leprosy"] = round(confidence_non_lep/num_non_lep,2) 
    #if num_non_lep is more than lep
    elif num_lep < num_non_lep:
        labels["Leprosy"] = round(confidence_lep/num_lep,2)
        labels["Non Leprosy"] = round(confidence_non_lep/num_non_lep,2)

    #if num_non_lep and num_lep is equal but they are equal coz they are both 0
    elif num_lep == num_non_lep and num_lep == 0:
        labels["Others"] = 0.9

    #if num_non_lep and num_lep is equal but they are equal coz they are both 0
    elif num_lep == num_non_lep:

        #incase of a tie in quantity we compare the mean probability of each
        confidence_lep = round(confidence_lep/num_lep,2)
        confidence_non_lep = round(confidence_non_lep/num_non_lep,2)
        
        if confidence_lep > confidence_non_lep:
        labels["Leprosy"] = confidence_lep
        labels["Non Leprosy"] = confidence_non_lep
        
        elif confidence_lep < confidence_non_lep:
        labels["Leprosy"] = confidence_lep
        labels["Non Leprosy"] = confidence_non_lep
        
        elif confidence_lep == confidence_non_lep:
        labels["Leprosy"] = confidence_lep
        labels["Non Leprosy"] = confidence_non_lep

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB),labels

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
    gr.components.Label(type="json", label="Labels with Confidence"),
]
interface_image = gr.Interface(
    fn=show_preds_image_and_labels,
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


# gr.TabbedInterface(
#     [interface_image],
#     tab_names=['Image inference']
# ).queue().launch()

gr.TabbedInterface(
    [interface_image],
    tab_names=['Image inference']
).queue().launch()