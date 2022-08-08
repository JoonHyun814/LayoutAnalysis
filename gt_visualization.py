from PIL import Image
import json
import utils

image_path = '../FUNSD/testing_data/images'
ann_path = '../FUNSD/testing_data/annotations'

img = Image.open(image_path+'/82092117.png').convert('RGB')

with open(ann_path+'/82092117.json','r') as f:
    ann_dict = json.load(f)

print(ann_dict['form'][0])

boxes = []

for gt_box_info in ann_dict['form']:
    x_min,y_min,x_max,y_max = gt_box_info['box']
    boxes.append([x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max])

    text_boxes = []
    for word in gt_box_info['words']:
        x_min,y_min,x_max,y_max = word['box']
        text_boxes.append([x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max])
    
    utils.plot_boxes(img,text_boxes)

utils.plot_boxes(img, boxes)
img.save('gt.png')