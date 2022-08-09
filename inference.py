import dataset
import model
import utils

import torch
from torchvision.transforms.functional import to_pil_image

infer_dir = '../FUNSD/test'
model_path = './pths/best_00_0.36.pth'
batch_size = 1

infer_dataset = dataset.FUNSD_Dataset(infer_dir,inference=True)
infer_dataloader = dataset.DataLoader(infer_dataset,batch_size=batch_size)

trans_model = model.LayoutClassification(64,5,8)
trans_model.load_state_dict(torch.load(model_path))


from torchvision.utils import save_image


for img,boxes,gt_mask,gt_class,gt_mask,raw_imgs,anns in infer_dataloader:
    with torch.set_grad_enabled(False):
        outputs = trans_model(img,boxes)
    
    for b in range(batch_size):
        raw_img = to_pil_image(raw_imgs[0])
        pred_mask = torch.zeros(raw_img.size[1],raw_img.size[0])
        print(outputs[b])
        print(torch.argmax(outputs[b],axis=1))
        for box,cls in zip(boxes[b],torch.argmax(outputs[b],axis=1)):
            x_min,y_min,_,_,x_max,y_max,_,_,_ = box
            pred_mask[int(y_min):int(y_max),int(x_min):int(x_max)] = cls
        utils.plot_boxes(raw_img,boxes[b],text = torch.argmax(outputs[b],axis=1),)
        raw_img.save('11.png')

        b_list = []
        for box,label in anns:
            x_min,y_min,x_max,y_max = box
            b_list.append([x_min,y_min,x_max,y_min,x_max,y_max,x_min,y_max])
            cropped = pred_mask[y_min:y_max,x_min:x_max]
            num_bg = sum(sum(cropped==0))
            num_other = sum(sum(pred_mask[int(y_min):int(y_max),int(x_min):int(x_max)]==1))
            num_header = sum(sum(pred_mask[int(y_min):int(y_max),int(x_min):int(x_max)]==2))
            num_question = sum(sum(pred_mask[int(y_min):int(y_max),int(x_min):int(x_max)]==3))
            num_answer = sum(sum(pred_mask[int(y_min):int(y_max),int(x_min):int(x_max)]==4))
            score_list = [num_bg,num_other,num_header,num_question,num_answer]
            class_num = score_list.index(max(score_list))

            print('---------------')
            print(score_list)
            print(class_num)
            print(label)
        utils.plot_boxes(raw_img,b_list,outline=(255,0,0),text=list(map(lambda x:x[1],anns)),text_color=(255,0,0))
        raw_img.save('22.png')
        
    break