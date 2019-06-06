from __future__ import division
from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import *
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import DataLoader

import numpy as np
from threading import Thread


class Yolov3:
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4
    _thread_cnt = 5
    model = None
    cuda = True

    def __init__(self, img_size=416, conf_thres=0.8, nms_thres=0.4, cuda=True):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        if torch.cuda.is_available():
            self.cuda = cuda
        else:
            self.cuda = False
        model = Darknet('yolo/config/yolov3.cfg', img_size=self.img_size)
        model.load_weights('yolo/weights/yolov3.weights')
        model.eval()
        model.share_memory()

        if self.cuda:
            torch.cuda.empty_cache()
            model.cuda()

        self.model = model

    def yolo(self, img_list, return_mask=True, classes=(0, 15, 16, 25, 26, 28, 37)):
        masks = []  # Stores image paths
        dataloader = DataLoader(ImageList(img_list, img_size=self.img_size),
                                batch_size=25, shuffle=False, num_workers=1)
        tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        img_detections = []  # Stores detections for each image index
        detections_result = []

        img_size = Image.open(img_list[0]).size
        img_size = (img_size[1],img_size[0])
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = input_imgs.type(tensor)
            if self.cuda:
                input_imgs = input_imgs.cuda()
            # # Get detections
            # with torch.no_grad():
            #     detections = model(input_imgs)
            #     detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
            # # Save image and detections
            # img = np.array(Image.open(img_paths[0]))
            # masks.append(torch.ones(img.shape[0], img.shape[1]))
            # img_detections.extend(detections)
            with torch.no_grad():

                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
                img_detections.append(detections)
            if return_mask:
                for i in range(len(img_paths)):
                    masks.append(torch.ones(img_size[0], img_size[1]))
            # for proc in psutil.process_iter():
            #     print(proc.open_files())
        #     if len(threads) >= self._thread_cnt:
        #         for thread in threads:
        #             thread.join()
        #             img_detections.append(thread.detections)
        #         threads = []
        # for thread in threads:

        for indx in range(0, len(img_detections)):

            if img_detections[indx] is not None:
                if return_mask:
                    mask = masks[indx]
                else:
                    detections = img_detections[indx]
                # The amount of padding that was added
                pad_x = max(img_size[0] - img_size[1], 0) * (self.img_size / max(img_size))
                pad_y = max(img_size[1] - img_size[0], 0) * (self.img_size / max(img_size))
                # Image height and width after padding is removed
                unpad_h = self.img_size - pad_y
                unpad_w = self.img_size - pad_x

                # Draw bounding boxes and labels of detections
                for indx2 in range(0, len(detections)):
                    res = []
                    if detections[indx2] is not None:
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[indx2]:
                            if cls_pred not in classes:
                                continue
                            if res is None:
                                res = []
                            # Rescale coordinates to original dimensions
                            box_h = int((((y2 - y1) / unpad_h) * img_size[0]).item())
                            box_w = int((((x2 - x1) / unpad_w) * img_size[1]).item())
                            y1 = int((((y1 - pad_y // 2) / unpad_h) * img_size[0]).item())
                            x1 = int((((x1 - pad_x // 2) / unpad_w) * img_size[1]).item())

                            if return_mask:
                                mask[y1:(y1 + box_h), x1:(x1 + box_w)] = 0

                            else:
                                res.append((int(x1), int(y1), int(x1 + box_w), int(y1 + box_h), conf.item(), cls_conf.item(), cls_pred.item()))
                    if not return_mask:
                        if len(res) == 0:
                            res = None
                        detections_result.append(res)
                if return_mask:
                       masks[indx] = mask

        if return_mask:
            return masks
        else:
            return detections_result


class YoloThread(Thread):

    def __init__(self, model, img, conf_thres, nms_thres):
        self.model = model
        self.detections = None
        self.img = img
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        super(YoloThread, self).__init__()

    def run(self):
        # Get detections
        print(self.img.size())
        with torch.no_grad():
            detections = self.model(self.img)
            detections = non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        # Save image and detections
        print(len(detections))
        self.detections = detections
