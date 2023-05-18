from threading import Thread
from copy import deepcopy

import traceback
import cv2
from queue import Queue
import time
from imutils.video import VideoStream
import imutils
import numpy as np

import pafy
# import matplotlib.pyplot as plt
# from utils import *
# from darknet import Darknet

class Camera(Thread):
    def __init__(self, url, ques): # normalQue, detectedQue):

        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        self.vs = VideoStream(src=best.url).start()

        Thread.__init__(self)
        self.__cam = self.vs
        self.ques = ques
        self.__shouldStop = False

    def __del__(self):
        self.__cam.release()
        print('Camera released')

    def run(self):
        while True:
            frame = self.__cam.read()

            if frame is not None:
                #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                for que in self.ques:
                    que.put(frame)

            if self.__shouldStop:
                break

    def stopCamera(self):
        self.__shouldStop = True


class NormalVideoStream(Thread):
    def __init__(self, framesQue):
        Thread.__init__(self)
        self.__frames = framesQue
        self.__img = None
        self.res = 240
        self.on = False


    def run(self):
        while True:
            if self.__frames.empty():
                continue
            self.__img = self.__frames.get()

    def gen(self):
        while True:
            try:
                if self.__img is None:
                    print('Normal stream frame is none')
                    continue
                frame = imutils.resize(self.__img, width=self.res, height=self.res)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #frame = cv2.GaussianBlur(frame, (7, 7), 0)
                # Detects cars of different sizes in the input image
                # cars = self.car_cascade.detectMultiScale(frame, 1.1, 1)
                      
                # To draw a rectangle in each cars
                # for (x,y,w,h) in cars:
                #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                  
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            except GeneratorExit:
                print('closed')
                return
            except:
                traceback.print_exc()
                print('Normal video stream genenation exception')

    def enable(self, do: bool):
        self.on = do

class FPSOverlayVideoStream(Thread):
    # The documentation says that get(CAP_PROP_FPS)
    # or get(CV_CAP_PROP_FPS) gives the frames per second.
    # Now that is true for video files, but not for webcams. 
    # For webcams and many other connected cameras, 
    # you have to calculate the frames per second manually

    def __init__(self, framesQue):
        Thread.__init__(self)
        self.__frames = framesQue
        self.__img = None
        self.fps = 0
        self.num_frames = 10
        self.on = False


    def run(self):
        num_frames = self.num_frames
        start = time.time()
        while True:
            if self.__frames.empty():
                continue
            if num_frames == 0:
                num_frames = self.num_frames
                # End time
                end = time.time()
                # Time elapsed
                seconds = end - start
                # Calculate frames per second
                self.fps = self.num_frames / seconds
                start = time.time()
            num_frames -= 1
            self.__img = self.__frames.get()



    def gen(self):
        while True:
            try:
                if self.__img is None:
                    print('FPS stream frame is none')
                    continue

                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,500)
                fontScale              = 1
                fontColor              = (255,255,255)
                thickness              = 2
                lineType               = 2

                cv2.putText(self.__img,f"FPS {self.fps}", 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

                 
                (flag, encodedImage) = cv2.imencode(".jpg", self.__img)
                if not flag:
                    continue
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            except GeneratorExit:
                return
            except:
                traceback.print_exc()
                print('Normal video stream genenation exception')

    def enable(self, do: bool):
        self.on = do


class ObjectsDetectionVideoStream(Thread):
    def __init__(self,
            framesQue: Queue,
            cam: Camera = None,
            objects=['car'], 
            count: bool = False,
            use_tiny_model: bool = True,
            ):

        Thread.__init__(self)
        
        self.__frames = framesQue
        self.__img = None
        self.on = False
        self.count = count
        # self.__faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.vs = cam.vs

        self.w = self.vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.h = self.vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        self.CONFIDENCE = 0.2
        self.SCORE_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.5
        tiny = '-tiny' if use_tiny_model else ''
        self.config_path = f"data/yolov3{tiny}.cfg"
        self.weights_path = f"data/yolov3{tiny}.weights"
        self.labels = open("data/coco.names").read().strip().split("\n")
        self.ids_to_display = []
        for i, obj in enumerate(self.labels):
            if obj in objects:
                self.ids_to_display.append(i)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.res = 416 if use_tiny_model else 618 # hardcoded from cfg

        # -------------- init for torch ------ 
        # self.nms_threshold = 0.4
        # self.iou_threshold = 0.4
        # cfg_file = "data/yolov3.cfg"
        # weight_file = "data/yolov3.weights"
        # namesfile = "data/coco.names"
        # self.m = Darknet(cfg_file)
        # self.m.load_weights(weight_file)
        # class_names = load_class_names(namesfile)

        # ----- for cascade ---- 
        # Trained XML classifiers describes some features of some object we want to detect
        # self.car_cascade = cv2.CascadeClassifier('cars.xml')

    def run(self):
        while True:
        # print(self.__frames.qsize(), end="//")
            if self.__frames.empty():
                continue

            if self.on:
                self.__img = self.__detectObjects()
            else:
                self.__img = self.__frames.get()

    def gen(self):
        while True:
            try:
                if self.__img is None:
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", self.__img)

                if not flag:
                    continue
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            except GeneratorExit:
                self.enable(False)
                return
            except:
                traceback.print_exc()
                print('Detection video stream genenation exception')

    def enable(self, do: bool):
        self.on = do
    
    # def detectObjectsUsingGPU(self):
    #     # current PC calculated in 117 sec. maybe try on supporting CUDA :)
    #     retImg = None
    #     try:
    #         start = time.perf_counter()
    #         original_image = self.__frames.get()
    #         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(original_image, (self.m.width, self.m.height))
    #         boxes = detect_objects(self.m, img, self.iou_threshold, self.nms_threshold)
    #         plot_boxes(original_image, boxes, class_names, plot_labels=True)
    #         retImg = original_image
    #     except:
    #         traceback.print_exc()
    #         print('Object detection exception')
    #     return retImg
        
    def __detectObjects(self):
        retImg = None
        try:
    
            image = self.__frames.get()
            w      = self.w
            h      = self.h
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.res, self.res), swapRB=True, crop=False)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.net.setInput(blob)
            
            ln = self.net.getLayerNames()
            ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
            start = time.perf_counter()
            layer_outputs = self.net.forward(ln)

            font_scale = 1
            thickness = 1
            boxes, confidences, class_ids = [], [], []
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > self.CONFIDENCE:
                        box = detection[:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.SCORE_THRESHOLD, self.IOU_THRESHOLD)
            count = 0
            if len(idxs) > 0:
                for i in idxs.flatten():
                    if class_ids[i] not in self.ids_to_display: #cars
                        continue
                    count += 1
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    color = [int(c) for c in self.colors[class_ids[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                    text = f"{self.labels[class_ids[i]]}: {confidences[i]:.2f}"
                    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                    text_offset_x = x
                    text_offset_y = y - 5
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                    overlay = image.copy()
                    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            if self.count:
                cv2.putText(image, f"Count: {count}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,255,255), thickness=5)
            retImg = image
            time_took = time.perf_counter() - start
            print(f"took: {time_took:.2f}s")

        except:
            traceback.print_exc()
            print('Car detection exception')

        return retImg
