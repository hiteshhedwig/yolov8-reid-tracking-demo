import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import os
import yaml
from easydict import EasyDict as edict
from pathlib import Path

import supervision as sv
from bytetrack.byte_tracker import BYTETracker
from strongsort.strong_sort import StrongSORT


SAVE_VIDEO = False

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)



class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        self.config_main = "./config.yml"
        self.main_cfg = get_config()
        self.main_cfg.merge_from_file(self.config_main)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

        reid_weights   = Path(self.main_cfg.demo_test.reid_model_path)
          
        if self.main_cfg.demo_test.tracker == "bytetrack":
            tracker_config = "bytetrack/configs/bytetrack.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
     
            self.tracker = BYTETracker(
                track_thresh=cfg.bytetrack.track_thresh,
                match_thresh=cfg.bytetrack.match_thresh,
                track_buffer=cfg.bytetrack.track_buffer,
                frame_rate=cfg.bytetrack.frame_rate
            )
        else :
            tracker_config = "strongsort/configs/strongsort.yaml"
            cfg = get_config()
            cfg.merge_from_file(tracker_config)
    
            self.tracker = StrongSORT (
                reid_weights,
                torch.device("cuda:0"),
                False,
                max_dist=cfg.strongsort.max_dist,
                max_iou_dist=cfg.strongsort.max_iou_dist,
                max_age=cfg.strongsort.max_age,
                max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
                n_init=cfg.strongsort.n_init,
                nn_budget=cfg.strongsort.nn_budget,
                mc_lambda=cfg.strongsort.mc_lambda,
                ema_alpha=cfg.strongsort.ema_alpha,
            )


    def load_model(self):
       
        model = YOLO(self.main_cfg.demo_test.detection_model_path)  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def draw_results(self, frame, results):
        xyxys = []
        confidences = []
        class_ids = []
        detections = []
        boxes = []
        for result in results:
            # return a list of class ids
            class_id = result.boxes.cls.cpu().numpy().astype(int) 
            print(result.boxes)
            if len(class_id) == 0:
                continue

            if len(class_id) > 1:
                class_id = class_id[0]
            
            if class_id == 0:  
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
                boxes.append(result.boxes)
                # Setup detections for visualization
                detections = sv.Detections(
                            xyxy=result.boxes.xyxy.cpu().numpy(),
                            confidence=result.boxes.conf.cpu().numpy(),
                            class_id=result.boxes.cls.cpu().numpy().astype(int),
                            )
        
    
            # Format custom labels
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame, boxes
       
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter(self.main_cfg.demo_test.save_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720))
        # setup tracker
        tracker = self.tracker

        # if tracker is using model then warmup
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        outputs = [None]
        curr_frames, prev_frames = None, None

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            frame, _ = self.draw_results(frame, results)
            
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:  # camera motion compensation
                    tracker.tracker.camera_update(prev_frames, curr_frames)

            for result in results:
                outputs[0] = tracker.update(result,frame)
                for j, (output) in enumerate(outputs[0]):
                    bbox = output[0:4]
                    tracked_id = output[4]
                    # cls = output[5]
                    # conf = output[6]
                    top_left = (
                        int(bbox[-2]-100),
                        int(bbox[1])
                    )
                    cv2.putText(
                        frame,
                        f"ID : {tracked_id}",
                        top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0,255,0), 
                        3
                    )

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('YOLOv8 Detection', frame)
            if SAVE_VIDEO:
                outputvid.write(frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        if SAVE_VIDEO:
            outputvid.release()
        cap.release()
        cv2.destroyAllWindows()
        
        


detector = ObjectDetection(capture_index=0)
detector()
