"""
Open-set detection and segmentation module originally designed for spot robot picking operation.
Given an image, it returns the bounding boxes of the objects of interest, and if the full model is usedsegments the region of the object within the bounding box,
to return a segmentation mask and a central point of the object.


Author: Dimitrios Arapis (DTAI)
Date: 2023-05-31
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import supervision as sv
from typing import List
from scipy.ndimage.measurements import center_of_mass
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model
from supervision import Detections

class NovoOpenDetector:
    
    def __init__(self, model, classes, box_threshold, text_threshold):   
        # Setting up paths and parameters
        self.model = model
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.HOME = os.path.dirname(os.path.abspath(__file__))
        SAM_CHECKPOINT_PATH = os.path.join(self.HOME, "weights", "sam_vit_b_01ec64.pth")
        GROUNDING_DINO_CONFIG_PATH = os.path.join(self.HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(self.HOME, "weights", "groundingdino_swint_ogc.pth")

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CLASSES = classes
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        
        #Loading segment anything model (for segmentation masks)
        SAM_ENCODER_VERSION = "vit_b"
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    def segment(self, image, xyxy):
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def enhance_class_name(self, class_names):
        return [
            f"{class_name}"
            for class_name
            in class_names
        ]
    # Main function
    
    def detect(self, image):
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.enhance_class_name(class_names=self.CLASSES),
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD
        )
        
        #If detections exists
        if len(detections) == 0:
            return (image, None, None, None)
        else:
            # Just keep most confident detection
            if len(detections) >= 2:
                # Get the attributes of the most confident detection for simplicity in picking
                max_confidence_index = np.argmax(detections.confidence)
                xyxy = detections.xyxy[max_confidence_index].reshape(1, -1)
                mask = None
                confidence = detections.confidence[max_confidence_index].reshape(1)
                class_id = detections.class_id[max_confidence_index].reshape(1)
                tracker_id = detections.tracker_id[max_confidence_index] if detections.tracker_id is not None else None

                # Create a new instance of the Detection class with the most confident detection
                detections = Detections(xyxy=xyxy, mask=mask, confidence=confidence, class_id=class_id, tracker_id=tracker_id)


            x_min, y_min, x_max, y_max = detections.xyxy[0] 
            centroid_x = (x_max - x_min) / 2 + x_min
            centroid_y = (y_max - y_min) / 2 + y_min
            centroid = (int(centroid_x), int(centroid_y))
            
            
            labels = [
                f"{self.CLASSES[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _ 
                in detections]
            
            object,confidence = labels[0].split(" ")
                
            box_annotator = sv.BoxAnnotator()
            
            annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            
            if self.model == 'full': #
                #FOR FULL MODEL YOU NEED GPU/CUDA
                # convert detections to masks
                detections.mask = self.segment(
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )

                mask_annotator = sv.MaskAnnotator()
                annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
                centroid = center_of_mass(detections.mask[0])
                centroid = (int(centroid[1]), int(centroid[0])) #inverse these as they come back y,x. Also integer for manipulation
                    
            cv2.circle(annotated_image, (int(centroid[0]), int(centroid[1])), 15, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(self.HOME, "results", 'result.jpg'), annotated_image)
            return (annotated_image, object, confidence, centroid)  
