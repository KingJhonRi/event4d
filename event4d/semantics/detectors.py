from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from event4d.types import Detection

DEFAULT_LABELS=["person","car","bicycle","motorcycle","bus","truck","traffic sign","traffic light","backpack"]

class GroundingDINODetector:
    def __init__(self, root: Path, labels=None, box_threshold=.35, text_threshold=.25):
        from groundingdino.util.inference import load_model
        self.labels=labels or DEFAULT_LABELS
        self.box_threshold=box_threshold; self.text_threshold=text_threshold
        config=root/"third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights=root/"models/groundingdino/groundingdino_swint_ogc.pth"
        self.model=load_model(str(config),str(weights),device="cpu")

    def detect(self, gray):
        from groundingdino.util.inference import predict
        from groundingdino.util import box_ops
        from groundingdino.datasets import transforms as T
        from PIL import Image
        rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        pil=Image.fromarray(rgb)
        transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        tensor,_=transform(pil,None)
        boxes, scores, phrases=predict(self.model,tensor, " . ".join(self.labels)+" .",self.box_threshold,self.text_threshold,device="cpu")
        h,w=gray.shape
        xyxy=box_ops.box_cxcywh_to_xyxy(boxes)*np.array([w,h,w,h])
        return [Detection(b.detach().cpu().numpy(),str(p),float(s)) for b,s,p in zip(xyxy,scores,phrases)]

class TransformersGroundingDINODetector:
    """Open-vocabulary fallback using the model authors' Hugging Face release."""
    def __init__(self,labels=None,box_threshold=.35):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.labels=labels or DEFAULT_LABELS; self.threshold=box_threshold
        self.processor=AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()

    def detect(self,gray):
        import torch
        from PIL import Image
        image=Image.fromarray(cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB))
        inputs=self.processor(images=image,text=[[x for x in self.labels]],return_tensors="pt")
        with torch.no_grad(): outputs=self.model(**inputs)
        result=self.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=self.threshold,text_threshold=.25,target_sizes=[image.size[::-1]])[0]
        return [Detection(b.cpu().numpy(),str(label),float(score)) for b,score,label in zip(result["boxes"],result["scores"],result["labels"])]

class MockDetector:
    """Diagnostic only; detections are synthetic and must never be called real semantics."""
    def detect(self,gray):
        h,w=gray.shape
        return [Detection(np.array([w*.4,h*.35,w*.6,h*.65]),"mock object",1.0)]
