from abc import ABC,abstractmethod
from event4d.types import ReconstructedFrame,StereoEventSlice

class VisualFrameProvider(ABC):
    @abstractmethod
    def frame(self,index:int)->ReconstructedFrame: ...

class DSECImageBackend(VisualFrameProvider):
    def __init__(self,adapter): self.adapter=adapter
    def frame(self,index): return self.adapter.image(index,"left")

class EventAccumulationBackend(VisualFrameProvider):
    def __init__(self,adapter,window_ms=20): self.adapter=adapter;self.window_ms=window_ms
    def frame(self,index):
        import cv2
        from event4d.visualization.draw import event_accumulation
        t=int(self.adapter.image_timestamps[index])
        packet=self.adapter.events("left",t-self.window_ms*1000,t)
        return ReconstructedFrame(t,cv2.cvtColor(event_accumulation(packet),cv2.COLOR_BGR2GRAY),"event_accumulation")

class E2VIDBackend(VisualFrameProvider):
    def frame(self,index): raise NotImplementedError("E2VID is reserved for the event reconstruction ablation")

class SpikeAdapter(ABC):
    @abstractmethod
    def stereo_slice(self,start_us:int,end_us:int): ...
