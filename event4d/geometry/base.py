from abc import ABC,abstractmethod
from event4d.types import GeometryFrame

class GeometryBackend(ABC):
    @abstractmethod
    def frame(self,index:int)->GeometryFrame: ...

class SpikeStereoGeometryBackend(GeometryBackend):
    def frame(self,index): raise NotImplementedError("Connect a calibrated spike stereo estimator")
