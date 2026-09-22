from .blazepose import BlazePose
from .movinet_a0 import MoViNetA0
from .tsm_mobilenetv2 import TSMMobileNetV2
from .vitpose import ViTPose


def create_models():
    return [
        BlazePose(),
        MoViNetA0(),
        TSMMobileNetV2(),
        ViTPose(),
    ]
