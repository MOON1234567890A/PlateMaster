from config import PLATE_CONFIGS
from typing import Optional
from dataclasses import dataclass
from typing import List

@dataclass
class PlateConfig:
    plate_type: str
    length: int
    bg_color: str
    bg_path:str
    fg_color: List[str]
    is_double: bool
    st: List[str]
    plate_width: int
    plate_height: int
    

class PlateType:
    def __init__(self, plate_type: str):
        if plate_type not in PLATE_CONFIGS:
            raise ValueError(f"不支持的车牌类型: {plate_type}")
        
        self.config_dict = PLATE_CONFIGS[plate_type]
        self.plate_type = plate_type

        self.len = self.config_dict["len"]
        self.bg_color = self.config_dict["bg_color"]
        self.bg_path =self.config_dict["bg_path"]
        self.fg_color = self.config_dict["fg_color"]
        self.isdouble = self.config_dict["isdouble"]
        self.st = self.config_dict["st"]
        self.plate_height = 220 if self.isdouble else 140
        self.plate_width = 440 if self.len == 7 else 480

    def get_config(self) -> PlateConfig:
        return PlateConfig(
            plate_type=self.plate_type,
            length=self.len,
            bg_color=self.bg_color,
            bg_path=self.bg_path,
            fg_color=self.fg_color,
            is_double=self.isdouble,
            st=self.st,
            plate_width=self.plate_width,
            plate_height=self.plate_height
        )
