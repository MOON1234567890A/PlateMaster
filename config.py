PLATE_CONFIGS = {
    "single_blue": {
        "len": 7,
        "bg_color": "blue",
        "fg_color": ["white"] * 7,
        "isdouble": False,
        "st": ['*'] * 7,
        "bg_path": "plate_model/blue_140.PNG"
    },
    "single_yellow": {
        "len": 7,
        "bg_color": "yellow",
        "fg_color": ["black"] * 7,
        "isdouble": False,
        "st": ['*'] * 7,
        "bg_path": "plate_model/yellow_140.PNG"
    },
    "small_new_energy": {
        "len": 8,
        "bg_color": "green",
        "fg_color": ["black"] * 8,
        "isdouble": False,
        "st": ['*'] * 8,
        "bg_path": "plate_model/green_car_140.PNG"
    },
    "police": {
        "len": 7,
        "bg_color": "white",
        "fg_color": ["black"] * 6 + ["red"],
        "isdouble": False,
        "st": ['*'] * 6 + ["警"],
        "bg_path":"plate_model/white_140.PNG"
    },
    "yingji": {
        "len": 8,
        "bg_color": "white",
        "fg_color": ["black", "red", "black", "black", "black", "black", "red", "red"],
        "isdouble": False,
        "st": ['*'] * 6 + ['应急'],
        "bg_path":"plate_model/white_140.PNG"
    },
    "xun": {
        "len": 7,
        "bg_color": "blue",
        "fg_color": ["white"] * 7,
        "isdouble": False,
        "st": ['巡'] + ['*'] * 6,
        "bg_path":"plate_model/blue_140.PNG"
    },
    "dian": {
        "len": 7,
        "bg_color": "green",
        "fg_color": ["black"] * 7,
        "isdouble": False,
        "st": ['电'] + ['*'] * 6,
        "bg_path":"plate_model/green_car_140.PNG"
    },
    "double_yellow_gua": {
        "len": 7,
        "bg_color": "yellow",
        "fg_color": ["black"] * 7,
        "isdouble": True,
        "st": ['*'] * 6 + ['挂'],
        "bg_path":"plate_model/yellow_220.PNG"
    },
    "double_yellow": {
        "len": 7,
        "bg_color": "yellow",
        "fg_color": ["black"] * 7,
        "isdouble": True,
        "st": ['*'] * 7,
        "bg_path":"plate_model/yellow_220.PNG"
    },
}
