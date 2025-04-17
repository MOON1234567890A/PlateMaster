import numpy as np
import cv2, os, argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from plate_number import random_select, generate_plate_number_white, generate_plate_number_yellow_xue
from plate_number import generate_plate_number_black_gangao, generate_plate_number_black_shi, generate_plate_number_black_ling
from plate_number import generate_plate_number_blue, generate_plate_number_yellow_gua
from plate_number import letters, digits
from PIL import ImageFont
from PIL import Image, ImageDraw, ImageFont
from plate_type import PlateConfig
from plate_type import PlateType 
def get_location_data(length=7, split_id=1, height=140):
    """
    获取车牌号码在底牌中的位置
    length: 车牌字符数，7或者8，7为普通车牌、8为新能源车牌
    split_id: 分割空隙
    height: 车牌高度，对应单层和双层车牌
    """
    # 字符位置
    location_xy = np.zeros((length, 4), dtype=np.int32)

    # 单层车牌高度
    if height == 140:
        # 单层车牌，y轴坐标固定
        location_xy[:, 1] = 25
        location_xy[:, 3] = 115
        # 螺栓间隔
        step_split = 34 if length == 7 else 49
        # 字符间隔
        step_font = 12 if length == 7 else 9

        # 字符宽度
        width_font = 45
        for i in range(length):
            if i == 0:
                location_xy[i, 0] = 15
            elif i == split_id:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_split
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            # 新能源车牌
            if length == 8 and i > 0:
                width_font = 43
            location_xy[i, 2] = location_xy[i, 0] + width_font
    else:
        # 双层车牌第一层
        location_xy[0, :] = [110, 15, 190, 75]
        location_xy[1, :] = [250, 15, 330, 75]

        # 第二层
        width_font = 65
        step_font = 15
        for i in range(2, length):
            location_xy[i, 1] = 90
            location_xy[i, 3] = 200
            if i == 2:
                location_xy[i, 0] = 27
            else:
                location_xy[i, 0] = location_xy[i - 1, 2] + step_font
            location_xy[i, 2] = location_xy[i, 0] + width_font

    return location_xy


def copy_to_image_multi(img, font_img, bbox, color="black"):
    """
    将字符图 font_img 贴到底图 img 的 bbox 区域，颜色支持名称字符串如 'red'、'yellow'、'white'
    
    :param img: 彩色底图
    :param font_img: 灰度字符图（白底黑字）
    :param bbox: 字符区域 (x1, y1, x2, y2)
    :param color: 字体颜色名 (如 'black', 'white', 'red', 'yellow')
    :return: 合成后的图像
    """

    COLOR_MAP = {
        "black": (0, 0, 0),
        "white": (0, 0, 0),
        "red":   (0, 0, 255),
        "blue":  (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
    }

    x1, y1, x2, y2 = bbox
    font_img = cv2.resize(font_img, (x2 - x1, y2 - y1))

    img_crop = img[y1: y2, x1: x2]

    # 获取字体颜色 RGB 值，默认为黑色
    color_rgb = COLOR_MAP.get(color.lower(), (0, 0, 0))

    # 字符区域 mask
    mask = font_img < 200

    # 字体区域上色
    img_crop[mask] = color_rgb

    return img




class MultiPlateGenerator:
    def __init__(self, adr_plate_model, adr_font):
        # 车牌底板路径
        self.adr_plate_model = adr_plate_model
        # 车牌字符路径
        self.adr_font = adr_font
        self.font_ch = ImageFont.truetype("./font/platech.ttf", 180, 0)  # 中文字体格式
        self.font_en = ImageFont.truetype('./font/platechar.ttf', 240, 0)  # 英文字体格式
        # 车牌字符图片，预存处理
        self.font_imgs = {}
        font_filenames = glob(os.path.join(adr_font, '*jpg'))
        for font_filename in font_filenames:
            font_img = cv2.imdecode(np.fromfile(font_filename, dtype=np.uint8), 0)

            if '140' in font_filename:
                font_img = cv2.resize(font_img, (45, 90))
            elif '220' in font_filename:
                font_img = cv2.resize(font_img, (65, 110))
            elif font_filename.split('_')[-1].split('.')[0] in letters + digits:
                font_img = cv2.resize(font_img, (43, 90))
            self.font_imgs[os.path.basename(font_filename).split('.')[0]] = font_img

        # 字符位置
        self.location_xys = {}
        for i in [7, 8]:
            for j in [1, 2, 4]:
                for k in [140, 220]:
                    self.location_xys['{}_{}_{}'.format(i, j, k)] = \
                        get_location_data(length=i, split_id=j, height=k)

    def get_char_image(self, char:str,pos:int ,cfg: PlateConfig) -> np.ndarray:
        """
        根据字符获取字体图像，优先使用缓存，没有则使用字体动态绘制。
        支持位置感知，如可根据 pos 或长度决定备用字体风格。
        """
        key = f'{cfg.plate_height}_{char}'
        if key in self.font_imgs:
            return self.font_imgs[key]
        
        # 新能源车牌字符特殊缓存
        if cfg.length == 8 and f'green_{char}' in self.font_imgs:
            return self.font_imgs[f'green_{char}']
        
        # 尝试旧逻辑 key fallback（兼容 220_up_X 之类的）
        if pos < 2 and f'220_up_{char}' in self.font_imgs:
            return self.font_imgs[f'220_up_{char}']
        if pos >= 2 and f'220_down_{char}' in self.font_imgs:
            return self.font_imgs[f'220_down_{char}']

        """
        生成刚好包住字体的灰度图像（白底黑字，无边距）
        :param char: 要绘制的中文字符
        :return: np.ndarray 灰度图
        """
        # 先创建一个临时大画布
        temp_img = Image.new("L", (500, 500), 255)
        draw = ImageDraw.Draw(temp_img)

        # 获取字符的边界框
        bbox = draw.textbbox((0, 0), char, font=self.font_ch)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 创建刚好包住字符的新画布
        tight_img = Image.new("L", (text_w, text_h), 255)
        tight_draw = ImageDraw.Draw(tight_img)

        # 在 (0, -bbox[1]) 位置绘制字符，让它刚好 fit 到画布顶端
        tight_draw.text((0, -bbox[1]), char, fill=0, font=self.font_ch)

        # return np.array(tight_img)
        # return np.array(tight_img)
        img=tight_img
        # 根据配置缩放为最终输出尺寸
        if cfg.length == 8:
            img = img.resize((43, 90))
        elif cfg.plate_height == 140:
            img = img.resize((45, 90))
        elif cfg.plate_height == 220:
            img = img.resize((65, 110))

        return np.array(img)


    # 获取字符位置
    def get_location_multi(self, plate_number, height=140):
        length = len(plate_number)
        if '警' in plate_number or '应'in plate_number:
            split_id = 1
        elif '使' in plate_number:
            split_id = 4
        else:
            split_id = 2
        return self.location_xys['{}_{}_{}'.format(length, split_id, height)]

    # 随机生成车牌号码，获取底板颜色、单双层
    def generate_plate_number(self):
        rate = np.random.random(1)
        if rate > 0.4:
            # 蓝牌
            plate_number = generate_plate_number_blue(length=random_select([7]))
        else:
            # 白牌、黄牌教练车、黄牌挂车、黑色港澳、黑色使、领馆
            generate_plate_number_funcs = [generate_plate_number_white,
                                           generate_plate_number_yellow_xue,
                                           generate_plate_number_yellow_gua,
                                           generate_plate_number_black_gangao,
                                           generate_plate_number_black_shi,
                                           generate_plate_number_black_ling]
            plate_number = random_select(generate_plate_number_funcs)()
            plate_number =random_select([generate_plate_number_yellow_gua])()
        # 车牌底板颜色
        bg_color = random_select(['blue'] + ['yellow'])

        if len(plate_number) == 8:
            bg_color = random_select(['green_car'] * 10 + ['green_truck'])
        elif len(set(plate_number) & set(['使', '领', '港', '澳'])) > 0:
            bg_color = 'black'
        elif '警' in plate_number or plate_number[0] in letters:
            bg_color = 'white'
        elif len(set(plate_number) & set(['学', '挂'])) > 0:
            bg_color = 'yellow'

        is_double = random_select([False] + [True] * 3)

        if '使' in plate_number:
            bg_color = 'black_shi'

        if '挂' in plate_number:
            # 挂车双层
            is_double = True
        elif len(set(plate_number) & set(['使', '领', '港', '澳', '学', '警'])) > 0 \
                or len(plate_number) == 8 or bg_color == 'blue':
            # 使领港澳学警、新能源、蓝色都是单层
            is_double = False

        # special，首字符为字母、单层则是军车
        if plate_number[0] in letters and not is_double:
            bg_color = 'white_army'

        return plate_number, bg_color, is_double

    # 随机生成车牌图片
    def generate_plate(self, enhance=False):
        plate_number, bg_color, is_double = self.generate_plate_number()
        bg_color='yellow'
        is_double=True
        height = 220 if is_double else 140

        # 获取底板图片
        # print(plate_number, height, bg_color, is_double)
        number_xy = self.get_location_multi(plate_number, height)
        img_plate_model = cv2.imread(os.path.join(self.adr_plate_model, '{}_{}.PNG'.format(bg_color, height)))
        img_plate_model = cv2.resize(img_plate_model, (440 if len(plate_number) == 7 else 480, height))

        for i in range(len(plate_number)):
            if len(plate_number) == 8:
                # 新能源
                font_img = self.font_imgs['green_{}'.format(plate_number[i])]
            else:
                if '{}_{}'.format(height, plate_number[i]) in self.font_imgs:
                    font_img = self.font_imgs['{}_{}'.format(height, plate_number[i])]
                else:
                    # 双层车牌字体库
                    if i < 2:
                        font_img = self.font_imgs['220_up_{}'.format(plate_number[i])]
                    else:
                        font_img = self.font_imgs['220_down_{}'.format(plate_number[i])]

            # 字符是否红色
            if (i == 0 and plate_number[0] in letters) or plate_number[i] in ['警', '使', '领']:
                is_red = True
            elif i == 1 and plate_number[0] in letters and np.random.random(1) > 0.5:
                # second letter of army plate
                is_red = True
            else:
                is_red = False

            if enhance:
                k = np.random.randint(1, 6)
                kernel = np.ones((k, k), np.uint8)
                if np.random.random(1) > 0.5:
                    font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                else:
                    font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))

            # 贴上底板
            img_plate_model = copy_to_image_multi(img_plate_model, font_img,
                                                  number_xy[i, :], bg_color, is_red)
            # # 底牌加车牌文字
            # img = cv2.bitwise_and(font_img, img_plate_model)
        img_plate_model = cv2.blur(img_plate_model, (3, 3))

        return img_plate_model, number_xy, plate_number, bg_color, is_double
   
    def generate_plate_special(self, plate_number, cfg: PlateConfig,bg=False,enhance=False):
        """
        生成特定号码、颜色车牌
        :param plate_number: 车牌号码
        :param is_double: 是否双层
        :param bg_color: 背景颜色，black/black_shi（使领馆）/blue/green_car（新能源轿车）/green_truck（新能源卡车）/white/white_army（军队）/yellow
        :param enhance: 图像增强
        :return: 车牌图
        """
        height = 220 if cfg.is_double else 140
        plate_width = 440 if len(plate_number) == 7 else 480
        # print(plate_number, height, bg_color, is_double)s
        number_xy = self.get_location_multi(plate_number, height)
        if bg==False:
            img_plate_model = np.ones((height, plate_width, 3), dtype=np.uint8) * 255
        else:
            img_plate_model = cv2.imread(cfg.bg_path)
        img_plate_model = cv2.resize(img_plate_model, (440 if len(plate_number) == 7 else 480, height))

        for i in range(len(plate_number)):
            char = plate_number[i]
            font_img = self.get_char_image(char,i,cfg)


            if enhance:
                k = np.random.randint(1, 6)
                kernel = np.ones((k, k), np.uint8)
                if np.random.random(1) > 0.5:
                    font_img = np.copy(cv2.erode(font_img, kernel, iterations=1))
                else:
                    font_img = np.copy(cv2.dilate(font_img, kernel, iterations=1))
            img_plate_model = copy_to_image_multi(img_plate_model, font_img,
                                                  number_xy[i, :], cfg.fg_color[i]) 
        img_plate_model = cv2.blur(img_plate_model, (3, 3))

        return img_plate_model


def parse_args():
    parser = argparse.ArgumentParser(description='中国车牌生成器')
    parser.add_argument('--number', default=10, type=int, help='生成车牌数量')
    parser.add_argument('--save-adr', default='multi_val', help='车牌保存路径')
    args = parser.parse_args()
    return args


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


if __name__ == '__main__':
    license_generator = MultiPlateGenerator('plate_model', 'font_model')
    pt = PlateType('dian')
    cfg = pt.get_config()
    img = license_generator.get_char_image("电",0,cfg)
    plt.imshow(img, cmap='gray')
    plt.title("生成的字符图像")
    plt.axis('off')
    plt.show()
    from PIL import ImageFont

    font = ImageFont.truetype("./font/platech.ttf", 180)

    # 查看 bbox 是否正常
    char = "电"
    try:
        bbox = font.getbbox(char)
        print(f"字符 '{char}' bbox: {bbox}")
        if bbox[2] - bbox[0] < 5:  # 宽度很小
            print(f"[问题] 字体可能不支持字符 '{char}'，绘制宽度过小")
    except Exception as e:
        print(f"[错误] 字符 '{char}' 绘制失败：{e}")

    # args = parse_args()
    # print(args)
    # # 随机生成车牌
    # print('save in {}'.format(args.save_adr))

    # mkdir(args.save_adr)
    # generator = MultiPlateGenerator('plate_model', 'font_model')

    # for i in tqdm(range(args.number)):
    #     img, number_xy, gt_plate_number, bg_color, is_double = generator.generate_plate()
    #     cv2.imwrite(os.path.join(args.save_adr, '{}_{}_{}.jpg'.format(gt_plate_number, bg_color, is_double)), img)