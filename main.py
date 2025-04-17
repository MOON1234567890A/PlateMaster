# -*- coding: utf-8 -*-
"""
Created on 2019/4/17
File generate_license_plate
@author:ZhengYuwei
1. 产生车牌号：generate_license_plate_number
    1.1 定义车牌号中每一位车牌元素：license_plate_elements
    2.1 产生车牌号标签：针对特定类型的车牌，从车牌元素中选择元素组成车牌号
2 为车牌号产生车牌号图形：generate_chars_image
3. 产生车牌底牌图片：generate_plate_template
4. 加扰动元素进行数据增强，拼装底牌和车牌号图片： augment_image
4. 保存图片
"""
import cv2
import os
import sys
import datetime
from generate_multi_plate import MultiPlateGenerator
from plate_number import generate_plate_number
from plate_type import PlateType 
from augment_image import ImageAugmentation
import matplotlib.pyplot as plt

license_generator = MultiPlateGenerator('plate_model', 'font_model')
class LicensePlateGenerator(object):
    
    @staticmethod
    def generate_license_plate_images(plate_type, batch_size, save_path, shift_index=0,zq=True):
        """ 生成特定数量的、指定车牌类型的车牌图片，并保存到指定目录下
        :param plate_type: 车牌类型
        :param batch_size: 车牌号数量
        :param save_path: txt文件根目录
        :param shift_index: 图片名称保存的前缀偏移量
        :return:
        """
        sys.stdout.write('\r>> 生成车牌号图片...')
        sys.stdout.flush()
        if batch_size==0:
            return
        # 生成车牌号
        pt = PlateType(plate_type)
        cfg = pt.get_config()
        plate_nums = [generate_plate_number(cfg.st) for _ in range(batch_size)]

        # 生成车牌号图片：白底黑字
        plate_images = []
        for plate in plate_nums:
            img = license_generator.generate_plate_special(plate,cfg,bg=not zq)
            plate_images.append(img)
            
        license_template_generator = cv2.imread(cfg.bg_path)
        license_template_generator = cv2.resize(license_template_generator, (cfg.plate_width, cfg.plate_height))  # 注意：宽度在前，高度在后
        # 数据增强及车牌字符颜色修正，并保存
        sys.stdout.write('\r>> 生成车牌图片...')
        sys.stdout.flush()
        prefix_len = 9 # 图片前缀位数，亿
        save_path = os.path.join(save_path, plate_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if zq==False:
            for index, char_image in enumerate(plate_images):
                image_name = str(shift_index + index).zfill(prefix_len) + '_' + plate_nums[index] + '.jpg'
                image_path = os.path.join(save_path, image_name)
                # image = augmentation.augment(char_image)
                # image = cv2.resize(image, (880, 220))
                cv2.imencode('.jpg', char_image)[1].tofile(image_path)
                if (index+1) % 100 == 0:
                    sys.stdout.write('\r>> {} done...'.format(index + 1))
                    sys.stdout.flush()
            return
            
        augmentation = ImageAugmentation(plate_type,'black' in cfg.fg_color,license_template_generator)
        
        # global plate_height
        plate_width = 880
        plate_height =220
        for index, char_image in enumerate(plate_images):
            image_name = str(shift_index + index).zfill(prefix_len) + '_' + plate_nums[index] + '.jpg'
            image_path = os.path.join(save_path, image_name)
            image = augmentation.augment(char_image)
            image = cv2.resize(image, (plate_width, plate_height))
            cv2.imencode('.jpg', image)[1].tofile(image_path)
            if (index+1) % 100 == 0:
                sys.stdout.write('\r>> {} done...'.format(index + 1))
                sys.stdout.flush()
        return
    
    
if __name__ == '__main__':
    # 迭代次数
    iter_times = 500
    # 保存文件夹名称
    file_path = os.path.join(os.getcwd(), 'plate_images')
    os.makedirs(file_path, exist_ok=True)
    start_index = 0
    sys.stdout.write('{}: total {} iterations ...\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                            iter_times))
    sys.stdout.flush()
    for i in range(iter_times):
        sys.stdout.write('\r{}: iter {}...\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i))
        sys.stdout.flush()
        for typ,num in {'double_yellow':100,'double_yellow_gua':100,'dian':0,'xun': 0,'police':0, 'yingji': 0, 'single_blue': 0,'single_yellow':100,'small_new_energy':0}.items():
            LicensePlateGenerator.generate_license_plate_images(typ,
                                                        batch_size=num,
                                                        save_path=file_path,
                                                        shift_index=start_index,
                                                        zq=True
                                                        )
            start_index += num
    sys.stdout.write('\r{}: done...\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iter_times))
    sys.stdout.flush()
