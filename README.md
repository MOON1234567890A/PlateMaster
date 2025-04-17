# PlateMaster - 中国车牌生成系统 🇨🇳🚗

PlateMaster 是一个灵活且强大的中国车牌合成图像生成系统，支持多种类型、多样规则和增强方式的车牌生成，广泛适用于计算机视觉任务中的训练数据合成、车牌识别算法测试与评估等场景。

---

## ✨ 功能特色

- ✅ **生成特定车牌**：支持自定义输入字符串，生成指定号码车牌图像
- 📦 **批量生成车牌**：支持批量配置生成多张车牌图像，用于数据集构建
- ⚙️ **规则自定义**：可自定义字符格式、位置、字体样式等生成规则
- 🎨 **多种车牌类型**：
  - 单层 / 双层车牌
  - 蓝色、绿色、黄色、白色、警用、应急等背景类型
- 🌪️ **图像增强支持**：支持对车牌图像加噪声、模糊、亮度调整等增强操作
- 📁 **输出灵活**：支持生成图像保存到指定目录，可配合标签输出

---

## 🔗 致谢与引用

本项目基于以下开源项目进行改进：

- [Pengfei8324/chinese_license_plate_generator](https://github.com/Pengfei8324/chinese_license_plate_generator)
- [zheng-yuwei/license-plate-generator](https://github.com/zheng-yuwei/license-plate-generator)

感谢这些优秀的开源项目，作为基础代码，我们在此基础上进行了扩展与改进，增加了更多自定义功能和生成选项。

---

## 📂 项目结构

```bash
.
├── main.py                     # 主入口：用于生成车牌图像
├── config.py                  # 配置文件：全局参数配置
├── plate_type.py              # 定义各种车牌类型及属性
├── plate_number.py            # 车牌号码生成规则
├── plate_model/               # 字符模板、图像生成核心逻辑
├── font_model/                # 字体加载与处理模块
├── font/                      # 车牌字体资源
├── background/                # 各类车牌底板模板
├── plate_images/              # 已生成车牌图像的默认保存目录
├── generate_multi_plate.py    # 支持批量、多类型车牌生成的脚本
├── augment_image.py           # 图像增强模块：模糊、加噪、变换等
├── images/                    # 示例图片目录
├── LICENSE                    # MIT 开源协议
├── README.md                  # 项目说明文档
```
