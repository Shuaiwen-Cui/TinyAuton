# 真实数据集选项（Real Dataset Options）

## 1. UCI HAR Dataset (Human Activity Recognition)
- **来源**: UCI Machine Learning Repository
- **数据**: 手机传感器数据（加速度计+陀螺仪）
- **类别**: 6种活动（走路、上楼、下楼、坐着、站着、躺着）
- **下载**: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- **特点**: 经典数据集，数据量大，适合MCU部署

## 2. WISDM Dataset (Wireless Sensor Data Mining)
- **来源**: WISDM Lab
- **数据**: 加速度计数据
- **类别**: 多种活动识别
- **下载**: http://www.cis.fordham.edu/wisdm/dataset.php
- **特点**: 真实手机传感器数据

## 3. Opportunity Dataset
- **来源**: Opportunity Activity Recognition Dataset
- **数据**: 多传感器数据（加速度计、陀螺仪、磁力计等）
- **类别**: 日常活动识别
- **下载**: https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition
- **特点**: 数据丰富，但较复杂

## 4. 自定义数据采集
- 使用手机或开发板（如ESP32）采集真实加速度计数据
- 自己定义手势类别
- 完全可控，最适合实际应用

## 推荐方案
对于MCU部署，建议：
1. **教学/测试**: 使用当前的合成数据（已实现）
2. **实际应用**: 使用UCI HAR数据集（简化版）或自己采集数据

