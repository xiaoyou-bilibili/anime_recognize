# 动漫人脸识别项目(开发中)

## 数据集准备

## 模型训练

### yolov5 模型训练
```bash
# 首先我们克隆一下yolov5项目
cd yolov5
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5m.pt
```

