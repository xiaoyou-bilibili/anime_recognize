import torch
import torchvision.transforms as T


class Config:
    # network settings
    backbone = 'fmobile'  # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    # 特征维度，这里我们设置为512维
    embedding_size = 512
    drop_ratio = 0.5

    # 数据处理部分
    # 输入模型的形状
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 数据集地址（训练的数据集地址）
    train_root = 'data/recognize/images'
    # test_root = "/data/lfw-align-128"
    # test_list = "/data/lfw_test_pair.txt"

    # 模型地址
    model_path = 'model/23.pth'

    # training settings
    checkpoints = "checkpoints"
    restore = False
    restore_model = ""
    test_model = "checkpoints/23.pth"

    train_batch_size = 64
    test_batch_size = 60

    # 迭代轮次
    epoch = 24
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss'  # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 4  # dataloader


config = Config()
