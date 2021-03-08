import sys
sys.path.append("../..")
from datasets import mnist

import paddle

#导入mnist数据集
from paddle.vision.transforms import Compose, Normalize
transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
train_dataset = mnist.MNIST(mode='train', transform=transform)
test_dataset = mnist.MNIST(mode='test', transform=transform)

import model as mod
from paddle.metric import Accuracy
model = paddle.Model(mod.LeNet())   # 用Model封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# 训练模型
model.fit(train_dataset,
        epochs=2,
        batch_size=64,
        verbose=1
    )
    
model.save('LetNet_MNIST_checkpoint/LetNet')
