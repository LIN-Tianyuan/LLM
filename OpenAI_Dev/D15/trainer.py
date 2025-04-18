"""
基于中文BERT模型进行文本分类任务的训练主程序
"""

import torch
from MyData import Mydataset            # Mydataset：自定义的数据集类，来自你的 MyData.py
from torch.utils.data import DataLoader
from net import Model                   # 前面定义的分类模型，来自 net.py
# BertTokenizer：BERT分词器，用于将文本转成模型输入。AdamW：是BERT推荐使用的优化器，带有权重衰减。
from transformers import BertTokenizer, AdamW

# 定义训练设备
"""
自动使用GPU（如果有）进行训练。
设置训练100轮（EPOCH=100）。
加载中文BERT的分词器。
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100

token = BertTokenizer.from_pretrained("bert-base-chinese")


# 自定义函数，对数据进行编码处理
# data 是一个 batch 的样本，里面是 (文本, 标签) 的格式。
def collate_fn(data):
    sentences = [i[0] for i in data]
    label = [i[1] for i in data]
    # 编码
    # 将每一条句子编码为 BERT 的输入格式，转为张量（pt）。
    # 最大长度设为350，多的截断、不足补0。
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentences,
        truncation=True,
        padding="max_length",
        max_length=350,
        return_tensors="pt",
        return_length=True,
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
train_dataset = Mydataset("train")
# 创建DataLoader
"""
加载自定义训练集（应该在 Mydataset 中定义了读取和 __getitem__ 的方式）。
batch_size=32，每次喂给模型32条数据。
collate_fn 是刚才定义的编码函数。
"""
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)
    # 模型移动到 GPU/CPU。
    model = Model().to(DEVICE)
    # 用 AdamW 优化器更新模型参数。
    optimizer = AdamW(model.parameters(), lr=5e-4)
    # 分类常用的交叉熵损失函数。
    loss_func = torch.nn.CrossEntropyLoss()

    # 开始进入训练状态，分轮训练。
    model.train()
    # 每一轮从 train_loader 中取出一个 batch。
    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将数据放到DEVICE上
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), \
                attention_mask.to(DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)

            # 执行前向计算得到输出
            # 前向传播得到模型输出。
            out = model(input_ids, attention_mask, token_type_ids)
            # 计算损失。
            loss = loss_func(out, labels)

            # 梯度清零 + 反向传播 + 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每5个 batch 打印一次：轮数、batch号、损失值、准确率。
            if i % 5 == 0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                print(epoch, i, loss.item(), acc)
        # 保存模型参数
        # 把当前 epoch 的模型保存到 params 文件夹里。
        torch.save(model.state_dict(), f"params/{epoch}bert.pt")
        print(epoch, "参数保存成功！")
