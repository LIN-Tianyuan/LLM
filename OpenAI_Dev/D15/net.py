"""
加载中文 BERT 预训练模型，并在它的基础上构建一个用于分类的下游任务模型
加载一个中文 BERT 预训练模型，把它作为“特征提取器”，再接一个全连接层进行文本分类（2类）。
"""

# 加载预训练模型
from transformers import BertModel  # 使用 transformers 库中的 BertModel 类，加载一个 BERT 预训练模型。
import torch                        # torch 是 PyTorch 框架，用于构建深度学习模型。

# 定义训练设备
# 判断你是否有 NVIDIA 显卡（CUDA），有的话就用 GPU 加速，没有就用 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 这里加载的是 中文版本的 BERT（bert-base-chinese）。
# .to(DEVICE) 表示把模型放到上面选择的设备（GPU 或 CPU）上。
# 这里只是 BERT 主干部分，不包括用于分类的层（也叫做 "下游任务层"）。
pretrained = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)
print(pretrained)


# 定义下游任务模型（将主干网络所提取的特征进行分类）
"""
创建一个新的模型 Model，继承自 torch.nn.Module。
添加了一个全连接层（Linear(768, 2)），将 BERT 的输出维度从 768 映射到 2类，用于分类任务（比如 情感分类：正面/负面）。
"""
class Model(torch.nn.Module):
    # 模型结构设计
    def __init__(self):
        super().__init__()
        # 接收 BERT 的输出，送入一个 Linear 层，输出 2 个类别（比如正面、负面）
        self.fc = torch.nn.Linear(768, 2)

    # forward 是模型调用时的执行逻辑。
    # 用 with torch.no_grad() 包住 BERT，这意味着：BERT 不参与训练，只提取特征（固定住了）。
    # 输入包括：input_ids, attention_mask, token_type_ids，这是 BERT 模型的标准输入。

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 上游任务不参与训练
        # 不计算 BERT 部分的梯度（冻结主干网络，只训练你定义的分类层 fc），目的是节省显存、加快训练速度
        with torch.no_grad():
            # 输入 input_ids、attention_mask、token_type_ids，标准 BERT 输入格式
            # 输出是一个对象（通常含有 last_hidden_state, pooler_output, ...）
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 下游任务参与训练
        # out.last_hidden_state[:, 0]：取的是 [CLS] 这个特殊标记的输出向量，表示整个句子的语义表示。
        out = self.fc(out.last_hidden_state[:, 0])
        # 然后通过全连接层 fc 映射成2个类别。
        # softmax(dim=1) 把输出转成两个类别的概率。
        out = out.softmax(dim=1)
        return out
