#-*- coding: utf-8 -*-
import torch

# 项目配置文件
class ParameterConfig():
    def __init__(self):
        # 判断是否使用GPU（1.电脑里面必须有显卡；2.必须安装cuda版本的pytorch）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 词典路径：在vocab文件夹里面
        self.vocab_path = './vocab/vocab.txt'
        # 训练集的pkl路径
        self.train_path = 'data/medical_train.pkl'
        # 验证集的pkl路径
        self.valid_path = 'data/medical_valid.pkl'
        # 模型配置文件
        self.config_json = './config/config.json'
        # 模型的保存路径
        self.save_model_path = 'save_model1'
        # 如果有预训练模型就写上路径（此处没有直接运行GPT2预训练好的模型，而是只用了该模型的架构）
        self.pretrained_model = ''
        # 保存对话语料
        self.save_samples_path = 'sample'
        # 忽略一些字符：句子需要长短补齐，针对补的部分，没有意义，所以一般不进行梯度更新
        self.ignore_index = -100
        # 历史对话句子的长度
        self.max_history_len = 1# "dialogue history的最大长度"
        # 每一个完整对话的句子最大长度
        self.max_len = 300  # '每个utterance的最大长度,超过指定长度则进行截断,默认25'
        self.repetition_penalty = 1.0 # "重复惩罚参数，若生成的对话重复性较高，可适当提高该参数"
        self.topk = 4 # '最高k选1。默认8'
        self.batch_size = 4 # 一个批次几个样本
        self.epochs = 4     # 训练几轮
        self.loss_step = 1 # 多少步汇报一次loss
        self.lr = 2.6e-5
        #   eps，为了增加数值计算的稳定性而加到分母里的项，其为了防止在实现中除以零
        self.eps = 1.0e-09
        self.max_grad_norm = 2.0
        # 累积梯度计算
        self.gradient_accumulation_steps = 4
        # 默认.warmup_steps = 4000
        self.warmup_steps = 100 # 使用Warmup预热学习率的方式,即先用最初的小学习率训练，然后每个step增大一点点，直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），采用最初设置的学习率进行训练（注：预热学习率完成后的训练过程，学习率是衰减的），有助于使模型收敛速度变快，效果更佳。


if __name__ == '__main__':
    pc = ParameterConfig()
    print(pc.train_path)
    print(pc.device)
    print(torch.cuda.device_count())
    """
    data/medical_train.pkl
    cpu
    0
    """