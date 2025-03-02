# -*- coding: utf-8 -*-
import pickle

from torch.utils.data import Dataset  # 导入Dataset模块，用于定义自定义数据集
import torch  # 导入torch模块，用于处理张量和构建神经网络


class MyDataset(Dataset):
    """
    自定义数据集类，继承自Dataset类
    """

    def __init__(self, input_list, max_len):
        super().__init__()
        """
        初始化函数，用于设置数据集的属性
        :param input_list: 输入列表，包含所有对话的tokenize后的输入序列
        :param max_len: 最大序列长度，用于对输入进行截断或填充
        """
        # print(f'input_list--->{len(input_list)}')
        self.input_list = input_list  # 将输入列表赋值给数据集的input_list属性
        self.max_len = max_len  # 将最大序列长度赋值给数据集的max_len属性

    def __len__(self):
        """
        获取数据集的长度
        :return: 数据集的长度
        """
        return len(self.input_list)  # 返回数据集的长度

    def __getitem__(self, index):
        """
        根据给定索引获取数据集中的一个样本
        :param index: 样本的索引
        :return: 样本的输入序列张量
        """
        # print(f"当前取出的索引是->{index}")
        # 当前取出的索引是->0
        input_ids = self.input_list[index]  # 获取给定索引处的输入序列
        # print(f'input_ids-> {input_ids}')
        """
        input_ids-> [101, 2364, 7032, 3481, 1363, 1217, 5341, 1394, 2519, 4638, 6774, 1221, 3780, 4545, 3300, 763, 784, 720, 8043, 102, 5341, 1394, 3780, 4545, 8039, 2434, 1908, 6378, 5298, 8039, 4495, 3833, 2844, 4415, 2900, 2193, 8039, 856, 7574, 7028, 1908, 5307, 7565, 4828, 1173, 4080, 3780, 4545, 102]
        """
        input_ids = input_ids[:self.max_len]  # 根据最大序列长度对输入进行截断或填充
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # 将输入序列转换为张量long类型
        return input_ids  # 返回样本的输入序列张量

if __name__ == '__main__':

    with open('../data/medical_train.pkl', 'rb') as f:
        train_input_list = pickle.load(f) # 从文件中加载输入列

    # print(f'train_input_list: {len(train_input_list)}')
    # print(f'train_input_list: {type(train_input_list)}')

    """
    train_input_list: 30177
    train_input_list: <class 'list'>
    """

    my_dataset = MyDataset(input_list=train_input_list, max_len=300)
    print(f'my_dataset-->{len(my_dataset)}')
    """
    my_dataset-->30177
    """
    # 30177样本中第一个样本的对话的数字表示
    result = my_dataset[0]
    print(result)
    """
    tensor([ 101, 2364, 7032, 3481, 1363, 1217, 5341, 1394, 2519, 4638, 6774, 1221,
        3780, 4545, 3300,  763,  784,  720, 8043,  102, 5341, 1394, 3780, 4545,
        8039, 2434, 1908, 6378, 5298, 8039, 4495, 3833, 2844, 4415, 2900, 2193,
        8039,  856, 7574, 7028, 1908, 5307, 7565, 4828, 1173, 4080, 3780, 4545,
         102])
    """