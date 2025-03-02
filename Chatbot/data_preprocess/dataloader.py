# -*- coding: utf-8 -*-
import torch.nn.utils.rnn as rnn_utils  # 导入rnn_utils模块，用于处理可变长度序列的填充和排序
from torch.utils.data import Dataset, DataLoader  # 导入Dataset和DataLoader模块，用于加载和处理数据集
import torch  # 导入torch模块，用于处理张量和构建神经网络
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
from dataset import *  # 导入自定义的数据集类

def load_dataset(train_path, valid_path):
    # print('进入函数')
    """
    加载训练集和验证集
    :param train_path: 训练数据集路径
    :return: 训练数据集和验证数据集
    """
    with open(train_path, "rb") as f:
        train_input_list = pickle.load(f)  # 从文件中加载输入列表

    with open(valid_path, "rb") as f:
        valid_input_list = pickle.load(f)  # 从文件中加载输入列表
    # 划分训练集与验证集
    # print(len(input_list))  # 打印输入列表的长度
    # print(input_list[0])
    #
    train_dataset = MyDataset(train_input_list, 300)  # 创建训练数据集对象
    val_dataset = MyDataset(valid_input_list, 300)  # 创建验证数据集对象
    return train_dataset, val_dataset  # 返回训练数据集和验证数据集

def collate_fn(batch):
    """
    自定义的collate_fn函数，用于将数据集中的样本进行批处理
    :param batch: 样本列表
    :return: 经过填充的输入序列张量和标签序列张量
    """
    # print(f'batch-->{batch}')
    """
    batch-->[tensor([ 101, 1920, 4636, 6956, 4638, 1914, 1355, 1765, 1277, 3221, 1525, 7027,
        8043,  102, 3887, 7377,  102]), tensor([ 101, 5541, 1366, 4563, 1963,  862, 3780, 4545, 8024, 3297, 6818, 5541,
        1366, 2600, 3221, 1139, 4385, 5541, 4578, 5541, 7315, 4638, 2658, 1105,
        8024, 1278, 7368, 3466, 3389, 6432, 3221, 1461, 1429, 6887, 4565, 4567,
        8024,  852, 3221, 1391, 5790,  679, 2582,  720, 5052, 4500, 8024, 3297,
        6818, 1495, 1644, 1217, 7028, 8024, 6435, 7309, 5541, 1366, 4563, 1963,
         862, 3780, 4545,  102,  872, 1962, 8038, 3418, 2945,  872, 2989, 6835,
        4638, 2658, 1105,  117, 2456, 6379, 1343, 1278, 7368,  976,  702, 2552,
        4510, 1745, 1469, 2552, 5552, 2506, 6631, 3466, 3389,  117, 5517, 7262,
        3466, 3389,  117, 2961, 7370, 2552, 5491, 5375, 6117, 4638, 1377, 5543,
        1469, 5517, 6956, 2772, 5442, 3867, 1265, 5143, 5320, 4565, 4567,  117,
        5440, 5991, 3221, 4868, 5307, 4578,  119, 2456, 6379, 2398, 3198, 3800,
        2692,  828, 2622,  117,  679, 6206, 4228, 1915,  117, 1772, 6130, 7650,
        7608,  119, 2361, 3307, 2769, 4638, 1726, 5031, 1377,  809, 2376, 1168,
         872,  117,  167, 8024, 2190,  754, 1461, 1429, 1079, 4906, 4565, 4567,
        4565, 4567, 4638, 1139, 4385, 8024, 2642, 5442, 3301, 1351,  812, 2418,
        6421,  976, 1168, 1350, 3198, 3780, 4545, 1333, 1156, 8024, 1728,  711,
        3193, 3309, 4638, 1461, 1429, 1079, 4906, 4565, 4567, 3221, 2159, 3211,
        2533, 1168, 2971, 1169, 4638,  511, 2642, 5442,  812,  679, 6206, 7231,
        6814, 3780, 4545, 4638, 1962, 3198, 3322,  511,  102]), tensor([ 101, 5511, 5310, 5688, 1147, 7370, 1377,  809, 1391, 3862, 7831, 1408,
        8024, 3187,  102,  872, 4638, 2658, 1105, 3297, 1962, 2797, 3318, 3780,
        4545, 8024, 2797, 3318, 3126, 3362, 3683, 6772, 4802, 1147, 8024, 5790,
        4289, 3780, 4545, 3187, 2692,  721, 8024, 3318, 1400, 7564, 7344,  839,
        1366, 2697, 3381, 8024, 3800, 2692, 6133, 1041,  831, 6574, 6028, 4635,
        6574,  914, 6822,  839, 1366, 2689, 1394,  511,  102]), tensor([  101,  3780,  4545,   749,  1184,  1154,  5593,  4142,  1400,  6820,
         3221,  2228,  2593,  2582,   720,  1215,  8024,  3780,  4545,   749,
         1184,  1154,  5593,  4142,  1400,  6820,  3221,  2228,  2593,  2228,
         3198,  2218,  6206,  1920,   912,  3198,  1468,  1726,   752,   749,
         8024,  1825,  3315,  1469,  3780,  4545,  1184,   671,  3416,  8024,
         3780,  4545,  6589,  4500,   749, 10450,  8129,  1039,  8024,  6820,
         3221,  1333,  3341,  2658,  1105,   102,   872,  1962,  8024,   872,
         4638,  2658,  1105,  1072,  3300,  1184,  1154,  5593,  4142,  2193,
         5636,  4638,  4568,  4307,  8024,   671,  5663,   833,  2193,  5636,
         2228,  7574,  2228,  2593,  1469,  2228,   679,  1112,  5023,  4568,
         4307,  4638,  8024,   872,  4638,  2658,  1105,  7444,  6206,  3300,
         1350,  5790,  4289,  1400,  3309,  4937,  1743,  3780,  4545,  4638,
         8024,  1469,  2552,  2658,  3300,  1068,  5143,  4638,  2900,  2193,
         2692,  6224,  8038,   872,  4638,  2658,  1105,  2769,  2456,  6379,
         7674,  1044,  2218,  6206,  3800,  2692,   828,  2622,  8024,  3926,
         3909,  3946,  4178,  3211,  6133,  1041,  7650,  7608,  2523,  7028,
         6206,  4638,  8024,  2456,  6379,   872,  4916,  3353,  3302,  5790,
         3780,  4545,  8024,  3683,  1963,  6432,  3302,  7608,  5540,  1718,
         1469,  2340,  3709,  3703,  3763,  3215,  4275,  1469,  3789,  2228,
         4130,  4275,  3780,  4545,  8024,  1914,  1215,   752,   711,  2139,
         4638,  8024,  3123,  3351,  2552,  2658,   711,  2139,  4638,  8024,
         4867,   872,   978,  2434,  1184,  1154,  5593,  4142,  4567,  2658,
         3211,  1353,  1353,  1908,  1908,  8024,  7444,  6206,  3300,  7270,
         3309,  4638,  5790,  4289,   924,  2898,   511,  5445,   684,  2190,
          754,  4511,  2595,  2642,  5442,  3341,  6432,  8024,  6206,  1350,
         3198,  1343,   683,   689,  3633,  6226,  4638,  1278,  7368,  6402,
         4567,  8024,  3418,  2945,   798,  5301,  3466,  3389,  5310,  3362,
         1086,  5440,  5991,  3780,  4545,   511,   102])]
    """
    print(f'batch的长度-->{len(batch)}')
    """
    batch的长度-->4
    """
    # print(f'batch的第一个样本的长度-->{batch[0].shape}')
    # print(f'batch的第二个样本的长度-->{batch[1].shape}')
    # print(f'batch的第三个样本的长度-->{batch[2].shape}')
    # print(f'batch的第四个样本的长度-->{batch[3].shape}')
    """
    batch的第一个样本的长度-->torch.Size([35])
    batch的第二个样本的长度-->torch.Size([35])
    batch的第三个样本的长度-->torch.Size([23])
    batch的第四个样本的长度-->torch.Size([24])
    """
    print('----------')
    # rnn_utils.pad_sequence: 将根据一个batch中，最大句子长度，进行补齐
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)  # 对输入序列进行填充，使其长度一致
    # print(f'input_ids-->{input_ids}')
    # print(f'batch的第一个样本的长度-->{input_ids[0].shape}')
    # print(f'batch的第二个样本的长度-->{input_ids[1].shape}')
    # print(f'batch的第三个样本的长度-->{input_ids[2].shape}')
    # print(f'batch的第四个样本的长度-->{input_ids[3].shape}')
    """
    batch的第一个样本的长度-->torch.Size([35])
    batch的第二个样本的长度-->torch.Size([35])
    batch的第三个样本的长度-->torch.Size([35])
    batch的第四个样本的长度-->torch.Size([35])
    """
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)  # 对标签序列进行填充，使其长度一致
    # print(f'labels-->{labels}')
    return input_ids, labels  # 返回经过填充的输入序列张量和标签序列张量

def get_dataloader(train_path, valid_path):
    """
    获取训练数据集和验证数据集的DataLoader对象
    :param train_path: 训练数据集路径
    :return: 训练数据集的DataLoader对象和验证数据集的DataLoader对象
    """
    train_dataset, val_dataset = load_dataset(train_path, valid_path)  # 加载训练数据集和验证数据集
    print(f'train_dataset: {len(train_dataset)}')
    print(f'val_dataset: {len(val_dataset)}')
    """
    train_dataset: 30177
    val_dataset: 413
    """
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4, # 一批要送入模型几个样本
                                  shuffle=True, # 随机读数据
                                  collate_fn=collate_fn,    # 自定义函数
                                  drop_last=True    # 扔掉最后一批
                                  )  # 创建训练数据集的DataLoader对象
    validate_dataloader = DataLoader(val_dataset,
                                     batch_size=4,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     drop_last=True)  # 创建验证数据集的DataLoader对象
    return train_dataloader, validate_dataloader  # 返回训练数据集的DataLoader对象和验证数据集的DataLoader对象


if __name__ == '__main__':
    train_path = '../data/medical_train.pkl'
    valid_path = '../data/medical_valid.pkl'
    # load_dataset(train_path)
    train_dataloader, validate_dataloader = get_dataloader(train_path, valid_path)
    for input_ids, labels in validate_dataloader:
        # print('你好')
        # print(f'input_ids--->{input_ids.shape}')
        # print(f'labels--->{labels.shape}')
        """
        input_ids--->torch.Size([4, 283])
        labels--->torch.Size([4, 283])
        """
        print('*'*80)