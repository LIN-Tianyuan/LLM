from torch.utils.data import Dataset
from datasets import load_from_disk


class Mydataset(Dataset):
    # 初始化数据
    def __init__(self, split):
        # 从磁盘加载数据
        self.dataset = load_from_disk("/Users/citron/Documents/GitHub/LLM/OpenAI_Dev/D15/data/ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            print("数据集名称错误！")

    # 获取数据集大小
    def __len__(self):
        return len(self.dataset)

    # 对数据做定制化处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label


if __name__ == '__main__':
    dataset = Mydataset("validation")
    for data in dataset:
        print(data)

"""
('位置尚可，但距离海边的位置比预期的要差的多，只能远远看大海，没有停车场', 0)
('整体来说，本书还是不错的。至少在书中描述了许多现实中存在的司法系统方面的问题，这是值得每个法律工作者去思考的。尤其是让那些涉世不深的想加入到律师队伍中的年青人，看到了社会特别是中国司法界真实的一面。缺点是：书中引用了大量的法律条文和司法解释，对于已经是律师或有一定工作经验的法律工作者来说有点多余，而且所占的篇幅不少，有凑字数的嫌疑。整体来说还是不错的。不要对一本书提太高的要求。', 1)
('5月8日付款成功，当当网显示5月10日发货，可是至今还没看到货物，也没收到任何通知，简不知怎么说好！！！', 0)
"""