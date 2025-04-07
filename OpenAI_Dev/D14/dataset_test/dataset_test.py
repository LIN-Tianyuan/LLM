from datasets import load_dataset,load_from_disk

# print(list_datasets())
# 在线加载数据集
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
# 转存为csv格式
# dataset.to_csv(path_or_buf="D:/PycharmProjects/day14_demo/data/hermes-function-calling-v1.csv")
# 加载磁盘数据
dataset = load_from_disk("../data/ChnSentiCorp")
# 加载csv文件数据
# dataset = load_dataset("csv",data_files="D:/PycharmProjects/day14_demo/data/hermes-function-calling-v1.csv")
# print(dataset)
# 取出训练集
train_data = dataset["train"]
print(train_data)
#查看数据
for data in train_data:
    print(data)