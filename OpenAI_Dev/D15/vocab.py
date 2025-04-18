# 把三条情感文本用中文BERT的分词器编码成模型输入格式，生成新的可训练的数据集。
from transformers import BertTokenizer
from datasets import Dataset

# 制作 Dataset
# 这是 HuggingFace datasets.Dataset 对象，可以后续 .map() 操作或 .save_to_disk() 保存。
dataset = Dataset.from_dict({
    'text': ['位置尚可，但距离海边的位置比预期的要差的多',
             '5月8日付款成功，当当网显示5月10日发货，可是至今还没看到货物，也没收到任何通知，简不知怎么说好！！！',
             '整体来说，本书还是不错的。至少在书中描述了许多现实中存在的司法系统方面的问题，这是值得每个法律工作者去思考的。尤其是让那些涉世不深的想加入到律师队伍中的年青人，看到了社会特别是中国司法界真实的一面。缺点是：书中引用了大量的法律条文和司法解释，对于已经是律师或有一定工作经验的法律工作者来说有点多余，而且所占的篇幅不少，有凑字数的嫌疑。整体来说还是不错的。不要对一本书提太高的要求。'],
    'label': [0, 1, 1]  # 0 表示负向评价，1 表示正向评价
})

# 加载 BERT 模型的 vocab 字典
# 加载的是 bert-base-chinese 模型对应的分词器（按“字”切词），能把文本转换成：
# input_ids
# attention_mask
# token_type_ids
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 将数据集中的文本转换为 BERT 模型所需的输入格式
# 表示对文本列表进行批量分词编码，返回 PyTorch Tensor 格式。
dataset = dataset.map(lambda x: tokenizer(x['text'], return_tensors="pt"),
                      batched=True)
# 查看数据集信息
print(dataset)