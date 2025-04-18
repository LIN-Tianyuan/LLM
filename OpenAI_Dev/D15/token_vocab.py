# 对字典进行定制化处理，对语句进行编码解码
from transformers import BertTokenizer

# 加载字典和分词工具
token = BertTokenizer.from_pretrained("bert-base-chinese")
# print(token)

sents = [
    "酒店太旧了， 大堂感觉象三星级的， 房间也就是的好点的三星级的条件， 在青岛这样的酒店是绝对算不上四星标准， 早餐走了两圈也没有找到可以吃的， 太差了",
    "已经贴完了，又给小区的妈妈买了一套。最值得推荐",
    "屏幕大，本本薄。自带数字小键盘，比较少见。声音也还过得去。usb接口多，有四个。独显看高清很好。运行速度也还可以，性价比高！",
    "酒店环境很好 就是有一点点偏 交通不是很便利 去哪都需要达车 关键是不好打 酒店应该想办法解决一下"]

# Test1
"""
# 获取字典
vocab = token.get_vocab()
# print(vocab)
print(len(vocab))
print("阳" in vocab)
print("光" in vocab)
print("阳光" in vocab)

# 添加新词
token.add_tokens(new_tokens="阳光")
vocab = token.get_vocab()
print(len(vocab))
print("阳光" in vocab)


21128
True
True
False
21129
True
"""

# Test2
# 获取字典
vocab = token.get_vocab()
# print(vocab)
print(len(vocab))
print("阳" in vocab)
print("光" in vocab)
print("阳光" in vocab)

# 添加新词
token.add_tokens(new_tokens=["阳光","大地"])
# 添加新的特殊符号
token.add_special_tokens({"eos_token":"[EOS]"})
vocab = token.get_vocab()
print(len(vocab))
print("阳光" in vocab,"大地" in vocab,"[EOS]" in vocab)

# 编码新句子
out = token.encode(text="阳光照在大地上[EOS]",
             text_pair=None,
             truncation=True,
             padding="max_length",
             max_length=10,
             add_special_tokens=True,
             return_tensors=None)
print(out)
# 解码为原字符串
print(token.decode(out))

"""
[101, 21128, 4212, 1762, 21129, 677, 21130, 102, 0, 0]
[CLS] 阳光 照 在 大地 上 [EOS] [SEP] [PAD] [PAD]
"""