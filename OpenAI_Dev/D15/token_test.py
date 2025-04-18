from transformers import BertTokenizer

# 加载字典和分词工具
# 加载的是 BERT 中文版（base）模型的分词器，这个分词器能把中文句子变成BERT模型可以处理的“token id”。
token = BertTokenizer.from_pretrained("bert-base-chinese")
print(token)

# 准备输入句子
sents = [
    "酒店太旧了， 大堂感觉象三星级的， 房间也就是的好点的三星级的条件， 在青岛这样的酒店是绝对算不上四星标准， 早餐走了两圈也没有找到可以吃的， 太差了",
    "已经贴完了，又给小区的妈妈买了一套。最值得推荐",
    "屏幕大，本本薄。自带数字小键盘，比较少见。声音也还过得去。usb接口多，有四个。独显看高清很好。运行速度也还可以，性价比高！",
    "酒店环境很好 就是有一点点偏 交通不是很便利 去哪都需要达车 关键是不好打 酒店应该想办法解决一下"]

# 批量编码句子，
out = token.batch_encode_plus(
    # 要编码的一批文本（也可以是两个文本对）只用了前两条
    batch_text_or_text_pairs=[sents[0], sents[1]],
    # 添加特殊符号 [CLS]、[SEP]
    add_special_tokens=True,
    # 当句子长度大于max_length时，截断
    truncation=True,
    # 一律补零到max_length长度
    padding="max_length",
    # 最大长度
    max_length=30,
    # 可取值tf,pt,np,默认为返回list
    return_tensors=None,
    # 返回attention_mask（padding 是 0，其它是 1）
    return_attention_mask=True,
    # 返回token_type_ids
    # 返回 token 类型（只有句对才有用）
    return_token_type_ids=True,
    # 返回special_tokens_mask 特殊符号标识
    # 返回特殊token的位置，比如 [CLS] 和 [SEP] 是 1，其他是 0
    return_special_tokens_mask=True,
    # 返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    # return_offsets_mapping=True,
    # 返回length 标识长度
    return_length=True,
)
# input_ids 就是编码后的词
# token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
# special_tokens_mask 特殊符号的位置是1,其他位置是0
# attention_mask pad的位置是0,其他位置是1
# length 返回句子长度
print(out)
for k, v in out.items():
    print(k, ":", v)

print(token.decode(out["input_ids"][0]), token.decode(out["input_ids"][1]))
