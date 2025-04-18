import torch
from net import Model
from transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = ["负向评价","正向评价"]
print(DEVICE)
model = Model().to(DEVICE)


token = BertTokenizer.from_pretrained("bert-base-chinese")

# 预处理文本
# 输入：一条字符串句子，比如 "位置尚可，但距离海边..."；
# 分词器编码成：input_ids、attention_mask、token_type_ids；
# padding 到 max_length=500；
# 返回这些张量给模型作为输入。
def collate_fn(data):
    sentes = []
    sentes.append(data)
    # print(sentes)
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=500,
                                   return_tensors="pt",
                                   return_length=True)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    # print(input_ids,attention_mask,token_type_ids)
    return input_ids,attention_mask,token_type_ids

def final():
    # model.load_state_dict(torch.load("params/2bert.pt"))
    # 加载训练好的模型权重；
    model.load_state_dict(torch.load("params/2bert.pt", map_location=DEVICE))
    # 设置模型为评估模式 eval()
    model.eval()
    while True:
        data = input("请输入测试数据(输入'q'退出)：")
        if data == "q":
            print("测试结束")
            break
        # 进行分词 → 模型推理 → 输出情感类别
        input_ids, attention_mask, token_type_ids= collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids,attention_mask,token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：",names[out],"\n")

if __name__ == '__main__':
    final()

"""
请输入测试数据(输入'q'退出)：位置尚可，但距离海边的位置比预期的要差的多，只能远远看大海，没有停车场
模型判定： 正向评价 

请输入测试数据(输入'q'退出)：5月8日付款成功，当当网显示5月10日发货，可是至今还没看到货物，也没收到任何通知，简不知怎么说好！！！
模型判定： 负向评价 
"""