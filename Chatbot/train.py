import torch
import os
# 时间
from datetime import datetime
import transformers
# 配置定义GPT2模型
from transformers import GPT2LMHeadModel, GPT2Config
# 使用Bert的分词器
from transformers import BertTokenizerFast
# 导入自定义的工具类函数（计算损失和准确率）
from functions_tools import *
# 导入项目的配置文件（训练数据集路径和训练的轮次参数等）
from parameter_config import *
# 导入数据：dataloader
from data_preprocess.dataloader import *


def train_epoch(model,
                train_dataloader,
                optimizer, scheduler,
                epoch, args):
    """
    :param model: GPT2模型
    :param train_dataloader: 训练数据集
    :param optimizer: 优化器：更新参数
    :param scheduler: 学习率预热
    :param epoch: 当前的轮次
    :param args: 模型配置文件的参数对象
    :return:
    """
    # 1. 指明模型训练
    model.train()
    device = args.device
    # 对于ignore_index的label token不计算梯度
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        # print(f'input_ids-->{input_ids.shape}')
        # print(f'labels-->{labels.shape}')
        # print(f'将数据送入模型中。。。。。。。。。。。。。。。。')
        # print(f'labels0---->{labels.shape}')
        """
        input_ids-->torch.Size([4, 216])
        labels-->torch.Size([4, 216])
        将数据送入模型中。。。。。。。。。。。。。。。。
        labels0---->torch.Size([4, 216])
        """
        # 如果对模型输入不仅包含input还包含标签，那么得到结果直接就有loss值
        outputs = model.forward(input_ids, labels=labels)
        # print(f'outputs-->{outputs}')
        # print(f'outputs-->{outputs.keys()}')
        # outputs-->odict_keys(['loss', 'logits', 'past_key_values'])
        # print(f'outputs.logits-->{outputs.logits.shape}')
        # outputs.logits-->torch.Size([4, 216, 13317]) # 预测出13317个概率值
        # print(f'outputs.loss-->{outputs.loss}') # outputs.loss-->9.579055786132812

        # outputs = model.forward(input_ids)
        # logits形状：【4,200,13317】
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        # 统计该batch的预测token的正确数与总数
        batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)

        # 计算该batch的accuracy
        batch_acc = batch_correct_num / batch_total_num

        # 统计该epoch的预测token的正确数与总数
        epoch_correct_num += batch_correct_num
        epoch_total_num += batch_total_num

        total_loss += loss.item()
        # self.gradient_accumulation_steps = 4， 累积的步数
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        # 梯度裁剪 # 避免梯度爆炸的方式。梯度乘以缩放系数。self.max_grad_norm = 2.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #
    #     # 进行一定step的梯度累计之后，更新参数
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 清空梯度信息
            optimizer.zero_grad()

        if (batch_idx + 1) % args.loss_step == 0:
            print(
                "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                    batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps, batch_acc, scheduler.get_lr()))

        del input_ids, outputs


    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    print(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    if epoch % 10 == 0 or epoch == args.epochs:
        print('saving model for epoch {}'.format(epoch + 1))
        model_path = os.path.join(args.save_model_path, 'bj_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # 保存预训练模型的方式
        model.save_pretrained(model_path)
        print('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        print('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, epoch, args):
    print("start validating")
    model.eval()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)

            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            total_loss += loss.item()
            del input_ids, outputs

        # 记录当前epoch的平均loss
        epoch_mean_loss = total_loss / len(validate_dataloader)
        print(
            "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
        epoch_finish_time = datetime.now()
        print('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
        return epoch_mean_loss


def train(model,  train_dataloader, validate_dataloader, args):
    # len(train_dataloader) --> 训练一次完整的数据，需要迭代多少步 7544
    # t_total 模型训练完毕，一共要迭代多少步
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    # eps，为了增加数值计算的稳定性而加到分母里的项，其为了防止在实现中除以零
    # 让模型更新参数
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    '''
    这里对于模型的参数，分别进行权重参数的衰减优化：防止过拟合，以及学习率预热处理优化：
    在初始阶段将学习率从较小的值逐步增加到设定的初始值，然后按照设定的学习率调整策略进行训练。
    学习率预热的目的是让模型在初始阶段更快地适应数据，避免训练过程中因为学习率过大导致的梯度爆炸等问题，
    从而提高模型的训练效果和泛化性能。
    optimizer： 优化器
    num_warmup_steps：初始预热步数
    num_training_steps：整个训练过程的总步数
    '''
    '''
    参数的解析如下：

optimizer：这个参数需要传入一个优化器对象（optimizer object）。它代表在训练过程中用于更新模型参数的优化器，比如Adam或SGD等。

num_warmup_steps：这个参数确定学习率在开始阶段从0线性增加到初始值的步数。在Transformer模型中，通过逐渐增加学习率来稳定和加速训练过程是常见的做法。通常，这个值是总训练步数的一小部分。

num_training_steps：这个参数指定了总的训练步数或迭代次数。它表示优化器将在给定数据集上进行多少次参数更新。
    '''
    # 学习率预热
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    """
    假设一共训练10000步
    我们设置warmup_steps = 1000
    训练刚开始时，从1步-1000步，学习率从0线性增加到预定的lr
    随后的9000步，将基于lr一直训练
    """
    print('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, args=args)
        train_losses.append(train_loss)
        # ========== validate ========== #
        # 验证的时候不需要更新参数
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            print('saving current best model for epoch {}'.format(epoch + 1))
            model_path = os.path.join(args.save_model_path, 'min_ppl_model_bj'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model.save_pretrained(model_path)


def main():
    # 初始化配置参数
    params = ParameterConfig()

    # 设置使用哪些显卡进行训练:默认为0
    # 如果电脑有大于1张的显卡，可以选择使用，数字0代表第一张显卡
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 数字0代表第一张显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '1' 数字1代表第二张显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1' 数字0, 1代表同时使用0和1两张显卡
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(params.vocab_path,
                                  sep_token="[SEP]",
                                  pad_token="[PAD]",
                                  cls_token="[CLS]")
    # tokenizer = BertTokenizerFast(params.vocab_path)
    # print(f'tokenizer-->{tokenizer.vocab_size}')
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    # print(f'sep_id-->{sep_id}')
    # print(f'pad_id-->{pad_id}')
    # print(f'cls_id-->{cls_id}')
    """
    tokenizer-->13317
    sep_id-->102
    pad_id-->0
    cls_id-->101
    """

    # 创建模型的输出目录
    # 如果没有创建会自动创建输出目录
    if not os.path.exists(params.save_model_path):
        os.mkdir(params.save_model_path)

    # 创建模型
    if params.pretrained_model:  # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(params.config_json)
        # print(model_config)
        model = GPT2LMHeadModel(config=model_config)
    # print(f'model-->{model}')
    """
        model-->GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(13317, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): Conv1D(nf=2304, nx=768)
              (c_proj): Conv1D(nf=768, nx=768)
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D(nf=3072, nx=768)
              (c_proj): Conv1D(nf=768, nx=3072)
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=13317, bias=False)
    )
    """
    model = model.to(params.device)
    # print(f'model.config.vocab_size-->{model.config.vocab_size}')
    # print(f'tokenizer.vocab_size-->{tokenizer.vocab_size}')
    """
    model.config.vocab_size-->13317
    tokenizer.vocab_size-->13317
    """
    # assert这里相当于确认
    assert model.config.vocab_size == tokenizer.vocab_size

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    # print(f'模型参数总量--->{num_parameters}')
    # 模型参数总量--->96069888

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataloader, validate_dataloader = get_dataloader(params.train_path,params.valid_path)
    # print(f'train_dataloader: {len(train_dataloader)}')
    """
    train_dataset: 30177
    val_dataset: 413
    train_dataloader: 7544
    """
    train(model, train_dataloader, validate_dataloader, params)
    """
    validate epoch 4: loss 2.992560781321479
    time for validating one epoch: 0:01:21.291755
    saving current best model for epoch 4
    """

if __name__ == '__main__':
    main()
