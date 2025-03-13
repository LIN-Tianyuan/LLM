# LLM

[1. Introduction to the Large Model Background](./01/README.md)

## Project

### Knowledge QA
#### 1. Context
Problem:
Models are trained based on past empirical data and do not have access to the latest knowledge, as well as knowledge that is private to each organization.

Method:
1. Leveraging enterprise private knowledge and fine-tuning based on open source grand models.
2. Q&A on Building a Local Knowledge Base Based on LangChain Integration Vector Database and LLM (RAG).

Build a local knowledge base based on logistics information and test the Q&A results.

#### 2. Principles

![Elovutionary Tree](/img/15.png)

 - Loading files
 - Reading files
 - Text segmentation
 - Text vectorization
 - Interrogative vectorization
 - Match the top_k of text vectors that are similar to the interrogative vectors
 - The matched text is added to the prompt as context along with the question
 - Submit to LLM to generate answers

#### 3. Configuration
```bash
# Python version
python3.8-Python3.11

# Requirements
pip install faiss-cpu
pip install langchain

# Model download or online
```
#### 4. Realisation
 - Customized Model Classes (model_ChatGLM.py)
 - Build Faiss Index (get_vector.py)
 - Implement QA Local Knowledge Base (main.py)
#### 5. Usage
 - Run main.py

## Notice（重点）
### 1. 大模型的微调手段
传统方式：Fine-Tuning：一般是全量参数的微调
Prompt_Tuning: Lora、P_tuning：部分参数微调
不改变模型参数的方式：Few-shot或者Zero-shot
### 2. GPT2医疗问诊机器人
文本处理的前置工作：
 - 数据预处理：把每一段对话（两句话）变成数字，连接到一块，变成完整的一句话直接送给模型（生成的pkl文件）
 - medical_train.pkl 和 medical_valid.pkl
 - 数据处理脚本（data_preprocess/preprocess.py）

构建dataset数据源：
 - 把前面处理的数据进行再次的封装
 - 变成张量的形式
 - 构建dataset脚本（data_preprocess/dataset.py）

dataloader:
 - 对话任务：前文生成后文，自回归任务
 - 一句话：ABCD
 - A --> B; AB --> C; ABC --> D
 - 构建dataloader脚本（data_preprocess/dataloader.py）
### 3. 电商评论文本分类
数据:
 - train.txt
 - prompt.txt
 - dev.txt
 - verbalizer.txt

项目配置文件：
 - pet_config.py

数据预处理(data_handle)：
 - template.py
 - data_preprocess.py
 - data_loader.py

工具类(utils):
 - verbalizer.py
### 4. 为什么要填充 encoded?(神经网络的张量计算要求相同形状的输入)
 - 保持序列长度一致：Transformer 模型（如 BERT、T5）需要固定长度输入，否则 batch 计算时会报错。
 - 提高计算效率：填充后可以使用 tensor 并行计算，不需要对每个样本单独处理。
 - 防止模型误解：用 pad_token_id 让模型知道哪些部分是无效填充，不应该被关注（例如，在 attention_mask 中会标记出来）。
### 5. 什么时候需要运行返回tensor?
 - 适用于 PyTorch、TensorFlow 训练模型(返回)
   - 如果要把数据 直接输入深度学习模型进行训练，通常需要返回张量（Tensor），这样可以更高效地利用 GPU 计算。
 - 适用于数据预处理、调试、非深度学习任务(不返回)
   - 如果只是想检查 tokenized_output，或者在训练前进行 预处理、数据分析，通常不需要返回 Tensor，而是返回普通的 Python dict 或 list。