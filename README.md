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
