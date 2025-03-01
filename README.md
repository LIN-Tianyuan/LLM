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
