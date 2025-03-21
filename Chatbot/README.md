# Medical Consultation Chatbot
Building a medical consultation robot based on GPT2

Usage:
 - Run app.py
 - Go to http://127.0.0.1:5000
 - Ask your question

## 1. Context
An intelligent medical Q&A system was constructed based on data from the medical field, with the aim of providing users with accurate, efficient, and high-quality medical Q&A services.

## 2. Environment
```bash
python >= 3.6
transformers >= 4.2.0
pytorch >= 1.7.0
```
## 3. Code
 - Text processing: data_preprocess/...
 - Model building(GPT2LMHeadModel(config=model_config))
 - Model train: train.py
 - Helper class function: functions_tools.py
 - Model predict: interact.py

## 4. Web interaction
app.py

index.html