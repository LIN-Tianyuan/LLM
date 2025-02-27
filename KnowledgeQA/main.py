# coding:utf-8
# 导入必备的工具包
from langchain.prompts import PromptTemplate
from get_vector import *
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 加载FAISS向量库
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.load_local('./faiss/camp', embeddings, allow_dangerous_deserialization=True)


def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace('\n\n', '\n'))
    return '\n'.join(related_content)

def define_prompt():
    question = '我买的商品来自于哪个仓库，从哪出发的，预计什么到达'
    # k=5 让FAISS返回5个文档
    docs = db.similarity_search(question, k=5)
    print(f'docs-->{docs}')
    related_docs = get_related_content(docs)

    # 构建模板
    PROMPT_TEMPLATE = """
           基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。
           已知内容:
           {context}
           问题:
           {question}"""
    prompt = PromptTemplate(input_variables=["context", "question"],
                            template=PROMPT_TEMPLATE)

    my_prompt = prompt.format(context=related_docs,
                              question=question)
    return my_prompt

def qa():
    model = ChatOpenAI(model_name="gpt-4o")
    my_prompt = define_prompt()
    result = model.invoke(my_prompt)
    return result.content

if __name__ == '__main__':
    result = qa()
    print(f'result-->{result}')

"""
result-->您购买的商品来自东方仓储中心，位于深圳市。预计运输时间为3天。
"""