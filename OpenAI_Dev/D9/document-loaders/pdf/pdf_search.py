# ⼀旦我们将PDF加载到LangChain的 Document 对象中，我们可以像通常⼀样对它们进⾏索引
# （例如，RAG应⽤程序）
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

file_path = "../../resource/TD1.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("How many exercises?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])