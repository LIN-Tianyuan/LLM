# pip install pymupdf
from langchain_community.document_loaders import UnstructuredPDFLoader


loader = UnstructuredPDFLoader('../../resource/TD1.pdf', mode="elements")
data= loader.load()
print(data[0])
