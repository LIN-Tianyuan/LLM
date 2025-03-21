# pip install pymupdf
from langchain_community.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader('../../resource/TD1.pdf')
data= loader.load()
print(data[0])
