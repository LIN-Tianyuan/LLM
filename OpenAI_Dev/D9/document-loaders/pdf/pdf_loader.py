# pip install pypdf
# 使⽤ pypdf 将PDF加载为⽂档数组，其中每个⽂档包含⻚⾯内容和带有 page 编号的元数据。
from langchain_community.document_loaders import PyPDFLoader

file_path = "../../resource/TD1.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
print(pages[0])
