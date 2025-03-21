# pip install rapidocr-onnxruntime
# 从图像中提取⽂本⼀些 PDF 包含⽂本图像，例如扫描⽂档或图表
# 将图像提取为⽂本
from langchain_community.document_loaders import PyPDFLoader

file_path = "../../resource/TD1.pdf"
loader = PyPDFLoader(file_path, extract_images=True)
pages = loader.load()
# 识别第9页图片文字
print(pages[8].page_content)
