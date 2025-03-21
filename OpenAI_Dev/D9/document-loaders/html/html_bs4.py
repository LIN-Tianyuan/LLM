# pip install bs4
from langchain_community.document_loaders import BSHTMLLoader

# 使⽤BeautifulSoup4使⽤ BSHTMLLoader 加载HTML⽂档。
# 将HTML中的⽂本提取到 page_content 中，并将⻚⾯标题提取到 metadata 的 title 中。
file_path = '../../resource/content.html'
loader = BSHTMLLoader(file_path, open_encoding="UTF-8")
data = loader.load()
print(data)