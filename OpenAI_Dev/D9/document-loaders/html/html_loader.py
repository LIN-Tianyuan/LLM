# pip install "unstructured[html]"
from langchain_community.document_loaders import UnstructuredHTMLLoader

# 将HTML⽂档加载到LangChain的Document对象中
# 使⽤Unstructured加载HTML
file_path = "../../resource/content.html"
loader = UnstructuredHTMLLoader(file_path, encodings="UTF-8")
data = loader.load()
print(data)

"""
[Document(metadata={'source': '../../resource/content.html'}, page_content='黄山\n\n黄山位于中国安徽省南部，是中国著名的风景名胜区，以奇松、怪石、云海和温泉“四绝”闻名\n\n大峡谷\n\n大峡谷位于美国亚利桑那州，是世界上最著名的自然景观之一，以其壮观的地质奇观和深邃的峡谷闻名。')]
"""