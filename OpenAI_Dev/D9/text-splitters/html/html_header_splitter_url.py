from langchain_text_splitters import HTMLHeaderTextSplitter

# 要直接从 URL 读取
url = "https://plato.stanford.edu/entries/goedel/"
# 指定要分割的标题
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url(url)
for element in html_header_splits[:1]:
    print(element)