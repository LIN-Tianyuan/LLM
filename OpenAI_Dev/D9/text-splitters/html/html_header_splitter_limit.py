from langchain_text_splitters import HTMLHeaderTextSplitter

url = "https://www.cnn.com/2023/09/25/weather/el-nino-winter-us-climate/index.html"
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url(url)
print(html_header_splits[1].page_content[:500])