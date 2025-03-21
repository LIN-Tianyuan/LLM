# pip install "unstructured[md]"
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "../../resource/test_markdown.md"
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()
print(f"文档数量：{len(data)}\n")
for document in data[:2]:
    print(f"{document}\n")

print(set(document.metadata["category"] for document in data))

"""
文档数量：17

page_content='测试 Markdown 文件' metadata={'source': '../../resource/test_markdown.md', 'category_depth': 0, 'languages': ['eng'], 'file_directory': '../../resource', 'filename': 'test_markdown.md', 'filetype': 'text/markdown', 'last_modified': '2025-03-19T21:03:03', 'category': 'Title', 'element_id': 'cbb469ee61ce690536d5d9ee1a648092'}

page_content='介绍' metadata={'source': '../../resource/test_markdown.md', 'category_depth': 1, 'languages': ['eng'], 'file_directory': '../../resource', 'filename': 'test_markdown.md', 'filetype': 'text/markdown', 'last_modified': '2025-03-19T21:03:03', 'parent_id': 'cbb469ee61ce690536d5d9ee1a648092', 'category': 'Title', 'element_id': 'f64f25698cdd18c998acfb18a104f19f'}

{'Title', 'Table', 'ListItem', 'UncategorizedText'}
"""