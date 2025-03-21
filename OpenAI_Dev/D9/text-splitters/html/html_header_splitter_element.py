from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
 <div>
 <h1>Foo</h1>
 <p>Some intro text about Foo.</p>
 <div>
 <h2>Bar main section</h2>
 <p>Some intro text about Bar.</p>
 <h3>Bar subsection 1</h3>
 <p>Some text about the first subtopic of Bar.</p>
 <h3>Bar subsection 2</h3>
 <p>Some text about the second subtopic of Bar.</p>
 </div>
 <div>
 <h2>Baz</h2>
 <p>Some text about Baz</p>
 </div>
 <br>
 <p>Some concluding text about Foo</p>
 </div>
</body>
</html>
"""
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on,
    # 返回每个元素以及其关联的标题
    return_each_element=True
)
html_header_splits_elements = html_splitter.split_text(html_string)
# 现在每个元素都作为⼀个独⽴的 Document 返回
for element in html_header_splits_elements[:2]:
    print(element)

"""
page_content='Foo' metadata={'Header 1': 'Foo'}
page_content='Some intro text about Foo.' metadata={'Header 1': 'Foo'}
"""

for element in html_header_splits_elements[:3]:
    print(element)

"""
page_content='Foo' metadata={'Header 1': 'Foo'}
page_content='Some intro text about Foo.' metadata={'Header 1': 'Foo'}
page_content='Bar main section' metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section'}
"""

