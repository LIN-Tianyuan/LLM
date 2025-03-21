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
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
print(html_header_splits)
"""
[Document(metadata={'Header 1': 'Foo'}, page_content='Foo'), Document(metadata={'Header 1': 'Foo'}, page_content='Some intro text about Foo.'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section'}, page_content='Bar main section'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section'}, page_content='Some intro text about Bar.'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section', 'Header 3': 'Bar subsection 1'}, page_content='Bar subsection 1'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section', 'Header 3': 'Bar subsection 1'}, page_content='Some text about the first subtopic of Bar.'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section', 'Header 3': 'Bar subsection 2'}, page_content='Bar subsection 2'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Bar main section', 'Header 3': 'Bar subsection 2'}, page_content='Some text about the second subtopic of Bar.'), Document(metadata={'Header 1': 'Foo', 'Header 2': 'Baz'}, page_content='Baz'), Document(metadata={'Header 1': 'Foo'}, page_content='Some text about Baz  \nSome concluding text about Foo')]
"""