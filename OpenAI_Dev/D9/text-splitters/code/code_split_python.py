from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

PYTHON_CODE = """
def hello_world():
 print("Hello, World!")
# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs)

"""
[Document(metadata={}, page_content='def hello_world():\n print("Hello, World!")'), Document(metadata={}, page_content='# Call the function\nhello_world()')]
"""