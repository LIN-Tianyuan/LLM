from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = (
    "../../resource/doc_search.csv"
)

# ⾃定义 CSV 解析和加载
loader = CSVLoader(
    file_path=file_path,
    encoding="UTF-8",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Name", "Species", "Age", "Habitat"]
    }
)
data = loader.load()
for record in data[:2]:
    print(record)

"""
page_content='Name: 名称
Species: 种类
Age: 年龄
Habitat: 栖息地' metadata={'source': '../../resource/doc_search.csv', 'row': 0}
page_content='Name: 狮子
Species: 哺乳动物
Age: 8
Habitat: 非洲草原' metadata={'source': '../../resource/doc_search.csv', 'row': 1}
"""