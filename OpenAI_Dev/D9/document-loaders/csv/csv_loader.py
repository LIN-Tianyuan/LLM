from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = (
    "../../resource/doc_search.csv"
)

# LangChain 实现了⼀个 CSV 加载器，可以将 CSV ⽂件加载为⼀系列 Document 对象。
# CSV ⽂件的每⼀⾏都会被翻译为⼀个⽂档。
loader = CSVLoader(file_path=file_path,encoding="UTF-8")
data = loader.load()
for record in data[:2]:
    print(record)

"""
page_content='名称: 狮子
种类: 哺乳动物
年龄: 8
栖息地: 非洲草原' metadata={'source': '../../resource/doc_search.csv', 'row': 0}
page_content='名称: 大熊猫
种类: 哺乳动物
年龄: 5
栖息地: 中国竹林' metadata={'source': '../../resource/doc_search.csv', 'row': 1}
"""