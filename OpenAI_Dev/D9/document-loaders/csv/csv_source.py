from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = (
    "../../resource/doc_search.csv"
)

# 指定⽤于标识⽂档来源的列
"""
Document 元数据中的 "source" 键可以使⽤ CSV 的某⼀列进⾏设置。使⽤ source_column 参数指定从每⼀⾏创建的⽂档的来源。
否则， file_path 将⽤作从 CSV ⽂件创建的所有⽂档的来源。
"""
loader = CSVLoader(file_path=file_path,encoding="UTF-8",source_column="栖息地")
data = loader.load()
for record in data[:2]:
    print(record)

"""
page_content='名称: 狮子
种类: 哺乳动物
年龄: 8
栖息地: 非洲草原' metadata={'source': '非洲草原', 'row': 0}
page_content='名称: 大熊猫
种类: 哺乳动物
年龄: 5
栖息地: 中国竹林' metadata={'source': '中国竹林', 'row': 1}
"""