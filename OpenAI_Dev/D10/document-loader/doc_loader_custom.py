from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import asyncio

# 创建⾃定义Document loader(⽂档加载器)
# 该加载器从⽂件中加载数据，并从⽂件的每⼀⾏创建⼀个⽂档。
class CustomDocumentLoader(BaseLoader):
    """一个从文件逐行读取的示例文档加载器。"""

    def __init__(self, file_path: str) -> None:
        """使用文件路径初始化加载器。
        Args:
            file_path: 要加载的文件的路径。
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- 不接受任何参数
        """逐行读取文件的惰性加载器。
        当您实现惰性加载方法时，应使用生成器逐个生成文档。
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                # yield 关键字用于定义生成器函数。
                # 生成器函数是一种特殊类型的函数，它允许你逐步生成一个序列的值，而不是一次性返回整个序列。
                # 与普通的函数不同，生成器函数在每次调用时不会从头开始执行，而是从上次离开的地方继续执行。
                # 这使得生成器非常适合处理需要逐步生成或处理大数据集的情况
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

    # alazy_load是可选的。
    # 如果您省略了实现，将使用默认实现，该实现将委托给lazy_load！
    async def alazy_load(
            self,
    ) -> AsyncIterator[Document]:  # <-- 不接受任何参数
        """逐行读取文件的异步惰性加载器。"""
        # 需要aiofiles
        # 使用`pip install aiofiles`安装
        # https://github.com/Tinche/aiofiles
        import aiofiles
        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            async for line in f:
                # async for 逐行异步读取文件内容，并通过 yield 关键字逐行生成 Document 对象。
                # 异步生成器允许你在异步环境中逐步生成值，这对于处理 I/O 密集型任务（如文件读取或网络请求）非常有用，
                # 因为它可以在等待 I/O 操作完成时释放事件循环，以便处理其他任务。
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1


with open("./meow.txt", "w", encoding="utf-8") as f:
    quality_content = "喵喵🐱 \n 喵喵🐱 \n 喵😻😻"
    f.write(quality_content)
loader = CustomDocumentLoader("./meow.txt")

## 测试延迟加载接口
for doc in loader.lazy_load():
    print()
    print(type(doc))
    print(doc)

"""
<class 'langchain_core.documents.base.Document'>
page_content='喵喵🐱 
' metadata={'line_number': 0, 'source': './meow.txt'}

<class 'langchain_core.documents.base.Document'>
page_content=' 喵喵🐱 
' metadata={'line_number': 1, 'source': './meow.txt'}

<class 'langchain_core.documents.base.Document'>
page_content=' 喵😻😻' metadata={'line_number': 2, 'source': './meow.txt'}
"""

# async def alazy():
#     async for doc in loader.alazy_load():
#         print()
#         print(type(doc))
#         print(doc)

# 测试异步实现
#asyncio.run(alazy())

# load() 在诸如 Jupyter Notebook 之类的交互式环境中很有用。
# 在生产代码中避免使用它，因为急切加载假定所有内容都可以放入内存中，
# 而这并不总是成立，特别是对于企业数据而言
# print(loader.load())
