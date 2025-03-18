from langchain_openai import ChatOpenAI
# pip install -qU langchain langchain-openai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import XMLOutputParser
# pip install defusedxml


model = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 还有一个用于提示语言模型填充数据结构的查询意图。
actor_query = "生成周星驰的简化电影作品列表，按照最新的时间降序"
# 设置解析器 + 将指令注入提示模板。
parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre"])
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
# print(parser.get_format_instructions())
chain = prompt | model
response = chain.invoke({"query": actor_query})
xml_output = parser.parse(response.content)
print(response.content)
"""
```xml
<movies>
    <actor>
        <name>周星驰</name>
        <film>
            <name>美人鱼</name>
            <genre>喜剧, 奇幻</genre>
        </film>
        <film>
            <name>西游·降魔篇</name>
            <genre>喜剧, 奇幻</genre>
        </film>
        <film>
            <name>长江七号</name>
            <genre>喜剧, 科幻</genre>
        </film>
        <film>
            <name>功夫</name>
            <genre>喜剧, 动作</genre>
        </film>
        <film>
            <name>少林足球</name>
            <genre>喜剧, 运动</genre>
        </film>
    </actor>
</movies>
```
"""
