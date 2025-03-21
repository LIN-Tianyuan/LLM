from langchain_community.document_loaders.csv_loader import CSVLoader
import tempfile


string_data = """
"Team", "Payroll (millions)", "Wins"
"Nationals",     81.34, 98
"Reds",          82.20, 97
"Yankees",      197.96, 95
"Giants",       117.62, 94
""".strip()

with tempfile.NamedTemporaryFile(delete=False, mode="w+") as temp_file:
    temp_file.write(string_data)
    temp_file_path = temp_file.name

# 当直接使⽤ CSV 字符串时，可以使⽤ Python 的 tempfile 。
loader = CSVLoader(file_path=temp_file_path)
data = loader.load()
for record in data[:2]:
    print(record)