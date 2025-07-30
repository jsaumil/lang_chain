from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'C:\Users\saumi\OneDrive\Documents\GitHub\lang_chain\final_main.pdf')

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[1].metadata)