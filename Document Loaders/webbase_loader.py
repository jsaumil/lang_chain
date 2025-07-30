from langchain_community.document_loaders import WebBaseLoader

# can extract multiple urls

url = 'https://medium.com/saarthi-ai/an-overview-of-speaker-recognition-with-sincnet-2a613a072ae5'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)