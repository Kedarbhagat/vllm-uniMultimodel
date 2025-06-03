from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path)
pages = []

def loadpdf():
    async for page in loader.alazy_load():
    pages.append(page)




