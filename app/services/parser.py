# from langchain_classic.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import ArxivLoader

def parser():
    #getting documents from the arxiv 
    papers = ArxivLoader(
        query= "LLM systems",
        load_max_docs=20
        
    )
    data = papers.load()
    return data[0].metadata
print(parser())
    