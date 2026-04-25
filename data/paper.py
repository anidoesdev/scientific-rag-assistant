#extracting paper from arxiv
# import asyncio
import arxiv

# from langchain_classic.document_loaders.directory import DirectoryLoader
# from langchain_classic.document_loaders import PyPDFDirectoryLoader,PyPDFLoader

client = arxiv.Client()
def extract_papers(topic:str):
    papers = arxiv.Search(
        query=topic,
        max_results=4,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for data in client.results(papers):
        # print(data.title)
        # print(data.authors)
        # print(data.entry_id)
        data.download_pdf(dirpath="./data/raw")
    return

extract_papers("RAG methods")
extract_papers("LLM agents / tool use")
extract_papers("Fine-tuning methods")
extract_papers("Evaluation methods")
extract_papers("Long-context or memory systems")


# docs = PyPDFDirectoryLoader(
#     path="./docs",
#     glob="**/[!.]*.pdf",
#     loader_cls = PyPDFLoader,

# )

# # metadata = load_docs()
# # print(metadata)
# #to load the data from docs
# data = docs.load()
# for doc in data:
#     print(f"content preview: {doc.page_content[:100]}")
    

        