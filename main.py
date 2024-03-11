import pickle
import os
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import (VectorStoreIndex, StorageContext, PromptTemplate, load_index_from_storage, Settings)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llama_pack import download_llama_pack
import subprocess
from llama_index.readers.file import FlatReader
from pathlib import Path
from llama_index.core.node_parser import (
    UnstructuredElementNodeParser,
)
import pickle
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path

pdf_path="./data/"
html_path="./html/"


def load_embedding_model():
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 
    print("embedding model loaded")
    
    
def convert_pdf_to_html():
    print("Pdf files converted to html")
    file_list = os.listdir(pdf_path)
    
    files_with_path = [pdf_path + s for s in file_list]
    files = ' '.join(files_with_path)

    
    command = f"pdfunite {files} {pdf_path}final.pdf"
    subprocess.call(command, shell=True)
        
    command = f"pdf2htmlEX {pdf_path}final.pdf --dest-dir {html_path}"
    subprocess.call(command, shell=True)
    
    command = f"rm {pdf_path}final.pdf"
    subprocess.call(command, shell=True)
    
def get_nodes():
    # Create a flat reader to read html files
    reader = FlatReader()
    docs = reader.load_data(Path("html/final.html"))   
    
    # Create a node parser object to parse the nodes of tables in the html file
    node_parser = UnstructuredElementNodeParser()
    
    # Create a pickle file and store the nodes
    if not os.path.exists("doc_nodes.pkl"):
        docs_raw_nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)
        pickle.dump(docs_raw_nodes, open("doc_nodes.pkl", "wb"))
    else:
        file = open("doc_nodes.pkl", "rb")
        docs_raw_nodes = pickle.load(file)
    
    # Get the base nodes and node mappings
    return node_parser.get_base_nodes_and_mappings(docs_raw_nodes)
     
    
def main():
    PERSIST_DIR = "./basic/storage"
    
    # Convert pdf files to html
    convert_pdf_to_html()
    
    # Get the nodes of tables in the html file
    base_nodes, node_mappings = get_nodes()
    
    # Get the embedding model
    embedding_model = load_embedding_model()
    Settings.embed_model = embedding_model
    # Create a vector store of base nodes
    index = VectorStoreIndex(base_nodes, embed_model=embedding_model)
    
    vector_retriever = index.as_retriever(similarity_top_k=3)
    
    llama = Ollama(
        model="llama2",
        request_timeout=800.0,
        )
    
    vector_query_engine = index.as_query_engine(similarity_top_k=3, llm=llama)
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=node_mappings)
    
    query_engine = RetrieverQueryEngine.from_args(recursive_retriever, llm=llama)
    
    qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )

    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
    print("Press ctr + c to exit")
    while True:
        query = input("Enter your query: ")
        response = query_engine.query(query)
        print(response)

    
if __name__ == "__main__":
    main()