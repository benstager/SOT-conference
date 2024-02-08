import os # operating system for files
from langchain_community.vectorstores import Chroma # vector db
from langchain.document_loaders import PyPDFLoader # pdf loader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter # text chunking
from langchain.embeddings import OpenAIEmbeddings # embedding text
from langchain.llms import OpenAI # LLM
from langchain.chains import RetrievalQA # chain
def text_processor(documents):
    
    # creating text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # splitting pdf
    documents = text_splitter.split_documents(documents)

    # initializing vector db using Chroma
    vectordb = Chroma.from_documents(
        documents, 
        embedding=OpenAIEmbeddings(),
        persist_directory='./data'
    )
    
    return vectordb

def run_model(db, query):
    # running openAI
    llm = OpenAI()

    qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=db.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True
    )   

    return qa_chain.invoke({'query': query})

os.environ["OPENAI_API_KEY"] = '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'

pdf = '/Users/benstager/Desktop/business_report.pdf'

loader = PyPDFLoader(pdf)

documents = loader.load()

db = text_processor(documents)

db.persist()

query = input('Please ask the bot a question: ')

while query != 'quit' and query != 'q':
    result = run_model(db, query)
    print(result['result'].lstrip())
    query = input('Please ask the bot a question: ')

print('Thanks for using the bot!')