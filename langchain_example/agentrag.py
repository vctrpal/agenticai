import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Set your API key
os.environ["GOOGLE_API_KEY"] = " "

# 2. Prepare Sample Source Documents
raw_documents = [
    Document(page_content="Project Titan is an internal initiative to migrate all legacy microservices to Kubernetes by Q3."),
    Document(page_content="The team lead for Project Titan is Alice Morgan, reporting to Director David Vance."),
    Document(page_content="The staging deployment freeze occurs annually between December 15 and January 5.")
]

# 3. Chunk the Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(raw_documents)

# 4. Embed and Store in Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. Define the Prompt Template
template = """Answer the question based strictly on the provided context:

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)

# Helper function to join retrieved doc chunks into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Build the RAG Chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Query the Chain
question = "Who leads Project Titan and when is the migration deadline?"
response = rag_chain.invoke(question)

print("Answer:\n", response)