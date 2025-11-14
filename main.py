# main.py
"""
AmbedkarGPT - Fully Compatible RAG System (latest LangChain community split)
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Build Chroma vectorstore

def build_vectorstore(speech_path, persist_dir="chroma_db"):
    print("Loading speech.txt ...")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    print("Splitting text into chunks ...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print("Generating embeddings ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building Chroma DB ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectordb.persist()
    return vectordb

# Build manual RAG chain (no RetrievalQA, no chains)

def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="mistral")

    prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below to answer the question.

<context>
{context}
</context>

Question: {question}

Answer in simple English.
""")

    # Document -> string function
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # RAG chain (manual)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain



# CLI Loop
def interactive_loop(rag_chain):
    print("\nAmbedkarGPT is ready! Ask your questions. ('exit' to quit)\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = rag_chain.invoke(query)
        print("\nAnswer:", response, "\n")
        print("=" * 80 + "\n")

# Main
def main():
    speech_path = "speech.txt"

    if not os.path.exists(speech_path):
        print("ERROR: speech.txt not found!")
        return

    vectordb = build_vectorstore(speech_path)
    rag_chain = build_rag_chain(vectordb)
    interactive_loop(rag_chain)


if __name__ == "__main__":
    main()

