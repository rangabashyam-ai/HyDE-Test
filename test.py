from flask import Flask, request, jsonify, render_template
from langchain.llms import Cohere
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document  # Ensure this import is correct
import fitz  # PyMuPDF
import os

app = Flask(__name__)

# Set up the LLM
os.environ["COHERE_API_KEY"] = "XWDxqLQyDNtqLLEFHSNtuYrz4kHhntwmBWXnQYO2"
llm = Cohere()

# Initialize HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}
embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

class PDFTextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        document = fitz.open(self.file_path)
        texts = []
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            texts.append(page.get_text())
        return texts

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)

vectorstore = None  # Initialize the vector store variable

@app.route('/', methods=['GET', 'POST'])
def home():
    global vectorstore  # Use the global vectorstore
    answer = None
    question = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error="No selected file")
            if file:
                file_path = f"/tmp/{file.filename}"
                file.save(file_path)

                # Load and process the PDF
                loader = PDFTextLoader(file_path)
                docs = loader.load()
                splits = text_splitter.split_documents([Document(page_content=doc) for doc in docs])

                # Index documents
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
                return render_template('index.html', message="File processed successfully")

        if 'question' in request.form:
            question = request.form['question']
            if not question:
                return render_template('index.html', error="Question is required")

            if vectorstore is None:
                return render_template('index.html', error="No documents uploaded yet.")

            template = """Please write a concise passage to answer the question
            Question: {question}
            Passage:"""
            prompt_hyde = ChatPromptTemplate.from_template(template)

            generate_docs_for_retrieval = (
                prompt_hyde | llm | StrOutputParser() 
            )

            generated_doc = generate_docs_for_retrieval.invoke({"question": question})

            retriever = vectorstore.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(generated_doc)

            template = """Answer the following question based on this context:
            {context}
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            final_rag_chain = (
                prompt | llm | StrOutputParser()
            )

            answer = final_rag_chain.invoke({"context": retrieved_docs, "question": question})

    return render_template('index.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True)
