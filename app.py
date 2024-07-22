from flask import Flask, request, jsonify
from langchain.llms import Cohere
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF
import os

app = Flask(__name__)

# Set up the LLM
os.environ["COHERE_API_KEY"] = "YOUR_COHERE_API_KEY"
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

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = f"/tmp/{file.filename}"
        file.save(file_path)
        
        # Load and process the PDF
        loader = PDFTextLoader(file_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        
        # Index documents
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
        retriever = vectorstore.as_retriever()
        
        return jsonify({"message": "File processed successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    template = """Please write a concise passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = (
        prompt_hyde | llm | StrOutputParser() 
    )
    
    generated_doc = generate_docs_for_retrieval.invoke({"question": question})
    
    retrieval_chain = generate_docs_for_retrieval | retriever
    retrieved_docs = retrieval_chain.invoke({"question": question})

    template = """Answer the following question based on this context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        prompt | llm | StrOutputParser()
    )

    output = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
    return jsonify({"answer": output})


@app.route('/')
def home():
    return "hi this is HyDE RAG App"

if __name__ == '__main__':
    app.run(debug=True)
