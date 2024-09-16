from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptHelper,load_index_from_storage,ListIndex
from langchain_openai import ChatOpenAI
from flask import Flask, render_template, request, jsonify
import os

os.environ["OPENAI_API_KEY"] = 'sk-'
app = Flask(__name__)

def construct_index(directory_path):
    # Assuming the PromptHelper parameters have changed
    max_input_size = 4096
    chunk_size_limit = 600
    chunk_overlap_ratio = 0.1  # Adjust based on actual documentation

    prompt_helper = PromptHelper(max_input_size,chunk_size_limit=chunk_size_limit, chunk_overlap_ratio=chunk_overlap_ratio)

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=512)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = VectorStoreIndex.from_documents(documents, llm=llm, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir='index.json')

    return index

index = construct_index("docs")

def chatbot(input_text):
    #index = VectorStoreIndex.load_from_disk('index.json')
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form['query']
    response = chatbot(query)  # Use the chatbot function to get the response
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
