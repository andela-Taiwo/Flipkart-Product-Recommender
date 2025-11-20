from flask import Flask, render_template, request
from flipkart.data_loader.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder


def create_app():
    app = Flask(__name__)
    data_ingestor = DataIngestor()
    vector_store = data_ingestor.ingest_data(load_existing_data=True)
    rag_chain_builder = RAGChainBuilder(vector_store)
    rag_chain = rag_chain_builder.build_rag_chain()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/chat', methods=['POST'])
    def chat():
        user_input = request.form['msg']
        
        response = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "users-session"}})["answer"]
        return response
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=8000, host='0.0.0.0')

