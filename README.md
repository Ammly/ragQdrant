### Instructions 

1. Clone the repo 
` gh repo clone Ammly/ragQdrant `
2. Install dependencies 
```shell
cd ragQdrant
poetry install
```
3. Start Qdrant database
` docker compose up -d `
4. Crete a vector store from your documents in 'data' folder
` poetry run python index_documents.py `
5. Run the app 
` poetry run python app.py `