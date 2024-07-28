### Instructions 

1. Clone the repo 
```shell 
gh repo clone Ammly/ragQdrant 
```
2. Install dependencies
```shell
cd ragQdrant
poetry install
```
3. Start Qdrant database
```shell 
docker compose up -d
```
4. Crete a vector store from your documents in `data` folder
```shell 
mkdir data
#Add your documents(pdf, ppt, md...) to this folder

poetry run python index_documents.py 
```
5. Run the app
```shell 
poetry run python app.py 
```