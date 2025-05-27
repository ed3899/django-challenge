# django-challenge
- Used conda as a package manager
- Built using codespaces
- Run conda env create -f environment.yml
- Make sure https://tesseract-ocr.github.io/tessdoc/Installation.html is installed
- Run docker compose up -d (spins up chroma db)
- run python manage.py shell
- Run
```
from core_processor.chromadb_manager import chroma_manager
chroma_manager.load_initial_dataset("/workspaces/django-challenge/dataset")
```
- You will be prompted for an open ai key, insert it
- Load the dataset
- Done in a synchronous way. Just to showcase skills, asyncio can potentially be used to speed up with concurrency

# Notes:
I'd wouldn't have any problem finishing this but my priorities have shifted to other opportunities and I don't want to dedicated as much time to this poc.

The missing parts would be the command line to call this directly from a python admin command instead of within the shell.

The other part would be the agent or chatbot to RAG given the inserted docs in Chroma to interact with when inserting new docs (different from the load_initial_dataset function, which is only used once at the very beginning)

Tests were skipped due to similar reasons.