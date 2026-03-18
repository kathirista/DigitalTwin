import json
import os
import chromadb
from google import genai
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

class NewGoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model_name="models/gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Note: gemini-embedding-001 uses 'models/' prefix
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=input
        )
        return [e.values for e in response.embeddings]

from dotenv import load_dotenv
load_dotenv()
import os
API_KEY = os.environ.get("GEMINI_API_KEY")
DB_PATH = "/home/sruthi_korlakunta/MediumPosts/knowledgeDB/ChromaDB_Blogs"
JSONL_PATH = "/home/sruthi_korlakunta/MediumPosts/knowledgeDB/database.jsonl"

embedding_fn = NewGoogleEmbeddingFunction(api_key=API_KEY)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="blogs", embedding_function=embedding_fn)

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        doc_id = data.get("id", f"blog_{i}")
        text_to_embed = f"Title: {data.get('title', '')}\nContent: {data.get('content', '')}"
        
        collection.add(
            ids=[doc_id],
            documents=[text_to_embed],
            metadatas=[{"url": data.get("url", ""), "title": data.get("title", "")}]
        )
        print(f"Indexed: {data.get('title')}")

print(f"\nDone! Database built with gemini-embedding-001.")