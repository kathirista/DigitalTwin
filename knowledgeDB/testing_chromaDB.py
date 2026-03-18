import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

# 1. Setup
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.environ.get("GEMINI_API_KEY")
client = chromadb.PersistentClient(path="/home/sruthi_korlakunta/MediumPosts/knowledgeDB/ChromaDB_Blogs") # Points to where your files are
embedding_fn = GoogleGenerativeAiEmbeddingFunction(api_key=api_key)

# 2. Connect to the collection you created
collection = client.get_collection(name="blogs", embedding_function=embedding_fn)

# 3. Ask a test question
query_text = "What does Sruthi say about having hate in her system?"
results = collection.query(
    query_texts=[query_text],
    n_results=1 # Give me the top 1 match
)

# 4. Print the result
print("\n--- Match Found ---")
print(f"Title: {results['metadatas'][0][0]['title']}")
print(f"Excerpt: {results['documents'][0][0][:200]}...")