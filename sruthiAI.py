import os
import chromadb
from google import genai
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

# 1. API Configuration
from dotenv import load_dotenv
load_dotenv()
import os
API_KEY = os.environ.get("GEMINI_API_KEY")
DB_PATH = "/home/sruthi_korlakunta/MediumPosts/knowledgeDB/ChromaDB_Blogs"

# 2. Embedding Function (Must match the one used during indexing)
class NewGoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model_name="models/gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=input
        )
        return [e.values for e in response.embeddings]

# 3. Initialize Clients
embedding_fn = NewGoogleEmbeddingFunction(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="blogs", embedding_function=embedding_fn)


genai_client = genai.Client(api_key=API_KEY)

def what_would_sruthi_say(user_query):
    # Retrieve top 3 matching blogs
    results = collection.query(query_texts=[user_query], n_results=3)
    context_text = "\n\n".join(results['documents'][0])
    
    # Simple prompt structure to avoid Pydantic errors
    full_prompt = f"""
    You are Sruthi Korlakunta's AI twin. Answer the question using ONLY the context provided below.
    If the answer isn't in the context, say "I haven't written about that yet, I talk about tech, 
    data engineering with python,
    immigration, career advice, fitness, bicycles, travel. Try these topics!!"
    
    Context from Sruthi's Blogs:
    {context_text}
    
    User Question: {user_query}
    """
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt
    )
    return response.text

if __name__ == "__main__":
    print("Sruthi AI is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        try:
            answer = what_would_sruthi_say(user_input)
            print(f"\nSruthi AI: {answer}")
        except Exception as e:
            print(f"\nError: {e}")