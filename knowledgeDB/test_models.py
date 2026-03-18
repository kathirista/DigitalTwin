from google import genai

client = genai.Client(api_key="AIzaSyA6320BdYESvx4cCAyvgDTyF2ev7biGIrw")

print("Available models for your key:")
for model in client.models.list():
    # We only care about models that support 'generateContent'
    if 'generateContent' in model.supported_actions:
        print(f" - {model.name}")