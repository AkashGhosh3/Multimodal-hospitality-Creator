import chromadb

client = chromadb.Client()
collection = client.create_collection(name="hospitality_prompts")

def store_prompt(prompt):
    collection.add(
        documents=[prompt],
        ids=[str(hash(prompt))]
    )

def retrieve_prompts(query):
    return collection.query(query_texts=[query], n_results=2)
