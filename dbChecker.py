import chromadb

# Connect to your persistent Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

for coll in client.list_collections():
    c = client.get_collection(coll.name)
    result = c.get(include=["metadatas"], limit=999999)
    total = len(result["ids"])
    with_meta = sum(1 for m in result["metadatas"] if m)
    print(f"{coll.name}: total={total}, with_metadata={with_meta}")
