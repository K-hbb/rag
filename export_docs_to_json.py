import json
from ragtest import load_documents

# === Load documents from the txt and pdf folders ===
documents = load_documents(
    txt_folder="procedures_text_output",
    pdf_folder="docs"
)

# === Prepare output format ===
output = [{"doc_id": doc["doc_id"], "content": doc["content"]} for doc in documents]

# === Write to JSON file ===
with open("docs_with_context.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully saved {len(output)} documents to docs_with_context.json")
