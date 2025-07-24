# eval_retriever.py

from retriever_wrapper import get_retrieved_docs
import json


def recall_at_k(retrieved, gold, k=5):
    """1 if any of the top-k retrieved docs is in gold, else 0."""
    return int(any(doc in gold for doc in retrieved[:k]))

def reciprocal_rank(retrieved, gold):
    """1/(rank of first relevant) or 0 if none found."""
    for i, doc in enumerate(retrieved):
        if doc in gold:
            return 1.0 / (i + 1)
    return 0.0

def main():
    with open("evaluation/data/queries_with_gold_docs.json", encoding="utf-8") as f:
        dataset = json.load(f)

    total_recall = 0
    total_mrr    = 0

    for item in dataset:
        query    = item["query"]
        gold_doc = item["gold_doc"]     # match your JSON key
        retrieved = get_retrieved_docs(query)

        total_recall += recall_at_k(retrieved, gold_doc, k=5)
        total_mrr    += reciprocal_rank(retrieved, gold_doc)

    n = len(dataset)
    print(f"Recall@5: {total_recall / n:.2f}")
    print(f"MRR: {total_mrr / n:.2f}")

if __name__ == "__main__":
    main()
