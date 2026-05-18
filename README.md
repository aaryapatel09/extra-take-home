gmail custom category builder

takes a natural language category description (like "work travel receipts from airlines") and classifies a pile of gmail messages as matching or not.

setup:
- python 3.10+
- ollama running locally (brew install ollama, ollama pull llama3.2)
- pip install -r requirements.txt (in a venv ideally)

how to run:
  python classifier.py messages.jsonl "work-related travel receipts from airlines"

or just:
  python classifier.py
  (it will ask you for the category)

how it works:
instead of sending every email to an LLM (expensive, slow, doesnt scale), we use the LLM once to generate synthetic example emails — some that match the category, some that dont, and some tricky edge cases. then we embed everything with sentence-transformers locally and classify each real email based on whether its more similar to the "yes" pile or the "no" pile.

basically we manufacture our own training data from the LLMs world knowledge and use embeddings to do the actual classification. 1 LLM call total regardless of how many emails you have.

output is a json array:
  [{"message_id": "...", "is_in_category": true/false, "explanation": "..."}]

still working on:
- accuracy on vague categories is rough
- explanations are just similarity scores rn
- need a margin threshold for borderline cases
