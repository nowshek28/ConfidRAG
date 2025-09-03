# pip install ollama
import ollama
from langchain_core.documents import Document

client = ollama.Client(host='http://127.0.0.1:11434')

def list_to_string_with_ollama(Data, Question, model="llama3.2:1b"):
    """
    Sends a list to Ollama locally and returns the model's string response.
    """
    all_texts   = []
    all_sources = []
    all_chunkID = []
    all_scores  = []

    for item in Data:
        # Handle case: sometimes it's just Document, sometimes (Document, score)
        doc, score = (item if isinstance(item, tuple) else (item, None))
        
        all_texts.append(doc.page_content)
        all_sources.append(doc.metadata.get("source"))
        all_chunkID.append(doc.metadata.get("chunk_id"))
        all_scores.append(score)

    # ✅ If you want all page_content combined into one long string:
    combined_text = " ".join(all_texts)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": f"From this data:/n{combined_text}/n Find this {Question}, and give a short response."}
        ],
        options={"num_ctx": 2048, "temperature": 0.2}
    )

    return response["message"]["content"], all_sources, all_chunkID, all_scores

if __name__ == "__main__":
  # ✅ Example usage
  my_list = [(Document(
   metadata={
     'source': 'C:\\Users\\nowsh\\AppData\\Roaming\\ABHISHEKS PROJECT\\ConfidRAG\\DataSet\\01_company_handbook.txt',
     'source_tag': 'C:\\Users\\nowsh\\AppData\\Roaming\\ABHISHEKS PROJECT\\ConfidRAG\\DataSet\\01_company_handbook.txt',
     'chunk_id': 0,
     'char_len': 866
   },
   page_content="Doc-ID: CH-01\nTitle: Acme Vision Labs — Company Handbook (v1.2, updated 2025-08-01)\n\nOverview\n--------\nThis handbook summarizes how we work at Acme Vision Labs (AVL), a fictional\ncomputer-vision startup. It is designed to be searchable. Use it as a\nknowledge base for policies, definitions, and contacts.\n\n1) Working Hours & Core Time\n • Standard hours: 9:00–18:00 JST (1 hour lunch).\n • Core time: 10:00–16:00 JST for meetings and synchronous work.\n • Remote work: Up to 3 days/week with manager approval.\n • Public holidays follow the Japan calendar.\n\n2) Leave Policy\n • Annual leave: 15 days/year; carry-over up to 5 days.\n • Sick leave: Separate from annual leave; no doctor's note required for ≤2 days.\n • Parenting leave: Up to 12 months; contact HR for details.\n • To request leave, submit on the HR portal at least 3 business days in advance."
 ),
 0.4121399
)]
  result,_,_,_ = list_to_string_with_ollama(my_list, "What is the leave policy?")
  print(result)
