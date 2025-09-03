# pip install ollama
import ollama

def list_to_string_with_ollama(Data, Question, model="llama3"):
    """
    Sends a list to Ollama locally and returns the model's string response.
    """
    input_text = ", ".join(map(str, input_list))

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": f"From this data:/n{Data}/n Find this {Question}, and give response."}
        ]
    )

    return response["message"]["content"]

if __name__ = "__main__":
  # ✅ Example usage
  my_list = ["apple", "banana", "cherry"]
  result = list_to_string_with_ollama(my_list)
  print(result)
