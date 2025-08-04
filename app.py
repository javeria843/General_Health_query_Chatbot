import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Use Falcon model (public and no auth needed)
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Safety filter
def is_safe_query(query):
    unsafe_keywords = ['dose', 'dosage', 'prescribe', 'prescription', 'take how much', 'overdose']
    return not any(word.lower() in query.lower() for word in unsafe_keywords)

# Generate response
def generate_response(user_query):
    if not is_safe_query(user_query):
        return "❗ Sorry, I cannot provide dosage or prescription advice. Please consult a licensed doctor."

    prompt = f"""You are a safe medical assistant. Answer the user's question in a friendly and clear way, but never give harmful or diagnostic medical advice.
User: {user_query}
Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_k=50, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.split("Assistant:")[-1].strip()

# Gradio app
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="user_query"),
    outputs=gr.Textbox(label="output"),
    title="🩺 Health Chatbot (Falcon-7B)",
    description="Ask general health questions. Safe, friendly answers. No harmful or prescription advice provided."
)

demo.launch()
