import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
 
torch.random.manual_seed(0)

model_path = "../Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
 
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": True,
    "temperature": 0.0,
    "do_sample": False,
}
 
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
