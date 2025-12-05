from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/workspace/reap/artifacts/Qwen3-VL-30B-A3B-Instruct/reap-calib-mix/pruned_models/reap-seed_42-0.40"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """ローマ帝国がなぜ反映したのか、小学生にもわかるように教えてください"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512,
                               do_sample=True,      # ← サンプリングを有効化
    temperature=0.7,     # ← 1.0より小さいほど保守的
    top_p=0.9,           # or top_k=50 など)
                              )
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
