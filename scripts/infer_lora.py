import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "/data/sjh/LLMwork/Qwen/finetune/models/Qwen-7B-Chat"
lora_path = "/data/sjh/LLMwork/Qwen/finetune/qwen-lora-output"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # 推荐
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, lora_path)

print(model.peft_config)

model.eval()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,  # 生成最大长度
        temperature=0.7,     # 随机性
        top_p=0.85,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

test_prompts = [
    "解释LoRA的低秩矩阵是如何工作的",
    "Qwen-7B用LoRA微调时，显存占用为什么只有12GB",
    "如何调整LoRA的r参数来平衡效果和显存"
]

for prompt in test_prompts:
    print(f"问题：{prompt}")
    print(f"回答：{generate_response(prompt)}\n")
    print("-"*50)