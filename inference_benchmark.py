import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Benchmark settings
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50
NUM_RUNS = 3

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Maximum tokens: {MAX_NEW_TOKENS}")
print(f"Test repetitions: {NUM_RUNS}")

# Loading the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if DEVICE == "cuda":
    model = model.cuda()
model.eval()

print("Model loaded")

# Benchmark function
def benchmark_inference(prompt, temperature=1.0, top_k=None, top_p=1.0, num_runs=NUM_RUNS):
    """
    Measures text generation speed.
    Returns: time to first token, average generation time, tokens/sec.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = inputs.cuda()

    input_length = inputs.shape[1]

    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True if temperature != 1.0 else False,
        pad_token_id=tokenizer.eos_token_id
    )

    times_first_token = []
    times_total = []
    tokens_generated = []

    for run in range(num_runs):
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(inputs, generation_config=generation_config)

        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end_time = time.time()

        total_time = end_time - start_time
        generated_tokens = outputs.shape[1] - input_length

        times_total.append(total_time)
        tokens_generated.append(generated_tokens)

    avg_total_time = sum(times_total) / len(times_total)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_second = avg_tokens / avg_total_time if avg_total_time > 0 else 0

    return {
        "avg_total_time": avg_total_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": tokens_per_second
    }


# Test prompts
test_prompts = [
    "How to cook a simple dinner?",
    "What is the best way to learn Python?",
    "I need help with my homework",
]

# Benchmark of various sampling parameters
# Configurations for testing
configs = [
    {"name": "Greedy (temperature=1.0)", "temperature": 1.0, "top_k": None, "top_p": 1.0},
    {"name": "Random (temperature=1.5)", "temperature": 1.5, "top_k": None, "top_p": 1.0},
    {"name": "Focused (temperature=0.7)", "temperature": 0.7, "top_k": None, "top_p": 1.0},
    {"name": "Top-K 50 (temperature=0.7)", "temperature": 0.7, "top_k": 50, "top_p": 1.0},
    {"name": "Top-P 0.9 (temperature=0.7)", "temperature": 0.7, "top_k": None, "top_p": 0.9},
]

print(f"\n{'Configuration':<35} | {'Tokens/sec':<12} | {'Time (s)':<10}")
print(f"{'-' * 35} | {'-' * 12} | {'-' * 10}")

results = []

for config in configs:
    prompt = test_prompts[0]
    metrics = benchmark_inference(
        prompt,
        temperature=config["temperature"],
        top_k=config["top_k"],
        top_p=config["top_p"]
    )

    results.append({
        "config": config["name"],
        "tokens_per_second": metrics["tokens_per_second"],
        "total_time": metrics["avg_total_time"]
    })

    print(f"{config['name']:<35} | {metrics['tokens_per_second']:<12.2f} | {metrics['avg_total_time']:<10.3f}")

# Benchmark on different prompts
print("\nSpeed on different prompts (temperature=0.7)")

print(f"\n{'Prompt':<50} | {'Token/sec':<12}")
print(f"{'-' * 50} | {'-' * 12}")

for prompt in test_prompts:
    metrics = benchmark_inference(prompt, temperature=0.7)
    display_prompt = prompt if len(prompt) <= 47 else prompt[:44] + "..."
    print(f"{display_prompt:<50} | {metrics['tokens_per_second']:<12.2f}")

# Text generation
print("\nText generation")

demo_prompt = "How to fix a slow computer?"
print(f"\nPrompt: '{demo_prompt}'")
print(f"\nGenerated response (temperature=0.7, top_k=50):")

inputs = tokenizer.encode(demo_prompt, return_tensors="pt")
if DEVICE == "cuda":
    inputs = inputs.cuda()

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

