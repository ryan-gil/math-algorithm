import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit DeepSeek-14B verification requires a CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute dtype: {dtype}")
    return dtype


def load_model(model_id):
    dtype = require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate_prompts(model, tokenizer, prompts, max_input_tokens, max_new_tokens):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    finished = []
    for token_ids in generated:
        finished.append(bool((token_ids == tokenizer.eos_token_id).any().item()))
    return list(zip(texts, finished))
