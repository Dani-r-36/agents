import os
import time
import torch
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM



"""
normal docker wiht llm hosted in api 
"""
# app = FastAPI()

# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# hf_token = os.getenv("HF_TOKEN")
# # Load tokenizer and model using PyTorch
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     token=hf_token,
#     dtype=torch.float32,
#     device_map="auto"
# )
# # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# class PromptRequest(BaseModel):
#     prompts: list[str]

# @app.post("/generate")
# async def generateResponse(request: PromptRequest):
#     # model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
#     # llm = LLM(model=model_name)
#     # outputs = llm.generate(request.prompts, sampling_params)
#     response = []
#     for prompt in request.prompts:
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=256)
#         generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         response.append({
#             "prompt": prompt,
#             "output": generated_text
#         })
#     return {"outputs": response}


MLFLOW_DB = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment("llama-3.2-fastapi-inference") 
# the set experiement puts all excuttion logs under single dashboard project llmam....

model = None
tokenizer = None
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    hf_token = os.getenv("HF_TOKEN")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    yield #pauses excustions after setup done so api can begin taking http request 

app = FastAPI(lifespan=lifespan)

# pydantic inputs using fastapi 
class PromptRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = 256

@app.post("/generate")
async def generateResponse(request: PromptRequest):
    start_time = time.time()
    response = []
    total_tokens_generated = 0

    # 2. Start an MLflow Run for this HTTP request
    with mlflow.start_run(run_name="generate_request"):
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_tokens", request.max_tokens)
        mlflow.log_param("batch_size", len(request.prompts))

        for prompt in request.prompts:
            messages = [{"role": "user", "content": prompt}]
            # apply_chat_template coverts string promotps into format for tokenizer and converts them into pytorch tensors 
            inputs = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]
            # no.grad stops pytorch autofrad engine for gradient tracking, used in training (speeds up computation)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=request.max_tokens,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            # generated_ids = outputs[0][inputs.shape[-1]:] #inputs.shape[-1] calcualtes number of tokens in original prompt
            generated_ids = outputs[0][prompt_len:]
            # above gets slice of arry of new generated token id
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # tokenizer.decorde turns ids into text and takes out special control tokens
            total_tokens_generated += len(generated_ids)
            response.append({
                "prompt": prompt,
                "output": generated_text
            })

        # 3. Log Performance Metrics to MLflow
        latency = time.time() - start_time
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("total_tokens_generated", total_tokens_generated)
        if latency > 0:
            mlflow.log_metric("tokens_per_second", total_tokens_generated / latency)

        # turns the raw input prmots and ouput into json using log_dict
        mlflow.log_dict({"request": request.dict(), "response": response}, "payload.json")
        
    return {"outputs": response}