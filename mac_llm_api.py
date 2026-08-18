# from vllm import LLM, SamplingParams

# def main():
#     conversations = [
#         [{"role": "user", "content": "Who is the president of the United States?"}],
#         # [{"role": "user", "content": "What is the capital of France?"}],
#     ]
#     sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
#     model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
#     llm = LLM(model=model_name)
#     outputs = llm.chat(conversations, sampling_params)
#     for i, output in enumerate(outputs):
#         user_prompt = conversations[i][0]["content"]
#         response_text = output.outputs[0].text.strip()

#         print("\n" + "═" * 60)
#         print(f"❓ PROMPT:\n{user_prompt}")
#         print("─" * 60)
#         print(f"🤖 RESPONSE:\n{response_text}")
#         print("═" * 60)

from fastapi import FastAPI
from pydantic import BaseModel
from mlx_lm import load, generate

app = FastAPI()
model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(model_name)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

class PromptRequest(BaseModel):
    prompts: list[str]

@app.post("/generate")
async def generateResponse(request: PromptRequest):
    # model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    # llm = LLM(model=model_name)
    # outputs = llm.generate(request.prompts, sampling_params)
    response = []
    for prompt in request.prompts:
        generated_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            # temp=0.8
        )
        response.append({
            "prompt": prompt,
            "output": generated_text
        })
    return {"outputs": response}


# if "__main__"==__name__:
#     main()