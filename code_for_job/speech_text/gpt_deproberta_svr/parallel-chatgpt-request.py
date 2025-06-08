import os
import openai
import asyncio
from flask_cors import CORS
from flask import Flask, request, jsonify
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

app = Flask(__name__)
CORS(app)

os.environ["OPENAI_API_KEY"] = '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' # for my personal account
openai.api_key = os.environ["OPENAI_API_KEY"]

MAX_TOKEN_ALLOWED_PER_PROMPT = 4000

async def chatgpt_request(prompt):
    if (len(tokenizer.encode(prompt)) > MAX_TOKEN_ALLOWED_PER_PROMPT):
        print("Prompt too long")
        return ""
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",

            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        return response['choices'][0]['message']['content']
    except: # NOTE: this is a bad practice
        print("Error in chatgpt_request")
        return ""

@app.route('/parallel-requests', methods=['POST'])
async def parallel_requests():
    # get data from body
    data = request.get_json()
    prompts = data['prompts']

    prompts = prompts.split("---")
    coroutines = [chatgpt_request(prompt) for prompt in prompts]
    gptResponses = await asyncio.gather(*coroutines)
    print(gptResponses)
    return jsonify({ 'response': gptResponses })

if __name__ == '__main__':
    app.run(debug=False, port=4100,host='0.0.0.0')

