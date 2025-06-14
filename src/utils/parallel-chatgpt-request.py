from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from transformers import GPT2TokenizerFast

import os
import openai
from dotenv import load_dotenv

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

app = Flask(__name__)
CORS(app)

# Load environment variables from a local .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Make sure the API key is found
if not api_key:
    raise ValueError("Error: OPENAI_API_KEY not found in environment variables. "
                     "Please create a .env file in the E-DAIC directory and add your key.")

openai.api_key = api_key

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

