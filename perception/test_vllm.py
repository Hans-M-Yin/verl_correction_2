from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18903/v1",
    api_key="EMPTY"
)

result = client.chat.completions.create(
    model="qwen3-vl-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)

print(result.choices[0].message.content)

response = client.responses.create(
    model="qwen3-vl-8b",
    instructions="You are a helfpul assistant.",
    input="Explain what MXFP4 quantization is."
)

print(response.output_text)