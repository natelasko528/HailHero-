import openai

# Remember to replace "your-z-ai-api-key" with your actual z.ai API key.
# For better security, load your API key from an environment variable.
# For example: api_key = os.environ.get("Z_AI_API_KEY")
api_key = "your-z-ai-api-key"

# The z.ai API endpoint URL.
base_url = "https://api.z.ai/api/paas/v4/"

try:
    # Initialize the OpenAI client with the z.ai API key and endpoint.
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Make a test call to the chat completions endpoint.
    # Using a model that is likely available on the z.ai platform.
    chat_completion = client.chat.completions.create(
        model="glm-4.5",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "Hello, please introduce yourself."
            }
        ]
    )

    # Print the response from the API.
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")