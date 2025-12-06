import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configure your API key
# Replace 'YOUR_API_KEY' with your actual API key
# It is recommended to store API keys securely, e.g., in environment variables
# genai.configure(api_key='AIzaSyAC90G8hBJLNVSk6EFgjvG40BI-mh2vVhA')

# Load environment variables
load_dotenv()

# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize the generative model
# 'gemini-pro' is a commonly used model for text generation
model = genai.GenerativeModel('gemini-2.0-flash')

genai.configure(api_key=GOOGLE_API_KEY)

# Define a prompt for the model
prompt = "Tell me a fun fact about space."

# Generate content based on the prompt
try:
    response = model.generate_content(prompt)

    # Print the generated text
    print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your API key is correct and you have network connectivity.")
