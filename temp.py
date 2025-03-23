import google.generativeai as genai

GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)


available_models = genai.list_models()
for model in available_models:
    print(model.name)
