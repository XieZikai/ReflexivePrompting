import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from utils_logiqa import *
from utils_truthfulqa import *
from utils_mmlu import *
from utils_bigbench import *


GOOGLE_API_KEY = '<PLEASE_USE_YOUR_API_KEY>'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


def get_gemini_result(prompt):
    response = model.generate_content("You are a helpful assistant. " + prompt,
                                      safety_settings={
                                          HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                          HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                          HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                                      })
    return response.text


if __name__ == "__main__":
    experiment_logiqa(get_gemini_result)
    experiment_truthfulqa(get_gemini_result)
    experiment_mmlu(get_gemini_result)
    experiment_bigbench(get_claude_result)
