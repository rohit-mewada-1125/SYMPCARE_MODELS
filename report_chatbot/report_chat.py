from openai import OpenAI
import json, re
import pandas as pd
from flask import Flask, request, jsonify

# ✅ Load cleaned reference range CSV
ranges_df = pd.read_csv("report_chatbot\data\lab_range.csv")

# Save last analyzed report here
last_report_data = None

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # ⚠️ Replace after regenerating
)

# ✅ Extract patient details from report
def extract_patient_details(text):
    name = re.search(r"(Name|Patient Name)[:\- ]+([A-Za-z ]+)", text, re.IGNORECASE)
    age = re.search(r"(Age)[:\- ]+(\d+)", text, re.IGNORECASE)
    gender = re.search(r"(Gender|Sex)[:\- ]+(Male|Female)", text, re.IGNORECASE)
    date = re.search(r"(Date|Collected On|Report Date)[:\- ]+([0-9\/\-.]+)", text, re.IGNORECASE)
    lab = re.search(r"(Lab|Hospital|Clinic)[:\- ]+([A-Za-z0-9 ,]+)", text, re.IGNORECASE)

    return {
        "name": name.group(2).strip() if name else "",
        "age": age.group(2) if age else "",
        "gender": gender.group(2) if gender else "",
        "test_date": date.group(2) if date else "",
        "lab_name": lab.group(2).strip() if lab else ""
    }

def lookup_range(test_name):
    row = ranges_df[ranges_df["Test Name"].str.lower() == test_name.lower()]
    if row.empty:
        return None, None
    r = row.iloc[0]
    return float(r["Normal_Min"]), float(r["Normal_Max"])

# ✅ Extract values from report text
def extract_parameters(text):
    parameters = {}
    lines = text.split("\n")
    for line in lines:
        match = re.findall(r"([A-Za-z][A-Za-z \-()]+)[\:\-\s]+(\d+\.?\d*)", line)
        if match:
            for name, value in match:
                try:
                    parameters[name.strip()] = float(value)
                except:
                    pass
    return parameters

def analyze_report_text(extracted_text):
    global last_report_data

    patient_details = extract_patient_details(extracted_text)
    extracted_params = extract_parameters(extracted_text)

    enriched_params = []
    for test_name, patient_value in extracted_params.items():
        normal_min, normal_max = lookup_range(test_name)
        enriched_params.append({
            "name": test_name,
            "patient": patient_value,
            "normal_min": normal_min,
            "normal_max": normal_max
        })

    system_prompt = {
    "role": "system",
    "content": (
        "You are a medical report analysis assistant. "
        "Explain the medical report in simple, friendly language. "
        "The explanation (summary) MUST be at least 100 words. Not less.\n\n"
        "Also provide exactly 2 image keywords that are relevant medical terms "
        "based on the conditions, symptoms, or organs involved in the report.\n\n"
        "Your response MUST be in the following JSON format ONLY:\n"
        "{\n"
        '  \"summary\": \"<simple medical explanation of at least 100 words>\",\n'
        '  \"image_keywords\": [\"keyword1\", \"keyword2\"]\n'
        "}\n\n"
        "STRICT RULES:\n"
        "- Do NOT include anything outside the JSON. No notes. No markdown.\n"
        "- The summary MUST be at least **100 words**. If it's less, rewrite it.\n"
        "- The image_keywords array MUST contain **exactly 2 keywords**, no more, no less.\n"
        "- The image_keywords must be short medical terms (disease, organ, symptom).\n"
        "- Do NOT use full sentences or descriptive phrases in image_keywords."
    )
}


    user_prompt = {
        "role": "user",
        "content": (
            f"Patient Details:\n{json.dumps(patient_details)}\n\n"
            f"Test Values:\n{json.dumps(enriched_params)}\n\n"
            "Now interpret the report and return JSON only."
        )
    }

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1-distill-llama-70b:free",
        messages=[system_prompt, user_prompt],
        temperature=0.3
    )
    
    response_text = completion.choices[0].message.content.strip()

    # ✅ Parse response JSON
    try:
        result = json.loads(response_text)
    except:
        # fallback in case model responds incorrectly
        result = {"summary": response_text, "image_keywords": []}

    # Store in memory for follow-up chat
    last_report_data = result["summary"]

    return result



chat_history = []
last_report_data = None

def chat_with_report_bot(user_input):
    global last_report_data, chat_history

    if last_report_data is None:
        reply = "Please upload a medical report first before asking questions 😊"
        return reply

    context_message = {
        "role": "system",
        "content": (
            f"Here is the patient's report summary:\n\n{last_report_data}\n\n"
            "When responding, explain your answers based specifically on this report."
        )
    }

    chat_history.append(context_message)
    chat_history.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=chat_history,
        temperature=0.5
    )

    reply = completion.choices[0].message.content.strip()

    chat_history.append({"role": "assistant", "content": reply})
    return reply

