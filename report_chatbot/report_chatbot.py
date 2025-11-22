import ollama
import json
import pandas as pd
import re
from flask import Flask, request, jsonify


# ✅ Load cleaned reference range CSV
ranges_df = pd.read_csv("report_chatbot\data\lab_range.csv")

# Save last analyzed report here
last_report_data = None

chat_history = [
    {
        "role": "system",
        "content": (
            "You are an AI Health Assistant. "
            "Speak clearly and simply. Do NOT output JSON here."
        )
    }
]


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


# ✅ RAG lookup
def lookup_range(test_name):
    row = ranges_df[ranges_df["Test Name"].str.lower() == test_name.lower()]
    if row.empty:
        return None, None
    r = row.iloc[0]
    return float(r["Normal_Min"]), float(r["Normal_Max"])


# ✅ Main function to analyze report


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
            "You are a medical report analysis assistant. Explain the report in simple language."
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            f"Patient Details:\n{json.dumps(patient_details)}\n\n"
            f"Test Values:\n{json.dumps(enriched_params)}\n\n"
            "Now provide a simplified interpretation summary and lifestyle tips."
        )
    }

    response = ollama.chat(model="llama3.1", messages=[system_prompt, user_prompt])
    summary = response["message"]["content"].strip()

    last_report_data = summary 

    # ✅ return raw summary instead of jsonify
    return {"summary": summary}



def chat_with_report_bot(user_input):
    global last_report_data, chat_history

    if last_report_data is None:
        reply = "Please upload a report first before asking questions 😊"
        chat_history.append({"role": "assistant", "content": reply})
        return reply

    # Add report context
    context_message = {
        "role": "system",
        "content": (
            f"Here is the patient's report data:\n{last_report_data}\n"
            "When responding, explain clearly based on these values."
        )
    }

    chat_history.append(context_message)
    chat_history.append({"role": "user", "content": user_input})

    # Collect streaming chunks into one final response
    full_reply = ""
    for chunk in ollama.chat(model="llama3.1", messages=chat_history, stream=True):
        full_reply += chunk["message"]["content"]

    chat_history.append({"role": "assistant", "content": full_reply})

    return full_reply
