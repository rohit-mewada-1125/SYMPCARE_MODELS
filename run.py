from flask import Flask, request, render_template, jsonify, Response
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
from collections import Counter
from flask_cors import CORS
from chatbot import chatbot
from mental_chat import role_models,questions_by_role,role_conditions
from ayur_centres import fetch_hospitals_from_osm
from report_chatbot.report_chat import analyze_report_text, chat_with_report_bot
from report_chatbot.ocr import extract_text_from_report
from symptoms import l1,symp132,disease,nl1

# from drug_disease_chatbot.drug_disease import build_or_load_faiss_index, generate_answer
# from drug_interaction_system.drug_interact import retrieve_context,load_sentences,build_or_load_faiss


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

label_encoders = {
    'gender': LabelEncoder(),
    'employment_status': LabelEncoder(),
    'family_history': LabelEncoder()
}

label_encoders['gender'].fit(['Male', 'Female', 'Other'])
label_encoders['employment_status'].fit(['Employed', 'Unemployed', 'Student'])
label_encoders['family_history'].fit(['Yes', 'No'])

clf_rf,clf_tree,clf_nb, model1, model,neural_model,accuracy,clf_rf1 = None, None, None,None,None,None,None,None


{# def load_trained_model():
#     global clf_tree, clf_rf, clf_nb, model1, model,neural_model,accuracy,clf_rf1
#     try:
#         with open('trained models\clf_nb.pkl', 'rb') as nb:
#             clf_nb= pickle.load(nb)
#         with open('trained models\clf_rf.pkl', 'rb') as rf:
#             clf_rf1= pickle.load(rf)
#         with open('trained models\clf_tree.pkl', 'rb') as tree:
#             clf_tree= pickle.load(tree)
#         with open('trained models\disease_prediction_rf_model.pkl', 'rb') as nn_rf:
#             neural_model= pickle.load(nn_rf)

#         # with open("disease_prediction_model\disease_model.pkl", "rb") as f:
#         #     model_data = pickle.load(f)
#         #     clf_rf = model_data["model"]
#         #     accuracy = model_data["accuracy"]

#         with open('trained models\mental_health_model.pkl', 'rb') as model_file:
#             model1 = pickle.load(model_file)
       
       
#         if os.path.exists('trained models\skin_disease_model.h5'):
#             model = load_model('trained models\skin_disease_model.h5')
#         else:
#             print("No trained model found for skin disease detection.")
#     except Exception as e:
#         print(f"Error loading models: {e}")
}

def predict_disease(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

@app.route("/")
def home():
    return " MODEL IS RUNNING "

@app.route('/skin-predict', methods=['POST'])
def predict_route():
    global clf_tree, clf_rf, clf_nb, model1, model,neural_model,accuracy,clf_rf1
    if os.path.exists('trained models\skin_disease_model.h5'):
            model = load_model("trained models/skin_disease_model.h5", compile=False)
            

    else:
            print("No trained model found for skin disease detection.")
    if model is None:
        return jsonify({"error": "Model is not trained yet."}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files['image']
    file_path = f'temp/{file.filename}'
    file.save(file_path)
    
    predicted_class = predict_disease(file_path)
    disease_classes = os.listdir('skin_disease_dataset')
    predicted_disease = disease_classes[predicted_class[0]]
    
    return jsonify({"prediction": predicted_disease})

@app.route('/symptoms-predict', methods=['POST'])
def predict_symptoms_manual():
    global clf_tree, clf_rf, clf_nb, model1, model,neural_model,accuracy,clf_rf1
    # with open('trained models\clf_nb.pkl', 'rb') as nb:
    #         clf_nb= pickle.load(nb)
    # with open('trained models\clf_rf.pkl', 'rb') as rf:
    #         clf_rf1= pickle.load(rf)
    # with open('trained models\clf_tree.pkl', 'rb') as tree:
    #         clf_tree= pickle.load(tree)
    with open('trained models\disease_prediction_rf_model.pkl', 'rb') as nn_rf:
            neural_model= pickle.load(nn_rf)
 # check model is loaded from the above function or not
    if  neural_model is None:
        return jsonify({"error": "Models are not loaded yet. Please try again later."}), 400

    # Get symptoms from the user inputs
    symptoms = request.json.get("symptoms", [])

    # Initialize symptom presence list
    l2 = [0] * len(l1)
    l3=[0]*len(symp132)

    # Update symptom  based on user input
    for symptom in symptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1
            l3[l1.index(symptom)] = 1

    # Convert input to a numpy array
    input_test = [l2]
    input_test_nn = [l3]

    # Predict using the loaded models
    # predicted_tree = disease[clf_tree.predict(input_test)[0]]
    # predicted_rf = disease[clf_rf1.predict(input_test)[0]]
    # predicted_nb = disease[clf_nb.predict(input_test)[0]]
    predict_neural=disease[neural_model.predict(input_test_nn)[0]]
    print(predict_neural)


    # # from here we get the most common predicted disease from the above three models
    # predictions = [predicted_tree, predicted_rf, predicted_nb,predict_neural]
    # most_common_prediction = Counter(predictions).most_common(1)[0][0]

    return jsonify({
        "Most Accurate Disease": predict_neural
    })

{# @app.route('/symptoms-predict', methods=['POST'])
# def predict_symptoms_manual():
#     global clf_rf, nl1, accuracy

#     if clf_rf is None:
#         return jsonify({"error": "Model is not loaded yet."}), 400

#     symptoms = request.json.get("symptoms", [])
#     symptoms = [s.lower().strip() for s in symptoms]  # Normalize input
#     nl1_clean = [s.lower().strip() for s in nl1]       # Ensure matching

#     # ✅ Critical medical safety override
#     if "melena" in symptoms and "blood in urine" in symptoms:
#         return jsonify({
#             "Most Accurate Disease": "Possible Systemic Internal Bleeding",
#             "Urgency": "HIGH",
#             "Recommendation": "Seek immediate medical evaluation (ER / Emergency)."
#         })

#     # ✅ Convert to one-hot vector
#     l2 = [1 if symptom in symptoms else 0 for symptom in nl1_clean]

#     # ✅ Get probabilities for Top 5 diseases
#     probabilities = clf_rf.predict_proba([l2])[0]
#     top_indices = probabilities.argsort()[-5:][::-1]

#     top_results = []
#     for idx in top_indices:
#         disease_name = clf_rf.classes_[idx]
#         probability = round(probabilities[idx] * 100, 2)
#         top_results.append({
#             "disease": disease_name,
#             "confidence": f"{probability}%"
#         })
#     print("Symptoms Selected:", symptoms)
#     print("One Hot Vector Sum:", sum(l2))


#     return jsonify({
#         "Top 5 Predicted Diseases": top_results,
#         "Model Accuracy": f"{accuracy:.2f}%" if accuracy else "N/A",
#         "Note": "This is an AI-based suggestion. Consult a doctor for confirmation."
#     })


# @app.route('/symptoms-predict', methods=['POST'])
# def predict_symptoms_manual():
#     global clf_rf, accuracy  

#     # Check if model is loaded
#     if clf_rf is None:
#         return jsonify({"error": "Model is not loaded yet. Please try again later."}), 400

#     # Get symptoms from request
#     symptoms = request.json.get("symptoms", [])

#     # Prepare input vector based on your master symptom list
#     l2 = [0] * len(nl1)  # nl1 = list of all symptoms in CSV
#     for symptom in symptoms:
#         if symptom in nl1:
#             l2[nl1.index(symptom)] = 1

#     input_test = [l2]

#     # Predict disease
#     predicted_index = clf_rf.predict(input_test)[0]  # predicted label
#     predicted_rf = predicted_index  # disease name



#     return jsonify({
#         "Most Accurate Disease": predicted_rf
#     })
}


@app.route('/mental-predict', methods=['POST'])
def predict():
    global clf_tree, clf_rf, clf_nb, model1, model,neural_model,accuracy,clf_rf1
    with open('trained models\mental_health_model.pkl', 'rb') as model_file:
            model1 = pickle.load(model_file)
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400
        
        data = request.get_json()
        required_fields = ['age', 'gender', 'employment_status', 'family_history', 'responses']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        if len(data['responses']) != 12:
            return jsonify({"error": "Invalid number of responses. Expected 12."}), 400
        
        age = int(data['age'])
        gender = label_encoders['gender'].transform([data['gender']])[0]
        employment_status = label_encoders['employment_status'].transform([data['employment_status']])[0]
        family_history = label_encoders['family_history'].transform([data['family_history']])[0]
        responses = [float(x) for x in data['responses']]
        
        all_features = [age, gender, employment_status, family_history] + responses
        user_input = np.array(all_features, dtype=float).reshape(1, -1)
        
        prediction = model1.predict(user_input)[0]
        conditions = ['Depression', 'Anxiety', 'Insomnia', 'Schizophrenia', 'Phobia']
        result = {condition: bool(pred) for condition, pred in zip(conditions, prediction)}
        
        return jsonify({"status": "success", "message": "Mental health assessment completed.", "prediction": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chats", methods=["POST"])
def chat_response():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        data = request.get_json()
        if "message" not in data: 
            return jsonify({"error": "Missing 'message' field in request"}), 400

        user_query = data["message"]  
        result = chatbot(user_query)
        
        return jsonify({"response": result}), 200  

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/assess', methods=['POST'])
def assess_mental_health():
    data = request.json
    role = data.get('role')
    responses = data.get('responses')

    if not role or not responses:
        return jsonify({"message": "Invalid data, 'role' and 'responses' are required"}), 400

    if not isinstance(responses, list) or len(responses) != len(questions_by_role.get(role, [])):
        return jsonify({"message": "Invalid response data"}), 400

    response_array = np.array(responses).reshape(1, -1)

    model_path = role_models.get(role)
    if not model_path:
        return jsonify({"message": "Model not found for the specified role"}), 400

    try:
        model = load_model(model_path)
    except Exception as e:
        return jsonify({"message": f"Error loading model: {str(e)}"}), 500

    prediction = model.predict(response_array)

    conditions = role_conditions.get(role, [])

    result = {}
    for i, condition in enumerate(conditions):
        result[condition] = float(prediction[0][i])

    return jsonify({"prediction": result}), 200


@app.route("/fetchAyurvedicCenters", methods=["GET"])
def get_ayurvedic_centers():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required."}), 400

    try:
        centers = fetch_hospitals_from_osm(lat, lon)
        return jsonify(centers)
    except Exception as e:
        return jsonify({"error": "Failed to fetch centers", "details": str(e)}), 500

UPLOAD_FOLDER = os.path.join("report_chatbot", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/analyze-report", methods=["POST"])
def analyze_report():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf']
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    extracted_text = extract_text_from_report(save_path)
    ai_result = analyze_report_text(extracted_text)

    return jsonify(ai_result), 200   # ✅ JSON is applied only here




@app.route("/report-chat", methods=["POST"])
def chat_report():
    data = request.get_json()
    user_msg = data.get("message", "")

    reply = chat_with_report_bot(user_msg)

    return jsonify({"response": reply}), 200


{# @app.route("/drug-interaction", methods=["POST"])
# def drug_interaction_api():
#     data = request.get_json()
    
#     if not data or "query" not in data:
#         return jsonify({"error": "Missing query field"}), 400
    
#     query = data["query"]

#     context_lines = retrieve_context(query, index, sentences)
#     context = "\n".join(context_lines)
#     answer = generate_answer(query, context)

#     return jsonify({
#         "query": query,
#         "context_used": context_lines,
#         "response": answer
#     }), 200
}

if __name__ == '__main__':
    app.run(debug=True)

