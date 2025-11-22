from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import ollama
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

questions_by_role = {
    "Student": [
        "Do you often feel overwhelmed by academic pressure?",
        "Do you struggle to concentrate during lectures or while studying?",
        "How often do you feel anxious before exams or assignments?",
        "Do you experience difficulty balancing academic and personal life?",
        "Have you lost interest in extracurricular activities you once enjoyed?",
        "Do you frequently procrastinate on assignments due to mental exhaustion?",
        "Do you feel socially isolated or disconnected from your peers?",
        "How often do you experience self-doubt about your abilities?",
        "Do you feel that you are not performing as well as you should be?",
        "Have you ever experienced difficulty sleeping due to academic stress?"
    ],
    "Working professional": [
        "Do you feel emotionally exhausted at the end of the workday?",
        "How often do you struggle with motivation for work-related tasks?",
        "Do you find it difficult to disconnect from work during personal time?",
        "How often do you experience stress due to deadlines and workload?",
        "Do you feel unappreciated for your contributions at work?",
        "How frequently do you worry about job security or career growth?",
        "Do you find it difficult to focus on tasks without getting distracted?",
        "How often do you experience physical symptoms like headaches due to work stress?",
        "Do you feel that your work-life balance is unhealthy?",
        "How frequently do you consider quitting due to mental exhaustion?"
    ],
    "Housewife homemaker": [
        "Do you feel that your daily responsibilities are overwhelming?",
        "How often do you experience feelings of loneliness or isolation?",
        "Do you feel underappreciated for the work you do at home?",
        "How frequently do you experience mood swings or irritability?",
        "Do you feel that your work at home goes unnoticed?",
        "How often do you neglect self-care due to family responsibilities?",
        "Do you find it hard to ask for help when you need it?",
        "Do you experience anxiety about managing household responsibilities?",
        "How often do you feel mentally drained even after a normal day?",
        "Do you feel that you don’t have time to focus on your personal goals?"
    ],
    "healthcare_professional": [
        "Do you frequently feel emotionally drained after patient interactions?",
        "How often do you experience stress due to long working hours?",
        "Do you struggle with maintaining empathy due to constant exposure to suffering?",
        "How often do you feel you cannot provide adequate care due to system limitations?",
        "Do you feel like you have no time for personal self-care?",
        "How often do you experience physical exhaustion at work?",
        "Do you feel that work-related stress affects your personal relationships?",
        "How frequently do you doubt your ability to help your patients?",
        "Do you feel a loss of motivation or passion for your job?",
        "How often do you think about quitting the healthcare profession?"
    ],
    "IT Tech Employee": [
        "How often do you experience stress due to tight project deadlines?",
        "Do you feel mentally exhausted from coding or troubleshooting issues?",
        "How frequently do you find yourself working outside office hours?",
        "Do you struggle to stay focused for long periods?",
        "How often do you feel anxious about making mistakes in code?",
        "Do you feel that long hours of screen time affect your mental well-being?",
        "How frequently do you experience imposter syndrome in your role?",
        "Do you find it difficult to balance work and personal time?",
        "How often do you experience burnout due to excessive workload?",
        "Do you feel pressure to constantly upskill to stay relevant in the industry?"
    ],

    "Teacher":[
        "Do you frequently feel exhausted after a day of teaching?",
        "How often do you experience stress due to student behavior or workload?",
        "Do you feel that your efforts in teaching go unrecognized?",
        "How frequently do you experience frustration or irritation at work?",
        "Do you feel emotionally drained after handling students’ problems?",
        "How often do you feel pressure to meet academic expectations?",
        "Do you struggle to maintain motivation for teaching?",
        "How frequently do you feel a lack of work-life balance?",
        "Do you experience anxiety about managing large classes?",
        "Have you ever thought of leaving teaching due to stress?"
    ],
    "Factory Worker":[
                
        "Do you frequently experience physical exhaustion from your work?",
        "How often do you feel that your work is monotonous and repetitive?",
        "Do you experience stress due to unsafe or difficult working conditions?",
        "How frequently do you feel unappreciated by your employers?",
        "Do you struggle with feelings of job insecurity?",
        "How often do you experience back pain, headaches, or physical discomfort from work?",
        "Do you feel emotionally or mentally exhausted at the end of the day?",
        "How frequently do you experience difficulty sleeping due to work-related stress?",
        "Do you feel that you lack control over your work conditions?",
        "How often do you think about quitting due to stress?"
    ],

    "Entrepreneurs":[
        "Do you often feel overwhelmed by business-related stress?",
       "How frequently do you struggle with financial anxiety related to your business?",
        "Do you experience self-doubt regarding your ability to succeed?",
        "How often do you feel isolated due to the pressures of running a business?",
        "Do you struggle with decision-making due to stress or anxiety?",
        "How frequently do you feel mentally or physically exhausted?",
        "Do you have difficulty sleeping due to business worries?",
        "Do you feel guilty about taking breaks or vacations?",
        "How often do you experience mood swings related to business performance?",
        "Do you feel that you are sacrificing personal happiness for business success?"
    ]


}

role_models = {
    "Student": "trained models\Students_mental_health_model.h5",
    "Working professional": "trained models\Working_Professionals_mental_health_model.h5",
    "Housewife homemaker": "trained models\Homemaker_mental_health_model.h5",
    "IT Tech Employee": "trained models\IT_Tech_mental_health_model.h5",
    "Teacher":"trained models\Teachers_mental_health.h5",
    "Factory Worker":"trained models\Faactory_worker_mental_health.h5",
    "Entrepreneurs":"trained models\Entrepreneurs_mental_heath.h5",
    "Healthcare professional":"trained models\Healthcare_mental_health_model.h5"
}
role_conditions = {
    "Student": ["ANXIETY", "DEPRESSION", "ADHD", "BURNOUT"], 
    "Working professional": ["BURNOUT", "ANXIETY", "DEPRESSION", "ADHD"], 
    "Housewife homemaker": ["BURNOUT", "DEPRESSION", "ANXIETY", "ESTEEM_ISSUE"], 
    "IT Tech Employee": ["BURNOUT", "ANXIETY", "ADHD", "DEPRESSION"], 
    "Teacher": ["BURNOUT", "ANXIETY", "COMPASSION_FATIGUE", "DEPRESSION"], 
    "Factory Worker": ["ANXIETY", "DEPRESSION", "OCD", "BURNOUT"],
    "Healthcare professional": ["BURNOUT", "DEPRESSION", "ANXIETY", "COMPASSION_FATIGUE"],
    "Entrepreneurs": ["ANXIETY", "BURNOUT", "DEPRESSION", "ESTEEM_ISSUE"]
}

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
        result[condition] = prediction[0][i]

    return jsonify({"prediction": result}), 200



if __name__ == "__main__":
    app.run(debug=True)