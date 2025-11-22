import ollama

conversation_history = [
    {
        "role": "system",
        "content": (
            "You are an expert professional and mental healthcare assistant. "
            "Only answer questions related to health, diseases, wellness, meditation, or treatment only as well as for the symptoms and diseases also. "
            "If the input is unrelated to healthcare (e.g., jokes, unauthenticated chats), kindly reply: "
            "'sorry I'm designed to assist only with healthcare-related queries. 🙏' "
            "For valid health issues, provide both Ayurvedic and traditional treatment suggestions in simple 2-line responses."
            "provide the result in short and concise and easy to understandable by normal person as well."
            "you are female gender."
            "Your name is Sympcare. and Developed by Rohit mewada and tulsiram pathe."
        )
    }
]

def generate_response(user_input):
    conversation_history.append({"role": "user", "content": user_input})
    
    response = ollama.chat(
        model="llama3.1",
        messages=conversation_history
    )
    
    ai_response = response['message']['content']
    conversation_history.append({"role": "assistant", "content": ai_response})
    
    return ai_response



def chatbot(user_message):
    if user_message.lower() in ["exit", "quit"]:
        return "Goodbye! Take care. 😊"
    else:
        return generate_response(user_message)

if __name__ == "__main__":
    while True:
        msg = input("You: ")
        if msg.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! 👋")
            break
        reply = chatbot(msg)
        print(f"Bot: {reply}")
