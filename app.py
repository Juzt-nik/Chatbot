from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import string

app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('popular', quiet=True)

# Load JSON data
with open('intents.json', 'r') as json_file:
    intents = json.load(json_file)

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenisation
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi", "Hey", "Hello", "Hello! How can I help you"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Function to check if the user's query is about groundwater data
def check_aqua_query(user_response):
    for topic, info in intents.items():
        if topic.lower() in user_response.lower():
            return f"{topic.capitalize()}: {info}"
    return None

# Generating response
def generate_response(user_response):
    # Check if the query is about groundwater data
    aqua_info = check_aqua_query(user_response)
    if aqua_info:
        return aqua_info
    
    # If not about groundwater, use the general response mechanism
    temp_sent_tokens = sent_tokens + [user_response]
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(temp_sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        return "I'm sorry, but I don't have enough information to answer that question. Can you please provide more context or ask about a specific groundwater topic?"
    else:
        return temp_sent_tokens[idx]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    
    if user_message.lower() in ['bye', 'goodbye', 'exit', 'quit']:
        return jsonify({'response': "Goodbye! Take care and feel free to come back if you have more questions about groundwater."})
    
    if user_message.lower() in ['thanks', 'thank you']:
        return jsonify({'response': "You're welcome! Is there anything else you'd like to know about groundwater?"})
    greeting_response = greeting(user_message)
    if greeting_response:
        return jsonify({'response': greeting_response})
    
    bot_response = generate_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)

