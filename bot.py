import nltk
import random
import string  # to process standard python strings
import warnings
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

nltk.download('popular', quiet=True)  

f= open('data.txt', 'r', errors='ignore')
checkpoint = "./chatbot_weights.ckpt"

raw = f.read()
raw = raw.lower()

#nltk.download('punkt')  # first-time use only
#nltk.download('wordnet')  # first-time use only

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)  

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hola!", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    trans = Translator()
    t = trans.translate(sentence)
    t = t.text
    for word in t.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


def chat(user_response):
    user_response = user_response.lower()

    if (user_response != 'ok bye'):
        if (user_response == 'ok thanks' or user_response == 'thank you'):
            flag = False
            return "You are welcome.."

        elif (greeting(user_response) != None):
            from googletrans import Translator
            trans = Translator()
            t = trans.translate(user_response)
            t = str(t.text)
            return greeting(t)
        elif (user_response == "how are you?"):
            return "I am fine."

        else:
            # print(response(user_response))
            from googletrans import Translator
            trans = Translator()
            t = trans.translate(user_response)
            t =str(t.text)
            return response(t)
            sent_tokens.remove(user_response)



    else:
        flag = False
        return "Bye! take care.."
