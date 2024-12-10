import numpy as np
import speech_recognition as sr
import nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')


nltk.download('punkt') # Download the Punkt sentence tokenizer
# nltk.data.path.append('/home/mkh/nltk_data')
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

import string

import re

# Download necessary NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stop words using nltk's stopword list
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# The ignore list containing words to exclude
ignore_list = [""]

# Function to clean contractions
def clean_contractions(word):
    return re.sub(r"['’]s$", "", word).replace("n't", "")

# Function to tokenize and lemmatize while filtering

def preprocess_words_fn(pattern, stop_words, punctuation):
    """
    Preprocess a sentence or a list of words:
    - Tokenizes if it's a string.
    - Filters stopwords and punctuation.
    - Lemmatizes the words.
    """
    all_word=[]
    if isinstance(pattern, str):  # Check if input is a string
        words = nltk.word_tokenize(pattern.lower())  # Tokenize and convert to lowercase
    elif isinstance(pattern, list):  # If it's already a list
        words = [word.lower() for word in pattern]  # Normalize case
    else:
        raise ValueError("Input must be a string or a list of words.")

    words = [clean_contractions(w) for w in words]  # Clean contractions like 's and n't
    words = [lemmatizer.lemmatize(w) for w in words]  # Lemmatize words
    words = [w for w in words if w not in stop_words and w not in punctuation]  # Filter
    # print("---<" , words)
    all_word.extend(words)
    # print("\n \n \n <<>>>>>" , len(all_words))
    # print("\n \n \n <<>>>>>" , len(sorted(set(all_words))))
    return sorted(set(all_word))  # Return unique words


import numpy as np
def bag_of_words(sentence, all_words, preprocess_words_fn, stop_words, punctuation):
    """
    Return a bag of words: 1 for each known word that exists in the sentence, 0 otherwise.
    sentence = "hello how are you"
    all_words = ["hi", "hello", "I", "bye", "thank", "cool"]
    bag = [0, 1, 0, 0, 0, 0]
    """
    # Preprocess the sentence (e.g., tokenization, lemmatization)
    sentence_words = preprocess_words_fn(sentence, stop_words, punctuation)

    # Initialize the bag with 0s for each word in all_words|
    bag = np.zeros(len(all_words), dtype=np.float32)
    # print(bag)
    # Loop through each word in all_words
    for idx, word in enumerate(all_words):
        if word in sentence_words:  # If the word exists in the sentence
            bag[idx] = 1  # Mark it as present

    return bag


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # Correct predictions
    accuracy = (correct / len(y_pred)) * 100  # Accuracy in percentage
    return accuracy

    
def train_fn(model: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Adam,
             loss_fn: torch.nn.Module,
             accuracy_fn, 
             device: torch.device = device,
             prnt=True):
    trainLoss, trainAcc = 0.0, 0.0  # Scalars
    model.to(device)

    model.train()  # Set model in training mode

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # 1- Forward pass
        y_pred = model(X)

        # 2- Calculate the loss
        loss = loss_fn(y_pred, y)
        trainLoss += loss.item()

        # 3- Calculate accuracy
        # Use the class indices directly by applying argmax on predictions
        trainAcc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 4- Optimizer zero_grad
        optimizer.zero_grad()

        # 5- Backpropagation
        loss.backward()

        # 6- Optimizer step
        optimizer.step()

    # Average loss and accuracy over all batches
    trainLoss /= len(data_loader)
    trainAcc /= len(data_loader)

    if prnt:
        print(f"Model training loss: {trainLoss:.5f} | Model accuracy: {trainAcc:.2f}%")

    return trainLoss, trainAcc



def convert_speach_to_text() :
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    text=""
    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)  # Reduce noise for better accuracy
        
        print("Please speak now...")
        try:
            # Capture audio from the microphone
            audio = recognizer.listen(source)  # Timeout after 5 seconds of silence
            
            # Convert audio to text using Google Speech Recognition
            text = recognizer.recognize_google(audio)
            print("You said: ", text)
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio.")
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    return text
    
            