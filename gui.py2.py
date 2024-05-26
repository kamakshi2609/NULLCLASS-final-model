import tkinter as tk
from tkinter import ttk
from keras.layers import TextVectorization
import re
import tensorflow as tf_strings
import json
import string
from keras.models import load_model
from tensorflow import argmax
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf

#eng to spanish translation
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

#load the english vectorization layer config
with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\eng_vectorization_config.json') as json_file:
    eng_vectorization_config = json.load(json_file)

#Recreate the eng vectorization layer w basic configuration
eng_vectorization = TextVectorization(
    max_tokens = eng_vectorization_config['max_tokens'],
    output_mode = eng_vectorization_config['output_mode'],
    output_sequence_length = eng_vectorization_config['output_sequence_length']
)

#apply the custom standardization function
eng_vectorization.standardize = custom_standardization

#load the spanish vectorization layer
with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\spa_vectorization_config.json') as json_file:
    spa_vectorization_config = json.load(json_file)

#Recreate the spa vectorization layer w basic configuration
spa_vectorization = TextVectorization(
    max_tokens = spa_vectorization_config['max_tokens'],
    output_mode = spa_vectorization_config['output_mode'],
    output_sequence_length = spa_vectorization_config['output_sequence_length'],
    standardize = custom_standardization
    )

#load and set the eng vocab 
with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\eng_vocab.json') as json_file:
    eng_vocab = json.load(json_file)
    eng_vectorization.set_vocabulary(eng_vocab)

#load and set the spanish vocab
with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\spa_vocab.json') as json_file:
    spa_vocab = json.load(json_file)
    spa_vectorization.set_vocabulary(spa_vocab)

#loading the spanish model
transformer_model = load_model(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\transformer_model')

spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sentence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer_model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy().item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

#ENGLISH TO FRENCHH 

#loading french model 
model = load_model(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\english_to_french_model')
with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)

with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)

#load max length
with open(r'C:\Users\Kamakshi\OneDrive\Documents\GitHub\ENG-TO-FRENCH\sequence_length.json') as f:
    max_length = json.load(f)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
  english_sentence = english_sentence.lower()

  english_sentence = english_sentence.replace(".", '')
  english_sentence = english_sentence.replace("?", '')
  english_sentence = english_sentence.replace("!", '')
  english_sentence = english_sentence.replace(",", '')

  english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
  english_sentence = pad(english_sentence, max_length)

  english_sentence = english_sentence.reshape((-1,max_length))

  french_sentence = model.predict(english_sentence)[0]
  french_sentence = [np.argmax(word) for word in french_sentence]

  french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
  print("French translation: ", french_sentence)

  return french_sentence

def translate_to_spanish(english_sentence):
    spanish_sentence = decode_sentence(english_sentence)
    print("Spanish translation:" ,spanish_sentence)

    return spanish_sentence.replace("[start]","").replace("[end]", "")

#function to handle translation req based on selected lang
def handle_translate():
    selected_language = language_var.get()
    english_sentence = text_input.get("1.0", "end-1c")

    if selected_language =="French":
        translation = translate_to_french(english_sentence)
    elif selected_language =="Spanish":
        translation = translate_to_spanish(english_sentence)
    
    translation_output.delete("1.0","end")
    translation_output.insert("end", f"{selected_language} translation: {translation}")

#setting up the main window
root = tk.Tk()
root.title("Language Translation")
root.geometry("550x600")

#front cofig
font_style = "Times New Roman"
font_size = 14

#frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

#heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=(font_style,font_size,'bold'))
input_heading.pack()

#text input for english sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

#language slectionn
language_var = tk.StringVar()
language_label = tk.Label(input_frame, height=10, width=50, font=(font_style, font_size))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["French","Spanish"],font=(font_style,font_size))
language_select.pack()


#submit button
submit_button = ttk.Button(root, text="Translate", command=handle_translate)
submit_button.pack()

#frame for output
output_frame = tk.Frame(root)
output_frame.pack()

#heading for output
output_heading = tk.Label(output_frame, text="Translation",font=(font_style, font_size))
output_heading.pack()

#text output for translation
translation_output = tk.Text(output_frame, height=10, width=50, font=(font_style,font_size))
translation_output.pack()

#running the application
root.mainloop()
