# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 01:13:46 2019

@author: LABA
"""

import pandas as pd
import re
from nltk.corpus import stopwords
from pickle import dump, load
import nltk
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def define_models(n_input, n_output, n_units): 
    # define training encoder 
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(n_input, 200)(encoder_inputs)
    encoder = LSTM(n_units, return_state=True) 
    encoder_outputs, state_h, state_c = encoder(encoder_embedding) 
    encoder_states = [state_h, state_c] 
    # define training decoder 
    decoder_inputs = Input(shape=(None,)) 
    decoder_embedding = Embedding(n_output, 200)(decoder_inputs)
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True) 
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states) 
    decoder_dense = Dense(n_output, activation='softmax') 
    decoder_outputs = decoder_dense(decoder_outputs) 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
    # define inference encoder 
    encoder_model = Model(encoder_inputs, encoder_states) 
    # define inference decoder 
    decoder_state_input_h = Input(shape=(n_units,)) 
    decoder_state_input_c = Input(shape=(n_units,)) 
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c] 
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs) 
    decoder_states = [state_h, state_c] 
    decoder_outputs = decoder_dense(decoder_outputs) 
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states) 
    # return all models 
    return model, encoder_model, decoder_model
    

def convert_data(text, word_dict, max_len):
    
    
    convert_all = []
    
    for t in text:
        convert = []
        for char in t:
            if char in word_dict:
                convert.append(word_dict[char])
            else:
                pass
        convert_all.append(convert)
    convert_result = pad_sequences(convert_all, max_len, padding='post')    
    return convert_result
    
def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_decoder_seq_length):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
        
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
            # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['\t']
        
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
                    
        decoded_sentence += sampled_char
        
            # Exit condition: either hit max length
            # or find stop character.
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        
                # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0,0 ] = sampled_token_index
        
                # Update states
        states_value = [h, c]
    
    return decoded_sentence


#nltk.download('stopwords')
#
#def clean_text(text, remove_stopwords=True): 
#    # Convert words to lower case 
#    text = text.lower() 
#    if True: 
#        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) 
#        text = re.sub(r'\<a href', ' ', text) 
#        text = re.sub(r'&', '', text) 
#        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text) 
#        text = re.sub(r'<br />', ' ', text) 
#        text = re.sub(r'\'', ' ', text) 
#        if remove_stopwords: 
#            text = text.split() 
#            stops = set(stopwords.words("english")) 
#        text = [w for w in text if not w in stops] 
#        text = " ".join(text) 
#    return text
#
#
#
#
#data = pd.read_csv("Reviews.csv")
#
#data = data[['Summary','Text']]
#
#data = data.dropna()
#
#clean_summaries = []
#for summary in  list(data['Summary']):
#    clean_summaries.append(clean_text(summary))
#
#clean_texts = []
#for t in list(data['Text']):
#    clean_texts.append(clean_text(t, remove_stopwords=True))
#    
#stories = list()
#
#for i, text in  enumerate(clean_texts):
#    stories.append({'story': text, 'highlights': clean_summaries[i]})
#
#dump(stories, open('dataset.pkl', 'wb'))

x = load(open('dataset.pkl', 'rb'))
x = x[:5000]
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

for d in x:
    input_text = d['story']
    target_text = d['highlights']
    input_texts.append(d['story'])
    target_texts.append('\t' + target_text + '\n')

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
        
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))        
        
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])        
        
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)      

input_token_index = dict()
target_token_index = dict()

for i, char in enumerate(input_characters):
    input_token_index[char] = i 
        
for  i, char in enumerate(target_characters):
    target_token_index[char] = i

reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


encoder_input = convert_data(input_texts, input_token_index, max_encoder_seq_length)
encoder_input = encoder_input[:len(encoder_input) - 10]
deocder_input = convert_data(target_texts, target_token_index, max_decoder_seq_length)
deocder_input = deocder_input[:len(deocder_input) - 10]

test_sentence = encoder_input[:-10]
test_summary_sentence = deocder_input[:-10]

train_encoder_input = encoder_input[:int(len(encoder_input) * 0.8)]
train_deocer_input = deocder_input[:int(len(deocder_input) * 0.8)]
test_encoder_input = encoder_input[int(len(encoder_input) * 0.8):]
test_deocer_input = deocder_input[int(len(deocder_input) * 0.8):]

target_data = [[deocder_input[n][i+1] for i in range(len(deocder_input[n])-1)] for n in range(len(deocder_input))]
target_data = pad_sequences(target_data, maxlen=max_decoder_seq_length, padding="post")
target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))
train_target_data = target_data[:int(len(target_data) * 0.8)]
test_target_data = target_data[int(len(target_data) * 0.8):]

model, encoder, encoder = define_models(num_encoder_tokens, num_decoder_tokens, 256)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([train_encoder_input, train_deocer_input], train_target_data,
          batch_size=32,
          epochs=10,
          validation_data=([test_encoder_input, test_deocer_input], test_target_data))

model.save('model.h5')








