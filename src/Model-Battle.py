#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Model-battle.py
# (intended to be used for battling a local model against bigger ones via API)

import tensorflow as tf
import os
import json
import sys
import threading
import random
import logging
import numpy as np
import model
import sample
import encoder
import requests
import datetime

# Import API key and system message functions
from get_api_key import get_api_key, get_system_message

# Path to the system message file
system_message_file = os.path.expanduser('~/NeuralNetwork/openai_system_message.txt')

# Path to save the chatlog file
chatlog_file = os.path.expanduser('~/NeuralNetwork/model_battle_chatlog.txt')

# Set the directory for models here
models_directory = os.path.expanduser('~/NeuralNetwork/vzgpt/__incheck/')

# Set the GPU you want to use here
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust this as needed to switch GPUs

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will hide INFO and WARNING messages
tf.get_logger().setLevel('ERROR')  # Suppresses info and warning messages

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)
logger = logging.getLogger(__name__)

# Model and conversation settings
input_prefix = "|k| "
output_prefix = "|v| "
starting_context = "\n"
debug = True
top = 0.77
degree = 1.0
mx = 0.00500
tok = 0
learning = ""
mode = True
learn = True
cache = ""
turns = []

class ModelWrapper:
    def __init__(self, models_dir, model_name=''):
        self.models_dir = models_dir
        self.model_name = model_name  # this could be an empty string if not using subdirectories
        self.turns = []  # Initialize turns inside the class        
        self.session = tf.compat.v1.Session(graph=tf.Graph())        
        self.load_model()
        self.api_key = get_api_key()  # Fetch the API key here for use in API calls

    def load_model(self):
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        with self.session.graph.as_default():
            self.enc = encoder.get_encoder(self.model_name, self.models_dir)  # Adjusted here
            self.hparams = model.default_hparams()
            with open(os.path.join(self.models_dir, 'hparams.json')) as f:
                self.hparams.override_from_dict(json.load(f))
            self.context = tf.compat.v1.placeholder(tf.int32, [1, None])
            self.output = sample.sample_sequence(
                hparams=self.hparams, length=1000, # hold ~24 tokens under 1000
                context=self.context,
                batch_size=1,
                temperature=degree, top_k=tok, top_p=top
            )
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.models_dir)
            saver.restore(self.session, ckpt)

    def decode(self, tokens):
        try:
            text = ''.join([self.enc.decoder[token] if token in self.enc.decoder else '[UNK]' for token in tokens])
        except KeyError as e:
            print(f"Unknown token ID {e.args[0]} encountered in the output.")
            text = "[Decode Error]"
        return text

    def interact_model(self, text_input):
        # Encode the new input and append it to the dialogue history
        new_input_tokens = self.enc.encode(input_prefix + text_input + '\n' + output_prefix)
        self.turns.append(new_input_tokens)
        
        # Flatten the list of token lists to a single list of tokens
        context_tokens = [token for sublist in self.turns for token in sublist]
        
        # Trim tokens from the beginning if exceeding the max context length
        max_context_length = 1000  # Adjust based on model capabilities; 1024 usual max, keep a few tokens spare.
        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[-max_context_length:]
        
        # Process the context
        out = self.session.run(self.output, feed_dict={self.context: [context_tokens]})[:, len(context_tokens):]
        
        try:
            text = self.enc.decode(out[0])
        except KeyError as e:
            print(f"Unknown token ID {e.args[0]} encountered in the output.")
            text = "[Decode Error]"
        
        return text.split('\n')[0]

    def generate_initial_message(self):
        # Use the output prefix to generate an initial message
        start_tokens = self.enc.encode(output_prefix)
        out = self.session.run(self.output, feed_dict={self.context: [start_tokens]})[:, len(start_tokens):]
        return self.enc.decode(out[0]).split('\n')[0]

    def get_chat_response(self, messages):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        response = requests.post(url, json=data, headers=headers)
        return response.json()

    def log_chat(self, role, message):
        file_path = chatlog_file
        # Check file size and rename if greater than a certain size (e.g., 5MB)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 5 * 1024 * 1024:
            os.rename(file_path, file_path.replace('.txt', f'_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt'))
        with open(file_path, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} {role}: {message}\n")

def main_loop():
    model_wrapper = ModelWrapper(models_directory)
    system_message = get_system_message(system_message_file)
    print("System message:", system_message)  # Display the system message
    model_wrapper.log_chat('System', system_message)

    # Prompt the user for initial input to start the conversation
    user_initial_input = input("You: ")
    formatted_initial_input = input_prefix + user_initial_input + '\n' + output_prefix
    initial_local_message = model_wrapper.interact_model(formatted_initial_input)
    clean_initial_local_message = initial_local_message.replace(output_prefix, "").strip()

    model_wrapper.log_chat('Local GPT-2', clean_initial_local_message)
    print("Local GPT-2:", clean_initial_local_message)

    # Initialize the conversation with the local model's message
    messages = [{"role": "assistant", "content": clean_initial_local_message}]

    while True:
        # Fetch response from OpenAI's GPT-3.5 Turbo model
        response = model_wrapper.get_chat_response(messages)
        latest_message_from_openai = response['choices'][0]['message']['content']
        model_wrapper.log_chat('OpenAI GPT-3.5', latest_message_from_openai)
        print("OpenAI GPT-3.5:", latest_message_from_openai)

        # Prepare the input for the local model using the correct prefixes
        formatted_input_for_local = input_prefix + latest_message_from_openai + '\n' + output_prefix
        response_from_local = model_wrapper.interact_model(formatted_input_for_local)
        clean_response_from_local = response_from_local.replace(output_prefix, "").strip()

        model_wrapper.log_chat('Local GPT-2', clean_response_from_local)
        print("Local GPT-2:", clean_response_from_local)

        # Update the message list for the next OpenAI call
        messages = [{"role": "assistant", "content": clean_response_from_local}]

        # User interaction to continue or exit
        user_input = input("Press ENTER to continue (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

if __name__ == '__main__':
    main_loop()
