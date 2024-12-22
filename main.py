from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model_bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

with open("data.json") as file:
    data = json.load(file)

embedded_data = []
for q, a in zip(data['questions'], data['answers']):
    question_vec = model_bert(**tokenizer(q, return_tensors='pt'))['last_hidden_state'][:, 0, :].detach().numpy()
    answer_vec = model_bert(**tokenizer(a, return_tensors='pt'))['last_hidden_state'][:, 0, :].detach().numpy()
    embedded_data.append([question_vec[0], answer_vec[0]])

embedded_data = np.array(embedded_data)

chat_model = tf.keras.models.load_model("model.keras")

while True:
    user_input = [input("Вы: ")]

    question_embedding = model_bert(**tokenizer(user_input, return_tensors='pt'))['last_hidden_state'][:, 0, :].detach().numpy()[0]
    predictions = []

    for idx in tqdm(range(embedded_data.shape[0])):
        answer_embedding = embedded_data[idx, 1]
        combined_vector = np.concatenate([question_embedding, answer_embedding])
        prediction_score = chat_model.predict(np.expand_dims(combined_vector, axis=0), verbose=False)[0, 0]
        predictions.append([idx, prediction_score])

    predictions = np.array(predictions)
    best_match_idx = np.argmax(predictions[:, 1])
    print(f"Чат-бот: {data['answers'][best_match_idx]}")
