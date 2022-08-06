import random
import json
import torch
from model import SNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", "r") as f:
    intents = json.load(f)

data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
model_state = data["model_state"]
all_words = data["all_words"]
tags = data["tags"]

model = SNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Barista"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    return "Sorry, I don't understand. Please ask another question."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)