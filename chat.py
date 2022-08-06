import random
import json
import torch
from model import SNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SNet(input_size, hidden_size, output_size).to(device)

