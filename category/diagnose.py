import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "label_encoder_90.pkl"), "rb") as f:
    obj = pickle.load(f)

print(obj.classes_)