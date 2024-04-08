import torch
from detectionapp.face_model import FaceModel

def load_emotion_model(path):
    model_emotion = FaceModel()
    model_emotion.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    model_emotion = model_emotion.eval()
    return model_emotion