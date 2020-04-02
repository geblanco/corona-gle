import os
import sys
import torch
import flair
flair.devide = torch.device('cuda:0')
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import DataLoader
from flair.data import Sentence
from flair.data import (
    segtok_tokenizer
)

curr_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(curr_path, "data")
logs_path = os.path.join(curr_path, "logs")

# mt = ModelTrainer.load_checkpoint(os.path.join(logs_path, "checkpoint.pt"), corpus)
# mt.model.save(os.path.join(logs_path, "model.pt"))
classifier = TextClassifier.load(os.path.join(logs_path, "final-model.pt"))
classifier.eval()

sentence = Sentence("""
Taiwan's National Laboratory System is one of the action packages of the Global Health Security Agenda, which was launched by the World Health Organization (WHO) to promote health security as an international priority and to encourage progress toward full implementation of the WHO International Health Regulations (IHR) 2005. The mission of each national laboratory system is to conduct real-time biosurveillance and effective laboratory-based diagnostics, as measured by a nationwide laboratory system able to reliably conduct diagnoses on specimens transported properly to designated laboratories from at least 80% of the regions in the country. In Taiwan, the national laboratory system for public health is well-established and coordinated by the Taiwan Centers for Disease Control (CDC), which is the government authority in charge of infectious disease prevention and intervention. Through the national laboratory system, Taiwan CDC effectively detects and characterizes pathogens that cause communicable diseases across the entire country, including both known and novel threats, and also conducts epidemiologic analyses of infectious diseases. In this article, we describe the national laboratory system for public health in Taiwan. We provide additional information on the national influenza laboratory surveillance network to demonstrate how our national laboratory systems work in practice, including descriptions of long-term seasonal influenza characterization and successful experiences identifying novel H7N9 and H6N1 influenza viruses.
""", use_tokenizer=segtok_tokenizer)
classifier.predict(sentence)

print(sentence.labels)