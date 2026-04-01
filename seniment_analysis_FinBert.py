import sys
print(sys.executable)
import feedparser
from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")
print(pipe('Stocks rallied and the British pound gained.'))