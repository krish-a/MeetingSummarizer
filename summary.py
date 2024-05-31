import requests
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def read_file(filename):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(5242880)
            if not data:
                break
            yield data
          
def VideoToText( audioFile, key ):
  audio_file= open(audioFile, "rb")
  auth_key = key
  headers = {
    "authorization": auth_key,
    "content_type": "application/json"
  }
  upload_response = requests.post("https://api.assemblyai.com/v2/upload", headers = headers, data=read_file("sd.mp3"))
  ur = upload_response.json()
  transcript_request = {
    "audio_url": 'https://cdn.assemblyai.com/upload/' + str(ur)
  }
  transcript_response = requests.post("https://api.assemblyai.com/v2/transcript", headers=headers, json=transcript_request)
  ponse = requests.get('https://api.assemblyai.com/v2/transcript/6o8z3hhyzi-bb30-4d36-9798-e1731812ab4b', headers=headers)
  return ponse.json()["text"]

def summary(audioFile, key):
    text = ponse.json()["text"]
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    sentences = sent_tokenize(text)
    sentenceValue = dict()
    for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    average = int(sumValues / len(sentenceValue))
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
    text= ponse.json()["text"]
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    summarizer = pipeline('summarization')
    summ = summarizer(VideoToText(audioFile, key), max_length = 30, min_length = 10, do_sample = False)
    return summ
