import re
import json
import spacy
from scispacy.abbreviation import AbbreviationDetector
from num2words import num2words

nlp = spacy.load('en_core_web_sm')

# def abbreviation_detection (data):
#     abbreviation_pipe = AbbreviationDetector(nlp)
#     nlp.add_pipe(abbreviation_pipe)
#     data = data.read().lower()
#     data = nlp(data)
#     print("Abbreviation", "\t", "Definition")
#     for abrv in data._.abbreviations:
#         print(f"{abrv} \t {abrv._.long_form}")

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
abbreviations = []

def file_processing(data):
    data = json.load(data)
    for item in data['articles']:
        text = item['text']
        text = re.sub(r'[^\w\s]', '', text)
        result = nlp(text)
        replace_acronyms(result)
        for token in result:
            if token.is_digit:
                print(token, convert_num2words(str(token)))
            if token in abbreviations:
                print(convert_abbreviations(token))
            else:
                print (token, token.lemma_)

def replace_acronyms(text):

    #altered_tok = [tok.text for tok in text]
    for abrv in text._.abbreviations:
        abbreviations.append(abrv)

def convert_abbreviations(abrv):
    altered_tok = [abrv.text]
    altered_tok[abrv.start] == str(abrv._.long_form)
    return(f"{abrv} \t {abrv._.long_form}")

def convert_num2words(number):
    return num2words(number)

file_processing(open('Corpus.json', 'r', encoding="utf-8"))
#abbreviation_detection(open('Corpus.txt', 'r', encoding="utf-8"))
# print(replace_acronyms(open('Corpus.json', 'r', encoding="utf-8")))