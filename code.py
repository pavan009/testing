#importing all the dependencies packages
import json
import csv
import nltk
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem     import PorterStemmer
from nltk.stem     import WordNetLemmatizer
import string
import re
import csv
import pandas as pd
import numpy as np
import sys
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
f = open('data.json')
end_json = json.load(f)
def extract_documents(sentence1,productName):
    #removing punctuations and stopwords to extract main relavant data
    document_list = []
    sentence1 = sentence1.lower() # lowercase text
    sentence1 = REPLACE_BY_SPACE_RE.sub(' ', sentence1) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    sentence1 = BAD_SYMBOLS_RE.sub('', sentence1) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    sentence1 = sentence1.replace('x', '')
    sentence1 = ' '.join(word for word in sentence1.split() if word not in STOPWORDS) # remove stopwors from text
    extracted_words_from_sentence = set()
    for key in end_json:
        for word in end_json[key]:
            pattern = r"\b" + re.escape(word) + r"\b"
            match = re.search(pattern, sentence1)
            if match:
                extracted_words_from_sentence.add(key)
        # print("words from sentence")
        # print(extracted_words_from_sentence)
    #comparisions based on products
    df = pd.read_excel (r'./test_sample_1.xlsx')
    data = df[['Keywords (m)','Product','Document Name','DOC ID']]
    data = data.sort_values(by=['Product'])
    data = data.dropna()
    data = data.replace(r'^\s*$', np.nan, regex=True)
    data['Product'] = data['Product'].apply(str.lower)
    data['Keywords (m)'] = data['Keywords (m)'].str.lower()
    data = data[data.Product.str.contains('none|n/a|no data|nan')==False]
    data = data[data.Product.str.contains(str(productName).lower())]
    data['new_keywords'] = ''
    data['score']=0.0
    #importing the keywords excel for comparision
    for sent in data['Keywords (m)']:
        t = re.sub(r'\s*,\s*',',',str(sent))
        data['Keywords (m)'] = data['Keywords (m)'].replace([sent],t)
    for idx1,sentence2 in data['Keywords (m)'].iteritems():
        string_words_new = ''
        extracted_words_from_key_words = set()
        if str(sentence2).lower() not in ('nan','n/a','none','no data'):
            for key in end_json:
                for word in end_json[key]:
                    new_sentence_split = np.array(sentence2.split(','))
                    if word in new_sentence_split:
                        extracted_words_from_key_words.add(key)
            for val in extracted_words_from_key_words:
                string_words_new = string_words_new+ ',' + val
            data.at[idx1, 'new_keywords'] = string_words_new
    for idx,sentence in data['new_keywords'].iteritems():
        if str(sentence).lower() not in ('nan','n/a','none','no data','','[none]'):
            array_toCompare = np.array(sentence.split(','))
            flt_sent2 = [w for w in array_toCompare if not w.lower() in STOPWORDS and w not in string.punctuation and w not in ('nan','n/a','none','no data','','[none]')]
            ac = [x.lower() for x in flt_sent2]
            common_elements = list(extracted_words_from_sentence.intersection(ac))
            union_elements = list(set(extracted_words_from_sentence).union(set(ac)))
            #Calculating the score
            score = len(common_elements)/len(union_elements)*100
            data.at[idx, 'score'] = score
    data.to_csv('pavan.csv')
    df_final = data.nlargest(3, 'score')
    result_final_output = df_final[["DOC ID"]].to_numpy().flatten()
    print(result_final_output)
    return result_final_output

if __name__ == "__main__":
    extract_documents(sys.argv[1],sys.argv[2])