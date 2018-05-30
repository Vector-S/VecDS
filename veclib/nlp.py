from nltk import word_tokenize
import subprocess
import gensim
import sklearn
import nltk
import pandas as pd
import stop_words



def lemmatize(word,pos='v'):
    wl=nltk.stem.WordNetLemmatizer()
    word = wl.lemmatize(word,pos=pos)
    return word


def doc2bow(doc):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = stop_words.get_stop_words('en')
    # Create p_stemmer of class PorterStemmer
    p_stemmer = nltk.stem.porter.PorterStemmer()
    # clean and tokenize document string
    raw = doc.lower()
    bow = tokenizer.tokenize(raw)
    # remove stop words from tokens
    bow = [i for i in bow if not i in en_stop]
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return bow


def nltk_parser(text):
    try:
        text = word_tokenize(text)
        tagged = nltk.pos_tag(text)
        df_tag = pd.DataFrame(tagged,columns = ['Word','Tag'])
        return df_tag
    except:
        print("Exception: nltk_parser(text)")
        return None