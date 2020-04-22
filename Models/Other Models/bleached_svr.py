import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, roc_curve
import numpy as np
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import sys

def dummy(x):
    return x

def combine_bleached(x):
    punctclist, shapelist, vowelslist, lengthlist = punctC(x).split(), shape(x).split(), vowels(x).split(), length(x).split()
    mylist = []
    for punctc1, shape1, length1, vowel1 in zip(punctclist, shapelist, lengthlist, vowelslist):
        mylist.append(punctc1 + " " + shape1 + " " + length1 + " " + vowel1)
    x = " ".join(mylist)
    return x


def punctC(x):
    tknzr = TweetTokenizer()
    punctclist = []
    xlist = tknzr.tokenize(x)
    for word in xlist:
        previous_alnum = False
        punctc = ""
        for char in word:
            if char.isalnum():
                if not previous_alnum:
                    punctc += 'W'
                previous_alnum = True
            else:
                previous_alnum = False
                punctc += char
        punctclist.append(punctc)
    x = " ".join(punctclist)
    return x


def shape(x):
    tknzr = TweetTokenizer()
    shapelist = []
    xlist = tknzr.tokenize(x)
    for word in xlist:
        shape_word = ""
        for char in word:
            if char.islower():
                shape_word += 'L'
            elif char.isupper():
                shape_word += 'U'
            elif char.isupper():
                shape_word += 'D'
            else:
                shape_word += 'O'
        shapelist.append(shape_word)
    x = " ".join(shapelist)
    return x

def vowels(x):
    tknzr = TweetTokenizer()
    vowels = []
    xlist = tknzr.tokenize(x)
    for word in xlist:
        bleached_word = ""
        for char in word:
            if char.lower() in ['a', 'e', 'o', 'u', 'i']:
                bleached_word += "V"
            elif char.lower() in ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']:
                bleached_word += "C"
            elif char in string.punctuation:
                bleached_word += "P"
            else:
                bleached_word += "O"
        vowels.append(bleached_word)
    x = " ".join(vowels)
    return x

def length(x):
    tknzr = TweetTokenizer()
    xlist = tknzr.tokenize(x)
    length = []
    for word in xlist:
        single_length = str(len(word))
        if len(single_length) == 1:
            single_length = '0' + single_length
        else:
            single_length = single_length
        length.append(single_length)
    x = " ".join(length)
    return x

def tokenizer(text):
    textlist = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            textlist.append(word)
    text = " ".join(textlist)
    return text

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))


def concatenate_features(data_df):
    data_df["features"] = data_df["Text1"].map(str) + " ---------- "+ data_df["Text2"].map(str)
    data_df["label"] = data_df["Value"]
    return data_df

def main():
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    print(trainfile)

    featurelist = ['Text1', 'Text2', 'Value']

    train_df = pd.read_excel(trainfile, header=0)
    train_df = train_df.fillna('-')
    print(train_df[featurelist])
    train_df = concatenate_features(train_df)
    Xtrain = train_df['features'].tolist()
    Ytrain = train_df['label'].tolist()


    test_df = pd.read_excel(testfile, header=0)
    if 'twin' in testfile.lower():
        Yrelationlabel = test_df['Relation'].tolist()
    test_df = test_df.fillna('-')
    print(test_df[featurelist])
    test_df = concatenate_features(test_df)
    Xtest = test_df['features'].tolist()
    Ytest = test_df['label'].tolist()



    punctC_vec = TfidfVectorizer(  preprocessor = punctC,
                                    analyzer = 'char',
                                    ngram_range = (1,5))

    shape_vec = TfidfVectorizer(  preprocessor = shape,
                                    analyzer = 'char',
                                    ngram_range = (5,10))


    vowels_vec = TfidfVectorizer(  preprocessor = vowels,
                                    analyzer = 'char',
                                    ngram_range = (10,15))

    length_vec = TfidfVectorizer( preprocessor = length,
                                    analyzer = 'char',
                                    ngram_range = (10,15))

    combined_bleach_vec = TfidfVectorizer(preprocessor = combine_bleached,
                                            analyzer = 'word',
                                            ngram_range = (1,3))



    vec = FeatureUnion([("punctC_vec", punctC_vec), ("shape_vec", shape_vec), ("vowels_vec", vowels_vec), ("length_vec", length_vec), ("combined_bleach_vec", combined_bleach_vec)])




    classifier = Pipeline( [('vec', vec),
                            ('cls', svm.LinearSVR(C=0.5))] )

    classifier.fit(Xtrain, Ytrain)


    try:
        coefs = classifier.named_steps['cls'].coef_
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 20)
        print()
    except:
        pass

    Yguess = classifier.predict(Xtest).tolist()
    print(Yguess)

    if 'reddit' in testfile.lower() or 'pan' in testfile.lower():
    #FOR TESTING ON PAN DATA
        Ycomparelist = []
        for pred in Yguess:
            if pred < 0:
                Ycomparelist.append(-1)
            else:
                Ycomparelist.append(1)

        print(confusion_matrix(Ytest, Ycomparelist))
        print(classification_report(Ytest, Ycomparelist))
        print(accuracy_score(Ytest,Ycomparelist))
    elif 'twin' in testfile.lower():
        total_random = 0
        count_random = 0
        total_sibling = 0
        count_sibling = 0
        total_itwin = 0
        count_itwin = 0
        total_utwin = 0
        count_utwin = 0
        for relation, guess in zip(Yrelationlabel, Yguess):
            if relation == 'R':
                r = 'RANDOM:'
                total_random += guess
                count_random += 1
            elif relation == 'S':
                r = 'SIBLING:'
                total_sibling += guess
                count_sibling += 1
            elif relation == 'I':
                r = 'IDENTICAL TWIN'
                total_itwin += guess
                count_itwin += 1
            elif relation == 'U':
                r = 'NONIDENTICAL TWIN'
                total_utwin += guess
                count_utwin += 1
            print(r, guess)
        avg_random = total_random / count_random
        avg_sibling = total_sibling / count_sibling
        avg_itwin = total_itwin / count_itwin
        avg_utwin = total_utwin / count_utwin
        print("\n\n\nAVERAGE IDENTICAL TWIN:", avg_itwin)
        print("AVERAGE NONIDENTICAL TWIN:", avg_utwin)
        print("AVERAGE SIBLING:", avg_sibling)
        print("AVERAGE RANDOM:", avg_random)



if __name__ == "__main__":
    main()
