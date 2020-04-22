from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from transformers import BertTokenizer, BertModel
import sys


def transform(filename):
        datafile = pd.read_excel(filename, header=0)
        datafile = datafile.fillna('-')
        datafile["text"] = datafile["Text1"]+ "----------"+ datafile["Text2"]
        datafile["labels"] = datafile["Value"]
        datafile_new = datafile[1:]

        transformer_dict = {-1: 0, 1: 1}
        datafile_new['labels'] = datafile_new['labels'].apply(lambda x: transformer_dict[x])

        datafile_new.insert(1, "alpha", "a")
        datafile_new = datafile_new[["labels", "alpha", "text"]]
        #print(datafile)
        return datafile, datafile_new

def transformer(train_df, eval_df, datafile):

    #tokenizer = BertTokenizer.from_pretrained("bert-base-dutch-cased")
    model = ClassificationModel("bert", "bert-base-dutch-cased", use_cuda = False, num_labels=2) # You can set class weights by using the optional weight argument

    # Train the model
    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    print(model_outputs)

    predlist = []
    model1_outputs = model_outputs.tolist()
    for output in model1_outputs:
        if output[0] > output[1]:
            prediction = 0
        else:
            prediction = 1
        predlist.append(prediction)

    labels = eval_df["labels"].tolist()
    print(labels)
    print(predlist)


    print(classification_report(labels, predlist))
    print(confusion_matrix(labels, predlist))
    print(accuracy_score(labels, predlist))


def main():
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    datafile, traindata = transform(trainfile)
    train = traindata
    train = train[["text", "labels"]]
    datafile, trialdata = transform(testfile)
    trial = trialdata
    trial = trial[["text", "labels"]]
    test = trial

    transformer(train, test, datafile)







if __name__ == "__main__":
    main()
