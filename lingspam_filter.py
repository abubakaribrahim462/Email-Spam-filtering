import os
import sys
import tkinter.filedialog
from collections import Counter
from tkinter import *

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB

root = Tk()
root.title('Email Spam Filtering system')

# Functions

def browse_button():
    global folder_path
    filename = tkinter.filedialog.askdirectory(parent=root, title='choose directory', initialdir='C:\\')
    folder_path.set(filename)
    print(filename)

def redirector(inputStr):
    text.insert(INSERT, inputStr)

sys.stdout.write = redirector

def delete():
   text.configure(state=NORMAL)
   text.delete(1.0, 'end')

def print_all():
    print('The result of the classification is: ')
    print(result)
    print('The confusion matrix is: ')
    print(confusion_matrix(test_labels, result))
    print('The accuracy of the model is: ', format(accuracy_score(test_labels,  result)))
    print('The precision of the model is: ', format(precision_score(test_labels, result)))
    print('The recall of the model is ', (recall_score(test_labels, result)))
    print('The F1_Score of the model is: ', (f1_score(test_labels, result, average=None)))

# GUI

topframe = Frame(root)
topframe.grid()

folder_path = StringVar()

upload = Button(topframe, text='BROWSE', command=browse_button)
upload.config(height=5, width=15)
upload.grid(row=0, column=0)

lbl1 = Label(topframe, textvariable=folder_path)
lbl1.config(width=107)
lbl1.grid(row=0, column=1, sticky=W)

clear = Button(topframe, text='CLEAR MESSAGE', command=delete)
clear.config(height=5, width=15)
clear.grid(row=0, column=2, sticky=N)

text = Text(topframe, height=20)
text.grid(row=1, column=1)

process = Button(topframe, text='PROCESS', command=print_all)
process.config(height=5, width=15)
process.grid(row=1, column=0, sticky=N)

exit = Button(topframe, text='EXIT', command=root.quit)
exit.config(height=5, width=15)
exit.grid(row=1, column=2, sticky=N)

label = Label(text='Abubakar Ibrahim, 2022', anchor='center')
label.grid()

# Machine Learning Aspect

# Creating of a word dictionary

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)

    list_to_remove = list(dictionary.keys())
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


# Feature extraction function

def extract_features(mail_dir):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    docID = 0;
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            docID = docID + 1
    return features_matrix


# Create a dictionary of words with its frequency

train_dir = r'./ling-spam/train-mails/'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training Naive bayes classifier
model = MultinomialNB()

# Fitting The Model
model.fit(train_matrix, train_labels)

#   Test the unseen mails for Spam
test_dir = r'./ling-spam/test-mails/'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1

result = model.predict(test_matrix)
root.mainloop()
