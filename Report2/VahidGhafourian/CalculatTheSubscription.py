import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

digiData = pd.read_csv("balance_classified_train.csv")
digiData = digiData.drop(columns=["Unnamed: 0", "Unnamed: 0.1"])

like = digiData[digiData["target"]==(0 or 2) ]
verylike = digiData[digiData["target"]==(1 or 3) ]

xtrain = verylike.comment
likes = like.comment
vocab_size = 19192
embedding_dim = 120
max_length = 120

tokenizer = Tokenizer(num_words = vocab_size, lower = False)
tokenizer.fit_on_texts(xtrain)
word_index = tokenizer.word_index
def encode(valid):
    valid_sequences = tokenizer.texts_to_sequences(valid)
    valid_data = pad_sequences(valid_sequences, padding = "post", maxlen = max_length)
    return valid_data

def getAVN (setOne, setTwo, numOfWords):
    eshterak = {}
    for i, v in enumerate(setOne):
        for j, k in enumerate(v):
            if k!=0:
                if (eshterak.get(str(k))==None):
                    eshterak[str(k)] = 0
                if (eshterak.get(str(k))==0):
                    for n, a in enumerate(setTwo):
                        for m, b in enumerate(a):
                            if b!=0 and k==b:
                                eshterak[str(k)] = eshterak.get(str(k))+1
                            if b==0:
                                break
            else:
                break
        # print(v)
        if i%500 == 0:
            print(i, end=" ")
    # print(type(eshterak))
    return sum(eshterak.values())/numOfWords

likeDislike = encode(digiData[digiData["target"]==0].comment)
verylikeDislike = encode(digiData[digiData["target"]==1].comment)
likeVeryDislike = encode(digiData[digiData["target"]==2].comment)
verylikeVeryDislike = encode(digiData[digiData["target"]==3].comment)

print("start...")
LDvLD = getAVN(likeDislike, verylikeDislike, vocab_size)
print("done. 1")
LDLvD = getAVN(likeDislike, likeVeryDislike, vocab_size)
print("done. 2")
LDvLvD = getAVN(likeDislike, verylikeVeryDislike, vocab_size)
print("done. 3")
vLDLvD = getAVN(verylikeDislike, likeVeryDislike, vocab_size)
print("done. 4")
vLDvLvD = getAVN(verylikeDislike, verylikeVeryDislike, vocab_size)
print("done. 5")
LvDvLvD = getAVN(likeVeryDislike, verylikeVeryDislike, vocab_size)
print("done. 6")

result = "LDvLD: " + str(LDvLD) + "\nLDLvD:" + str(LDLvD) + "\nLDvLvD:" + str(LDvLvD) + "\nvLDLvD:" + str(vLDLvD) + "\nvLDvLvD:" + str(vLDvLvD) + "\nLvDvLvD:" + str(LvDvLvD)

f = open("subscription.txt", "a")
f.write("Result:\n" + result)
f.close()
