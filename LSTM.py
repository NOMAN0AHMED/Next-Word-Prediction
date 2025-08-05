#NLP sikhne aur practice karne ke liye humein real data chahiye hota hai.
#Lekin har bar internet se data lana mushkil hota hai. 
# Is liye nltk hamein Gutenberg jaise ready-made text corpus deta hai — 
# taake hum direct practice kar saken.
#Ham gutenberg corpus is liye use kar rahay hain taake humein NLP sikhne ke liye ready-made 
# aur clean English text mil jaye. Isse hum apne code par test karte hain aur concepts ko deeply 
# samajhne mein madad milti hai.
import nltk
nltk.download('gutenberg')

from nltk.corpus import gutenberg
import pandas as pd
#load data set
#gutenberg.raw() function se hum "shakespeare-hamlet.txt" file ka raw text load kart
#: raw() method pure text ko string ke form mein deta hai, jo NLP tasks ke liye starting point hota hai.
data = gutenberg.raw("shakespeare-hamlet.txt")

# Save to file
with open("hamlet.txt", "w") as file:
    file.write(data)
#data preprosessing
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
#load data
with open("hamlet.txt", "r") as file:
    text=file.read().lower()
#Tokenizer  the text meaning text convert into vector
tokenizer=Tokenizer()
#fit_on_texts([text]) se tokenizer pura text
#  scan karta hai aur har unique word ko ek integer assign karta hai.
tokenizer.fit_on_texts([text])
#agr +1 na likta to 0 index sa start hota +1 likha ab 1 index sa start ho ga
# +1 isliye, kyunki indexing 1 se hoti hai, aur padding ke liye 0 reserved hota hai. 
totel_word=len(tokenizer.word_index)+1

print(totel_word)
#tokenizer.word_index check the word ,which index is any word 
print(tokenizer.word_index)
#create input sequence
input_sequences=[]
for line in text.split("\n"):
#tokenizer.texts_to_sequences()_List of text sentences leta hai
#Har word ko integer token mein convert karta hai
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
#Ye approach text generation tasks ke liye common hai.
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)
print(input_sequences)
#hamma pata ni hota katna line ka sentencs haan issi liya ham loop laga data haan taka hamma fixed lenght mil sakha
#max_length = 50 manually likho:
#Agar actual sequence ka length < 50 ho → to system zero se pad karega (that’s fine).
#Lekin agar koi sequence > 50 ho → to cut ho jayega (important data loss ho sakta hai) issi liya ham max lenght lata haan
max_lenght=max([len(x) for x in input_sequences])
print(max_lenght)
input_sequence = pad_sequences(input_sequences, maxlen=max_lenght, padding='pre')
print(input_sequence)
#create predicitors and lable
import tensorflow as tf
#x independ featur and y in dependent mean ya is lable target variable
#X input_sequence[:,:-1] take all the word remove last word 
#y input_sequence[:,-1] all the word  take last word 

#Har sequence ka last word prediction hota hai, aur pehle wale word(s) input hote hain.

#har row ka last word only (output)
x,y=input_sequence[:,:-1],input_sequence[:,-1]
#y ko one-hot format mein convert kiya.
y=tf.keras.utils.to_categorical(y,num_classes=totel_word)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout,GRU
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
#Embedding(input_dim, output_dim, input_length)
#Last word ko target label banate hain
model.add(Embedding(totel_word,100,input_length=max_lenght-1))
#return_sequences=True means ye poori sequence return karega (not just last output).
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
#Ye final LSTM layer hai jo sequence ko compress karke ek output vector mein convert karta hai
#Pichle LSTM se sequence leke ek final compressed feature vector banata hai.
model.add(LSTM(100))
model.add(Dense(totel_word,activation="softmax"))
print(model.summary())
import tensorflow as tf
#comile 
#optimizer="adam"	Efficient optimizer to adjust weights
opt=tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])

#earlyStopping=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
print(x_train.shape)  # Yeh line run karo

#train
history=model.fit(
#validation_data har epoch ke baad model ki performance yahan check hoti hai
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1,
    #callbacks=[earlyStopping]
)

model.save("LSTM.keras")
import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
