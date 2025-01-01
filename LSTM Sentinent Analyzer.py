import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , LSTM , Dense

# these import numpy and keras needing to build lstm layers and uses embedding to embed data into a 16 dimensional vector.

dataset = [
    ("I love this product! It's amazing.", "positive"),
    ("This is the worst experience I've ever had.", "negative"),
    ("It's okay, not great but not terrible.", "neutral"),
    ("Absolutely fantastic service!", "positive"),
    ("I'm extremely disappointed.", "negative"),
    ("The item arrived as expected.", "neutral"),
    ("Highly recommend this to everyone!", "positive"),
    ("The quality is awful, do not buy.", "negative")
]

# this is the dataset that the model will train on and it will also show whta statments are postive , negative or neutral.

validation_data = [
    ("This product exceeded my expectations.", "positive"),
    ("I hate this item; it's a waste of money.", "negative"),
    ("It's neither good nor bad, just average.", "neutral"),
    ("What an incredible experience!", "positive"),
    ("Terrible quality and poor service.", "negative"),
    ("The delivery time was acceptable.", "neutral"),
    ("A great choice, I'm very satisfied!", "positive"),
    ("Not worth the price at all.", "negative")
]

# this data is the validation data that will introduce new words that do not appear in the dataset for teh machine to learn examples that use more advanced words.

texts = [sample[0] for sample in dataset]
labels = [sample[1] for sample in dataset]

# These lines extract the text and from the dataset and put in one list and extract the labels from one list and put in another list.
tokenizer = tf.keras.layers.TextVectorization(max_tokens = 50)

# a tokenizer converts text into sequences of tokens ot numeric indices that cna be fed therough a machine learning model.
#max tokens limits the size of the vocabulary.
# only the 50 most common words will be included in the vocabulary. any words after this are OOV and are replaced with a special token.
tokenizer.adapt(texts)

# this line will pass our list of strings into the tokenizer and it will build a vocabulary of 50 most common words as out parameter is set to 50.
sequences = tokenizer(texts)
# the tokenizer will convert the text into tokenized sequences ans it assigns to 'sequences' variable.

print(sequences)

#run this to see the sequences
# as you can see it is a list of a list of numbers.
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)
#padding a sequence will add zeros by defaukt to make them the same length as the longer sequences in the input.
# also it willl truncate a sequence if it is longer than the maximum desired length.
# when this is printed all teh numpy arrays will have the same length.
print(padded_sequences)
model = Sequential([
    Embedding(50,16),
    LSTM(16),
    Dense(1,activation='sigmoid')
])
# this line builds a sequential mode, which is a model where the output of one layer is the input to the next layer.
# The list inside Sequential defines the order of layers in the model.
# Embedding maps each word(represented as a integer index) into a dense vector of fixed size (an embedding).
# The first parameter (50) is the size of the voacbulary (the number of unique tokens in the dataset) This should match the vocab size in the tokenizer.
# The second parameter (16) is the size of the embedding vector each word will be represnted by a 16 dimensional vector.
#The second line adds a Long short term memroy layer to the model.
#LSTM's are a type of Recurrent neural network that hnadle sequential data by remembering past informationa and learning dependencies over time.
# The paramter (16) is the number of hidden units (or dimensions) in the lstm cell .This controls the size of the output vector from the LSTM.
# The LSTM processes the input sequence from the (embedding layer) produces a single output vector representing the sequences learned features.
# The last line adds a fully connected dense layer with (the first paramter creating a single output neuron for classification)
#The second paramter applie sthe sigmoid function to the output which maps the reuslt to a value between 0 and 1.
# This produces the output which is below 0.5 is negative and greater than 0.5 is postive.
# MODEL ARCHITECTURE
# Input -> Embedding Layer -> LSTM Layer -> Dense (Sigmoid) Layer -> Output
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.compile configures the model for training.
# the loss function is binary crossentropy which measures the diffrence between predicted probabilties and actual labels.
# This chooses the adam optimizer which adjusts the models weights based on past gradients helping to minimize loss efficiently.
# this chooses accuracy as the metric to evaluate model performance during training, showing the percentage of correct predictions.
label_to_id = {'positive': 1 , 'negative': 0, 'neutral' : 0.5 }
numeric_labels  = np.array([label_to_id[label] for label in labels])

# The first line converts the text into numbers and takes positive as 1 and negative as 0 in the dictionary.
# For each label in the labels list the second line converts retrives the corresponding numeric value (0 or 1).
# the np.array converts the list of lists into a numpy array.
# Positive --> 1
# Negative --> 0
print(len(padded_sequences))
print(len(numeric_labels))
model.fit(padded_sequences,numeric_labels,epochs=100,verbose=1,validation_split=0.2)

# this line is training the model for 100 epochs.
