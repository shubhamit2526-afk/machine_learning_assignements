import nltk 

paragraph = "`int8`, `int16`, `int32`, and `int64` are integer data types that differ in the amount of memory they use and the range of values they can store. The number after `int` represents the number of bits used to store the integer. For example, `int8` uses 8 bits (1 byte) and can store values from -128 to 127, while `int16` uses 16 bits and stores larger values. Similarly, `int32` and `int64` can store much bigger numbers because they use more memory. Smaller integer types save memory and are faster for small data, whereas larger types are used when very large numbers need to be stored. These data types are commonly used in libraries like NumPy and Pandas for efficient memory management and faster computation."

#cleaning the texts

import re # re library will use for regular expression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



ps = PorterStemmer()
Wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

corpus = []

# create the empty list name as corpus  becaue after cleaned the dataset corpus 

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [Wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

#creating the bag of words model

#Also we called as document matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_tf = tf.fit_transform(corpus).toarray()







