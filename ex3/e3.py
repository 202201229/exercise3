import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download necessary resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')
nltk.download('wordnet')

# Read Moby Dick from Gutenberg dataset
moby_dick = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Parts-of-Speech (POS) tagging
pos_tags = nltk.pos_tag(filtered_tokens)

# POS frequency
pos_freq = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_freq.most_common(5)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token, pos=tag[0].lower()) if tag[0].lower() in ['n', 'v'] else token for token, tag in pos_tags][:20]

# Plotting frequency distribution
pos_freq.plot()

# Display the results
print("Most common parts of speech:")
for pos, count in top_pos:
    print(pos, count)

print("Lemmatized tokens:")
print(lemmatized_tokens)

# Show the plot
plt.show()