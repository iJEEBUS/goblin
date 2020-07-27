import nltk, ssl, random, re, string
from nltk import classify, FreqDist, NaiveBayesClassifier
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pickle

class Analyzer:

    def __init__(self):
        self.model = None
        self.dataset = None

    def preprocess_data(self):
        self.pos_tweets = twitter_samples.strings('positive_tweets.json')
        self.neg_tweets = twitter_samples.strings('negative_tweets.json')
        self.text = twitter_samples.strings('tweets.20150430-223406.json')
        self.pos_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        self.neg_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
        self.pos_cleaned_tokens_list = [self.remove_noise(tokens) for tokens in self.pos_tweet_tokens]
        self.neg_cleaned_tokens_list = [self.remove_noise(tokens) for tokens in self.neg_tweet_tokens]
        self.all_pos_words = self.get_all_words(self.pos_cleaned_tokens_list)
        self.all_neg_words = self.get_all_words(self.neg_cleaned_tokens_list)
        self.freq_dist_pos = FreqDist(self.all_pos_words)
        self.freq_dist_neg = FreqDist(self.all_neg_words)
        self.pos_tokens_for_model = self.get_tweets_for_model(self.pos_cleaned_tokens_list)
        self.neg_tokens_for_model = self.get_tweets_for_model(self.neg_cleaned_tokens_list)

        self.pos_dataset = [(tweet_dict, "Positive") for tweet_dict in self.pos_tokens_for_model]
        self.neg_dataset = [(tweet_dict, "Negative") for tweet_dict in self.neg_tokens_for_model]
        self.dataset = self.pos_dataset + self.neg_dataset
        random.shuffle(self.dataset)
        mid = len(self.dataset) // 2
        self.train_data = self.dataset[:mid]
        self.test_data = self.dataset[mid:]

    def lemmatize_sentence(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []

        for word, tag in pos_tag(tokens):

            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))

        return lemmatized_sentence

    def remove_noise(self, tokens, stop_words=stopwords.words('english')):
        cleaned_tokens = []

        for token, tag in pos_tag(tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def get_all_words(self, cleaned_tokens):
        for tokens in cleaned_tokens:
            for token in tokens:
                yield token
        
    def get_tweets_for_model(self, cleaned_tokens):
        for tweet_tokens in cleaned_tokens:
            yield dict([token, True] for token in tweet_tokens)

    def train_model(self, data):
        self.model = NaiveBayesClassifier.train(data)
    
    def dump(self):
        print(classify.accuracy(self.model, self.test_data))
        print(self.model.show_most_informative_features(10))
        custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
        custom_tweet = "Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies"
        custom_tokens = self.remove_noise(word_tokenize(custom_tweet))
        print(self.model.classify(dict([token, True] for token in custom_tokens)))

    def save(self, file='model/cc.pickle'):
        """Save the current model and dataset
        """
        model_and_data = [self.model, self.dataset]
        f = open(file, 'wb')
        pickle.dump(model_and_data, f)
        f.close()

    def load(self, file='model/cc.pickle'):
        """Load a model and it's data

        This also setups the self.test_data and self.train_data variables
        """
        f = open(file, 'rb')
        self.model, self.dataset = pickle.load(f)
        f.close()

        mid = len(self.dataset) // 2
        random.shuffle(self.dataset)
        self.train_data = self.dataset[:mid]
        self.test_data = self.dataset[mid:]

    def classify(self, sentence):
        tokens = self.remove_noise(word_tokenize(sentence))
        return self.model.classify(dict([token, True] for token in tokens))

## Testing
if __name__ == "__main__":

    a = Analyzer()
    # Create a new NLP model
    # a.preprocess_data()
    # a.train_model(a.train_data)
    # a.save()
    # a.dump()

    # Load an NLP model
    a.load()
    a.dump()




# This might be needed for the sake of downloading other datasets
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk libraries that are used
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('twitter_samples')