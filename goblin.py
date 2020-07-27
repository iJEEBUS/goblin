from analyzer import Analyzer  # for sentiment analysis
from parser import Parser  # for reddit comment extraction
from config import user_agent, client_id, client_secret, username, password

class Goblin:
    def __init__(self):
        # Parsing agent that will extract comments from a subreddit
        self.parser = Parser(user_agent, client_id, client_secret, username, password)
        self.data = None

        # Analyzing agent that will be used to classify comments
        self.analyzer = Analyzer()
        self.analyzer.load()

        # Collect the tickers that are mentioned in comments
        self.tickers = dict()
        

    def extract_new_data(self, subreddit):
        """Extract comments from any subreddit
        
        Load comments into the goblin instance
        """
        self.parser.connect()
        self.parser.add_subreddit(subreddit)
        self.parser.get_all_submissions()
        self.parser.populate_buckets()
        self.data = self.parser.get_data()
    
    def load_data(self, filename):
        pass

    def classify(self):
        """Classify article titles
        """
        for ticker, submissions in self.data.items():
            for sub in submissions:
                print(self.analyzer.classify(sub.title))


## Testing
if __name__ == "__main__":
    g = Goblin()
    g.extract_new_data("RobinHoodPennyStocks")
    g.classify()
