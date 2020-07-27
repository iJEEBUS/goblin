import praw
import re
from collections import defaultdict
from config import user_agent, client_id, client_secret, username, password

class Parser:

    def __init__(self, user_agent, client_id, client_secret, username, password):
        self.user_agent = user_agent
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.subreddits = defaultdict()
        self.symbol_buckets = defaultdict(list)
        self.r_agent = None

    def connect(self):
        """Connect to a Reddit instance
        """
        self.r_agent = praw.Reddit(
            user_agent=self.user_agent,
            client_id=self.client_id,
            client_secret=self.client_secret,
            username=self.username,
            password=self.password
        )

    def disconnect(self):
        """Destroy the reddit instance
        """
        del(self.r_agent)
        self.r_agent = None

    def add_subreddit(self, subreddit):
        """Add subreddit to to the objects memory

        Accepts a single subreddit. If a list is passed then it is routed to the
        add_subreddits() method.
        """

        sub_type = type(subreddit)
        if sub_type == str:
            self.subreddits[subreddit] = []
        elif sub_type == list:
            self.add_subreddits(subreddit)
        else:
            raise ValueError(
                f"Cannot search for a subreddit of type {sub_type}")

    def add_subreddits(self, subreddits):
        """Adds multiple subreddits to to the objects memory

        Accepts a list of subreddits. If a single subreddit is passed then it is 
        routed to the add_subreddit() method.
        """

        sub_type = type(subreddits)
        if sub_type == list:
            self.subreddits = {sub: [] for sub in subreddits}
        elif sub_type == str:
            self.add_subreddit(subreddits)
        else:
            raise ValueError(
                f"Cannot search for a subreddit of type {sub_type}")

    # def get_all_comments(self):
    #     """Scrape all comments from subreddits
    #     """
    #     for sub in self.subreddits.keys():
    #         self.subreddits[sub]['comments'] = self.r_agent.subreddit(f"{sub}").comments(limit=None)
    
    def get_all_submissions(self):
        """Scrape all article titles from subreddits
        """
        for sub in self.subreddits.keys():
            self.subreddits[sub] = self.r_agent.subreddit(f"{sub}").hot(limit=15)
            # for s in self.subreddits[sub]:
            #     print(s.title)


    def populate_buckets(self):
        """Filter out submissions that do not have a ticker value
        """
        regex = re.compile(r'[A-Z]{3,4}')

        filtered = []
        symbols = []

        # Extract only the posts that have a symbol in their title
        for sub, sub_container in self.subreddits.items():
            for submission in sub_container:
                if regex.search(submission.title):
                    filtered.append(submission)

        # Figure out what the ticker is then move the submission accordingly
        for submission in filtered:
            temp = submission.title.split()

            for word in temp:
                
                # Potential symbol found
                if word.startswith('$'):
                    symbol = word[1:]

                    # Only add symbol if it's made of chars
                    is_number = any(char.isdigit() for char in symbol)
                    if is_number is False:
                        self.symbol_buckets[symbol].append(submission)


    def get_data(self):
        """Returns submissions keyed by their ticker symbol
        """
        return self.symbol_buckets

                
## Testing
if __name__ == "__main__":

    p = Parser(user_agent, client_id, client_secret, username, password)
    p.add_subreddit("RobinHoodPennyStocks")
    p.connect()
    p.get_all_submissions()
    p.populate_buckets()
    print(p.symbol_buckets.items())