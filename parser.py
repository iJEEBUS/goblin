import praw


class Parser:

    def __init__(self, user_agent, client_id, client_secret, username, password):
        self.user_agent = user_agent
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.subreddits = dict()
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
        """Add subreddit(s) to to the objects membory

        Can input a string for a single subreddit or a list of subreddits (as strings)
        """

        sub_type = type(subreddit)

        if sub_type == list:
            self.subreddits = {sub: [] for sub in subreddit}
        elif sub_type == str:
            self.subreddits[subreddit] = []
        else:
            raise ValueError(
                f"Cannot search for a subreddit of type {sub_type}")

    def get_all_comments(self):
        """Scrape all comments from ubreddits
        """
        for sub in self.subreddits.keys():
            self.subreddits[sub] = self.r_agent.subreddit(
                f"{sub}").comments(limit=5)
