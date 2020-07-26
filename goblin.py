import praw
import nltk
from .analyzer import Analyzer  # for sentiment analysis
from .parser import Parser  # for reddit comment extraction


class Goblin:

    def __init__(self):
        ua = "Comment Extraction and Classification (by /u/USERNAME)"
        c_id = ""
        c_secret = ""
        u_name = ""
        u_pswd = ""
        self.parser = Parser(ua, c_id, c_secret, u_name, u_pswd)
        self.analyzer = Analyzer()
        self.data = dict()

    def get_comments(self):

        # Testing
if __name__ == "__main__":
    g = Goblin()
