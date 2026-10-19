"""
Module: text_analyzer.py
Description: Natural Language Processing (NLP) text mining and sentiment analysis
for open-ended survey comments, feedback fields, and subjective responses.
Extracts sentiment polarity, common keywords, and prepares word clouds.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple, Any
import pandas as pd

# Standard survey stopwords list
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "provided", "comment", "none", "na", "survey", "poll", "also", "overall", "need"
])

POSITIVE_WORDS = set([
    "great", "exceptional", "good", "love", "smooth", "snappy", "seamless", "value", "top", "clean",
    "crucial", "significantly", "must", "promotes", "sustainable", "ideal", "equilibrium", "saved",
    "boosts", "creative", "synergy", "flexibility", "improved", "well-being", "high", "routines",
    "adequate", "balanced", "reliable", "fast", "friendly", "helpful", "convenient", "efficient", "happy"
])

NEGATIVE_WORDS = set([
    "fast", "overpriced", "nothing", "average", "drains", "unresponsive", "delayed", "bloatware",
    "degrades", "plasticky", "fragile", "budget", "overruns", "lack", "maintenance", "high", "unacceptable",
    "damages", "retention", "distraction", "disappointing", "spiked", "micromanagement", "poor",
    "terrible", "frustrating", "slow", "broken", "worst", "bad", "difficult", "problem", "issues"
])

class TextAnalyzer:
    """
    Lightweight, dependency-free text processor for open-ended feedback sentiment and theme extraction.
    """

    def __init__(self, texts: List[str]):
        self.raw_texts = [str(t) for t in texts if pd.notna(t) and str(t).strip() != ""]

    def clean_token_stream(self) -> List[str]:
        """
        Tokenizes text, strips punctuation, normalizes case, and filters stopwords.
        """
        all_words = []
        for text in self.raw_texts:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            filtered = [w for w in tokens if w not in STOPWORDS]
            all_words.extend(filtered)
        return all_words

    def get_top_keywords(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Returns top N frequent meaningful keywords from comments.
        """
        words = self.clean_token_stream()
        counter = Counter(words)
        return counter.most_common(top_n)

    def analyze_sentiments(self) -> Dict[str, Any]:
        """
        Categorizes each text comment into Positive, Neutral, or Negative sentiment
        based on lexicon polarity density.
        """
        pos_count = 0
        neg_count = 0
        neu_count = 0
        scored_comments = []

        for text in self.raw_texts:
            words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
            pos_matches = words.intersection(POSITIVE_WORDS)
            neg_matches = words.intersection(NEGATIVE_WORDS)

            score = len(pos_matches) - len(neg_matches)
            if score > 0:
                sentiment = "Positive"
                pos_count += 1
            elif score < 0:
                sentiment = "Negative"
                neg_count += 1
            else:
                sentiment = "Neutral"
                neu_count += 1

            scored_comments.append({
                "comment": text,
                "sentiment": sentiment,
                "score": score
            })

        total = len(self.raw_texts) if self.raw_texts else 1
        return {
            "total_comments": len(self.raw_texts),
            "sentiment_counts": {
                "Positive": pos_count,
                "Neutral": neu_count,
                "Negative": neg_count
            },
            "sentiment_percentages": {
                "Positive": round((pos_count / total) * 100, 1),
                "Neutral": round((neu_count / total) * 100, 1),
                "Negative": round((neg_count / total) * 100, 1)
            },
            "scored_sample": scored_comments[:10]
        }
