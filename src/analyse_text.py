import numpy as np
from typing import List
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
analyser = SentimentIntensityAnalyzer()


def get_sentiment_score_VDR(text: List[str],
                            score_type: str = "compound") -> List[float]:
    """
    Calculates nltk Vader sentiment analysis score (score_type: 'compound' default, 'pos', 'neg')
    for each sentence in a paragraph text. The input must be a list of string sentences.

    Return a list of scores (as float), one score for each sentence in the paragraph text.
    If text is empty, return NaN.

    Args:
        text :          a paragraph of text as a list of string sentences. ["I think.", "Therefore, I am."]

        score_type :    'compound' (default), 'all' (i.e., pos, neg, neu)

    Returns:
        A list of sentiment scores in the [-1.0, 1.0] range.
    """

    if score_type == 'all':
        score_type = ['pos', 'neu', 'neg']

    try:
        score = np.nan if len(text) == 0 else {
            k: v
            for k, v in analyser.polarity_scores(text).items()
            if k in score_type
        }

        return score

    except TypeError as e:
        return e


def get_sentiment_score_TB(text: str) -> float:
    """
    Calculates sentiment analysis score using TextBlob
    for the provided text.

    If text is empty, return NaN.

    Args:
        text :          a paragraph of text as string.

    Returns:
        A sentiment scores in the [-1.0, 1.0] range.
    """

    score = np.nan if len(text) == 0 else TextBlob(text).sentiment.polarity

    return score


if __name__ == "__main__":

    test_text = [
        "I love running.", "But oh how much I hate stretching afterwards!", ""
    ]

    print(get_sentiment_score_VDR(test_text))

    print([get_sentiment_score_TB(text) for text in test_text])
