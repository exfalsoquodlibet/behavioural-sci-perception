import numpy as np
from typing import List
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


def get_sentiment_score_VDR(text: List[str],
                            score_type: str = "compound") -> List[float]:
    """
    Calculate nltk Vader sentiment analysis score (score_type: 'compound' default, 'pos', 'neg')
    for each sentence in a paragraph text. The input must be a list of string sentences.

    Return a list of scores (as float), one score for each sentence in the paragraph text.
    If text is empty, return NaN.

    Args:
        text :          a paragraph of text as a list of string sentences. ["I think.", "Therefore, I am."]

        score_type :    'compound' (default), 'pos' or 'neg'

    Returns:
        A list of sentiment scores. E.g., [0.0, 0.1]
    """

    try:
        scores = (np.nan if len(text) == 0 else
                  [analyser.polarity_scores(s)[score_type] for s in text])
        return scores

    except TypeError as e:
        return e


if __name__ == "__main__":

    test_text = [
        "I love running.", "But oh how much I hate stretching afterwards!"
    ]

    print(get_sentiment_score_VDR(test_text))
