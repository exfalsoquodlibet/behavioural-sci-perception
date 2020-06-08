import pytest
from nltk.corpus import wordnet

from src.utils import chain_functions

from src.preproc_text import tokenise_sent, tokenise_word, \
    _get_wordnet_pos, tag_pos, lemmatise, remove_stopwords, _fix_neg_auxiliary, remove_punctuation, \
    clean_vacancies_quibbles, flatten_irregular_listoflists, detokenise_list

args_tokenise_sent = [("I don't think. Then you shouldn't talk.",
                       ["I don't think.", "Then you shouldn't talk."]),
                      ("I don't think... Then you shouldn't talk.",
                       ["I don't think... Then you shouldn't talk."]),
                      ("I don't think... .", ["I don't think... ."]), ("", [])]


@pytest.mark.parametrize("text,expected", args_tokenise_sent)
def test_tokenise_sent_expected(text, expected):
    assert tokenise_sent(text) == expected


def test_tokenise_sent_errors():
    with pytest.raises(TypeError):
        tokenise_sent(True)
    with pytest.raises(TypeError):
        tokenise_sent(["I want a string!"])


args_tokenise_word = [(["I think.", "You do?"], [["I", "think", "."],
                                                 ["You", "do", "?"]]),
                      (["I think...", "You do?"], [["I", "think", "..."],
                                                   ["You", "do", "?"]]),
                      (["", "You do?"], [[], ["You", "do", "?"]]), ([], [])]


@pytest.mark.parametrize("text,expected", args_tokenise_word)
def test_tokenise_word_expected(text, expected):
    assert tokenise_word(text) == expected


def test_tokenise_word_errors():
    with pytest.raises(TypeError):
        tokenise_word(True)
    with pytest.raises(TypeError):
        tokenise_word("I want a list of strings!")


args_tokenise_sent_and_tokenise_word_expected = [
    ("", []),
    ("I don't think. Then you shouldn't talk!",
     [["I", "do", "n't", "think", "."],
      ["Then", "you", "should", "n't", "talk", "!"]]),
    ("I think...", [["I", "think", "..."]])
]


@pytest.mark.parametrize("text,expected",
                         args_tokenise_sent_and_tokenise_word_expected)
def test_tokenise_sent_and_tokenise_word_expected(text, expected):
    assert tokenise_word(tokenise_sent(text)) == expected


@pytest.mark.parametrize("treebank,expected", [("J", wordnet.ADJ), ("", ""),
                                               ("S", wordnet.ADJ_SAT),
                                               ("R", wordnet.ADV),
                                               ("N", wordnet.NOUN),
                                               ("V", wordnet.VERB),
                                               ("M", wordnet.VERB)])
def test_get_wordnet_pos_expected(treebank, expected):
    assert _get_wordnet_pos(treebank) == expected


args_tag_POS_and_tokenise_word = [
    (["The striped bats are hanging on their feet for best"],
     [[('The', 'DT'), ('striped', 'JJ'), ('bats', 'NNS'), ('are', 'VBP'),
       ('hanging', 'VBG'), ('on', 'IN'), ('their', 'PRP$'), ('feet', 'NNS'),
       ('for', 'IN'), ('best', 'JJS')]]),
    (["I like apples.", "Do you?"], [[('I', 'PRP'), ('like', 'VBP'),
                                      ('apples', 'NNS'), ('.', '.')],
                                     [('Do', 'VB'), ('you', 'PRP'),
                                      ('?', '.')]]), ([], [])
]


@pytest.mark.parametrize("text,expected", args_tag_POS_and_tokenise_word)
def test_tag_POS_and_tokenise_word(text, expected):
    assert tag_pos(tokenise_word(text)) == expected


def test_lemmatise_expected():
    assert isinstance(
        lemmatise([[('I', 'PRP'), ('do', 'VBP'), ('.', '.')],
                   [('Cars', 'NNS'), ('go', 'VBP')]]), list)
    assert lemmatise([[('thinking', 'VBP'),
                       ('thought', 'VBP'), ('think', 'VBP'), ('the', 'DT'),
                       ('', '')]]) == [['think', 'think', 'think', 'the', '']]


@pytest.mark.parametrize("text,expected", [
    ("I am thinking to go. You should think better thoughts and not going!", [[
        "I", "be", "think", "to", "go", "."
    ], ["You", "should", "think", "good", "thought", "and", "not", "go", "!"]])
])
def test_lemmatise_and_preprocessing_funs(text, expected):
    text_preproc = chain_functions(tokenise_sent, tokenise_word, tag_pos)
    assert lemmatise(text_preproc(text)) == expected


@pytest.mark.parametrize("text,expected", [
    ([["i", "won", "go"], ["i", "haven't", "either"]
      ], [['i', 'not', 'go'], ['i', 'not', 'either']]),
    ([["i", "won", "go"], ["you", "could", "not"]], [['i', 'not', 'go'],
                                                     ['you', 'could', 'not']]),
    ([["Wouldn't", "you", "think", "?"]], [["Wouldn't", "you", "think", "?"]])
])
def test__fix_neg_auxiliary(text, expected):
    assert _fix_neg_auxiliary(text) == expected


def test_remove_stopwords():
    dialogue = "I won win. Will you not try? I haven't said. Isn't it weird? Dad does not know. No way."
    dialogue_sw_removed_kn = remove_stopwords(tokenise_word(
        tokenise_sent(dialogue.lower())),
                                              keep_neg=True)
    dialogue_sw_removed_rn = remove_stopwords(tokenise_word(
        tokenise_sent(dialogue.lower())),
                                              keep_neg=False)
    dialogue_sw_removed_wk = remove_stopwords(tokenise_word(
        tokenise_sent(dialogue.lower())),
                                              words_to_keep=['i', 'you'])
    dialogue_sw_removed_es = remove_stopwords(
        tokenise_word(tokenise_sent(dialogue.lower())),
        extra_stopwords=['win', 'said', 'weird', 'way'])
    assert dialogue_sw_removed_kn == [["not", "win", "."], ["not", "try", "?"],
                                      ["not", "said", "."],
                                      ["not", "weird", "?"],
                                      ["dad", "not", "know", "."],
                                      ["no", "way", "."]]
    assert dialogue_sw_removed_rn == [["win", "."], ["try", "?"],
                                      ["said", "."], ["weird", "?"],
                                      ["dad", "know", "."], ["way", "."]]
    assert dialogue_sw_removed_wk == [["i", "not", "win", "."],
                                      ["you", "not", "try", "?"],
                                      ["i", "not", "said", "."],
                                      ["not", "weird", "?"],
                                      ["dad", "not", "know", "."],
                                      ["no", "way", "."]]
    assert dialogue_sw_removed_es == [["not", "."], ["not", "try", "?"],
                                      ["not", "."], ["not", "?"],
                                      ["dad", "not", "know", "."], ["no", "."]]


def test_remove_punctuation():
    dialogue = [["do", "you", "?!"], ["yes", "!!!"], ["oh", ",", "wow", "..."]]
    dialogue_removed_pkt = remove_punctuation(dialogue)
    dialogue_removed_pkt_except = remove_punctuation(dialogue,
                                                     punct_to_keep="!")
    assert dialogue_removed_pkt == [["do", "you"], ["yes"], ["oh", "wow"]]
    assert dialogue_removed_pkt_except == [["do", "you", "!"], ["yes", "!!!"],
                                           ["oh", "wow"]]


@pytest.mark.parametrize(
    "text,expected",
    [("do you?! yes!!! oh, wow...", [["do", "you"], ["yes"], ["oh", "wow"]])])
def test_remove_punctuation_and_text_preprocess(text, expected):
    assert remove_punctuation(tokenise_word(tokenise_sent(text))) == expected


@pytest.mark.parametrize("text,expected", [(
    "<strong>Hello**</strong>, \n please visit\n www.random_site.come or http://www.another_random_site.gov.uk or email random_email@random.gov.uk &#1234; .* Thanks! Best regards*",
    "Hello, please visit or or email . Thanks! Best regards")])
def test_clean_vacancies_quibbles(text, expected):
    assert clean_vacancies_quibbles(text) == expected


@pytest.mark.parametrize("text,expected", [(
    "<strong>Hello**</strong>, \n please visit\n www.random_site.come or http://www.another_random_site.gov.uk or email random_email@random.gov.uk &#1234; .* Thanks! Best regards*",
    ["Hello, please visit or or email .", "Thanks!", "Best regards"])])
def test_clean_vacancies_quibbles_and_tokenise_sent(text, expected):
    assert tokenise_sent(clean_vacancies_quibbles(text)) == expected


@pytest.mark.parametrize("input,expected",
                         [(['a', 'b', [], ['c'], [['d']], [['e'], ['f']]], [
                             'a',
                             'b',
                             'c',
                             'd',
                             'e',
                             'f',
                         ]), (['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])])
def test_flatten_irregular_listoflists(input, expected):
    assert list(flatten_irregular_listoflists(input)) == expected


@pytest.mark.parametrize(
    "input,expected",
    [(["i", "do", "not", "think", "then", "you", "should", "not", "talk"
       ], "i do not think then you should not talk")])
def test_detokenise_list(input, expected):
    assert detokenise_list(input) == expected
