"""
NLP 텍스트 전처리 유틸리티 모듈

이 모듈은 자연어 처리를 위한 다양한 전처리 함수들을 제공합니다.
정수 인코딩, 토큰화, 정규화 등의 기능을 포함합니다.

Author: NLP Study Team
Created: 2025-10-17
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_text_for_encoding(text, min_word_length=2, language="english"):
    """
    텍스트를 정수 인코딩을 위해 전처리하는 함수

    Parameters:
    -----------
    text : str
        원본 텍스트
    min_word_length : int, default=2
        최소 단어 길이. 이보다 짧은 단어는 제거됨
    language : str, default='english'
        불용어 언어 설정 ('english', 'korean' 등)

    Returns:
    --------
    vocab : dict
        단어별 빈도수 딕셔너리 {단어: 빈도수}
    preprocessed_sentences : list
        전처리된 문장들의 리스트 (각 문장은 단어 리스트)

    Example:
    --------
    >>> text = "Hello world. This is a test."
    >>> vocab, sentences = preprocess_text_for_encoding(text)
    >>> print(vocab)
    {'hello': 1, 'world': 1, 'test': 1}
    """
    # 문장 토큰화
    sentences = sent_tokenize(text)

    # 불용어 리스트
    try:
        stop_words = set(stopwords.words(language))
    except OSError:
        print(
            f"⚠️ 경고: '{language}' 불용어 데이터를 찾을 수 없습니다. 빈 불용어 리스트를 사용합니다."
        )
        stop_words = set()

    # 단어 사전(key=단어, value=빈도수)
    vocab = {}

    # 토큰화/정제/정규화 처리 결과
    preprocessed_sentences = []

    # 각 문장을 처리
    for sent in sentences:
        sent = sent.lower()  # 대소문자 정규화(소문자 변환)
        words = word_tokenize(sent)  # 단어(공백 기준) 토큰화

        # 불용어 제거 및 단어 길이 필터링
        filtered_tokens = [
            word
            for word in words
            if word not in stop_words
            and len(word) > min_word_length
            and word.isalpha()  # 알파벳만 (구두점 제거)
        ]

        # vocab 사전에 단어 빈도수 추가
        for word in filtered_tokens:
            vocab[word] = vocab.get(word, 0) + 1

        preprocessed_sentences.append(filtered_tokens)

    return vocab, preprocessed_sentences


def create_tokenizer_and_sequences(
    preprocessed_sentences, num_words=None, oov_token="<OOV>"
):
    """
    전처리된 문장들로부터 토크나이저를 생성하고 정수 시퀀스로 변환하는 함수

    Parameters:
    -----------
    preprocessed_sentences : list
        전처리된 문장들의 리스트 (각 문장은 단어 리스트 또는 문자열)
    num_words : int, optional, default=None
        단어 집합의 최대 크기. None이면 모든 단어 사용
    oov_token : str, default='<OOV>'
        OOV(Out-Of-Vocabulary) 토큰 설정

    Returns:
    --------
    tokenizer : Tokenizer
        학습된 Tokenizer 객체
    sequences : list
        정수 시퀀스로 변환된 문장들

    Example:
    --------
    >>> sentences = [['hello', 'world'], ['test', 'sentence']]
    >>> tokenizer, sequences = create_tokenizer_and_sequences(sentences)
    >>> print(sequences)
    [[2, 3], [4, 5]]
    >>> print(tokenizer.word_index)
    {'<OOV>': 1, 'hello': 2, 'world': 3, 'test': 4, 'sentence': 5}
    """
    # 토크나이저 생성
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    # 리스트 형태의 문장들을 문자열로 변환 (필요한 경우)
    text_data = []
    for sent in preprocessed_sentences:
        if isinstance(sent, list):
            text_data.append(" ".join(sent))
        else:
            text_data.append(sent)

    # 단어 집합 구축
    tokenizer.fit_on_texts(text_data)

    # 문장들을 정수 시퀀스로 변환
    sequences = tokenizer.texts_to_sequences(text_data)

    return tokenizer, sequences


def create_padded_sequences(
    sequences, maxlen=None, padding="post", truncating="post", value=0
):
    """
    정수 시퀀스들을 패딩하여 동일한 길이로 만드는 함수

    Parameters:
    -----------
    sequences : list
        정수 시퀀스들의 리스트
    maxlen : int, optional, default=None
        패딩 후 최대 시퀀스 길이. None이면 가장 긴 시퀀스 길이 사용
    padding : str, default='post'
        패딩 위치 ('pre': 앞쪽, 'post': 뒤쪽)
    truncating : str, default='post'
        잘라내기 위치 ('pre': 앞쪽, 'post': 뒤쪽)
    value : int, default=0
        패딩에 사용할 값

    Returns:
    --------
    padded_sequences : numpy.ndarray
        패딩된 시퀀스 배열

    Example:
    --------
    >>> sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> padded = create_padded_sequences(sequences, maxlen=5)
    >>> print(padded)
    [[1 2 3 0 0]
     [4 5 0 0 0]
     [6 7 8 9 0]]
    """
    padded_sequences = pad_sequences(
        sequences, maxlen=maxlen, padding=padding, truncating=truncating, value=value
    )

    return padded_sequences


def tokenize_and_pad(
    preprocessed_sentences,
    num_words=None,
    oov_token="<OOV>",
    maxlen=None,
    padding="post",
    truncating="post",
    pad_value=0,
):
    """
    토크나이저 생성, 시퀀스 변환, 패딩을 한 번에 수행하는 통합 함수

    Parameters:
    -----------
    preprocessed_sentences : list
        전처리된 문장들의 리스트 (각 문장은 단어 리스트 또는 문자열)
    num_words : int, optional, default=None
        단어 집합의 최대 크기
    oov_token : str, default='<OOV>'
        OOV 토큰 설정
    maxlen : int, optional, default=None
        패딩 후 최대 시퀀스 길이
    padding : str, default='post'
        패딩 위치
    truncating : str, default='post'
        잘라내기 위치
    pad_value : int, default=0
        패딩에 사용할 값

    Returns:
    --------
    tokenizer : Tokenizer
        학습된 Tokenizer 객체
    padded_sequences : numpy.ndarray
        패딩된 정수 시퀀스 배열

    Example:
    --------
    >>> sentences = [['hello', 'world'], ['test']]
    >>> tokenizer, padded = tokenize_and_pad(sentences, maxlen=3)
    >>> print(padded)
    [[2 3 0]
     [4 0 0]]
    """
    # 토크나이저 생성 및 시퀀스 변환
    tokenizer, sequences = create_tokenizer_and_sequences(
        preprocessed_sentences, num_words, oov_token
    )

    # 패딩 적용
    padded_sequences = create_padded_sequences(
        sequences, maxlen, padding, truncating, pad_value
    )

    return tokenizer, padded_sequences
