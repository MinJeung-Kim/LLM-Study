"""
NLP 텍스트 전처리 유틸리티 모듈

이 모듈은 자연어 처리를 위한 다양한 전처리 함수들을 제공합니다.
정수 인코딩, 토큰화, 정규화 등의 기능을 포함합니다.

Author: NLP Study Team
Created: 2025-10-17
"""

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


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
        words = word_tokenize(sent)  # 단어 토큰화

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


def create_word_to_index_mapping(vocab, max_vocab_size=None):
    """
    단어 빈도수 딕셔너리를 기반으로 단어-인덱스 매핑을 생성

    Parameters:
    -----------
    vocab : dict
        단어별 빈도수 딕셔너리
    max_vocab_size : int, optional
        최대 어휘 사전 크기. None이면 모든 단어 사용

    Returns:
    --------
    word_to_index : dict
        단어를 인덱스로 매핑하는 딕셔너리
    index_to_word : dict
        인덱스를 단어로 매핑하는 딕셔너리
    """
    # 빈도수 기준으로 정렬 (높은 순서)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    # 최대 어휘 크기 제한
    if max_vocab_size:
        sorted_vocab = sorted_vocab[:max_vocab_size]

    # 매핑 딕셔너리 생성 (인덱스 1부터 시작, 0은 보통 패딩용으로 예약)
    word_to_index = {"<PAD>": 0, "<UNK>": 1}  # 특수 토큰
    index_to_word = {0: "<PAD>", 1: "<UNK>"}

    for idx, (word, _) in enumerate(sorted_vocab, start=2):
        word_to_index[word] = idx
        index_to_word[idx] = word

    return word_to_index, index_to_word


def encode_sentences(sentences, word_to_index):
    """
    전처리된 문장들을 정수로 인코딩

    Parameters:
    -----------
    sentences : list
        전처리된 문장들 (단어 리스트의 리스트)
    word_to_index : dict
        단어-인덱스 매핑 딕셔너리

    Returns:
    --------
    encoded_sentences : list
        정수로 인코딩된 문장들
    """
    encoded_sentences = []

    for sentence in sentences:
        encoded_sentence = []
        for word in sentence:
            # 단어가 사전에 있으면 해당 인덱스, 없으면 <UNK> 인덱스
            index = word_to_index.get(word, word_to_index["<UNK>"])
            encoded_sentence.append(index)
        encoded_sentences.append(encoded_sentence)

    return encoded_sentences
