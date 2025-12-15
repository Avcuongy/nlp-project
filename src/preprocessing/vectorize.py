from typing import Iterable, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    analyzer: str = "word",
    min_df: int = 3,
    max_df: float = 0.95,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    max_features: Optional[int] = 30000,
) -> TfidfVectorizer:
    """Create a configured TfidfVectorizer for Vietnamese ABSA.

    Args:
        analyzer (str, optional): _description_. Defaults to "word".
        min_df (int, optional): _description_. Defaults to 3.
        max_df (float, optional): _description_. Defaults to 0.95.
        ngram_range (Tuple[int, int], optional): _description_. Defaults to (3, 5).
        sublinear_tf (bool, optional): _description_. Defaults to True.
        max_features (Optional[int], optional): _description_. Defaults to 30000.

    Returns:
        TfidfVectorizer
    """

    return TfidfVectorizer(
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
    )


def fit_transform_corpus(corpus: Iterable[str]):
    """Convenience helper to build the vectorizer and fit_transform a corpus.

    Args:
        corpus (Iterable[str]): The text corpus.

    Returns:
        X: The transformed feature matrix.
        vectorizer: The fitted TfidfVectorizer.
    """
    vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
