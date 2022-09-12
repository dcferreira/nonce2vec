from pathlib import Path
from typing import List, Tuple

import pytest
from gensim.models import Word2Vec

from nonce2vec.main_new import read_sentences, train_w2v_models


@pytest.fixture
def assets_path() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture
def test_sentences(assets_path) -> List[List[str]]:
    return read_sentences(assets_path / "test_sentences.csv")


@pytest.fixture
def train_sentences(assets_path) -> List[List[str]]:
    return read_sentences(assets_path / "train_sentences.csv")


@pytest.fixture
def w2v_models(train_sentences) -> Tuple[Word2Vec, Word2Vec]:
    return train_w2v_models(
        train_sentences,
        min_count=1,
    )
