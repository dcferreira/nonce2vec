import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import typer
from gensim.models import Word2Vec

from nonce2vec.models.informativeness import FilterType, Informativeness, SortBy
from nonce2vec.models.nonce2vec import LearningRateFunction, Nonce2Vec
from nonce2vec.utils.files import get_model_path

app = typer.Typer()


def read_sentences(path: Path) -> List[List[str]]:
    with path.open("r") as fd:
        sentences = [[w.strip() for w in line.split(" ")] for line in fd]
    return sentences


def train_w2v_models(
    train_sentences: Sequence[Sequence[str]], **kwargs
) -> Tuple[Word2Vec, Word2Vec]:
    sg_model = Word2Vec(sentences=train_sentences, sg=1, **kwargs)
    cbow_model = Word2Vec(sentences=train_sentences, sg=0, **kwargs)

    return sg_model, cbow_model


@app.command()
def train_w2v(
    train_data: Path,
    output_path: Path,
    vector_size: int = 100,
    alpha: float = 0.025,
    window: int = 5,
    min_count: int = 5,
    max_vocab_size: Optional[int] = None,
    sample: float = 0.001,
    seed: int = 1,
    workers: int = 3,
    min_alpha: float = 0.0001,
    hs: int = 0,
    negative: int = 5,
    ns_exponent: float = 0.75,
    cbow_mean: int = 1,
    epochs: int = 5,
    null_word: int = 0,
    sorted_vocab: int = 1,
    batch_words: int = 10000,
    compute_loss: bool = False,
    max_final_vocab: Optional[int] = None,
    shrink_windows: bool = True,
):
    train_sentences = read_sentences(train_data)
    sg_model, cbow_model = train_w2v_models(
        train_sentences,
        vector_size=vector_size,
        alpha=alpha,
        window=window,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
        sample=sample,
        seed=seed,
        workers=workers,
        min_alpha=min_alpha,
        hs=hs,
        negative=negative,
        ns_exponent=ns_exponent,
        cbow_mean=cbow_mean,
        epochs=epochs,
        null_word=null_word,
        sorted_vocab=sorted_vocab,
        batch_words=batch_words,
        compute_loss=compute_loss,
        max_final_vocab=max_final_vocab,
        shrink_windows=shrink_windows,
    )

    if not output_path.exists():
        os.makedirs(output_path)

    sg_model_path = Path(
        get_model_path(
            os.fspath(train_data),
            os.fspath(output_path),
            train_mode="skipgram",
            alpha=alpha,
            neg=negative,
            window_size=window,
            sample=sample,
            epochs=epochs,
            size=vector_size,
            min_count=min_count,
        )
    )
    with sg_model_path.open("wb+") as fd:
        pickle.dump(sg_model, fd)

    cbow_model_path = Path(
        get_model_path(
            os.fspath(train_data),
            os.fspath(output_path),
            train_mode="cbow",
            alpha=alpha,
            neg=negative,
            window_size=window,
            sample=sample,
            epochs=epochs,
            size=vector_size,
            min_count=min_count,
        )
    )
    with cbow_model_path.open("wb+") as fd:
        pickle.dump(cbow_model, fd)


@app.command()
def train_n2v(
    model: Path,
    info_model: Path,
    test_data: Path,
    reload: bool = False,
    train_with: LearningRateFunction = LearningRateFunction.CWI,
    lambda_decay: float = 70,
    kappa: int = 1,
    beta: int = 1000,
    sample_decay: float = 1.9,
    window_decay: int = 15,
    sum_only: bool = False,
    reduced: bool = False,
    sum_over_set: bool = False,
    weighted: bool = False,
    train_over_set: bool = False,
    with_stats: bool = False,
    shuffle: bool = False,
    sum_filter: Optional[FilterType] = None,
    sum_thresh: Optional[int] = None,
    train_filter: Optional[FilterType] = None,
    train_thresh: Optional[int] = None,
    sort_by: Optional[SortBy] = None,
):
    info_model_obj = Informativeness(
        os.fspath(info_model),
        sum_filter=sum_filter,
        sum_thresh=sum_thresh,
        train_filter=train_filter,
        train_thresh=train_thresh,
        sort_by=sort_by,
    )
    n2v_model = Nonce2Vec(
        model=model,
        info_model=info_model_obj,
        reload=reload,
        train_with=train_with,
        lambda_decay=lambda_decay,
        kappa=kappa,
        beta=beta,
        sample_decay=sample_decay,
        window_decay=window_decay,
        sum_only=sum_only,
        reduced=reduced,
        sum_over_set=sum_over_set,
        weighted=weighted,
        train_over_set=train_over_set,
        with_stats=with_stats,
        shuffle=shuffle,
    )
    test_sentences = read_sentences(test_data)
    n2v_model.add_nonces(test_sentences)
    if n2v_model.new_nonces is None:
        raise RuntimeError(
            "No new nonces were added to the Nonce2Vec model. "
            "Are you sure new nonces are included in the data?"
        )
    for nonce in n2v_model.new_nonces:
        print(nonce, n2v_model.model.wv.get_vector(nonce))


if __name__ == "__main__":
    app()
