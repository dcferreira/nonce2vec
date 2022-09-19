import copy
import os
import pickle
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Iterator

import numpy as np
import typer
from gensim.models import Word2Vec
from loguru import logger

from nonce2vec.main import (
    _get_rank,
    _compute_average_sim,
    _update_rr_and_count,
    _display_density_stats,
)
from nonce2vec.models.informativeness import FilterType, Informativeness, SortBy
from nonce2vec.models.nonce2vec import LearningRateFunction, Nonce2Vec
from nonce2vec.utils.files import get_model_path, Samples

app = typer.Typer()


class IncludedDataset(str, Enum):
    definitions = "definitions"
    l2 = "l2"
    l4 = "l4"
    l6 = "l6"


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


def _replace_words_in_list(sentence: Sequence[str], replace: str, replace_by: str) -> List[str]:
    return list(map(lambda x: x if x != replace else replace_by, sentence))


def _test_on_definitions(
    n2v: Nonce2Vec,
    shuffle: bool = False,
    reload: bool = False,
    with_stats: bool = False,
):
    """Test the definitional nonces."""
    ranks = []
    sum_10 = []
    sum_25 = []
    sum_50 = []
    relative_ranks = 0.0
    count = 0
    samples = Samples(source="def", shuffle=shuffle)
    total_num_sent = sum(1 for line in samples)
    logger.info(
        "Testing Nonce2Vec on the nonces dataset containing "
        "{} sentences".format(total_num_sent)
    )
    num_sent = 1
    n2vmodel = copy.deepcopy(n2v) if reload else n2v
    for sentences, nonce, probe in samples:
        logger.info("-" * 30)
        logger.info("Processing sentence {}/{}".format(num_sent, total_num_sent))
        if reload:
            del n2vmodel
            # reset model to before seeing the nonces
            n2vmodel = copy.deepcopy(n2v)

        logger.debug("Adding sentence...")
        replace_fn = partial(_replace_words_in_list, replace=nonce, replace_by=f"{nonce}_true")
        n2vmodel.add_nonces(list(map(replace_fn, sentences)))
        logger.debug("Finished sentence!")
        vocab_size = len(n2vmodel.model.wv)
        logger.info("vocab size = {}".format(vocab_size))
        logger.info("nonce: {}".format(nonce))
        logger.info("sentence: {}".format(sentences))
        if nonce not in n2vmodel.model.wv:
            logger.error(
                "Nonce '{}' not in gensim.word2vec.model vocabulary".format(nonce)
            )
            continue

        nns = n2vmodel.model.wv.most_similar(nonce, topn=vocab_size)
        logger.info("10 most similar words: {}".format(nns[:10]))
        rank = _get_rank(probe, nns)
        ranks.append(rank)
        if with_stats:
            gold_nns = n2vmodel.model.wv.most_similar(
                f"{nonce}_true", topn=vocab_size
            )
            sum_10.append(_compute_average_sim(gold_nns[:10]))
            sum_25.append(_compute_average_sim(gold_nns[:25]))
            sum_50.append(_compute_average_sim(gold_nns[:50]))
        relative_ranks, count = _update_rr_and_count(relative_ranks, count, rank)
        num_sent += 1
        median = np.median(ranks)
    logger.info("Final MRR =  {}".format(relative_ranks / count))
    logger.info("Median Rank = {}".format(median))
    if with_stats:
        _display_density_stats(ranks, sum_10, sum_25, sum_50)


@app.command()
def train_n2v(
    model: Path,
    info_model: Path,
    dataset: Optional[IncludedDataset] = None,
    dataset_path: Optional[Path] = None,
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
    sum_threshold: Optional[int] = None,
    train_filter: Optional[FilterType] = None,
    train_threshold: Optional[int] = None,
    sort_by: Optional[SortBy] = None,
):
    info_model_obj = Informativeness(
        os.fspath(info_model),
        sum_filter=sum_filter,
        sum_threshold=sum_threshold,
        train_filter=train_filter,
        train_threshold=train_threshold,
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
    )
    if dataset_path is not None:
        test_sentences = read_sentences(dataset_path)
        n2v_model.add_nonces(test_sentences)
        if n2v_model.new_nonces is None:
            raise RuntimeError(
                "No new nonces were added to the Nonce2Vec model. "
                "Are you sure new nonces are included in the data?"
            )
        for nonce in n2v_model.new_nonces:
            print(nonce, n2v_model.model.wv.get_vector(nonce))
    elif dataset == IncludedDataset.definitions:
        _test_on_definitions(
            n2v_model, shuffle=shuffle, reload=reload, with_stats=with_stats
        )


if __name__ == "__main__":
    app()
