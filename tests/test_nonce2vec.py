from copy import deepcopy

import numpy as np

from nonce2vec.models.informativeness import Informativeness
from nonce2vec.models.nonce2vec import Nonce2Vec


def test_add_nonce_reduced(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=True, epochs=1)

    nonce = "giraffes"
    n2v_model.add_nonces([test_sentences[0]])
    assert nonce in n2v_model.model.wv.key_to_index


def test_add_nonce_reduced_correct(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=True, epochs=100)

    nonce = "giraffes"
    n2v_model.add_nonces(test_sentences)
    assert (
        n2v_model.model.wv.similarity("giraffes", "cats") > 0.55
    ), n2v_model.model.wv.similar_by_key(nonce, topn=20)


def test_add_nonce(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=False)

    nonce = "giraffes"
    n2v_model.add_nonces(test_sentences)
    assert nonce in n2v_model.model.wv.key_to_index


def test_add_nonces(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=False)

    n2v_model.add_nonces(test_sentences)
    for nonce in ["giraffes", "elephants"]:
        assert nonce in n2v_model.new_nonces
        assert nonce in n2v_model.model.wv.key_to_index


def test_add_nonces_correctness(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=False)

    n2v_model.add_nonces(test_sentences)
    assert n2v_model.model.wv.similarity("giraffes", "elephants") > 0.99
    assert n2v_model.model.wv.similarity("giraffes", "yet") < 0.5


def test_determinism(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model1 = Nonce2Vec(deepcopy(sg_model), info_model_obj, reduced=True)
    n2v_model1.add_nonces(test_sentences)
    n2v_model2 = Nonce2Vec(deepcopy(sg_model), info_model_obj, reduced=True)
    n2v_model2.add_nonces(test_sentences)

    np.testing.assert_array_equal(
        n2v_model1.model.wv.get_vector("cats"), n2v_model2.model.wv.get_vector("cats")
    )
    np.testing.assert_array_equal(
        n2v_model1.model.wv.get_vector("giraffes"),
        n2v_model2.model.wv.get_vector("giraffes"),
    )


def test_add_nonces_multiple_occurences(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model1 = Nonce2Vec(deepcopy(sg_model), info_model_obj)
    n2v_model2 = Nonce2Vec(deepcopy(sg_model), info_model_obj)

    n2v_model1.add_nonces([test_sentences[0]])
    n2v_model2.add_nonces(test_sentences)

    # these vectors should be the same
    np.testing.assert_array_equal(
        n2v_model1.model.wv.get_vector("cats"), n2v_model2.model.wv.get_vector("cats")
    )
    with np.testing.assert_raises(AssertionError):
        # this should raise an error, the vectors should be different
        np.testing.assert_array_equal(
            n2v_model1.model.wv.get_vector("giraffes"),
            n2v_model2.model.wv.get_vector("giraffes"),
        )
