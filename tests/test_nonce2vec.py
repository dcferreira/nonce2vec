from nonce2vec.models.informativeness import Informativeness
from nonce2vec.models.nonce2vec import Nonce2Vec


def test_add_nonce_reduced(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(
        sg_model, info_model_obj, reduced=True, shuffle=False, epochs=1
    )

    nonce = "giraffes"
    n2v_model.add_nonces([test_sentences[0]])
    assert nonce in n2v_model.model.wv.key_to_index


def test_add_nonce_reduced_correct(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(
        sg_model, info_model_obj, reduced=True, shuffle=False, epochs=100
    )

    nonce = "giraffes"
    n2v_model.add_nonces(test_sentences)
    assert (
        n2v_model.model.wv.similarity("giraffes", "cats") > 0.55
    ), n2v_model.model.wv.similar_by_key(nonce, topn=20)


def test_add_nonce(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=False, shuffle=False)

    nonce = "giraffes"
    n2v_model.add_nonces(test_sentences)
    assert nonce in n2v_model.model.wv.key_to_index


def test_add_nonces(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=False, shuffle=False)

    n2v_model.add_nonces(test_sentences)
    for nonce in ["giraffes", "elephants"]:
        assert nonce in n2v_model.new_nonces
        assert nonce in n2v_model.model.wv.key_to_index


def test_add_nonces_correctness(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=False, shuffle=False)

    n2v_model.add_nonces(test_sentences)
    assert n2v_model.model.wv.similarity("giraffes", "elephants") > 0.99
    assert n2v_model.model.wv.similarity("giraffes", "yet") < 0.5
