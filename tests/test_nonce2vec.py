from nonce2vec.models.informativeness import Informativeness
from nonce2vec.models.nonce2vec import Nonce2Vec


def test_add_nonce_reduced(w2v_models, test_sentences):
    sg_model, cbow_model = w2v_models
    info_model_obj = Informativeness(cbow_model)
    n2v_model = Nonce2Vec(sg_model, info_model_obj, reduced=True, shuffle=False)

    nonce = "giraffes"
    n2v_model.add_nonce(nonce, test_sentences)
    assert nonce in n2v_model.model.wv.key_to_index
