import base64
import io

import pytest
from PIL import Image
from decision_index.engines.base import Unsupported, validate
from saina.media import ImagePolicy

from saina_decision_index import engine as module
from test_engine import FakeScorer, REVISION


@pytest.fixture
def vision(monkeypatch):
    scorer = FakeScorer()
    scorer.image_policy = ImagePolicy(max_pixels=262144)
    monkeypatch.setattr(module, 'load_scorer', lambda *a, **k: scorer)
    seen = []

    def evaluate(request, scorer):
        seen.append(request)
        answers = {}
        for key, q in request.questions.items():
            keys = list(q.criteria)
            answers[key] = {'type': 'choice', 'choice': keys[0],
                            'probabilities': {k: 1 / len(keys) for k in keys}}
        return {'answers': answers}

    monkeypatch.setattr(module, 'evaluate', evaluate)
    return module.SainaVisionEngine(revision=REVISION, device='cpu'), seen


def png(size):
    buf = io.BytesIO()
    Image.new('RGB', size).save(buf, format='PNG')
    return buf.getvalue()


def test_image_inputs_budget_and_no_metadata(vision, tmp_path):
    engine, seen = vision
    small, large = png((32, 64)), png((2000, 1000))
    path = tmp_path / 'image.png'
    path.write_bytes(small)
    refs = [path, large, base64.b64encode(small).decode(),
            'data:image/png;base64,' + base64.b64encode(small).decode()]
    questions = {'answer': {'type': 'choice', 'instructions': 'Read the image',
                            'criteria': {'A': 'A', 'B': 'B'}}}
    response, raw = engine({}, questions, images=refs,
                           prompt={'options_in_image': True, 'readout': 'DO NOT SEND', 'benchmark': 'SECRET'})
    validate(questions, response)
    request = seen[0]
    assert request.state == {}
    assert request.questions['answer'].criteria == {'A': 'A', 'B': 'B'}
    assert 'DO NOT SEND' not in request.model_dump_json()
    assert 'SECRET' not in request.model_dump_json()
    decoded = [base64.b64decode(u.split(',', 1)[1]) for u in request.images]
    assert decoded[0] == decoded[2] == decoded[3] == small
    sizes = [Image.open(io.BytesIO(b)).size for b in decoded]
    assert raw['image_pixels'] == [w * h for w, h in sizes]
    assert max(raw['image_pixels']) <= 1638400
    assert sizes[1] == (1810, 905)
    assert raw['image_tokens'] == sum(engine.scorer.image_policy.tokens(*s) for s in sizes)
    assert engine.provenance['processor_image_budget']['max_pixels'] == 1638400
    assert 'image_pixels' not in response


def test_vision_option_limit(vision):
    engine, seen = vision
    with pytest.raises(Unsupported, match='More than 4 options'):
        engine({}, {'q': {'type': 'choice', 'criteria': dict.fromkeys('ABCDE', 'x')}})
    assert not seen


@pytest.mark.parametrize('cap', [None, 0, 1638401, True, 1.5])
def test_invalid_cap(cap):
    with pytest.raises(ValueError, match='max_pixels'):
        module.SainaVisionEngine(max_pixels=cap)
