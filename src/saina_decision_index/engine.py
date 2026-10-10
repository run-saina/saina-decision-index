"""Decision Index engine for Saina models pulled from the Hugging Face Hub.

Requests go through `saina.jev.evaluate`, the same System One adapter that serves
`/v1/systemone` on api.saina.run, so the benchmark measures what users get. One fixed
rendering for every benchmark; no labels, benchmark names or metadata reach the model.
"""
import hashlib
import re
from importlib.metadata import version
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported
from pydantic import ValidationError
from saina.errors import InputTooLong
from saina.jev import SystemOneRequest, evaluate
from saina.loader import load_scorer

DEFAULT_MODEL = "run-saina/saina-helm-0.8b"


class SainaEngine(Engine):
    name = "saina"

    def __init__(self, model=DEFAULT_MODEL, revision=None, device="cuda", max_length=8192, repo_commit=None, **options):
        super().__init__(**options)
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pass --option revision=<40-char Hub commit>: runs must name an immutable checkpoint")
        self.model_id, self.revision = model, revision
        self.scorer = load_scorer(model, revision=revision, device=device, max_length=int(max_length))
        self.provenance = {
            "model": model,
            "revision": revision,
            "saina_version": version("saina"),
            "adapter": "saina.jev.evaluate (System One adapter served at /v1/systemone)",
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "repo_commit": repo_commit,
            "max_length": self.scorer.max_length,
            "max_options": len(self.scorer.codes),
            "gold_access": False,
            "no_truncation": True,
        }

    def __call__(self, state, questions):
        for q in questions.values():
            if q.get("type") not in ("choice", "noul"):
                raise Unsupported("Unsupported question type " + str(q.get("type")))
            if q["type"] == "choice" and len(q.get("criteria") or {}) > len(self.scorer.codes):
                raise Unsupported(f"More than {len(self.scorer.codes)} options")
        try:
            request = SystemOneRequest(model=self.model_id, state=state, questions=questions)
        except ValidationError as e:
            # The /v1/systemone contract's declared limits (option and question counts).
            raise Unsupported("Outside the System One request contract: " + str(e.errors()[0]["msg"])) from e
        try:
            response = evaluate(request, self.scorer)
        except InputTooLong as e:
            raise Unsupported(str(e)) from e
        response["model"] = f"{self.model_id}@{self.revision}"
        return response, response

    def warmup(self):
        super().warmup()
        self.synchronize()

    def synchronize(self):
        torch = self.scorer.torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def runtime(self):
        torch = self.scorer.torch
        params = {id(p): p for m in (self.scorer.model, self.scorer.head) for p in m.parameters()}
        return {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "loaded_parameters": sum(p.numel() for p in params.values()),
        }
