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
CONV_KERNEL_REPO = "kernels-community/mamba-ssm"


def use_conv_kernel(revision):
    """Swap the reference causal convolution for the prebuilt CUDA kernel from the Hub (pinned).

    Same kernel and revision pinning as Helm training; needs no CUDA compiler. Outputs match the
    reference path within bfloat16 rounding.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("conv_kernel must be a 40-char commit of " + CONV_KERNEL_REPO)
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from kernels import get_kernel

    kernel = get_kernel(CONV_KERNEL_REPO, revision=revision)._causal_conv1d

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        return kernel.causal_conv1d_fn(hidden_states, weight, bias, activation=activation)

    modeling.causal_conv1d_fn = causal_conv1d_fn
    return f"{CONV_KERNEL_REPO}@{revision}"


class SainaEngine(Engine):
    name = "saina"

    def __init__(self, model=DEFAULT_MODEL, revision=None, device="cuda", max_length=None, repo_commit=None,
                 conv_kernel=None, **options):
        super().__init__(**options)
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pass --option revision=<40-char Hub commit>: runs must name an immutable checkpoint")
        self.model_id, self.revision = model, revision
        conv = use_conv_kernel(str(conv_kernel)) if conv_kernel else None
        # Unset means the checkpoint's own context limit.
        limit = {} if max_length is None else {"max_length": int(max_length)}
        self.scorer = load_scorer(model, revision=revision, device=device, **limit)
        self.provenance = {
            "model": model,
            "revision": revision,
            "saina_version": version("saina"),
            "adapter": "saina.jev.evaluate (System One adapter served at /v1/systemone)",
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "repo_commit": repo_commit,
            "conv_kernel": conv,  # None: transformers' reference causal convolution
            "max_length": self.scorer.max_length,
            "max_options": len(self.scorer.codes),
            "shared_trunk_pass": getattr(self.scorer, "shared_trunk_pass", False),
            "gold_access": False,
            "no_truncation": True,
        }

    def __call__(self, state, questions):
        return self._evaluate(state, questions)

    def _evaluate(self, state, questions, **inputs):
        for q in questions.values():
            if q.get("type") not in ("choice", "noul"):
                raise Unsupported("Unsupported question type " + str(q.get("type")))
            if q["type"] == "choice" and len(q.get("criteria") or {}) > len(self.scorer.codes):
                raise Unsupported(f"More than {len(self.scorer.codes)} options")
        try:
            request = SystemOneRequest(model=self.model_id, state=state, questions=questions, **inputs)
        except ValidationError as e:
            # The /v1/systemone contract's declared limits (option and question counts).
            raise Unsupported("Outside the System One request contract: " + str(e.errors()[0]["msg"])) from e
        self._release_cached_memory()
        try:
            response = evaluate(request, self.scorer)
        except InputTooLong as e:
            raise Unsupported(str(e)) from e
        response["model"] = f"{self.model_id}@{self.revision}"
        return response, response

    def _release_cached_memory(self, slack=4 << 30):
        """Return cached-but-unused GPU blocks after very long requests (over 4 GiB idle), so a later long
        request does not hit allocator fragmentation; normal requests leave the cache alone. Scores are unchanged."""
        torch = self.scorer.torch
        if torch.cuda.is_available() and torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > slack:
            torch.cuda.empty_cache()

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


class SainaVisionEngine(SainaEngine):
    """System One option probabilities with the official Vision image budget."""

    name = "saina-vision"

    def __init__(self, model="run-saina/saina-helm-2-omni-0.8b", max_pixels=1638400,
                 image_root=None, **options):
        from dataclasses import asdict, replace
        from decision_index.vision.images import MAX_PIXELS
        from decision_index.vision.engines import resize_policy

        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or not 65536 <= max_pixels <= MAX_PIXELS:
            raise ValueError("max_pixels must be an integer between 65536 and 1638400")
        super().__init__(model=model, **options)
        if not hasattr(self.scorer, "image_policy"):
            raise ValueError("SainaVisionEngine requires a multimodal checkpoint")
        self.max_pixels, self.image_root = max_pixels, image_root
        # Saina passes this policy explicitly to its processor on every request.
        self.scorer.image_policy = replace(self.scorer.image_policy, max_pixels=max_pixels)
        self.provenance.update(
            max_pixels=max_pixels,
            image_resize=resize_policy(max_pixels) + "; then Saina's native patch-grid resize",
            processor_image_budget=asdict(self.scorer.image_policy),
            readout="option probabilities; prompt.readout ignored",
        )

    def __call__(self, state, questions, images=(), prompt=None):
        import io
        from PIL import Image
        from decision_index.vision import images as img
        from saina.media import image_data_url

        urls, image_pixels, image_tokens = [], [], 0
        for ref in images or ():
            raw = img.capped(str(ref) if isinstance(ref, Path) else ref,
                             root=self.image_root, max_pixels=self.max_pixels)
            urls.append(image_data_url(raw))
            with Image.open(io.BytesIO(raw)) as im:
                image_pixels.append(im.width * im.height)
                image_tokens += self.scorer.image_policy.tokens(*im.size)
        # The runner's prompt/metadata never enters the System One request.
        # Letter-to-letter criteria for options_in_image pass through unchanged.
        response, raw = self._evaluate(state, questions, images=urls)
        return response, {**raw, "image_pixels": image_pixels, "image_tokens": image_tokens}
