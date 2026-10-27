"""Instruction and task-id text for the SmolVLM tokenizer."""
# pyrefly: ignore-errors
# (torch/lerobot/transformers are optional and loosely typed; checked by the tests)

import torch

from letorifighter.tasks import Definitions

DEFAULT_VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def instruction_task_ids(definitions: Definitions | None = None) -> dict[str, int]:
    """Map every atomic instruction of the definition file (train and held-out
    paraphrases) to its task id. Compositions are not in the map."""
    definitions = definitions or Definitions.load()
    mapping: dict[str, int] = {}
    for task in definitions.tasks.values():
        for paraphrases in task.paraphrases.values():
            for text in paraphrases:
                mapping[text] = task.id
    return mapping


def task_id_text(task_id: int) -> str:
    """The text that stands in for the instruction under task-id conditioning."""
    return f"task {int(task_id)}"


class TextEncoder:
    """Tokenizes a string like LeRobot's SmolVLA processor (trailing newline,
    right padding to a fixed length) so batches stack."""

    def __init__(self, model_name: str = DEFAULT_VLM, max_length: int = 48) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(tokens (L,) long, attention_mask (L,) bool)`."""
        if text not in self._cache:
            out = self.tokenizer(
                text.rstrip("\n") + "\n",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            self._cache[text] = (
                out["input_ids"][0].long(),
                out["attention_mask"][0].bool(),
            )
        return self._cache[text]
