import torch

from moe.routed_lora_training import CompletionRouteCollator


class TinyChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and not return_dict
        if add_generation_prompt:
            return [10, 11, 99]
        assert messages[-1]["role"] == "assistant"
        return [10, 11, 99, 20, 21]


def test_completion_route_collator_masks_prompt_and_builds_route_targets():
    collator = CompletionRouteCollator(TinyChatTokenizer(), max_length=16, num_experts=3)
    batch = collator(
        [
            {"id": "a", "system": "s", "user": "u", "target": "x", "route_targets": [0, 2]},
            {"id": "b", "system": "s", "user": "u", "target": "y", "route_targets": [1]},
        ]
    )

    assert batch["input_ids"].shape == (2, 5)
    assert torch.equal(batch["labels"][:, :3], torch.full((2, 3), -100))
    assert torch.equal(batch["labels"][:, 3:], torch.tensor([[20, 21], [20, 21]]))
    assert torch.equal(
        batch["route_target_mask"],
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    assert torch.all(batch["attention_mask"] == 1)
