from __future__ import annotations

import types
import unittest

import torch

from scripts.check_dcl_critic_equivalence import (
    FIXTURES,
    classify_case_status,
    extract_logits,
    ordinary_guard_tokenization,
    tensor_difference,
)
from train.dcl_critic import (
    assemble_critic_embeddings,
    probabilities_to_embeddings,
    safe_unsafe_token_ids,
    unsafe_score_from_logits,
)


class CharacterTokenizer:
    """Small deterministic tokenizer with a Guard-like chat template."""

    chat_template = "taxonomy + user + assistant"

    @staticmethod
    def _render(messages):
        return (
            "<BEGIN UNSAFE CONTENT CATEGORIES>\nViolence and Hate\n"
            "<END UNSAFE CONTENT CATEGORIES>\n"
            f"User: {messages[0]['content']}\n"
            f"Agent: {messages[1]['content']}\n"
            "Answer safe or unsafe:"
        )

    @staticmethod
    def _ids(text, add_special_tokens):
        ids = [ord(character) + 10 for character in text]
        return ([1] + ids) if add_special_tokens else ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt=False,
    ):
        self.assert_no_generation_prompt = not add_generation_prompt
        rendered = self._render(messages)
        return self._ids(rendered, False) if tokenize else rendered

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        ids = self._ids(text, add_special_tokens)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = [(0, 0)] if add_special_tokens else []
            offsets.extend((index, index + 1) for index in range(len(text)))
            result["offset_mapping"] = offsets
        return result

    def encode(self, text, add_special_tokens=False):
        label_ids = {"safe": [2], "unsafe": [3]}
        if text in label_ids:
            return label_ids[text]
        return self._ids(text, add_special_tokens)


class DclCriticEquivalenceTests(unittest.TestCase):
    def test_fixture_edge_cases_are_literal(self):
        self.assertEqual(FIXTURES[4][1], "First line.\nSecond line is here.")
        self.assertTrue(FIXTURES[5][1].startswith("  "))
        self.assertIn("Café", FIXTURES[6][1])

    def test_one_hot_embedding_uses_direct_matrix_product(self):
        table = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
        )
        one_hot = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        actual = probabilities_to_embeddings(one_hot, table)
        expected = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_assembly_and_score_match_production_rule(self):
        header = torch.tensor([[[1.0]]])
        middle = torch.tensor([[[2.0], [3.0]]])
        footer = torch.tensor([[[4.0]]])
        assembled = assemble_critic_embeddings(header, middle, footer)
        torch.testing.assert_close(
            assembled, torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        )
        tokenizer = CharacterTokenizer()
        safe_id, unsafe_id = safe_unsafe_token_ids(tokenizer)
        logits = torch.zeros((1, 4))
        logits[:, safe_id] = -1.0
        logits[:, unsafe_id] = 1.0
        expected = torch.sigmoid(torch.tensor([2.0]))
        torch.testing.assert_close(
            unsafe_score_from_logits(logits, safe_id, unsafe_id), expected
        )

    def test_ordinary_guard_route_uses_chat_template_and_finds_response(self):
        tokenizer = CharacterTokenizer()
        result = ordinary_guard_tokenization(tokenizer, "hello", "world")
        self.assertEqual(result["source"], "critic_tokenizer.apply_chat_template(user, assistant)")
        self.assertTrue(result["contains_explicit_taxonomy"])
        self.assertTrue(result["response_span_verifiable"])
        decoded_ids = [ord(character) + 10 for character in "world"]
        self.assertEqual(result["response_ids"], decoded_ids)

    def test_extracted_logits_do_not_retain_a_view(self):
        full = torch.arange(24.0).reshape(1, 3, 8)
        output = types.SimpleNamespace(logits=full)
        extracted = extract_logits(output)
        full[:, -1, :].zero_()
        self.assertFalse(torch.equal(extracted, full[0, -1]))
        torch.testing.assert_close(extracted, torch.arange(16.0, 24.0))

    def test_status_rules(self):
        same = tensor_difference(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            atol=1e-3,
            rtol=1e-3,
        )
        structural = {
            "ordinary_vs_production_ids_equal": True,
            "ordinary_response_span_matches_production": True,
            "ordinary_length": 3,
            "production_length": 3,
            "safe_unsafe_labels_are_single_tokens": True,
        }
        status, reasons = classify_case_status(
            structural=structural,
            ab=same,
            ac=same,
            bc_embeddings=same,
            finite={"all": True},
            aa_repeat=same,
        )
        self.assertEqual((status, reasons), ("PASS", []))

        structural["ordinary_vs_production_ids_equal"] = False
        status, reasons = classify_case_status(
            structural=structural,
            ab=same,
            ac=same,
            bc_embeddings=same,
            finite={"all": True},
            aa_repeat=same,
        )
        self.assertEqual(status, "FAIL_INTERFACE")
        self.assertTrue(any("guard-template" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
