import unittest

import torch

from pytorchocr.diagnostics import (
    ErrorAccumulator,
    compare_output_sets,
    compute_recognition_pair,
    ctc_collapse,
    fake_quant_state,
    new_recognition_pair,
    observer_qparams,
    operator_delta,
    outputs_as_tuple,
    random_stage_comparison,
    state_preservation,
    tensor_record,
    update_recognition_pair,
)


class StageComparisonTest(unittest.TestCase):
    def test_error_accumulator_aggregates_batches(self):
        accumulator = ErrorAccumulator()
        accumulator.update(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 2.0]))
        accumulator.update(torch.tensor([4.0]), torch.tensor([2.0]))

        self.assertEqual(accumulator.compute(), {"mae": 1.0, "max_abs": 2.0})

    def test_output_tuple_normalization(self):
        tensor = torch.ones(1)
        self.assertEqual(outputs_as_tuple(tensor), (tensor,))
        self.assertEqual(outputs_as_tuple([tensor, tensor]), (tensor, tensor))
        self.assertEqual(
            outputs_as_tuple({"ctc": tensor, "ctc_neck": tensor})[0],
            tensor,
        )
        maps = (tensor, tensor, tensor)
        self.assertEqual(outputs_as_tuple({"maps": maps})[0], tensor)
        base_dict = {"neck_out": tensor, "head_out": {"ctc": tensor}}
        self.assertEqual(outputs_as_tuple(base_dict)[0], tensor)

    def test_output_set_comparison_reports_each_random_stage_output(self):
        reference = (
            torch.tensor([[[0.0, 2.0, 1.0]]]),
            torch.tensor([[1.0, 3.0]]),
        )
        actual = (
            reference[0] + 0.25,
            reference[1].clone(),
        )

        records = compare_output_sets(reference, actual, ("ctc", "gtc"))

        self.assertEqual([record["name"] for record in records], ["ctc", "gtc"])
        self.assertEqual(records[0]["mae"], 0.25)
        self.assertEqual(records[0]["max_abs"], 0.25)
        self.assertEqual(records[0]["argmax_agreement"], 1.0)
        self.assertEqual(records[1]["mae"], 0.0)
        self.assertTrue(all(record["finite"] for record in records))

    def test_random_stage_comparison_includes_quantonnx_boundaries(self):
        fake_off = (torch.tensor([[1.0, 2.0]]),)
        fake_on = (torch.tensor([[1.0, 1.0]]),)
        converted = (torch.tensor([[1.0, 1.0]]),)

        result = random_stage_comparison(
            {
                "prepared_fake_off": fake_off,
                "prepared_fake_on": fake_on,
            },
            converted,
            ("logits",),
            [converted[0].numpy()],
        )

        comparisons = result["comparisons"]
        self.assertIn("prepared_fake_on_to_converted_pt2e", comparisons)
        self.assertIn("converted_pt2e_to_quantonnx", comparisons)
        self.assertEqual(
            comparisons["converted_pt2e_to_quantonnx"][0]["max_abs"],
            0.0,
        )

    def test_ctc_collapse_removes_blanks_and_adjacent_duplicates(self):
        self.assertEqual(ctc_collapse([1, 1, 0, 1, 2, 2, 0]), (1, 1, 2))

    def test_float_preservation_pair_stats_include_ctc_behavior(self):
        stats = new_recognition_pair()
        reference = torch.tensor([[[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]]])
        actual = reference.clone()

        update_recognition_pair(stats, reference, actual)

        result = compute_recognition_pair(stats)
        self.assertEqual(result["logits"], {"mae": 0.0, "max_abs": 0.0})
        self.assertEqual(result["argmax_agreement"], 1.0)
        self.assertEqual(result["ctc_sequence_agreement"], 1.0)

    def test_fake_quant_state_reports_switches(self):
        fake_quant = torch.ao.quantization.FakeQuantize()
        model = torch.nn.Sequential(fake_quant)

        enabled = fake_quant_state(model)
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.ao.quantization.disable_fake_quant)
        disabled = fake_quant_state(model)

        self.assertEqual(enabled["modules"], 1)
        self.assertEqual(enabled["observer_enabled"], 1)
        self.assertEqual(enabled["fake_quant_enabled"], 1)
        self.assertEqual(disabled["observer_enabled"], 0)
        self.assertEqual(disabled["fake_quant_enabled"], 0)

    def test_state_and_operator_deltas_are_reported(self):
        class Exported(torch.nn.Module):
            def forward(self, inputs):
                return torch.relu(inputs)

        class Prepared(torch.nn.Module):
            def forward(self, inputs):
                return torch.sigmoid(torch.relu(inputs))

        exported = torch.fx.symbolic_trace(Exported())
        prepared = torch.fx.symbolic_trace(Prepared())

        state = state_preservation(exported, prepared)
        operators = operator_delta(exported, prepared)

        self.assertEqual(state["changed_common_keys"], [])
        self.assertEqual(len(operators), 1)
        self.assertEqual(list(operators.values()), [1])

    def test_observer_qparams_include_ranges(self):
        fake_quant = torch.ao.quantization.FakeQuantize()
        fake_quant(torch.tensor([-2.0, 1.0]))

        qparams = observer_qparams(torch.nn.Sequential(fake_quant))

        self.assertEqual(qparams["modules"], 1)
        self.assertEqual(qparams["nonfinite"], [])
        self.assertEqual(qparams["records"][0]["min_val"]["values"], [-2.0])
        self.assertEqual(qparams["records"][0]["max_val"]["values"], [1.0])

    def test_tensor_record_preserves_shape_and_values(self):
        record = tensor_record(torch.tensor([[1, 2]], dtype=torch.int32))

        self.assertEqual(record["dtype"], "torch.int32")
        self.assertEqual(record["shape"], [1, 2])
        self.assertEqual(record["values"], [1, 2])


if __name__ == "__main__":
    unittest.main()
