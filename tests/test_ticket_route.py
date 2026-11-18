import unittest

from ticket_route import DATA, load_examples, route, train


class TicketRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train()

    def test_routing_for_all_queues(self):
        examples = {
            "account_access": "I forgot my password and lost the login code",
            "billing": "Please refund my duplicate card payment",
            "delivery": "The courier has my parcel but the tracking is missing",
            "product_bug": "The app crashes when I click the export button",
        }
        for expected, ticket in examples.items():
            with self.subTest(queue=expected):
                result = route(self.model, ticket)
                self.assertEqual(result.queue, expected)
                self.assertAlmostEqual(sum(result.scores.values()), 1)
                self.assertTrue(result.supporting_terms)

    def test_unknown_ticket_requires_review_even_with_zero_threshold(self):
        result = route(self.model, "quasar intergalactic nebula", threshold=0)
        self.assertEqual(result.queue, "human_review")
        self.assertEqual(result.supporting_terms, [])

    def test_abstention_preserves_the_suggested_queue(self):
        result = route(self.model, "My invoice includes an extra charge", threshold=1)
        self.assertEqual(result.queue, "human_review")
        self.assertEqual(result.suggested_queue, "billing")

    def test_invalid_inputs(self):
        for ticket, threshold in [(" ", 0.5), ("x" * 2001, 0.5), ("invoice", float("nan"))]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    route(self.model, ticket, threshold)

    def test_holdout_examples_are_separate(self):
        train_texts, train_labels = load_examples(DATA / "train.csv")
        test_texts, test_labels = load_examples(DATA / "holdout.csv")
        self.assertTrue(set(train_texts).isdisjoint(test_texts))
        self.assertEqual(set(train_labels), set(test_labels))
        predicted = self.model.predict(test_texts)
        self.assertGreaterEqual(sum(actual == label for actual, label in zip(predicted, test_labels)), 9)


if __name__ == "__main__":
    unittest.main()
