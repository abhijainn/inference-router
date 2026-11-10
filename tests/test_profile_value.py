from copy import deepcopy
import unittest
from examples.profile_signal_scenarios import controlled_scenarios
from scripts.evaluate_profile_value import VARIANTS,category,real_cases,run_variant,routing_difference
from scripts.evaluate_combined_policy import JOBS


class ProfileValueTests(unittest.TestCase):
    def test_useful_nomination_changes_equal_depth_placement(self):
        spec=controlled_scenarios()['nomination_breaks_equal_depth_tie']
        context=run_variant(spec,'context_only');combined=run_variant(spec,'combined')
        self.assertEqual(context['tasks']['a-small']['backend'],'hosted')
        self.assertEqual(combined['tasks']['z-analysis']['backend'],'hosted')
        self.assertEqual(combined['metrics']['foreground_finish_seconds'],15.5)
        self.assertEqual(context['metrics']['foreground_finish_seconds'],40.5)
        self.assertEqual(category(context['metrics'],combined['metrics']),'better')

    def test_length_mechanism_needs_spare_capacity(self):
        spec=controlled_scenarios()['length_changes_wait_or_hosted_choice']
        nomination=run_variant(spec,'nomination_only');combined=run_variant(spec,'combined')
        self.assertEqual(nomination['tasks']['new-long']['start_seconds'],5)
        self.assertEqual(combined['tasks']['new-long']['start_seconds'],1)
        self.assertEqual(combined['metrics']['foreground_finish_seconds'],7)
        self.assertEqual(combined['metrics']['hosted_calls'],nomination['metrics']['hosted_calls'])
        one_slot=run_variant(spec,'combined',1)
        one_slot_no_length=run_variant(spec,'nomination_only',1)
        self.assertEqual(one_slot['metrics'],one_slot_no_length['metrics'])

    def test_wrong_signal_can_reverse_benefit(self):
        spec=controlled_scenarios()['wrong_nomination_hurts']
        context=run_variant(spec,'context_only');combined=run_variant(spec,'combined')
        self.assertEqual(category(context['metrics'],combined['metrics']),'worse')
        self.assertTrue(routing_difference(context,combined))

    def test_ablation_does_not_mutate_shared_inputs(self):
        spec=controlled_scenarios()['nomination_breaks_equal_depth_tie'];original=deepcopy(spec)
        for variant in VARIANTS:run_variant(spec,variant)
        self.assertEqual(spec,original)

    def test_two_metric_categories_and_tolerance(self):
        base={'foreground_finish_seconds':10,'hosted_calls':1}
        for finish,calls,expected in [(9,1,'better'),(10,0,'better'),(11,1,'worse'),(9,2,'tradeoff'),(11,0,'tradeoff'),(10+1e-10,1,'equal')]:
            self.assertEqual(category(base,{'foreground_finish_seconds':finish,'hosted_calls':calls}),expected)

    def test_complete_sweep_and_both_pair_orders(self):
        profiles={job['id']:{'preference':'uncertain','length_class':'medium'} for job in JOBS}
        paired={key:{'local':1,'hosted':.5} for key in profiles}
        cases=list(real_cases(profiles,paired))
        self.assertEqual(len(cases),4056)
        self.assertEqual(len({case['id'] for case in cases}),4056)
        self.assertEqual(sum(case['family']=='pair_matrix' for case in cases),2184)
        self.assertEqual(sum(case['family']=='occupied_lanes' for case in cases),1872)
        ordered={(case['first'],case['second']) for case in cases if case['family']=='pair_matrix'}
        self.assertTrue(all((b,a) in ordered for a,b in ordered))
