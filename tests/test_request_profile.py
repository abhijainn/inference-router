import copy
import unittest
from inference_router.request_profile import combine_predictions, ProfileRules


def predictions(task='Code Generation', length='medium', reasoning=.4):
    n={'task_type':task,'task_probabilities':{task:.8,'Other':.2},'reasoning':reasoning,'constraint_ct':.6,'truncated':False}
    c={'length_class_probabilities':{key:.8 if key==length else .1 for key in ['short','medium','long']}}
    return n,c


class RequestProfileTests(unittest.TestCase):
    def test_short_local(self):
        self.assertEqual(combine_predictions(*predictions('Classification','short'))['preference'],'prefer_local')

    def test_long_background_and_acceleration_candidates(self):
        self.assertEqual(combine_predictions(*predictions(length='long',reasoning=.1))['preference'],'prefer_local')
        self.assertEqual(combine_predictions(*predictions())['preference'],'consider_hosted')

    def test_disagreement_and_truncation_are_uncertain(self):
        result=combine_predictions(*predictions('Classification','long'))
        self.assertEqual(result['preference'],'uncertain')
        self.assertIn('task_and_length_disagree',result['issues'])
        n,c=predictions();n['truncated']=True
        self.assertEqual(combine_predictions(n,c)['preference'],'uncertain')

    def test_low_confidence_preserves_distribution(self):
        n,c=predictions();c['length_class_probabilities']={'short':.34,'medium':.33,'long':.33}
        result=combine_predictions(n,c)
        self.assertEqual(result['length_probabilities'],c['length_class_probabilities'])
        self.assertIn('uncertain_output_length',result['issues'])

    def test_bad_inputs_and_rule_boundaries(self):
        for value in [float('nan'),True,-1,1.1]:
            n,c=predictions();n['reasoning']=value
            with self.assertRaises(ValueError):combine_predictions(n,c)
        n,c=predictions();c['length_class_probabilities']['long']=float('nan')
        with self.assertRaises(ValueError):combine_predictions(n,c)
        with self.assertRaises(ValueError):combine_predictions(*predictions(),ProfileRules(min_task_probability=-1))
        n,c=predictions(reasoning=.3)
        self.assertEqual(combine_predictions(n,c)['preference'],'consider_hosted')

    def test_actual_durations_do_not_enter_profile(self):
        n,c=predictions();before=combine_predictions(n,c)
        c['estimated_seconds']=99999;n['client_elapsed_seconds']=.01
        self.assertEqual(before,combine_predictions(n,c))
