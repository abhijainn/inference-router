import copy
import math
from pathlib import Path
import tempfile
import unittest
from inference_router.ltr import length_order_score, prompt_text, LTRPredictor
from inference_router.scheduler import choose_dispatch, simulate
from scripts.evaluate_ltr import rank_metrics, variant_tasks


class LTRTests(unittest.TestCase):
    def test_inverse_length_score_and_invalid_values(self):
        self.assertLess(length_order_score(4), length_order_score(1))
        for value in [float('nan'), float('inf'), True, '4']:
            with self.assertRaises(ValueError): length_order_score(value)

    def test_plain_prompt_is_preserved_and_messages_validated(self):
        self.assertEqual(prompt_text([{'role':'user','content':' x\ny '}]), ' x\ny ')
        self.assertEqual(prompt_text([{'role':'system','content':'a'},{'role':'user','content':'b'}]), 'system: a\n\nuser: b')
        for values in [[], [{'role':'tool','content':'x'}], [{'role':'user','content':None}]]:
            with self.assertRaises(ValueError):prompt_text(values)

    def test_missing_assets_fail_without_network_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'download_ltr'):
                LTRPredictor(Path(directory))

    def test_rank_metrics_correct_inverse_and_ties(self):
        perfect = rank_metrics([1.,2.,3.], [10.,20.,30.])
        reverse = rank_metrics([3.,2.,1.], [10.,20.,30.])
        self.assertAlmostEqual(perfect['spearman'], 1.)
        self.assertEqual(perfect['pairwise_accuracy'],1.)
        self.assertAlmostEqual(reverse['spearman'],-1.)
        self.assertEqual(reverse['pairwise_accuracy'],0.)
        tied = rank_metrics([0.,0.,0.],[1.,1.,2.])
        self.assertIsNone(tied['spearman'])
        self.assertEqual(tied['pairs'],2)
        self.assertEqual(tied['pairwise_accuracy'],.5)
        for pred,actual in [([1.],[1.,2.]),([math.nan,1.],[1.,2.])]:
            with self.assertRaises(ValueError):rank_metrics(pred,actual)

    def test_continuous_length_changes_only_overlap_decision(self):
        def task(key,score):
            return {'id':key,'arrival_seconds':0,'dependencies':[], 'profile':{
                'preference':'uncertain','length_class':'medium','length_order_score':score,'length_order_source':'ltr'}}
        visible={'a':task('a',0.),'c':task('c',1.)}
        running=[{'id':'a','backend':'hosted','started_seconds':0}]
        available={'local':0,'hosted':1}
        self.assertEqual(choose_dispatch(visible,['c'],running,available,'profile_dynamic')[:2],('c','hosted'))
        # Same coarse classes retain the original wait behavior.
        coarse=copy.deepcopy(visible)
        for task in coarse.values():
            task['profile'].pop('length_order_score');task['profile'].pop('length_order_source')
        self.assertIsNone(choose_dispatch(coarse,['c'],running,available,'profile_dynamic'))
        self.assertIsNone(choose_dispatch(visible,['c'],running,available,'context_only'))
        # Unrelated scales cannot be compared as if their logits have the same units.
        visible['c']['profile']['length_order_source']='other'
        self.assertIsNone(choose_dispatch(visible,['c'],running,available,'profile_dynamic'))

    def test_invalid_scheduler_score_is_rejected(self):
        base={'id':'a','profile':{'preference':'uncertain','length_class':'medium'}}
        for update in [{'length_order_score':math.nan,'length_order_source':'ltr'},
                       {'length_order_score':1.}, {'length_order_source':'ltr'},
                       {'length_order_score':True,'length_order_source':'ltr'}]:
            task=copy.deepcopy(base);task['profile'].update(update)
            with self.assertRaises(ValueError):simulate([task],{'a':{'local':1.,'hosted':.5}})

    def test_ablation_preserves_frozen_nominations_and_does_not_mutate(self):
        tasks=[{'id':'a','profile':{'predictor_job_id':'p','length_class':'long','preference':'consider_hosted'}}]
        original=copy.deepcopy(tasks);scores={'p':{'scores':{'ltr':-3.,'clairvoyant_score':1.5}}}
        hybrid=variant_tasks(tasks,'ltr_frozen_nominations',scores)[0]['profile']
        length_only=variant_tasks(tasks,'ltr_length_only',scores)[0]['profile']
        self.assertEqual(hybrid['preference'],'consider_hosted')
        self.assertEqual(length_only['preference'],'uncertain')
        self.assertEqual(hybrid['length_order_score'],-3.)
        self.assertEqual(tasks,original)


if __name__=='__main__':unittest.main()
