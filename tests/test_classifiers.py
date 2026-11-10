import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from inference_router.classifiers import prompt_text, decode_nvidia, classify_job, ATTRIBUTE_LABELS, assets
from inference_router.classifier_assets import ROOT


class ClassifierTests(unittest.TestCase):
    def test_prompt_serialization_preserves_roles(self):
        self.assertEqual(prompt_text([{'role':'user','content':'hello'}]), 'hello')
        self.assertEqual(prompt_text([{'role':'system','content':'rules'}, {'role':'user','content':'hello'}]), 'system: rules\n\nuser: hello')
        for messages in [[], None, [{'role':'tool','content':'x'}], [{'role':'user','content':3}]]:
            with self.assertRaises(ValueError):
                prompt_text(messages)

    def test_weighted_heads_and_composite_match_published_conversion(self):
        names = ['creativity_scope','reasoning','constraint_ct','domain_knowledge','contextual_knowledge','number_of_few_shots']
        config = {'task_type_map':{'0':'Extraction','1':'Other'},'weights_map':{k:[0,1] for k in names},'divisor_map':{k:1 for k in names}}
        probabilities = {'task_type':[.9,.1], **{k:[.75,.25] for k in names}}
        result = decode_nvidia(probabilities,config)
        self.assertEqual(result['task_type'],'Extraction')
        self.assertAlmostEqual(result['reasoning'],.25)
        self.assertAlmostEqual(result['prompt_complexity_score'],.25)
        probabilities['number_of_few_shots']=[.99,.01]
        self.assertEqual(decode_nvidia(probabilities,config)['number_of_few_shots'],0)

    def test_repeat_keeps_predictions_without_prompt_text(self):
        class Fake:
            def predict(self,messages):
                return {'task_type':'Extraction','elapsed_seconds':.1}
        row=classify_job({'id':'one','messages':[{'role':'user','content':'private prompt'}]}, {'nvidia':Fake()},3)
        self.assertEqual(row['predictions']['nvidia']['timing_samples_seconds'],[.1,.1,.1])
        self.assertNotIn('private prompt',json.dumps(row))
        with self.assertRaises(ValueError):
            classify_job({'id':'one','messages':[]},{},0)

    def test_missing_assets_gives_setup_instruction(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory, patch('inference_router.classifiers.ROOT',Path(directory)):
            with self.assertRaisesRegex(ValueError,'download_classifiers'):
                assets('nvidia')

    def test_cli_help_needs_no_optional_runtime(self):
        result=subprocess.run([sys.executable,'-m','inference_router','classify','--help'],capture_output=True,text=True,check=True)
        self.assertIn('--classifiers',result.stdout)


HAS_CLASSIFIERS = all(importlib.util.find_spec(name) for name in ['torch','gliclass','transformers']) and all((ROOT/name/'model.safetensors').is_file() for name in ['nvidia','gliclass'])

@unittest.skipUnless(HAS_CLASSIFIERS,'optional classifier dependencies/weights unavailable')
class FrozenClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inference_router.classifiers import build_classifiers
        cls.models=build_classifiers(['nvidia','gliclass'])

    def test_nvidia_matches_published_example(self):
        result=self.models['nvidia'].predict([{'role':'user','content':'Write a Python script that uses a for loop.'}])
        self.assertEqual(result['task_type'],'Code Generation')
        self.assertAlmostEqual(result['task_probability'],.766669,places=4)
        self.assertAlmostEqual(result['reasoning'],.0632,places=4)

    def test_gliclass_matches_library_pipeline_and_complete_scores(self):
        model=self.models['gliclass']
        messages=[{'role':'user','content':'Return only the category billing or account: I was charged twice.'}]
        result=model.predict(messages)
        expected=model.pipeline(messages[0]['content'],list(ATTRIBUTE_LABELS.values()),threshold=0)[0]
        self.assertEqual(len(result['task_probabilities']),11)
        self.assertAlmostEqual(sum(result['task_probabilities'].values()),1,places=5)
        for key,label in ATTRIBUTE_LABELS.items():
            self.assertAlmostEqual(result['attribute_scores'][key],next(p['score'] for p in expected if p['label']==label),places=5)
        self.assertEqual(set(result['input_tokens_including_labels']),{'task','attributes'})

    def test_gliclass_declines_oversize_input_before_labels_are_lost(self):
        result=self.models['gliclass'].predict([{'role':'user','content':'Explain this sentence. '*1000}])
        self.assertEqual(result['status'],'input_too_long')
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['task_type'])
        self.assertEqual(result['forward_passes'],0)
