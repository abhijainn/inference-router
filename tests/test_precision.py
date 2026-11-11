from copy import deepcopy
import unittest
from inference_router.precision import aggregate_pairs,normalized_precision,verify_bf16_pair


def identities():
    return ({'backend':'local_ollama','model':'gemma3:4b-it-bf16','digest':'local-digest','details':{'quantization_level':'BF16'}},
            {'backend':'deepinfra','model':'google/gemma-3-4b-it','digest':None,'details':{'quantization_level':'bfloat16'}})


def rows():
    local,hosted=identities()
    result=[]
    for identity in [local,hosted]:
        result.append([{'job_id':'a','repeat':i,'backend':identity['backend'],'model_identity':identity,
                        'success':True,'done_reason':'stop','request_features':{'request_sha256':'same'},
                        'client_elapsed_seconds':v,'output_tokens':10,'model_returned':identity['model']} for i,v in enumerate([1,5,2])])
    return result


class PrecisionTests(unittest.TestCase):
    def test_declared_format_match_does_not_claim_runtime_attestation(self):
        check=verify_bf16_pair(*identities())
        self.assertTrue(check['matched_declared_weight_precision'])
        self.assertFalse(check['compute_dtype_equality_verified'])
        self.assertFalse(check['checkpoint_revision_equality_verified'])

    def test_fp16_is_not_bf16_and_missing_or_q4_is_rejected(self):
        for value in ['F16','float16','Q4_K_M',None]:
            local,hosted=identities();local['details']['quantization_level']=value
            with self.assertRaises(ValueError):verify_bf16_pair(local,hosted)
        self.assertNotEqual(normalized_precision({'details':{'quantization_level':'F16'}}),'BF16')

    def test_wrong_models_or_missing_digest_are_rejected(self):
        for field,value in [('model','gemma3:4b-it-q4_K_M'),('digest',None)]:
            local,hosted=identities();local[field]=value
            with self.assertRaises(ValueError):verify_bf16_pair(local,hosted)

    def test_median_keeps_all_repeats(self):
        table=aggregate_pairs(*rows(),['a'])
        self.assertEqual(table[0]['local']['median_seconds'],2)
        self.assertEqual(table[0]['local']['seconds'],[1,5,2])

    def test_missing_duplicate_truncated_mismatch_and_wrong_precision_fail(self):
        for mode in ['missing','duplicate','truncated','hash','precision','returned']:
            local,hosted=deepcopy(rows())
            if mode=='missing':local.pop()
            if mode=='duplicate':local.append(local[0])
            if mode=='truncated':local[0]['done_reason']='length'
            if mode=='hash':hosted[0]['request_features']['request_sha256']='other'
            if mode=='precision':local[0]['model_identity']['details']['quantization_level']='Q4_K_M'
            if mode=='returned':local[0]['model_returned']='wrong-model'
            with self.subTest(mode=mode),self.assertRaises(ValueError):aggregate_pairs(local,hosted,['a'])

    def test_runner_rejects_mismatch_before_warmup_or_paid_generation(self):
        import tempfile
        from unittest.mock import patch,MagicMock
        from scripts.run_matched_precision import main
        local_identity,hosted_identity=identities()
        local_identity['details']['quantization_level']='Q4_K_M'
        local=MagicMock();local.inspect.return_value=local_identity
        hosted=MagicMock();hosted.inspect.return_value=hosted_identity
        with tempfile.TemporaryDirectory() as directory, patch('sys.argv',['run_matched_precision','--output-dir',directory]), patch('scripts.run_matched_precision.load_dotenv'), patch('scripts.run_matched_precision.OllamaClient',return_value=local), patch('scripts.run_matched_precision.DeepInfraClient',return_value=hosted):
            with self.assertRaises(ValueError):main()
        local.chat.assert_not_called();hosted.chat.assert_not_called()

    def test_metadata_only_check_makes_no_generation_request(self):
        from unittest.mock import patch,MagicMock
        from scripts.run_matched_precision import main
        local_identity,hosted_identity=identities()
        local=MagicMock();local.inspect.return_value=local_identity
        hosted=MagicMock();hosted.inspect.return_value=hosted_identity
        with patch('sys.argv',['run_matched_precision','--check-only']),patch('scripts.run_matched_precision.load_dotenv'),patch('scripts.run_matched_precision.OllamaClient',return_value=local),patch('scripts.run_matched_precision.DeepInfraClient',return_value=hosted),patch('builtins.print'):
            main()
        local.chat.assert_not_called();hosted.chat.assert_not_called()

    def test_explicit_resume_preserves_failure_audit_and_skips_completed(self):
        import tempfile,json
        from pathlib import Path
        from scripts.run_matched_precision import resume_completed
        local,hosted=deepcopy(rows())
        local_row=local[0];local_row['run_id']='test-run'
        failed=hosted[0];failed.update(run_id='test-run',success=False,error='nodename nor servname provided, or not known')
        manifest={'run_id':'test-run','repeats':3,'request_hashes':{'a':'same'},'identities':dict(zip(['local','hosted'],identities()))}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'local.jsonl').write_text(json.dumps(local_row)+'\n')
            (root/'hosted.jsonl').write_text(json.dumps(failed)+'\n')
            completed=resume_completed(root,manifest)
            self.assertEqual(completed['local'],{('a',0)})
            self.assertEqual(completed['hosted'],set())
            self.assertEqual((root/'hosted.jsonl').read_text(),'')
            self.assertFalse(json.loads((root/'failed-attempts.jsonl').read_text())['success'])
            self.assertEqual(len(list(root.glob('hosted-interrupted-*.jsonl'))),1)
            resume_completed(root,manifest)
            self.assertEqual(len((root/'failed-attempts.jsonl').read_text().splitlines()),1)

    def test_resume_refuses_other_failure_types(self):
        import tempfile,json
        from pathlib import Path
        from scripts.run_matched_precision import resume_completed
        failed=rows()[1][0];failed.update(run_id='test-run',success=False,error='HTTP 403')
        manifest={'run_id':'test-run','repeats':3,'request_hashes':{'a':'same'},'identities':dict(zip(['local','hosted'],identities()))}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'hosted.jsonl').write_text(json.dumps(failed)+'\n')
            with self.assertRaises(ValueError):resume_completed(root,manifest)

    def test_environment_invalidated_run_cannot_be_treated_as_complete(self):
        from inference_router.precision import ensure_timing_valid
        with self.assertRaisesRegex(ValueError,'invalidated'):
            ensure_timing_valid({'completed_at':'some-time','timing_validity_status':'invalidated_environment'})
        with self.assertRaises(ValueError):ensure_timing_valid({'resume_allowed':False})
        ensure_timing_valid({})

    def test_elapsed_clock_gap_flags_possible_suspension(self):
        from inference_router.precision import check_elapsed_clocks
        self.assertAlmostEqual(check_elapsed_clocks(10.01,10),.01)
        for wall,client in [(260,10),(1,10),(float('nan'),10)]:
            with self.assertRaises(ValueError):check_elapsed_clocks(wall,client)
