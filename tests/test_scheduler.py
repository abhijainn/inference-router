import copy
import unittest
from inference_router.scheduler import simulate,choose_dispatch,STRATEGIES


def task(key, length='short', preference='prefer_local', **kwargs):
    return dict(id=key, arrival_seconds=0,dependencies=[],profile={'length_class':length,'preference':preference},**kwargs)


class SchedulerTests(unittest.TestCase):
    def test_background_work_does_not_block_new_foreground_arrivals(self):
        tasks=[task('a',background=True),task('b')];tasks[1]['arrival_seconds']=1
        result=simulate(tasks,{'a':{'local':20,'hosted':5},'b':{'local':1,'hosted':.5}})
        self.assertEqual(result['tasks']['a']['backend'],'local')
        self.assertEqual(result['tasks']['b']['backend'],'hosted')
        self.assertEqual(result['metrics']['foreground_finish_seconds'],1.5)

    def test_short_tasks_wait_locally_while_long_hosted_work_runs(self):
        tasks=[task('a','long','consider_hosted'),task('b'),task('c')]
        results=simulate(tasks,{'a':{'local':40,'hosted':15},'b':{'local':1,'hosted':.3},'c':{'local':1,'hosted':.3}})
        self.assertEqual(results['metrics']['hosted_calls'],1)
        self.assertEqual(results['tasks']['c']['start_seconds'],1)
        self.assertEqual(results['metrics']['foreground_finish_seconds'],15)

    def test_dependencies_and_capacity(self):
        tasks=[task('a'),task('b'),task('c')];tasks[2]['dependencies']=['a','b']
        outcomes={key:{'local':4,'hosted':2} for key in ['a','b','c']}
        for strategy in STRATEGIES:
            result=simulate(tasks,outcomes,strategy,hosted_slots=1)
            self.assertGreaterEqual(result['tasks']['c']['start_seconds'],max(result['tasks'][key]['finish_seconds'] for key in ['a','b']))
            for backend in ['local','hosted']:
                rows=sorted((row['start_seconds'],row['finish_seconds']) for row in result['tasks'].values() if row['backend']==backend)
                for previous,current in zip(rows,rows[1:]):self.assertGreaterEqual(current[0],previous[1])

    def test_future_arrivals_are_hidden_and_not_preempted(self):
        tasks=[task('a',background=True),task('b','long','consider_hosted')];tasks[1]['arrival_seconds']=100
        result=simulate(tasks,{'a':{'local':10,'hosted':2},'b':{'local':20,'hosted':5}})
        first=next(event for event in result['trace'] if event['event']=='dispatched')
        self.assertEqual(first['visible_tasks'],['a'])
        self.assertEqual(result['tasks']['a']['finish_seconds'],10)

    def test_dispatch_sees_no_outcome_times(self):
        visible={'a':task('a','long','consider_hosted'),'b':task('b')}
        first=choose_dispatch(visible,['a','b'],[],{'local':1,'hosted':1},'profile_dynamic')
        self.assertEqual(first[:2],('a','hosted'))
        # Decision depends on neither a running job's actual finish nor future jobs.
        self.assertNotIn('outcomes',visible['a'])

    def test_unknown_profiles_still_complete(self):
        result=simulate([task('a',preference='uncertain')],{'a':{'local':1,'hosted':.5}})
        self.assertEqual(result['tasks']['a']['backend'],'local')

    def test_invalid_workflows(self):
        tasks=[task('a')];outcomes={'a':{'local':1,'hosted':.5}}
        for mutation in ['cycle','missing','nan','slots']:
            values=copy.deepcopy(tasks);durations=copy.deepcopy(outcomes);slots=2
            if mutation=='cycle':values[0]['dependencies']=['a']
            if mutation=='missing':values[0]['dependencies']=['unknown']
            if mutation=='nan':durations['a']['local']=float('nan')
            if mutation=='slots':slots=True
            with self.assertRaises(ValueError):simulate(values,durations,hosted_slots=slots)

    def test_outcomes_do_not_change_first_dispatch(self):
        tasks=[task('a','long','consider_hosted'),task('b')]
        first=[]
        for durations in [dict(local=1,hosted=100),dict(local=100,hosted=1)]:
            result=simulate(tasks,{key:durations for key in ['a','b']})
            first.append(next(event for event in result['trace'] if event['event']=='dispatched'))
        self.assertEqual(first[0],first[1])

    def test_cli_simulation_needs_no_model_runtime_or_credentials(self):
        import json,subprocess,sys,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'workflow.json'
            path.write_text(json.dumps({'tasks':[task('a')],'outcomes':{'a':{'local':1,'hosted':.5}}}))
            result=subprocess.run([sys.executable,'-m','inference_router','--base-url','http://invalid.invalid','simulate',str(path)],capture_output=True,text=True,check=True)
            self.assertEqual(set(json.loads(result.stdout)),set(STRATEGIES))
            self.assertEqual(json.loads(result.stdout)['always_local']['metrics']['hosted_calls'],0)

    def test_context_ablation_does_not_use_acceleration_preference(self):
        visible={'a':task('a','long','consider_hosted')}
        self.assertEqual(choose_dispatch(visible,['a'],[],{'local':1,'hosted':1},'context_only')[1],'local')
        self.assertEqual(choose_dispatch(visible,['a'],[],{'local':1,'hosted':1},'profile_dynamic')[1],'hosted')
