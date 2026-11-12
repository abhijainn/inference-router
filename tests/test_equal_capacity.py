from collections import Counter
import copy
import unittest
from scripts.evaluate_equal_capacity import cases_once, run_capacity_variant, VARIANTS
from scripts.evaluate_combined_policy import JOBS
from inference_router.scheduler import simulate


def task(key,background=False,dependencies=None):
    return {'id':key,'arrival_seconds':0,'dependencies':dependencies or [],'background':background,
            'profile':{'preference':'uncertain','length_class':'medium'}}


class EqualCapacityTests(unittest.TestCase):
    def test_two_hosted_is_compared_to_two_total_mixed_slots(self):
        spec={'tasks':[task('a'),task('b')],
              'outcomes':{'a':{'local':15,'hosted':10},'b':{'local':15,'hosted':10}}}
        hosted=run_capacity_variant(spec,'always_hosted')
        mixed=run_capacity_variant(spec,'combined')
        self.assertEqual(hosted['metrics']['foreground_finish_seconds'],10)
        self.assertEqual(hosted['metrics']['hosted_calls'],2)
        self.assertEqual(mixed['metrics']['foreground_finish_seconds'],15)
        self.assertEqual(mixed['metrics']['hosted_calls'],1)
        self.assertEqual(mixed['metrics']['hosted_calls_avoided'],1)
        self.assertEqual(hosted['hosted_slots'],2)
        self.assertEqual(mixed['hosted_slots'],1)

    def test_controls_prioritize_foreground_without_using_local(self):
        spec={'tasks':[task('a-background',True),task('b-background',True),task('z-foreground')],
              'outcomes':{k:{'local':40,'hosted':10} for k in ['a-background','b-background','z-foreground']}}
        fifo=run_capacity_variant(spec,'always_hosted')
        control=run_capacity_variant(spec,'hosted_context')
        self.assertEqual(fifo['tasks']['z-foreground']['start_seconds'],10)
        self.assertEqual(control['tasks']['z-foreground']['start_seconds'],0)
        self.assertTrue(all(r['backend']=='hosted' for r in control['tasks'].values()))

    def test_capacity_and_dependencies_hold_for_every_variant(self):
        tasks=[task('a'),task('b'),task('join',dependencies=['a','b'])]
        spec={'tasks':tasks,'outcomes':{k:{'local':5,'hosted':3} for k in ['a','b','join']}}
        for variant in VARIANTS:
            result=run_capacity_variant(spec,variant)
            records=result['tasks']
            self.assertGreaterEqual(records['join']['start_seconds'],max(records[k]['finish_seconds'] for k in ['a','b']))
            limit=2 if variant in {'always_hosted','hosted_context','hosted_profile'} else 1
            for time in {r['start_seconds'] for r in records.values()}:
                hosted=sum(r['backend']=='hosted' and r['start_seconds']<=time<r['finish_seconds'] for r in records.values())
                local=sum(r['backend']=='local' and r['start_seconds']<=time<r['finish_seconds'] for r in records.values())
                self.assertLessEqual(hosted,limit)
                self.assertLessEqual(local,1)
                self.assertLessEqual(hosted+local,2)

    def test_workloads_are_not_double_counted(self):
        profiles={j['id']:{'preference':'uncertain','length_class':'medium'} for j in JOBS}
        paired={key:{'local':2.,'hosted':1.} for key in profiles}
        cases=list(cases_once(profiles,paired))
        self.assertEqual(Counter(c['family'] for c in cases),{'pair_matrix':1092,'occupied_lanes':936})
        self.assertEqual(len({c['id'] for c in cases}),2028)
        self.assertEqual(sum(len(c['spec']['tasks']) for c in cases),6006)

    def test_priority_control_does_not_see_future_arrival(self):
        a=task('a',True);b=task('b');b['arrival_seconds']=100
        spec={'tasks':[a,b],'outcomes':{'a':{'local':20,'hosted':10},'b':{'local':20,'hosted':10}}}
        result=run_capacity_variant(spec,'hosted_profile')
        first=next(e for e in result['trace'] if e['event']=='dispatched')
        self.assertEqual(first['visible_tasks'],['a'])
        self.assertEqual(first['time'],0)


if __name__=='__main__':unittest.main()
