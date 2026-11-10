"""Controlled mechanism checks, not predictions from the pretrained models."""
from inference_router.request_profile import combine_predictions


def profile(task_type, length, reasoning=.1):
    nvidia={'task_type':task_type,'task_probabilities':{task_type:.95,'Other':.05},
            'reasoning':reasoning,'constraint_ct':.5,'truncated':False}
    clairvoyant={'length_class_probabilities':{key:.9 if key==length else .05 for key in ['short','medium','long']}}
    return combine_predictions(nvidia,clairvoyant)


def controlled_scenarios():
    small=profile('Classification','short')
    nominee=profile('Code Generation','medium',.6)
    long_local=profile('Text Generation','long',.1)
    def task(key, request_profile, dependencies=None, arrival=0, background=False):
        return {'id':key,'profile':request_profile,'dependencies':dependencies or [],
                'arrival_seconds':arrival,'background':background}
    return {
        'nomination_breaks_equal_depth_tie': {
            'tasks':[task('a-small',small),task('z-analysis',nominee),task('join',small,['a-small','z-analysis'])],
            'outcomes':{'a-small':{'local':1,'hosted':.3},'z-analysis':{'local':40,'hosted':15},'join':{'local':.5,'hosted':.2}},
        },
        'length_changes_wait_or_hosted_choice': {
            'tasks':[task('a-hosted',nominee),task('background',long_local,background=True),task('new-long',long_local,arrival=1)],
            'outcomes':{'a-hosted':{'local':20,'hosted':5},'background':{'local':40,'hosted':10},'new-long':{'local':18,'hosted':6}},
        },
        'wrong_nomination_hurts': {
            'tasks':[task('a-large',small),task('z-small',nominee),task('join',small,['a-large','z-small'])],
            'outcomes':{'a-large':{'local':40,'hosted':15},'z-small':{'local':1,'hosted':.3},'join':{'local':.5,'hosted':.2}},
        },
    }
