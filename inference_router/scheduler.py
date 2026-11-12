"""Online decisions over visible task state; offline event-driven replay engine.

Service times are private to simulate(), never supplied to choose_dispatch().
"""
from copy import deepcopy
import heapq
import math

STRATEGIES = ('always_local', 'always_hosted', 'occupancy', 'context_only', 'profile_dynamic')


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Times must be finite, nonnegative numbers')
    return value


def goal_depths(tasks):
    """Longest visible dependency chain to a foreground result; no time units."""
    children = {key: [] for key in tasks}
    for task in tasks.values():
        for parent in task['dependencies']:
            if parent in children:
                children[parent].append(task['id'])
    cache = {}
    def depth(key):
        if key not in cache:
            own = 0 if tasks[key].get('background', False) else 1
            downstream = [depth(child) for child in children[key]]
            cache[key] = max([own] + [1 + value for value in downstream if value])
        return cache[key]
    return {key: depth(key) for key in tasks}


def length_covers(running_profile, waiting_profile):
    """Compare only predictor scores with the same declared scale."""
    if ('length_order_score' in running_profile and 'length_order_score' in waiting_profile
            and running_profile.get('length_order_source') == waiting_profile.get('length_order_source')):
        return running_profile['length_order_score'] >= waiting_profile['length_order_score']
    order = {'short': 0, 'medium': 1, 'long': 2}
    return order[running_profile['length_class']] >= order[waiting_profile['length_class']]


def choose_dispatch(visible, ready, running, available, strategy):
    """Return one (task, backend, reason) using no observed service outcomes.

    `visible` includes submitted blocked tasks, but never future arrivals.
    Unknown profiles are a neutral preference, not automatic hosted escalation.
    """
    if strategy not in STRATEGIES:
        raise ValueError('Unknown scheduling strategy')
    if not ready:
        return None
    if strategy == 'context_only':
        neutral = deepcopy(visible)
        for task in neutral.values():
            task['profile'] = {'preference': 'uncertain', 'length_class': 'medium'}
        return choose_dispatch(neutral, ready, running, available, 'profile_dynamic')
    ordered = sorted(ready, key=lambda key: (visible[key]['arrival_seconds'], key))
    if strategy in {'always_local', 'always_hosted'}:
        backend = 'local' if strategy == 'always_local' else 'hosted'
        return (ordered[0], backend, strategy) if available[backend] else None
    if strategy == 'occupancy':
        for backend in ['local', 'hosted']:
            if available[backend]:
                return ordered[0], backend, 'Use local capacity first, otherwise hosted.'
        return None
    depths = goal_depths(visible)
    ordered.sort(key=lambda key: (-int(depths[key] > 0), -depths[key], -int(visible[key]['profile']['preference'] == 'consider_hosted'), visible[key]['arrival_seconds'], key))
    critical = ordered[0]
    task = visible[critical]
    covered_chain = any(r['backend'] == 'hosted' and depths.get(r['id'], 0) >= depths[critical] for r in running)
    if available['hosted'] and depths[critical] > 1 and not covered_chain:
        return critical, 'hosted', 'Accelerate a visible dependency chain that unlocks foreground work.'
    # When independent foreground work competes, use profiles to select an
    # acceleration candidate; keep local work overlapping that hosted execution.
    contenders = [key for key in ordered if depths[key] and visible[key]['profile']['preference'] == 'consider_hosted']
    if available['hosted'] and contenders and not any(depths.get(r['id'], 0) and r['backend'] == 'hosted' for r in running):
        return contenders[0], 'hosted', 'Profile identifies an acceleration candidate for a foreground result.'
    if available['local']:
        return critical, 'local', 'Run ready work locally while retaining hosted capacity.'
    if available['hosted'] and depths[critical]:
        hosted_cover = any(r['backend'] == 'hosted' and depths.get(r['id'], 0) >= depths[critical]
                           and length_covers(visible[r['id']]['profile'], task['profile']) for r in running)
        if hosted_cover:
            return None  # Reconsider after the next completion/arrival; no finish-time oracle.
        return critical, 'hosted', 'Local is occupied; serve waiting foreground work on hosted.'
    return None


def validate(tasks, outcomes, hosted_slots):
    if isinstance(hosted_slots, bool) or not isinstance(hosted_slots, int) or hosted_slots < 1:
        raise ValueError('hosted_slots must be a positive integer')
    indexed = {}
    for original in tasks:
        task = deepcopy(original)
        key = task.get('id')
        if not isinstance(key, str) or key in indexed:
            raise ValueError('Task ids must be unique strings')
        _seconds(task.get('arrival_seconds', 0))
        task.setdefault('arrival_seconds', 0)
        task.setdefault('dependencies', [])
        if not isinstance(task['dependencies'], list) or len(set(task['dependencies'])) != len(task['dependencies']):
            raise ValueError('Dependencies must be a list without duplicates')
        if not isinstance(task.get('background', False), bool):
            raise ValueError('background must be boolean')
        profile = task.get('profile', {})
        if profile.get('preference') not in {'prefer_local', 'consider_hosted', 'uncertain'} or profile.get('length_class') not in {'short', 'medium', 'long'}:
            raise ValueError('Task requires a valid request profile')
        if 'length_order_score' in profile or 'length_order_source' in profile:
            value = profile.get('length_order_score')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not isinstance(profile.get('length_order_source'), str) or not profile['length_order_source']:
                raise ValueError('Continuous length ordering requires a finite score and named source')
        if key not in outcomes or set(outcomes[key]) != {'local', 'hosted'}:
            raise ValueError('Every task needs a local/hosted replay outcome')
        for duration in outcomes[key].values():
            if _seconds(duration) <= 0:
                raise ValueError('Service durations must be positive')
        indexed[key] = task
    if not indexed:
        raise ValueError('No tasks supplied')
    visiting, visited = set(), set()
    def visit(key):
        if key not in indexed:
            raise ValueError('Unknown dependency')
        if key in visiting:
            raise ValueError('Dependency cycle')
        if key in visited:
            return
        visiting.add(key)
        for parent in indexed[key]['dependencies']:
            visit(parent)
        visiting.remove(key); visited.add(key)
    for key in indexed:
        visit(key)
    return indexed


def simulate(tasks, outcomes, strategy='profile_dynamic', hosted_slots=2):
    """Replay arrivals/completions; fixed service outcomes assume no interference."""
    tasks = validate(tasks, outcomes, hosted_slots)
    events = [(task['arrival_seconds'], 1, key) for key, task in tasks.items()]
    heapq.heapify(events)
    visible, finished, active, records, trace = {}, {}, {}, {}, []
    now = 0
    while events:
        now = events[0][0]
        while events and events[0][0] == now:
            _, kind, key = heapq.heappop(events)
            if kind == 0:
                active.pop(key)
                finished[key] = now
                records[key]['finish_seconds'] = now
                trace.append({'time': now, 'event': 'completed', 'task': key})
            else:
                visible[key] = tasks[key]
                trace.append({'time': now, 'event': 'arrived', 'task': key})
        while True:
            ready = [key for key, task in visible.items() if key not in active and key not in finished and all(parent in finished for parent in task['dependencies'])]
            available = {'local': int(not any(r['backend'] == 'local' for r in active.values())),
                         'hosted': hosted_slots - sum(r['backend'] == 'hosted' for r in active.values())}
            decision = choose_dispatch(deepcopy(visible), ready, list(active.values()), available, strategy)
            if decision is None:
                break
            key, backend, reason = decision
            # Only the engine looks up outcomes, after dispatch is chosen.
            completion = now + outcomes[key][backend]
            if not math.isfinite(completion):
                raise ValueError('Completion timestamp overflow')
            heapq.heappush(events, (completion, 0, key))
            active[key] = {'id': key, 'backend': backend, 'started_seconds': now}
            task = visible[key]
            ready_since = max([task['arrival_seconds']] + [finished[parent] for parent in task['dependencies']])
            records[key] = {'backend': backend, 'start_seconds': now, 'queue_wait_seconds': now-ready_since, 'reason': reason}
            trace.append({'time': now, 'event': 'dispatched', 'task': key, 'backend': backend, 'reason': reason, 'visible_tasks': sorted(visible)})
    if len(finished) != len(tasks):
        raise ValueError('Scheduler failed to complete all tasks')
    foreground = [key for key, task in tasks.items() if not task.get('background', False)]
    return {'strategy': strategy, 'hosted_slots': hosted_slots, 'tasks': records, 'trace': trace,
            'metrics': {'hosted_calls': sum(r['backend'] == 'hosted' for r in records.values()),
                        'hosted_calls_avoided': sum(r['backend'] == 'local' for r in records.values()),
                        'all_tasks_finish_seconds': max(finished.values()),
                        'foreground_finish_seconds': max([finished[key] for key in foreground], default=0),
                        'mean_foreground_flow_seconds': sum(finished[key]-tasks[key]['arrival_seconds'] for key in foreground)/len(foreground) if foreground else 0,
                        'total_queue_wait_seconds': sum(r['queue_wait_seconds'] for r in records.values())}}
