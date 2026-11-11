"""Explicit billable paired Gemma BF16 experiment; refuses precision mismatch."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.make_requests import CALIBRATION_JOBS,VALIDATION_JOBS
from inference_router.env import load_dotenv
from inference_router.ollama import OllamaClient,OllamaError
from inference_router.deepinfra import DeepInfraClient
from inference_router.precision import verify_bf16_pair,ensure_timing_valid,check_elapsed_clocks


def resume_completed(destination,manifest):
    """Preserve successful rows and audit a resolver failure before explicit resume."""
    completed={'local':set(),'hosted':set()}
    for backend in completed:
        path=destination/f'{backend}.jsonl'
        raw=path.read_text() if path.exists() else ''
        rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
        good=[];failures=[]
        for row in rows:
            key=(row['job_id'],row['repeat'])
            if row.get('run_id')!=manifest['run_id'] or key[0] not in manifest['request_hashes'] or not 0<=key[1]<manifest['repeats']:
                raise ValueError('Existing measurement is incompatible with this run')
            if row.get('request_features',{}).get('request_sha256')!=manifest['request_hashes'][key[0]]:
                raise ValueError('Existing prompt hash mismatch')
            identity=row.get('model_identity',{})
            if backend=='local':
                verify_bf16_pair(identity,manifest['identities']['hosted'])
                if identity['digest']!=manifest['identities']['local']['digest']:
                    raise ValueError('Existing local measurement uses another artifact')
            else:
                verify_bf16_pair(manifest['identities']['local'],identity)
            if row.get('success') is not True:
                error=row.get('error','')
                if backend!='hosted' or not any(term in error for term in ['nodename nor servname','Name or service not known','Temporary failure in name resolution']):
                    raise ValueError('Resume supports only explicit DNS-failure recovery; inspect other failures separately')
                failures.append(row);continue
            if row.get('done_reason')!='stop' or key in completed[backend]:
                raise ValueError('Truncated or duplicate successful row cannot be resumed')
            completed[backend].add(key);good.append(row)
        if failures:
            backup=destination/f'{backend}-interrupted-{uuid.uuid4().hex[:8]}.jsonl'
            backup.write_text(raw)
            with (destination/'failed-attempts.jsonl').open('a') as output:
                for row in failures:output.write(json.dumps(row)+'\n')
            temporary=path.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row)+'\n' for row in good));temporary.replace(path)
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file')
    parser.add_argument('--output-dir')
    parser.add_argument('--check-only',action='store_true',help='Verify metadata only; make no generation requests')
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--resume',action='store_true',help='Explicitly resume after a documented DNS lookup failure')
    parser.add_argument('--timeout',type=float,default=600)
    args=parser.parse_args()
    if args.repeat<1:raise ValueError('repeat must be positive')
    load_dotenv(args.env_file)
    if not args.check_only and not args.output_dir:
        raise ValueError('--output-dir is required for a measured run')
    destination=Path(args.output_dir) if args.output_dir else None
    if not args.check_only:
        destination.mkdir(parents=True,exist_ok=True)
        if not args.resume and any((destination/name).exists() for name in ['local.jsonl','hosted.jsonl','manifest.json']):
            raise ValueError('Choose a new output directory; never overwrite or append to old experiments')
    local=OllamaClient(model='gemma3:4b-it-bf16',timeout=args.timeout)
    hosted=DeepInfraClient(timeout=args.timeout)
    identities={'local':local.inspect(),'hosted':hosted.inspect()}
    evidence=verify_bf16_pair(identities['local'],identities['hosted'])
    if args.check_only:
        print(json.dumps({'precision_check':evidence,'identities':identities},indent=2))
        return
    jobs=CALIBRATION_JOBS+VALIDATION_JOBS
    completed={'local':set(),'hosted':set()}
    if args.resume:
        manifest=json.loads((destination/'manifest.json').read_text())
        ensure_timing_valid(manifest)
        if manifest.get('completed_at') or manifest.get('repeats')!=args.repeat:
            raise ValueError('Run already complete or repetition configuration changed')
        if identities['local']['digest']!=manifest['identities']['local']['digest']:
            raise ValueError('Local artifact changed since interrupted run')
        completed=resume_completed(destination,manifest)
        manifest.setdefault('explicit_resumptions',[]).append({'at':datetime.now(timezone.utc).isoformat(),'reason':'DNS lookup failure; public connectivity and precision rechecked','completed_before_resume':{name:len(keys) for name,keys in completed.items()}})
        run_id=manifest['run_id']
    else:
        run_id=str(uuid.uuid4())
        manifest={'run_id':run_id,'started_at':datetime.now(timezone.utc).isoformat(),'identities':identities,
                  'precision_check':evidence,'repeats':args.repeat,'jobs':len(jobs),
                  'settings':{'temperature':0,'seed':42,'concurrency':1,'streaming':False,'local_num_ctx':4096,
                              'output_cap':None,'hosted_output_limit':'provider default, may still impose a maximum',
                              'prompt_order':'seeded shuffle per repetition; first backend alternates per pair'},
                  'platform':platform.platform(),'python':platform.python_version(),
                  'request_hashes':{job['id']:hashlib.sha256(json.dumps(job['messages'],sort_keys=True).encode()).hexdigest() for job in jobs}}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Precision preflight passed: both report BF16. Local warmup begins.',flush=True)
    warmup=local.chat([{'role':'user','content':'Reply with exactly OK.'}],max_tokens=16)
    if args.resume:
        manifest.setdefault('resume_warmups',[]).append({k:v for k,v in warmup.items() if k!='response'})
    else:
        manifest['local_warmup']={k:v for k,v in warmup.items() if k!='response'}
    manifest['local_runtime_after_warmup']=local._request('/api/ps')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Warmup complete.',json.dumps({'elapsed':warmup['client_elapsed_seconds'],'running':manifest['local_runtime_after_warmup']}),flush=True)
    clients={'local':local,'hosted':hosted}
    for repeat in range(args.repeat):
        identities={'local':local.inspect(),'hosted':hosted.inspect()}
        verify_bf16_pair(identities['local'],identities['hosted'])
        ordered=list(jobs);random.Random(20261111+repeat).shuffle(ordered)
        for index,job in enumerate(ordered):
            order=['local','hosted'] if (repeat*len(jobs)+index)%2==0 else ['hosted','local']
            for backend in order:
                if (job['id'],repeat) in completed[backend]:
                    continue
                row={'run_id':run_id,'job_id':job['id'],'repeat':repeat,
                     'started_at':datetime.now(timezone.utc).isoformat(),
                     'backend':'local_ollama' if backend=='local' else 'deepinfra',
                     'model_identity':identities[backend],'concurrency':1,'warmup_performed':backend=='local',
                     'request_features':{'request_sha256':manifest['request_hashes'][job['id']]},
                     'precision_check':evidence,'pair_backend_order':order}
                print(f"Starting repeat={repeat} {job['id']} {backend}",flush=True)
                started=time.perf_counter()
                wall_started=time.time()
                try:
                    result=clients[backend].chat(job['messages'],max_tokens=-1)
                    row.update(result,success=True)
                    row['request_wall_seconds']=time.time()-wall_started
                    try:
                        row['timing_clock_gap_seconds']=check_elapsed_clocks(row['request_wall_seconds'],row['client_elapsed_seconds'])
                    except ValueError:
                        manifest['timing_validity_status']='invalidated_environment'
                        manifest['resume_allowed']=False
                        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
                        raise
                    row['observed_output_proxy_units']=len(result['response'])//4
                    row['response_sha256']=hashlib.sha256(result['response'].encode()).hexdigest()
                    row['response']=result['response'] # private synthetic-output logs, ignored by Git
                    if backend=='local':row['local_runtime_after_request']=local._request('/api/ps')
                except (OllamaError,ValueError) as exc:
                    row.update(success=False,error=str(exc),client_elapsed_seconds=time.perf_counter()-started)
                with (destination/f'{backend}.jsonl').open('a') as output:
                    output.write(json.dumps(row)+'\n')
                print(f"Completed repeat={repeat} {job['id']} {backend}: success={row['success']} elapsed={row['client_elapsed_seconds']:.3f}s tokens={row.get('output_tokens')} stop={row.get('done_reason')}",flush=True)
                if not row['success'] or row.get('done_reason')!='stop':
                    raise ValueError('Stopped on failure or output truncation. No retry or further request was made.')
    manifest['completed_at']=datetime.now(timezone.utc).isoformat()
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Completed paired BF16 measurements:',destination,flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,OllamaError) as exc:
        print('Experiment stopped:',exc,file=sys.stderr)
        raise SystemExit(1)
