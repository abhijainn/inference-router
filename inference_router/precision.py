"""Precision checks and paired-run aggregation; never infer dtype from model size."""
import math
import statistics


def normalized_precision(identity):
    value=(identity.get('details') or {}).get('quantization_level')
    if not isinstance(value,str):
        return None
    return {'BF16':'BF16','BFLOAT16':'BF16','F16':'FP16','FLOAT16':'FP16','FP16':'FP16'}.get(value.strip().upper(),value.strip().upper())


def verify_bf16_pair(local,hosted):
    if local.get('backend')!='local_ollama' or hosted.get('backend')!='deepinfra':
        raise ValueError('Expected separate local Ollama and DeepInfra identities')
    if local.get('model')!='gemma3:4b-it-bf16' or hosted.get('model')!='google/gemma-3-4b-it':
        raise ValueError('Unexpected model; no substitution is allowed')
    if normalized_precision(local)!='BF16' or normalized_precision(hosted)!='BF16':
        raise ValueError('Both deployments must explicitly report BF16; FP16 and 4-bit are not equivalent')
    if not local.get('digest'):
        raise ValueError('Local artifact digest is required')
    return {'matched_declared_weight_precision':True,'precision':'BF16',
            'local_evidence':'Ollama model metadata and digest',
            'hosted_evidence':'DeepInfra public catalog; not runtime attestation',
            'provider_artifact_digest_verified':False,'checkpoint_revision_equality_verified':False,
            'compute_dtype_equality_verified':False,
            'scope':'Same model family/size/instruction variant and reported weight format; serving stacks, prompts/templates and hardware are not identical.'}


def aggregate_pairs(local_rows,hosted_rows,job_ids,expected_repeats=3):
    if isinstance(expected_repeats,bool) or not isinstance(expected_repeats,int) or expected_repeats<1:
        raise ValueError('Expected repeats must be positive')
    groups={'local':{},'hosted':{}}
    for label,rows,backend in [('local',local_rows,'local_ollama'),('hosted',hosted_rows,'deepinfra')]:
        for row in rows:
            key=(row.get('job_id'),row.get('repeat'))
            if key in groups[label] or key[0] not in job_ids or isinstance(key[1],bool) or not isinstance(key[1],int):
                raise ValueError('Duplicate or unknown measurement')
            if row.get('backend')!=backend or row.get('success') is not True or row.get('done_reason')!='stop':
                raise ValueError('Use only successful naturally stopped measurements')
            duration=row.get('client_elapsed_seconds')
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
                raise ValueError('Invalid measured duration')
            if row.get('model_returned')!=row.get('model_identity',{}).get('model'):
                raise ValueError('Returned model does not match the inspected model')
            if normalized_precision(row.get('model_identity',{}))!='BF16':
                raise ValueError('Measurement is not explicitly BF16')
            groups[label][key]=row
    expected={(job,repeat) for job in job_ids for repeat in range(expected_repeats)}
    if set(groups['local'])!=expected or set(groups['hosted'])!=expected:
        raise ValueError('Incomplete paired repetitions; never silently exclude missing or truncated requests')
    for key in expected:
        left,right=groups['local'][key],groups['hosted'][key]
        verify_bf16_pair(left['model_identity'],right['model_identity'])
        lhash=left.get('request_features',{}).get('request_sha256')
        rhash=right.get('request_features',{}).get('request_sha256')
        if not lhash or lhash!=rhash:
            raise ValueError('Paired prompt hashes do not match')
    table=[]
    for job in job_ids:
        result={'job_id':job}
        for label in groups:
            samples=[groups[label][(job,repeat)] for repeat in range(expected_repeats)]
            durations=[row['client_elapsed_seconds'] for row in samples]
            tokens=[row.get('output_tokens') for row in samples]
            result[label]={'seconds':durations,'median_seconds':statistics.median(durations),
                           'min_seconds':min(durations),'max_seconds':max(durations),
                           'output_tokens':tokens,'median_output_tokens':statistics.median(tokens) if all(isinstance(v,int) for v in tokens) else None}
        table.append(result)
    return table


def ensure_timing_valid(manifest):
    if manifest.get('timing_validity_status')=='invalidated_environment' or manifest.get('resume_allowed') is False:
        raise ValueError('This timing run was invalidated by environmental conditions. Start a fresh run after resolving them; do not resume or relabel it.')


def check_elapsed_clocks(wall_seconds,client_seconds,max_gap_seconds=5):
    gap=wall_seconds-client_seconds
    if not math.isfinite(gap) or abs(gap)>max_gap_seconds:
        raise ValueError('Wall and client clocks differ by more than five seconds; possible system suspension or clock adjustment. Timing run is invalid.')
    return gap
