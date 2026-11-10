"""Commands: check, chat, run. The run command is one sequential local worker."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

from .ollama import OllamaClient, OllamaError
from .env import load_dotenv


def load_jobs(path, max_tokens_override=None):
    jobs = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        job = json.loads(line)
        if not isinstance(job, dict):
            raise ValueError(f"Line {line_number}: expected a JSON object")
        job_id = job.get("id", f"line-{line_number}")
        if not isinstance(job_id, str) or job_id in seen:
            raise ValueError(f"Line {line_number}: id must be a unique string")
        seen.add(job_id)
        messages = job.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str) for m in messages
        ):
            raise ValueError(f"Line {line_number}: invalid messages")
        cap = max_tokens_override if max_tokens_override is not None else job.get("max_tokens", 128)
        if isinstance(cap, bool) or not isinstance(cap, int) or (cap != -1 and not 0 < cap < 4096):
            raise ValueError(f"Line {line_number}: max_tokens must be -1 (no output cap) or between 1 and 4095")
        jobs.append({"id": job_id, "messages": messages, "max_tokens": cap})
    if not jobs:
        raise ValueError("No jobs found")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["ollama", "deepinfra"], default="ollama")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=180, help="HTTP socket timeout in seconds; 0 disables it; not an execution deadline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check server, model digest, and thinking controls")
    chat = sub.add_parser("chat", help="Make one measured request")
    chat.add_argument("prompt")
    chat.add_argument("--max-tokens", type=int, default=128, help="Output cap; -1 generates until the model stops")
    run = sub.add_parser("run", help="Process a JSONL file sequentially, logging each attempt")
    run.add_argument("input")
    run.add_argument("--output", default=None)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=None, help="Override all job output caps; -1 generates until the model stops")
    run.add_argument("--no-warmup", action="store_true", help="Skip warmup; does not force a cold start")
    run.add_argument("--save-responses", action="store_true", help="Save generated text; prompts are never written to logs")
    estimate = sub.add_parser("estimate", help="Predict output length and approximate duration without calling Ollama")
    source = estimate.add_mutually_exclusive_group(required=True)
    source.add_argument("prompt", nargs="?")
    source.add_argument("--input", help="JSONL jobs; no language-model execution")
    estimate.add_argument("--timing-profile", action="append", help="Deployment profile; repeat to compare backend estimates without execution")
    profile = sub.add_parser("build-timing-profile", help="Summarize existing saved response timings; no model training")
    profile.add_argument("measurements")
    profile.add_argument("--output", required=True)
    run.add_argument("--estimate", action="store_true", help="Log a pretrained prediction before each request")
    run.add_argument("--timing-profile", help="Profile used for approximate duration estimates")
    run.add_argument("--update-timing-profile", action="store_true", help="Refresh measured speed after each completion; classifier stays frozen")
    evaluate = sub.add_parser("evaluate-predictions", help="Summarize predictions recorded before completed requests")
    evaluate.add_argument("measurements")
    policy = sub.add_parser("policy", help="Choose a backend without executing inference")
    policy.add_argument("--budget-seconds", required=True, type=float)
    policy.add_argument("--safety-factor", type=float, default=1.25)
    policy.add_argument("--local-seconds", type=float)
    policy.add_argument("--hosted-seconds", type=float)
    policy.add_argument("--local-queue-seconds", type=float, default=0)
    policy.add_argument("--hosted-queue-seconds", type=float, default=0)
    policy.add_argument("--prompt")
    policy.add_argument("--local-profile")
    policy.add_argument("--hosted-profile")
    replay_parser = sub.add_parser("replay-policy", help="Compare policy and baselines using paired logs; no inference calls")
    replay_parser.add_argument("--local-log", required=True)
    replay_parser.add_argument("--hosted-log", required=True)
    replay_parser.add_argument("--budgets", nargs="+", type=float, default=[5, 20, 60])
    replay_parser.add_argument("--safety-factor", type=float, default=1.25)
    replay_parser.add_argument("--mode", choices=["recorded", "leave-one-out"], default="recorded")
    replay_parser.add_argument("--calibration-local")
    replay_parser.add_argument("--calibration-hosted")
    replay_parser.add_argument("--requests")
    replay_parser.add_argument("--output")
    replay_parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()
    try:
        load_dotenv()
        if args.command == "policy":
            from .policy import BackendEstimate, choose_backend
            predictions = {}
            if args.prompt is not None:
                if not (args.local_profile and args.hosted_profile) or args.local_seconds is not None or args.hosted_seconds is not None:
                    raise ValueError("Prompt mode requires both profiles and cannot mix numeric duration estimates")
                from .estimator import DurationEstimator, TimingProfile
                estimator = DurationEstimator()
                for backend, path, expected in [("local", args.local_profile, "local_ollama"), ("hosted", args.hosted_profile, "deepinfra")]:
                    estimator.profile = TimingProfile.load(path)
                    if estimator.profile.configuration.get("backend") != expected:
                        raise ValueError("Profile backend does not match policy backend")
                    predictions[backend] = estimator.predict([{"role": "user", "content": args.prompt}], -1)
                args.local_seconds = predictions["local"]["estimated_seconds"]
                args.hosted_seconds = predictions["hosted"]["estimated_seconds"]
            elif args.local_profile or args.hosted_profile:
                raise ValueError("Use --prompt with timing profiles")
            result = choose_backend(BackendEstimate(args.local_seconds, args.local_queue_seconds),
                                    BackendEstimate(args.hosted_seconds, args.hosted_queue_seconds),
                                    args.budget_seconds, args.safety_factor)
            if predictions:
                result["duration_predictions"] = predictions
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "replay-policy":
            from .policy_replay import read_rows, replay, leave_one_out_predictions, markdown_report
            local, hosted = read_rows(args.local_log), read_rows(args.hosted_log)
            if args.mode == "leave-one-out":
                if not (args.calibration_local and args.calibration_hosted and args.requests):
                    raise ValueError("Leave-one-out requires both calibration logs and --requests prompt catalog")
                local, hosted = leave_one_out_predictions(local, hosted, read_rows(args.calibration_local),
                                                          read_rows(args.calibration_hosted), load_jobs(args.requests))
            result = replay(local, hosted, args.budgets, args.safety_factor)
            result["mode"] = args.mode
            rendered = markdown_report(result) if args.format == "markdown" else json.dumps(result, indent=2)
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rendered + "\n")
            else:
                print(rendered)
            return 0
        if args.command == "evaluate-predictions":
            rows = [json.loads(line) for line in Path(args.measurements).read_text().splitlines() if line.strip()]
            groups = {}
            for row in rows:
                error = row.get("duration_prediction_error")
                if row.get("success") is True and row.get("done_reason") == "stop" and error:
                    groups.setdefault(row.get("backend"), []).append(error)
            print(json.dumps({backend: {
                "scored_requests": len(errors),
                "mean_absolute_error_seconds": sum(e["absolute_error_seconds"] for e in errors) / len(errors),
                "mean_absolute_percentage_error": sum(e["absolute_percentage_error"] for e in errors) / len(errors),
                "underestimation_fraction": sum(e["underestimated"] for e in errors) / len(errors),
            } for backend, errors in groups.items()}, indent=2))
            return 0
        if args.command == "build-timing-profile":
            from .estimator import TimingProfile
            rows = [json.loads(line) for line in Path(args.measurements).read_text().splitlines() if line.strip()]
            profile = TimingProfile.from_rows(rows, source=str(Path(args.measurements).resolve()))
            profile.save(args.output)
            print(json.dumps({"saved": args.output, **profile.summary()}, indent=2))
            return 0
        if args.command == "estimate":
            from .estimator import DurationEstimator, TimingProfile
            profiles = [(path, TimingProfile.load(path)) for path in (args.timing_profile or [])]
            estimator = DurationEstimator()
            jobs = load_jobs(args.input) if args.input else [{"id": "prompt", "messages": [{"role": "user", "content": args.prompt}], "max_tokens": -1}]
            for job in jobs:
                if len(profiles) > 1:
                    predictions = []
                    for path, profile in profiles:
                        estimator.profile = profile
                        predictions.append({"profile_path": path, **estimator.predict(job["messages"], job["max_tokens"])})
                    print(json.dumps({"job_id": job["id"], "backend_predictions": predictions}))
                else:
                    estimator.profile = profiles[0][1] if profiles else None
                    print(json.dumps({"job_id": job["id"], **estimator.predict(job["messages"], job["max_tokens"])}))
            return 0
        estimator = None
        profile = None
        if args.command == "run":
            if (args.timing_profile or args.update_timing_profile) and not args.estimate:
                raise ValueError("Use --estimate with timing profile options")
            if args.update_timing_profile and not args.timing_profile:
                raise ValueError("--update-timing-profile requires --timing-profile")
            if args.estimate:
                from .estimator import DurationEstimator, TimingProfile
                profile = TimingProfile.load(args.timing_profile) if args.timing_profile and Path(args.timing_profile).exists() else None
                if args.timing_profile and profile is None and not args.update_timing_profile:
                    raise ValueError("Timing profile does not exist. Build it first or use --update-timing-profile to bootstrap.")
                estimator = DurationEstimator(profile)
            if args.repeat < 1:
                raise ValueError("repeat must be positive")
            jobs = load_jobs(args.input, args.max_tokens)
        if args.backend == "deepinfra":
            from .deepinfra import DeepInfraClient
            client = DeepInfraClient(args.base_url or "https://api.deepinfra.com/v1/openai",
                                     args.model or "google/gemma-3-4b-it", args.timeout)
        else:
            client = OllamaClient(args.base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                                  args.model or os.getenv("OLLAMA_MODEL", "gemma3:4b-it-q4_K_M"), args.timeout)
        identity = client.inspect()
        if args.command == "check":
            print(json.dumps(identity, indent=2))
            return 0
        if args.command == "chat":
            print(json.dumps(client.chat([{"role": "user", "content": args.prompt}], args.max_tokens), indent=2))
            return 0
        if args.command == "run" and args.update_timing_profile and profile is None:
            from .estimator import TimingProfile, fingerprint
            profile = TimingProfile(fingerprint(identity))
            estimator.profile = profile
        if profile is not None:
            profile.check_configuration(identity)
        destination = Path(args.output or ("runs/deepinfra.jsonl" if args.backend == "deepinfra" else "runs/local.jsonl"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        warmup_performed = not args.no_warmup and args.backend == "ollama"
        if warmup_performed:
            client.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=8)
        run_id = str(uuid.uuid4())
        failures = 0
        with destination.open("a", encoding="utf-8") as output:
            for repeat in range(args.repeat):
                for job in jobs:
                    row = {
                        "run_id": run_id, "job_id": job["id"], "repeat": repeat,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "backend": "deepinfra" if args.backend == "deepinfra" else "local_ollama", "model_identity": identity,
                        "platform": platform.platform(), "python": platform.python_version(),
                        "concurrency": 1, "warmup_performed": warmup_performed,
                        "max_output_tokens": job["max_tokens"],
                        "request_features": {
                            "input_characters": sum(len(m["content"]) for m in job["messages"]),
                            "message_count": len(job["messages"]),
                            "request_sha256": hashlib.sha256(json.dumps(job["messages"], sort_keys=True).encode()).hexdigest(),
                        },
                    }
                    if estimator:
                        row["duration_prediction"] = estimator.predict(job["messages"], job["max_tokens"])
                    started = time.perf_counter()
                    try:
                        result = client.chat(job["messages"], job["max_tokens"])
                        row.update(result, success=True)
                        row["observed_output_proxy_units"] = len(result.get("response", "")) // 4
                        if estimator and row.get("done_reason") == "stop":
                            from .estimator import prediction_error
                            row["duration_prediction_error"] = prediction_error(row["duration_prediction"], row["client_elapsed_seconds"])
                        if args.update_timing_profile:
                            row["timing_profile_updated"] = profile.update(row)
                        if not args.save_responses:
                            row.pop("response", None)
                    except (OllamaError, ValueError) as exc:
                        failures += 1
                        row.update(success=False, error=str(exc), client_elapsed_seconds=time.perf_counter() - started)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    if args.update_timing_profile and row.get("timing_profile_updated"):
                        profile.save(args.timing_profile)
                    print(f"{job['id']} repeat={repeat} success={row['success']} elapsed={row['client_elapsed_seconds']:.3f}s", file=sys.stderr)
                    # No automatic retry: avoid hiding failures or overlapping a timed-out call.
                    if not row["success"]:
                        print("Stopped after failure. Check the server before restarting; in-flight work may remain.", file=sys.stderr)
                        return 1
        print(f"Saved attempts to {destination}")
        return 1 if failures else 0
    except (OllamaError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
