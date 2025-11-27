import argparse
import random
import time
import os
import sys
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import requests

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
except Exception:
    class _C:
        def __getattr__(self, _): return ""
    Fore = Style = _C()

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def query_prometheus(base: str, q: str, session: requests.Session, timeout=5) -> Tuple[Optional[str], Optional[str]]:
    url = f"{base.rstrip('/')}/api/v1/query"
    try:
        r = session.get(url, params={"query": q}, timeout=timeout)
        if r.status_code != 200:
            return None, f"{r.status_code} {r.reason}"
        try:
            data = r.json()
        except ValueError:
            return None, "non-json response"
        if data.get("status") != "success":
            return None, f"prom status {data.get('status')}"
        results = data["data"]["result"]
        if not results:
            return "0", ""
        # If numeric values, return a comma-separated list of label=value for visibility
        entries = []
        for item in results:
            val = item["value"][1]
            labels = item.get("metric", {})
            # build label summary (instance/job if present)
            label_ident = ""
            if "instance" in labels:
                label_ident = labels["instance"]
            elif "job" in labels:
                label_ident = labels["job"]
            else:
                # include up to two label k=v pairs if present
                pairs = [f"{k}={v}" for k, v in list(labels.items())[:2]]
                label_ident = ",".join(pairs) if pairs else ""
            if label_ident:
                entries.append(f"{label_ident}:{val}")
            else:
                entries.append(val)
        return "; ".join(entries), ""
    except Exception as e:
        return None, str(e)

def tail_file_lines(path: str):
    """Generator that yields new lines appended to file (handles truncation)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        last_size = os.path.getsize(path)
        while True:
            where = f.tell()
            line = f.readline()
            if line:
                yield line.rstrip("\n")
                last_size = os.path.getsize(path)
            else:
                # detect truncation/rotation
                try:
                    cur_size = os.path.getsize(path)
                except OSError:
                    cur_size = 0
                if cur_size < last_size:
                    # file rotated/truncated: reopen
                    try:
                        f.close()
                        f = open(path, "r", encoding="utf-8", errors="replace")
                        yield f.readline().rstrip("\n")
                        last_size = os.path.getsize(path)
                    except Exception:
                        # wait and retry
                        time.sleep(0.2)
                else:
                    time.sleep(0.05)

def print_metric_line(query: str, value: str):
    ts = now_ts()
    print(f"{Fore.YELLOW}{ts}{Style.RESET_ALL} {Fore.CYAN}[METRIC]{Style.RESET_ALL} {query} = {value}")

def print_error_line(source: str, err: str):
    ts = now_ts()
    print(f"{Fore.YELLOW}{ts}{Style.RESET_ALL} {Fore.RED}[ERROR]{Style.RESET_ALL} {source} -> {err}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Tail a log file and stream Prometheus instant queries in real-time.")
    parser.add_argument("--prometheus", "-p", required=True, help="Prometheus base URL (e.g. http://localhost:9090)")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="Prometheus poll interval (seconds)")
    parser.add_argument("--queries", "-q", nargs="+", required=True, help="PromQL instant queries to poll")
    parser.add_argument("--logfile", "-l", required=True, help="Path to the log file to tail (real server log)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    if args.no_color:
        global Fore, Style
        class _C: 
            def __getattr__(self, _): return ""
        Fore = Style = _C()

    logfile = args.logfile
    if not os.path.isfile(logfile):
        print_error_line("startup", f"log file not found: {logfile}")
        sys.exit(1)

    sess = requests.Session()
    queries: List[str] = args.queries
    poll_interval = max(0.1, args.interval)

    # start tail generator
    tail_gen = tail_file_lines(logfile)
    next_poll = time.perf_counter()

    try:
        print(f"{Fore.GREEN}{now_ts()}{Style.RESET_ALL} Starting real stream. Tailing: {logfile}  Prometheus: {args.prometheus}")
        while True:
            # print any available log lines (drain small burst)
            printed = False
            for _ in range(50):  # cap per loop to avoid starvation
                try:
                    line = next(tail_gen)
                except StopIteration:
                    break
                except Exception:
                    break
                if line is not None:
                    # print raw log line with timestamp (preserve original content)
                    ts = now_ts()
                    print(f"{Fore.YELLOW}{ts}{Style.RESET_ALL} {Fore.WHITE}{line}{Style.RESET_ALL}")
                    printed = True
                else:
                    break

            now = time.perf_counter()
            if now >= next_poll:
                # poll Prometheus queries and print results
                for q in queries:
                    v, err = query_prometheus(args.prometheus, q, sess)
                    if v is not None:
                        print_metric_line(q, v)
                    else:
                        print_error_line(f"prometheus:{q}", err)
                next_poll = now + poll_interval

            # small sleep to yield
            if not printed:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\n{now_ts()} Stopped by user.")
    except Exception as e:
        print_error_line("fatal", str(e))

if __name__ == "__main__":
    main()