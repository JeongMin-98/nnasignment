import os
import itertools
import multiprocessing
import time
import argparse

# ====== 실험 설정 ======
models = ['resnet18','resnet34','resnet50']
optimizers = ['sgd', 'adam']
batch_size = 32
epochs = 20
lr = 0.001
use_wandb = True
max_concurrent = 3
max_retries = 1

# ====== 경로 설정 ======
log_dir = "log"
status_dir = "status"
success_log_path = "success_runs.txt"
failed_log_path = "failed_runs.txt"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(status_dir, exist_ok=True)

# ====== 세마포어 & 락 ======
semaphore = multiprocessing.Semaphore(max_concurrent)
lock = multiprocessing.Lock()

def write_status(run_id: str, status: str):
    with open(os.path.join(status_dir, f"{run_id}.status"), "w") as f:
        f.write(status)

def run_and_retry(model: str, optimizer: str):
    run_id = f"{model}-{optimizer}"
    log_path = os.path.join(log_dir, f"{run_id}.txt")
    attempt = 0
    success = False

    while attempt < max_retries and not success:
        attempt += 1
        write_status(run_id, f"running (attempt {attempt})")
        print(f"🚀 [{run_id}] 시도 {attempt} 시작")

        with semaphore:
            cmd = f"python main.py --model {model} --optimizer {optimizer} --batch_size {batch_size} --epochs {epochs} --lr {lr}"
            if use_wandb:
                cmd += " --use_wandb"
            full_cmd = f"{cmd} > {log_path} 2>&1"
            result = os.system(full_cmd)

        if result == 0:
            success = True
            write_status(run_id, "success")
            with lock:
                with open(success_log_path, 'a') as sf:
                    sf.write(f"{run_id}\n")
            print(f"✅ [{run_id}] 성공")
        else:
            write_status(run_id, f"fail (attempt {attempt})")
            print(f"❌ [{run_id}] 실패 (시도 {attempt})")
            if attempt < max_retries:
                time.sleep(5)

    if not success:
        with lock:
            with open(failed_log_path, 'a') as ff:
                ff.write(f"{run_id}\n")
        write_status(run_id, f"failed after {max_retries} attempts")

def monitor_status(total_runs: int):
    while True:
        status_files = sorted(os.listdir(status_dir))
        os.system('clear' if os.name != 'nt' else 'cls')
        print("📡 실험 상태 모니터링 중...\n")
        summary = {"running": 0, "success": 0, "fail": 0}
        for fname in status_files:
            with open(os.path.join(status_dir, fname)) as f:
                content = f.read().strip()
                run_id = fname.replace('.status', '')
                print(f"🔹 {run_id}: {content}")
                if "success" in content:
                    summary["success"] += 1
                elif "fail" in content:
                    summary["fail"] += 1
                elif "running" in content:
                    summary["running"] += 1
        print(f"\n📊 요약: ✅ {summary['success']}  ❌ {summary['fail']}  🔄 {summary['running']}")
        if summary["success"] + summary["fail"] >= total_runs:
            print("\n🧩 모든 실험 완료")
            break
        time.sleep(3)

def load_completed_runs() -> set:
    completed = set()
    if os.path.exists(success_log_path):
        with open(success_log_path) as f:
            completed.update([line.strip() for line in f])
    return completed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='이미 성공한 실험은 건너뛰기')
    args = parser.parse_args()

    # 실험 목록 생성
    run_combinations = list(itertools.product(models, optimizers))
    total_runs = len(run_combinations)

    # 이전 상태 유지 or 초기화
    if not args.resume:
        for path in [success_log_path, failed_log_path]:
            if os.path.exists(path):
                os.remove(path)
        for f in os.listdir(status_dir):
            os.remove(os.path.join(status_dir, f))
        completed_runs = set()
    else:
        completed_runs = load_completed_runs()
        print(f"🔁 재시작 모드: {len(completed_runs)}개는 생략됨")

    # 실험 조합 필터링
    jobs = []
    filtered_runs = []
    for model, optimizer in run_combinations:
        run_id = f"{model}-{optimizer}"
        if run_id in completed_runs:
            continue
        filtered_runs.append((model, optimizer))

    # 모니터 프로세스
    monitor_proc = multiprocessing.Process(target=monitor_status, args=(len(filtered_runs),))
    monitor_proc.start()

    # 병렬 실험 실행
    for model, optimizer in filtered_runs:
        p = multiprocessing.Process(target=run_and_retry, args=(model, optimizer))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
    monitor_proc.join()

    print("\n✅ 전체 실험 종료 완료")
