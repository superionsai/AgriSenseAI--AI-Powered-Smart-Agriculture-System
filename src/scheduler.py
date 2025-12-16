# src/scheduler.py
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_pipeline():
    os.chdir(ROOT)
    print("🔁 Fetching data")
    subprocess.run(["python", "src/data_fetch.py"], check=False)

    print("🧹 Preprocessing + weighted fusion")
    subprocess.run(["python", "src/preprocess.py"], check=False)

    print("🤖 Predicting + advisory")
    subprocess.run(["python", "src/predict_and_advisory.py"], check=False)

if __name__ == "__main__":
    sched = BlockingScheduler()
    sched.add_job(run_pipeline, "interval", hours=1)
    print("⏱ Scheduler running (hourly)")
    sched.start()
