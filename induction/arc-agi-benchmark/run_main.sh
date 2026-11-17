#!/bin/bash
CFG=claude-opus-5-5-medium-claudecode
for i in $(seq 1 60); do
  echo "=== Round $i: $(date) ===" >> run_log.txt
  uv run cli/run_all.py --config $CFG --data_dir data/evaluation \
    --save_submission_dir submissions/main/original --max-tasks-per-run 20 --num_attempts 1 > last1.txt 2>&1
  uv run cli/run_all.py --config $CFG --data_dir disguised \
    --save_submission_dir submissions/main/disguised --max-tasks-per-run 20 --num_attempts 1 > last2.txt 2>&1
  cat last1.txt last2.txt >> run_log.txt
  if grep -q "All tasks already completed" last1.txt && grep -q "All tasks already completed" last2.txt; then
    echo "=== ALL DONE: $(date) ===" >> run_log.txt
    break
  fi
  sleep 1800
done
