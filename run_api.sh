python run.py \
    --gpu-indices 0 \
    --max-tasks-per-gpu 1 \
    --competitions random-acts-of-pizza \
    --docker-image mle-dojo \
    --task-runtime-seconds 43200 \
    --kaggle-data ./data/prepared \
    --log-dir ./results/logs \
    --output-dir ./results/output 