# 4 Training Cameras at extremities
#python gui.py -s /data/Condense_v2/scenes/B2/ --port 6057 --expname "condense/test" --configs arguments/condense/b1.py
#CUDA_LAUNCH_BLOCKING=1 python gui.py -s /data/Condense_v2/scenes/B2/ --expname "condense/testog" --configs arguments/condense/bench.py

CUDA_LAUNCH_BLOCKING=1 python gui.py -s /data/Condense_v2/scenes/A1/ --expname "condense_3_tune/bench" --configs arguments/condense/tuning/bench.py --test_iterations 1000

# python render.py --model_path "output/condense/test5" --skip_train --configs arguments/dynerf/flame_salmon_1.py
# Metrics
# python render.py --model_path "output/condense/test" --skip_train --configs arguments/dynerf/$ARGS