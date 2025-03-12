EXP='test'
SCENES=("Pony" "Bassist" "Cymbals" "Pianist" "Fruity" "Curling")
SCENES=("Pony")

for SCENE in "${SCENES[@]}"; do
  python gui_nodpg.py -s /data/Condense_v2/scenes/${SCENE} --port 6017 --expname "condense/${SCENE}_${EXP}" --configs arguments/dynerf/${SCENE}.py
  #python render.py --model_path "output/dynerf/${SCENE}_${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py
  #python metrics.py --model_path "output/dynerf/${SCENE}_${EXP}"
done

#EXP='FSa_TEST'
#SCENE="flame_salmon"
#python render.py --model_path "output/dynerf/${EXP}/"  --skip_train --configs arguments/dynerf/${SCENE}.py

