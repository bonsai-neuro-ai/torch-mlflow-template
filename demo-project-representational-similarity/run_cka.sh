#!/usr/bin/env bash


MODELS=(
  "resnet18"
  "resnet34"
  "resnet50"
)

for MODELA in "${MODELS[@]}"; do
  LAYERSA=($(python -m model_info $MODELA --layers | grep "add"))
  for MODELB in "${MODELS[@]}"; do
    if [ "$MODELA" == "$MODELB" ]; then
      continue;
    fi
    LAYERSB=($(python -m model_info $MODELB --layers | grep "add"))
    for LAYERA in "${LAYERSA[@]}"; do
      for LAYERB in "${LAYERSB[@]}"; do
        echo "Running CKA for $MODELA layer $LAYERA and $MODELB layer $LAYERB"
        python -m compare_layers \
          --comparator="comparators.LinearCKA" \
          --modelA="$MODELA" \
          --modelB="$MODELB" \
          --layerA="$LAYERA" \
          --layerB="$LAYERB" \
          --dataset="imagenet" \
          --m=1000
      done
    done
  done
done