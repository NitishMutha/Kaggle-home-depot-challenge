#!/bin/bash

echo "Comparing All Models created"

MODEL_DIR='Data/Output/Models'
BASELINE_MODEL='ln.txt'

echo "Model Directory: $MODEL_DIR"
echo "Baseline Model: $BASELINE_MODEL"

java -cp bin/RankLib.jar ciir.umass.edu.eval.Analyzer -all "$MODEL_DIR" -base "$BASELINE_MODEL" > full_analysis.txt


