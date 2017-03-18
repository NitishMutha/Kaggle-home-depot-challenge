#!/bin/bash

TRAIN_INPUT='Data/Input/dummy.txt'
TEST_INPUT='Data/Input/ranklib_test_input.txt'
VAL_INPUT='Data/Input/ranklib_test_input.txt'
MODEL_DIR='Data/Output/Models'
RESULT_DIR='Data/Output/Results'

if [ "$1" = "train" ]
then
  if [ "$2" = "RankNet" ]
  then
    echo "Training model RankNet"
    java -jar bin/RankLib.jar -train "$TRAIN_INPUT" -validate "$VAL_INPUT" -ranker 1 -epoch $3 -layer $4 -node $5 -save "$MODEL_DIR/rn_epoch_$3_layer_$4_node_$5.txt"
    echo "Completed Training RankNet"
  elif [ "$2" = "RankBoost" ]
  then
    echo "Training model RankBoost"
    java -jar bin/RankLib.jar -train "$TRAIN_INPUT" -validate "$VAL_INPUT" -ranker 2 -round $3 -tc $4  -save "$MODEL_DIR/rb_round_$3_tc_$4.txt"
    echo "File Saved: $MODEL_DIR/rb_round_$3_tc_$4.txt"
    echo "Completed Training RankBoost"
  elif [ "$2" = "AdaRank" ]
  then
    echo "Training model AdaRank"
    java -jar bin/RankLib.jar -train "$TRAIN_INPUT" -validate "$VAL_INPUT" -ranker 3 -round $3 -tolerance $4 -max $5  -save "$MODEL_DIR/ar_round_$3_tolerance_$4_max_$5.txt"
    echo "File Saved: $MODEL_DIR/ar_round_$3_tolerance_$4_max_$5.txt"
    echo "Completed Training AdaRank"
  elif [ "$2" = "LambdaMART" ]
  then
    echo "Training model LambdaMART"
    java -jar bin/RankLib.jar -train "$TRAIN_INPUT" -validate "$VAL_INPUT" -ranker 6 -save "$MODEL_DIR/model.txt"
    echo "Completed Training LambdaMART"
  elif [ "$2" = "MART" ]
  then
    echo "Training model MART"
    java -jar bin/RankLib.jar -train "$TRAIN_INPUT" -validate "$VAL_INPUT" -ranker 0 -tree $3 -leaf $4 -shrinkage $5 -tc $6 -estop $7  -save "$MODEL_DIR/mart_tree_$3_leaf_$4_shrinkage_$5_tc_$6_estop_$7.txt"
    echo "File Saved: $MODEL_DIR/mart_tree_$3_leaf_$4_shrinkage_$5_tc_$6_estop_$7.txt"
    echo "Completed Training MART"
  elif [ "$2" = "ListNet" ]
  then
    echo "Training model ListNet"
    java -jar bin/RankLib.jar -train "$TRAIN_INPUT" -validate "$VAL_INPUT" -ranker 7 -epoch $3 -shrinkage $4 -save "$MODEL_DIR/ln_epoch_$3_shrinkage_$4.txt"
    echo "File Saved: $MODEL_DIR/ln_epoch_$3_shrinkage_$4.txt"
    echo "Completed Training ListNet"
  else
    echo "Something did not work, the model selected did not train."
  fi
elif [ "$1" = "test" ]
then
  if [ "$2" = "RankNet" ]
  then
    echo "Testing model RankNet"
    echo "Model: Data/Output/Models/rn_epoch_$3_layer_$4_node_$5.txt"
    echo "Test File: $TEST_INPUT"
    java -jar bin/RankLib.jar -load "$MODEL_DIR/rn_epoch_$3_layer_$4_node_$5.txt" -test "$TEST_INPUT" -metric2T ERR@10
      echo "Testing model RankNet complete"
  elif [ "$2" = "MART" ]
  then
    echo "Testing model MART"
    echo "Model: $MODEL_DIR/mart_tree_$3_leaf_$4_shrinkage_$5_tc_$6_estop_$7.txt"
    echo "Test File: $TEST_INPUT"
    java -jar bin/RankLib.jar -load "$MODEL_DIR/mart_tree_$3_leaf_$4_shrinkage_$5_tc_$6_estop_$7.txt" -test "$TEST_INPUT" -metric2T ERR@10
      echo "Testing model MART complete"
  elif [ "$2" = "ListNet" ]
  then
    echo "Testing model ListNet"
    echo "Model: $MODEL_DIR/ln_epoch_$3_shrinkage_$4.txt"
    echo "Test File: $TEST_INPUT"
    java -jar bin/RankLib.jar -load "$MODEL_DIR/ln_epoch_$3_shrinkage_$4.txt" -test "$TEST_INPUT" -metric2T ERR@10
    echo "Testing model ListNet complete"
  elif [ "$2" = "AdaRank" ]
  then
    echo "Testing model AdaRank"
    java -jar bin/RankLib.jar -load "$MODEL_DIR/ar_round_$3_tolerance_$4_max_$5.txt" -test "$TEST_INPUT" -metric2T ERR@10
    echo "Testing model AdaRank complete"
  elif [ "$2" = "LambdaMART" ]
  then
    echo "Testing model LambdaMART"
    java -jar bin/RankLib.jar -load "$MODEL_DIR/lm_tree_$3_leaf_$4_shrinkage_$5_tc_$6_estop_$7.txt" -test "$TEST_INPUT" -metric2T ERR@10
    echo "Testing model LambdaMART complete"
  elif [ "$2" = "RankBoost" ]
  then
    echo "Testing model RankBoost"
    java -jar bin/RankLib.jar -load "$MODEL_DIR/rb_round_$3_tc_$4.txt" -test "$TEST_INPUT" -metric2T ERR@10
    echo "Testing model RankBoost complete"
  fi
else
  echo "Model testing did not occur."
fi
