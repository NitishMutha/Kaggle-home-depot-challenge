import subprocess

TRAIN_INPUT='../../data/input/ranklib_train_input.txt'
TEST_INPUT='../../data/input/ranklib_test_rl_input.txt'
VAL_INPUT='../../data/input/ranklib_valc_input.txt'

MODEL_DIR='../../data/output/Models/'
RESULT_DIR='../../data/output/Results'

MODELS=['RankNet', 'RankBoost', 'AdaRank', 'LambdaMART', 'MART', 'ListNet']
RANKER = 1
EPOCH = 20
LAYER = 2
NODE = 20

### Call to Java to train model

result= subprocess.run(["java", "-jar",
                "./RankLib_2p1/bin/RankLib.jar",
                "-train",TRAIN_INPUT,
                "-validate",VAL_INPUT,
                "-test",TEST_INPUT,
                "-ranker", str(RANKER),
                "-metric2t", 'ERR@2',
                "-epoch", str(EPOCH),
                "-layer", str(LAYER),
                "-node", str(NODE),
                "-save", MODEL_DIR + MODELS[RANKER-1]+'_rn_epoch_' + str(EPOCH) + '_layer_' + str(LAYER) + '_node_' + str(NODE) + '.txt'], stdout=subprocess.PIPE)

print(result.stdout.decode())

### Call to Java to predict rank from trained model

result= subprocess.run(["java", "-jar",
                "./RankLib_2p1/bin/RankLib.jar",
                "-load", MODEL_DIR + MODELS[RANKER-1]+'_rn_epoch_' + str(EPOCH) + '_layer_' + str(LAYER) + '_node_' + str(NODE) + '.txt',
                "-rank",TEST_INPUT,
                "-score", MODEL_DIR + MODELS[RANKER-1]+'_rn_epoch_' + str(EPOCH) + '_layer_' + str(LAYER) + '_node_' + str(NODE) + '_test_rank.txt'], stdout=subprocess.PIPE)

print(result.stdout.decode())