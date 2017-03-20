
RankNet:

./model_train_test.sh train RankNet 50 3 25 0.00005

[ -epoch <T> ]  The number of epochs to train (default=100)
[ -layer <layer> ]  The number of hidden layers (default=1)
[ -node <node> ]  The number of hidden nodes per layer (default=10)
[ -lr <rate> ]  Learning rate (default=0.00005)

[+] RankBoost's Parameters:
No. of epochs: 50
No. of hidden layers: 3
No. of hidden nodes per layer: 25
Learning rate: 5.0E-5


RankBoost:

[+] RankBoost's Parameters:
No. of rounds: 150
No. of threshold candidates: 35

Command Line Input:

./model_train_test.sh train RankBoost 150 35

Best Parameters:

 -round <T> ] The number of rounds to train (default=300)
  [ -tc <k> ]

AdaRank: Working

./model_train_test.sh train AdaRank 50 0.2 5

[+] AdaRank's Parameters:
No. of rounds: 50
Train with 'enequeue': Yes
Tolerance: 0.2
Max Sel. Count: 5

Best Parameters:

  [ -round <T> ]  The number of rounds to train (default=500)
  [ -noeq ] Train without enqueuing too-strong features (default=unspecified)
  [ -tolerance <t> ]  Tolerance between two consecutive rounds of learning (default=0.002)
  [ -max <times> ]


LambdaMART: Working

./model_train_test.sh train LambdaMART 1000 10 0.1 -1 100

Best Parameters:

  [ -tree <t> ] Number of trees (default=1000)
  [ -leaf <l> ] Number of leaves for each tree (default=10)
  [ -shrinkage <factor> ] Shrinkage, or learning rate (default=0.1)
  [ -tc <k> ] Number of threshold candidates for tree spliting. -1 to use all feature values (default=256)
  [ -mls <n> ]  Min leaf support -- minimum #samples each leaf has to contain (default=1)
  [ -estop <e> ]  Stop early when no improvement is observed on validaton data in e consecutive rounds (default=100)

[+] LambdaMART's Parameters:
No. of trees: 1000
No. of leaves: 10
No. of threshold candidates: -1
Learning rate: 0.1
Stop early: 100 rounds without performance gain on validation data


MART: Working

./model_train_test.sh train MART 1000 10 0.1 -1 100

Best Parameters:

  [ -tree <t> ] Number of trees (default=1000)
  [ -leaf <l> ] Number of leaves for each tree (default=10)
  [ -shrinkage <factor> ] Shrinkage, or learning rate (default=0.1)
  [ -tc <k> ] Number of threshold candidates for tree spliting. -1 to use all feature values (default=256)
  [ -mls <n> ]  Min leaf support -- minimum #samples each leaf has to contain (default=1)
  [ -estop <e> ]  Stop early when no improvement is observed on validaton data in e consecutive rounds (default=100)

No. of trees: 1000
No. of leaves: 10
No. of threshold candidates: -1
Learning rate: 0.1
Stop early: 100 rounds without performance gain on validation data


ListNet: Working

./model_train_test.sh train ListNet 20 0.00001

-epoch
- shrinkage

[+] ListNet's Parameters:
No. of epochs: 20
Learning rate: 1.0E-5




