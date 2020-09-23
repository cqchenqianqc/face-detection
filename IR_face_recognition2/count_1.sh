#/bin/bash
for i in `ls trainingData831/train`
do
echo /trainingData831/train/$i
ls ./trainingData831/train/$i -lR | grep '^-' |wc -l
done
