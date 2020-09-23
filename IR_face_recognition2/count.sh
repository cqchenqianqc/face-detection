#/bin/bash
for i in `ls trainingData820`
do
echo /trainingData820/$i
ls ./trainingData820/$i -lR | grep '^-' |wc -l
done
