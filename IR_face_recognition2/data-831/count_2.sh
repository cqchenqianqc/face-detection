#/bin/bash
for i in `ls train`
do
echo /train/$i
ls ./train/$i -lR | grep '^-' |wc -l
done

