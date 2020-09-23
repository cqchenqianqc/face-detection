mkdir ./data/fake
mkdir ./data/real

mv ./data/train/fake/* ./data/fake/
mv ./data/valid/fake/* ./data/fake/
mv ./data/test/fake/* ./data/fake/

mv ./data/train/real/* ./data/real/
mv ./data/valid/real/* ./data/real/
mv ./data/test/real/* ./data/real/

rm -rf ./data/train
rm -rf ./data/valid
rm -rf ./data/test