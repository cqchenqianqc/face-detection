import os
import random
 
num = 0;
for image_name in os.listdir("./trainingData831/gray"):
    feed = random.randint(0, 915)
    if feed <= 500:
        os.remove(os.path.join("./trainingData831/gray", image_name));
        #print(feed)
        #print(os.path.join("./trainingData831/green", image_name))
        num = num + 1
print(num)