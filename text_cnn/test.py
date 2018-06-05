
import numpy as np
from keras import Sequential

from keras.layers import Embedding



# model = Sequential()
# model.add(Embedding(1000, 64, input_length=10))
#
# input_array = np.random.randint(1000, size=(32, 10))
#
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)
# print(output_array)

content=[]
with open("test",encoding="utf-8") as f:
    content=f.readlines()
flatten=""
for c in content:
    flatten+=c.replace("\n","").strip()
print(flatten)
