import numpy as np
import data_aggregation as da
import model

data = da.Data()
softmax_model = model.Model(0.0005)

def accuracy():
    predictions = (np.argmax(softmax_model.predict(data.test_imgs), axis=0) == data.test_lbls)
    print(np.sum(predictions) / predictions.size)

accuracy()

for i in range(1000):
    imgs, lbls, lbls_logits = data.next_batch(size=250)
    softmax_model.minimize(imgs, lbls_logits)
    if i%50 == 0:
        accuracy()

accuracy()