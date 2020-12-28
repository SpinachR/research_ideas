# Attentional Pooling for Action Recognition
Our attention module can be trained with or without extra supervision.

This form of 'hard-coded attention' (i.e., human pose keypoints, object/person bounding boxes) helps improve performance, but **requires labeling**. Moreover, these methods assume that focusing on the human or its parts is always useful for discriminating actions. This might not necessarily be true; some actions might be easier to distinguish by **background or context**.

Our work can learn attention maps which focus computation on specific parts of the input relevant to the task. The key idea can be seen as a natural extension of average pooling into a *'weighted' average pooling with image-specific weights*.

Bilinear pooling -- Second order pooling: ensemble two feature maps heterogenerously or homogenerously.
