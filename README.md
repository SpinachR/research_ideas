# Few-shot for object detection
1. Incremental Few-Shot Object Detection
2. Few-shot Object Detection via Feature Reweighting
3. Frustratingly Simple Few-Shot Object Detection

# Generative model
1. U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for I2I
2. Real or not real, that is the question: change the output of discriminator to a distribution with a vector  
3. On stabilizing Generative Adversarial Training with Noise  
4. Spectral Regularization for Combating Mode Collapse in GANs  
5. COCO-FUNIT (few-shot, example-guided episodic training): Few shot Unsupervised I2I generalize the model to an unseen domain by legeraing examples at inference time.    
   Few-shot I2I models find it difficult to preserve the **structure of the input image** while emulating the appearance of unseen domain (the content loss)   
   Content conditioned style encoder
   
   
6. Example-guided Style-Consistent Image Synthesis from Semantic Labeling   
   a. style consistency discriminator to determine whether a pair of images are consistent in style    
   b. adaptive semantic consistency loss     
   c. training data sampling strategy   
   
   Need (x, I, F(I)) as input for the Generator: x is label, F(I) is the label of examplar I. However, in most case, F(.) is missing.    
7. Example-guided Image Synthesis across Arbitrary Scenes using Masked Spatial-Channel Attention and Self-Supervision   
   Also need (x, I, F(I))
   
8. Semantic image synthesis with spatially-adaptive normalization (**SPADE**)    
   Semantic layout as input is suboptimal as the normalization layers tend to 'wash away' semantic information. Therefore, we propose to use the layout for modulating the activations in normalization layers through a spatially-adaptive learned transformation.    
   The semantic is controlled via a label map, the style is controlled via the reference style image. Spatially-adaptive normalization is a conditional normalization layer that modulates the activations using input semantic layouts through a spatially-adaptive transformation.   
   For style transfer tasks, the affine parameters are used to control the global style of the output, and hence are uniform across spatial coordinates. The proposed normalization layer applies a spatially-varying affine transformation, masking it suitable for image synthesis from semantic masks.    
   Spatially-Adaptive DEnormalization: uses external data to denormalize the normalized activations, i.e., conditional to style images. With the SPADE, there is no need to feed the segmentation map to the first layer of the generator. The encoder can be omitted, which results in a more lightweight network. Input is noise.

9. Conditional Image-to-Image Translation: assume the image inderits some domain-specific features and domain-independent features   
10. Semi-parametric Image Synthesis: combines the complementary strengths of parametric and non parametric techniques.

11. Image Synthesis with Semantic Region-Adaptive Normalization (**SEAN normalization**)    
    Control the layout of the generated image using a segmentation mask that has labels for each semantic region and 'add' realistic styles to each region according to their labels. SPADE does not allow using a different style input image for each region individually, SEAN accepts one style image per region (or per region instance) as input.     
    Inserting style information only in the beginning of a network is not a good architecture choice. Higher quality results can be obtained if style information is injected as normalization parameters in multiple layers in the network.   
    The spatially varying normalization parameters are dependent on the segmentation mask as well as the style input images.  
    Style can be encoded in three places: 1) statistics of image features; 2) neural network weights; 3) parameters of a network normalization layers.
    Need input both segmentation maps and style images. They won't be paired.
    
    
12. Semantic Bottleneck Scene Generation: two steps, z->semantic layout, semantic layout->image

13. You Only Need Adversarial Supervision for Semantic Image Synthesis
   

# GAN inversion
1. In-Domain GAN Inversion for Real Image Editing  
   Existing iversion methods typically focus on reconstructing the target image by pixel values yet fail to land the inverted code in the semantic domain of the original latent space. One is learning-based (train an encoder), the other is optimization-based (deal with a single instance at one time, optimize the latent code z). Some work combines these two ideas by using the encoder to generate an initialization z for optimization.    
   Expoliting deep generative prior: optimize the z and G together. Our work jointly train the encoder and the discriminator, the G is fixed. The second step is similar to 'deep generative prior', but not optimize G and use encoder to constrain instead.



# Multi-label classification
1. Learning a Deep ConvNet for Multi-label Classification with Partial Labels  
2. Label-refinery: Incorporating taxonomic information in related work. The networks is trained over the same set of images, but uses labels generated by previous version. Do we need to retrain the network from scratch?

# Image retrieval
1. NetVLAD with attention: attention can be generated by segmentation, we can use this attention to erase 'something' that have negative effects on retrival (such as person).
2. Unsupervised Part-based Weighting Aggregation of Deep Convolutional Features for Image Retrieval
3. ATTENTION-AWARE AGE-AGNOSTIC VISUAL PLACE RECOGNITION
4. Deep Attentional Structured Representation Learning for Visual Recognition (Some regions provide more discriminative information)

# 3D reconstruction
1. Neural Cages for Detail-Preserving 3D Deformations (3d-to-3d based on 3d)  
2. Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild (2d-to-3d directly, with 2d as supervision)
3. C-Flow: Conditional Generative Flow Models for Images and 3D Point Clouds (2d-to-3d, 3d-to-2d)
4. Learning Free-Form Deformations for 3D object Reconstruction (2d+3d to 3d)
5. Learning to Generate and Reconstruct 3D Meshes with only 2d Supervison (2d-to-3d)
6. Learning to Infer Semantic Parameters for 3D Shape Editing
7. DiscoNet Shapes Learning on Disconnected Manifolds for 3D Editing
8. Few-shot Generalization for Single-Image 3D Reconstruction via Priors
9. Image2Mesh: do not rely on landmarks, using model deformation
10. DeformSyncNet
11. 3D-CODED: 3D Correspondences by Deep Deformation
12. 3DN: 3D Deformation Network
13. Im2Struct: Reconvering 3D Shape Structure from a Single RGB Image


# Domain adaptation
1. GradMix: Multi-source Transfer across Domains and Tasks   
2. Cluster Alignment with a Teacher for Unsupervised Domain Adaptation   
   Cluster alignment with a teacher can effectively incorporate the discriminative clustering structures in both domains for better adaptation.     
   CAT leverages a teacher model to reliably discover the class-conditional structure in the feature space. Then, it forces the features of both the source and the target domains to form discriminative class-conditional clusters and aligns the corresponding clusters across domains.      
   a. Discover the class-conditional structures in feature space and reshape them to be discriminative clusters. Source domain is ok since we have labels, target domain is completed by pseudo labels obtained by ensembled models. Clustering-loss: two features belongs to the same clusters should be close, different class features should be far apart (a hinge loss) SNTG loss -- teacher graph, gathering low-dimensional feature mapping.     
   b. Cluster alignment via conditional feature matching: the source features and target features belong to the same class should be similar, similar to feature matching.     
   c. Combining CAT with marginal distributional alignment, i.e., DANN or RevGrad
   
   Thanks to the classconditional discriminative alignment between the source and target domains, CAT outperforms MCD and VADA.   
   Office31 is more realistic and high-dimensional images, providing a good complement to the digits adaptation task.
   
3. FixBi: Bridging Domain Spaces for Unsupervised Domain Adaptation    
4. Larger Norm More Transferable: an adaptive feature norm.    
   Erratic discrimination of the target domain mainly stems from its much smaller feature norms.    
   
5. Unsupervised Domain Adaptation in the Absence of Source Data (Roshni Sahoo): outperforms fine-tuning with LIMITED labeled data. Find a transformation that maximize the difference, update classifier to minimize their difference. Like MCD.

6. Domain2Vec: a technique to measure domain similarity

7. Deep Co-Training with Task Decomposition for *Semi-Supervised Domain Adaptation*: two tasks (semi-supervised learning + unsupervised domain adaptation)

8. Catastrophic Forgetting Meets Negative Transfer: Batch Spectral Shrinkage. Minimizing the spectral norm of transferable features

9. Light-weight calibration a separable component for unsupervised domain adaptation   
   Instead of modifing the model, this work modifies the inputs to achieve domain adaptation, which is flexible to adapt to arbitrary new domains after being deployed and compressed. Data calibrator is responsible for achieving domain adaptation via unsupervised training.   
   Instead of directly generating source-like data, it generates source-like perturbation

# Domain Adaptation Applications
1. Domain Adaptation for Image Dehazing


# Multi-tasks or Multi-domains or Lifelong-learning
1. Learning multiple visual domains with residual adapters: __Multiple domains VGG group__, sharing the same backbone but adding some specific residual connections for the different domains. This is some related with lifelong learning without forgetting, and it can deal with very differnt tasks. Check domain-guided dropout.

2. Efficient parametrization of multi-domain deep neural networks: it shows that it is necessary to adapt both shallow and deep layers of a deep network, but required changes are very small. We also show that these universal parametrization are very effective for transfer learning.  

3. Lifelong learning with dynamically EXPANDABLE networks：DEN that can dynamically decide its network capacity as it trains on a sequence of tasks.   
   1). Retraining the entire network while regularizing it to prevent large deviation from the original model.
   2). Non-retraining models expand the entire network for the new task.
   3). Partial retraining selectively retrains the old network, expanding its capacity when necessary.
   
   Preventing catastrophic forgetting: 1) regularization the model parameters, e.g., L2, Elastic Weight Consolidation via fisher information matrix, also consider the entire learning trajectory rather than the final parameter value. 2) completely block any modifications.


# Style Transfer
1. StyleBank: An Explicit Representation for Neural Image Style Transfer. (The features from content image can be clustered based on K-means, similar to segmentation results. This result can be used for REGULARIZATION or specific region transfer)    
2. Deep Painterly Harmonization: Copying an element from a photo and pasting it into a painting.   
3. Deep Photo Style Transfer: Neural style algorithm can transfer colors, but introduces distortions that make the output like a painting, which is undesirable in the context of PHOTO STYLE Transfer.  
4. Photorealistic style transfer via wavelet transforms: there will be a reconstruction step that warm-up the encoder.



# Object detection
1. Gradient Harmonized Single-stage Detector: deal with imbalanced samples based on gradient.
2. DUNIT: Detection-based Unsupervised Image-to-Image Translation, style transfer, but with encoding the detected object.



# Semantic segmentation
1. BUDA: Boundless Unsupervised Domain Adaptation in Semantic Segmentation (instead of predicting never-seen classes as 'unknown' in open-set problem, BUDA aims to recognize never-seen classes based on zero-shot learning). Zero-shot learning recognizes unseen classes based on their semantic associations with seen classes.    
2. Multichannel Semantic Segmentation with Unsupervised Domain Adaptation: using RGB + Depth + Instance Boundary for adaptation.    
3. ColorMapGAN for map segmentation (maybe useful when data are scarce): the generator does not perform any convolution or pooling operations, it transform the colors with only one element-wise matrix multiplication and one matrix addition. Spectral distribution is similar.   
4. Feature to Adapt (https://github.com/omerlandau/FeatureToAdapt): regularizing backbone's features to have larger L2 norm.   
5. Consistency regularization with high-dimensional non-adversarial source-guided perturbation.   
6. FDA - Fourier Domain Adaptation. Apply FFT to source and target images, replace the low frequency part of the source amplitude with that from the target. NO DEEP NETWORKS for style transfer. This work has a good examples for self-training.    
7. Domain Adaptation for Semantic Segmentation with Maximum Squares Loss: the gradient of entropy is biased towards samples that are easy to transfer. To balance the gradient of well-classified target samples, we propose the maximum squares loss. Besides, we introduce the image-wise weighting ratio to alleviate the class imbalance.    
8. Taking a closer look at domain shift: the weights of each class is obtained by the discriminator.   
9. ADVENT - adversarial entropy minimization: based on the entropy of the pixel-wise predictions.
   Model tends to produce over-confident predictions (like border) on source-like images and under-confident predictions (very noisy) on target-like ones. Therefore, enforcing high prediction certainty on target predictions as well:
   - direct entropy minimization. (independent pixel-wise prediction) Training on the 'hard' or 'most-confused' pixels produces better performance. No selection.
   - indirect entropy minimization using an adversarial loss. (aims at globally matching by weighted self-information, based on that fact the source and target domain share strong similarities in semantic layout)
   - Noted label statistics from source domain should be considered.
   - It is also useful for object detection with single-shot multibox detector. **GOOD WORK**
   
10. DADA - Depth-Aware Domain Adaptation in Semantic Segmentation.   
    Introducing additional depth-specific adaptation brings complementary effects to further bridge the performance between source and target at test time. Depth is an additional source domain supervision during training.   
    Therefore, we introduce a new depth-aware adversarial training protocol based on fusion of the network outputs, different depth levels should be treated differently. Using depth information to fuse the entropy of the pixel-wise predictions to help ADVENT.
    

11. Self-Ensembling with GAN-based Data Augmentation: heavily-tuned manual data augmentation used in self-ensembling is not useful to reduce the large domain gap in the semantic segmentation. GAN augmentation + self-ensembling.    
    Cycle-consistency has two limitations: needs redundant modules; too strong when target data are scarce.   
    Cycle-free data augmentation: Xg = G(Xs, Xt)  Xt provide style information based on Adaptive Instance Normalization with semantic constraint.   
    Self-ensembling: EMA, mean square loss for the probability maps which is after the softmax layer. Gaussian noise + Network perturbation.
    
12. Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision  
    Most previous works directly adapt models from the source data to the unlabeled target domain by reducing the **inter-domain gap**. These techniques do not consider the large distribution gap among the target data itself (intra-domain gap). In this work, there are two-step self-supervised domain adaptations to minimize the inter-domain and intra-domain gaps together.   
    Intra-domain gap: some target samples are easy, while the others are difficult (which exist large domain gap). In our model, there is an entropy-based ranking system to separate target data into the easy and hard split, and using pseudo labels from the easy subdomain.   
    It is a TwoStage-AdventNet (using entropy-based ranking to separate target domain into easy and hard splits as a new domain adaptation task). Rank score is computed by the sum of entropy mapping.
    
13. Adaptation across extreme variations using unlabeled bridges: introducing unlabeled bridging domains that connect the source and target domains. Can we automatically obtain the bridging domains by spliting the target domain as done in 12.

14. Confidence Regularized Self-Training (CRST)   
    A predominant stream of adversarial learning based UDA methods were proposed to reudce the discrepancy between source and target domain features. 1) self-training without confidence regularization: retrains with hard pseudo-labels (output is sharp which is unexpected in some case refineryNet); 2) Label regularized self-training introduces soft pseudo-labels (output is smooth which attenuates the misleading effect); 3) Model regualarized self-training: retrain with hard pseudo-labels, but incorporates a regularizer to directly promote output smoothness.    
    At high-level, the major goal of CRST is still aligned with entropy minimization, the confidence regularization serves as a safety measure to prevent infinite entropy minimization and degraded performance.    
    The pseudo-label are treated as discrete learnable latent variables being either one-hot or all-zero (not selected one). For each class k, lambda_k is determinied by the confidence value selecting the most confident p protion of class k predictions in the entire target set.   
    The label regularizer is related to model and pseudo labels
   
15. Unsupervised Scene Adaptation with Memory Regularization in vivo   
    Existing methods focus on minoring the inter-domain gap between source and target domains. However, the intra-domain knowledge and inherent uncertainty learned by the network are under-explored.   
    Memory regularization: segmentation model as the memory module and minor the discrepancy of the two classifiers (primary classifier and the auxiliary classifier), to reduce the prediction inconsistency.   
    Using the auxiliary classifier to pinpoint the intra-domain uncertainty. The predictions of the source domain input are relatively consistent, the unlabeled input from the target domain suffers from the uncertain prediction. The model provides different class predictions for the same pixel, implies the intra-domain consistency is under-explored.
    **Consistency loss three formats:** 1) perturbation on the input (data augmentation); 2) perturbation on the output (mutual learning); 3) perturbation on the networks (dropput, noise to weights and gradient, or *auxiliary heads*): we could not provide a right class prediction, it is more likely to act as a teacher providing the class distribution based on the historical experience.    
    Domain adaptation can be conducted on different levels, such as pixel level, feature level and semantic level. The brute-force alignment drives the model to learn the domain agnostic shared features of both domains. This line is sub-optimal since it ignores the domain-specific feature learning on the target domain (FOCUS on the target domain)  
    **Memory-based methods:** 1) MA-DNN applies an extra memory module to save the class prediction while training. 2) Mean teacher and mutual learning: apply one external model to memorize predictions and regularize the training. 3) We leverage the running network itself as the memory model.
    Mining the target domain knowledge: pseudo-label (select samples based on prediction model or discriminator), consistency loss, split the target domain into easy and hard split.   
    The pseudo labels contain noise, the consistency regularization could prevent the model from overfitting to the noise in pseudo labels. In stage I: domain-agnostic learning; In stage II: domain-specific learning. Compared to CBST, we do not introduce any threshold.
    
16. An Adversarial Perturbation Oriented Domain Adaptation Approach for Semantic Segmentation   
    Aligning the distribution across two domains globally fails in adapting the representations of the tail classes or small objects since the alignment is dominated by head categories or large objects.   
    We propose to perturb the intermediate feature maps with several attack objectives on each individual position for both domains, the classifier is trained to be invariant to the perturbations. Perform perturbation on feature space. 
    
17. Adapting to Changing Environments for Semantic Segmentation (ACE): dynamically adapts to changing environments over time. By aligning the distribution of labeled training data from the original domain with the distribution of incoming data in the target domain. (This is lift-long learning)   
    'lifelong learning' is non-trivial for DNN as 1) new data domains come in at real time without labels; and 2) deep networks suffer from catastrophic forgetting. We consider adapting the pre-trained model to dynamically changing environments whose distributions reflect disparate lighting and weather conditions. We consider the difficulties posed by learning over time, in which target environments appear sequentially.   
    Instead of using GAN, ACE saves a memory bank (feature statistics) which can transfer the contents of image from source task but in the style of a target task. Style transfer is achieved by renormalizing feature maps of source images so they have first- and second-order feature statistics. The light-weight memory statistics can be saved without the burden of storing a library of historical images. To avoid forgetting knowledge from past environments, we introduce a memory that stores feature statistics from previously seen domains. These statistics can be used to replay images in any of the previously domains, thus preventing catastrophic forgetting.   
    Encoder is vgg19 (fixed), decoder acts as an inverse for the encoder. Encoder should map the decoded image onto the features that produced it.

18. Self-Correction for Human Parsing: to progressively promote the reliability of the supervised labels as well as the learned models. (Extend this writing style for improved SUIT: first improve the generation, second improve with the self-training)    
    Starting from a model trained with inaccurate annotations as initialization, then iteratively aggregating the current learned model with the former one in an online manner; the corresponding corrected labels can in turn to further boost the performance.   
    Reference: CE2P is a well-performing framework for the human parsing task that joinly trains the parsing and edge detection. For parsing: cross-entropy + lovasz-softmax; For edge detection: balanced cross-entropy; With additional consistency loss: parsing results matches the predicted edge.    
    Online model aggregation: aggregate the model during each training cycle (learning rate cyclical period), e.g., Stochastic weight average   
    Online label refinement: update the ground truth of training labels (Temporal ensemble for the label prediction)
 
19. Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation: using Random AdaIN to generate more and more aggressive samples

20. Constructing Self-motivated Pyramid Curriculums for Cross-Domain Semantic Segmentation: A Non-Adversarial Approach
    

17. Domain Randomization and Pyramid Consistency - Generalization without Accessing Target Domain Data.


# Semantic Part Segmentation
segment parts within an object as opposed to objects within a scene as in semantic segmentation. This is more challenging because two parts sometimes do not have a visible boundary between them. Segment the same object, which has the same multiple parts, e.g., car object (with many components) or face object (with many components)

1. Repurposing GANs for One-shot Semantic Part Segmentation   
2. Scops: Self-supervised co-part segmentation   


# Distillation or self-training
1. Data Distillation: Towards Omni-Supervised Learning. 
2. Lessons from building acoustic models with a million hours of speech  
3. Automatic adaptation of object detectors to new domains using self-training  
4. Are labels required for improving adversarial robustness? 
5. Learning to self-train for semi-supervised few-shot classification  
6. Ensemble, distill, and fuse for easy video labeling


# Training or Architecture Improvement
1. Fixing the train-test resolution discrepancy  
2. Making Convolutional Networks Shift-Invariant Again
3. Gated Channel Transformation for Visual Recognition 

# Fine-grainded recognition
1. Local Temporal Bilinear Pooling for Fine-Grained Action Parsing. https://medium.com/syncedreview/local-temporal-bilinear-pooling-for-fine-grained-action-parsing-24a9bdcc1355


# Semi-supervised learning
1. Consistency-based: ReMixMatch (distribution alignment: prevent the model's prediction from collapsing) and FixMatch (new sota: use weakly-augmented samples to produce pseudo-labels whose largest class probability are above a threshold and train the model on strongly-augmented samples)
2. Graph-based (label propagation or use mean-teacher to obtain more reliable feature to construct graph):

   a. Low-shot learning with large-scale diffusion
   
   b. Transductive Semi-Supervised Deep Learning using Min-Max Features
   
   c. Label propagation for deep semi-supervised learning
   
   d. Smooth Neighbors on Teacher Graphs for Semi-supervised Learning
   
   e. Data-efficient semi-supervised learning by reliable edge mining
   
3. CoMatch (Contrastive Graph Regularization, jointly learning class probabilities from classification head and image representations from self-supervised learning head): unify consistency regularization, entropy minimization, contrastive learning, graph-based SSL.

   a. Pseudo-labeling heavily rely on the quality of model's prediction, where the prediction mistakes would accumulate.
   
   b. Self-supervised learning are task-agnostic, the representations may be suboptimal.
   
   c. Previous graph-based methods built graph on the features which are highly-correlated with the class predictions; due to high dimensionality, Euclidean distance becomes less meaningful (CoMatch considers the relative distance); Computation cost is high.
   
   d. In CoMatch, the self-supervised branch and classification branch are collaborative with each other, rather than separated parts used in many multi-tasks learning.
   
   e. Procedure: first, use weakly-augmentations to produce pseudo-graph, then use the pseudo-graph as target to training the embedding graph, which measures the similarity of strongly-augmented samples in the embedding space.

   
# Self-supervised learning
1. Self-supervised Representation Learning using 360° Data
2. Prototypical Contrastive Learning of Unsupervised Representations
3. Unsupervised Learning of View-invariant Action Representations
4. Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning
5. MEAL V2: Boosting Vanilla ResNet-50 to 80%+ Top-1 Accuracy on ImageNet without Tricks
   
# Noise label learning
1. A Simple yet Effective Baseline for Robust Deep Learning with Noisy Labels 

   Relation to semi-supervised learning: noise label will hurt training while unlabeled data will not
   
   Relation to deep kernel learning: Minimizing the predictive variance can be explained in the framework of posterior regularization, which lead to the solution to be of some specific form. Training samples are far from the decision boundaries and tend to cluster together.
   
   Local intrinsic dimensionality (LID) of adversarial example is significantly higher than that of normal data. Ref: Characterizing adversarial subspaces using local intrinsic dimensionality
   
2. Bootstrap-hard:  a self-learning technique that use a convex combination of the given label y and model prediction y_hat as the training target.
3. Forward-correction:  a loss correction method based on the noise transition matrix Tˆ estimated by a pre-trained network.
4. Generalized Cross Entropy: noise-robust loss function.
5. Co-teaching: maintains two networks simultaneously, and cross-trains on instances screened by the “small loss” criteria.
6. MentorNet: a meta-learning model that assigns different weights of training examples based on metalearned curriculum with clean validation dataset.
7. Learning to Reweight: reweight the samples based on clean validation set.
8. AugMix: mixup among k different augmentation.
9. Training deep neural-networks using a noise adaptation layer.
10. Co-Mining: Deep Face Recognition with Noisy Labels. (Mutual learning with two nets)
11. O2U-Net: adjust hyperparameters (learning rate) from Overfitting to Underfitting. The noise labels are usually memorized at the late stage of training as the "hard" samples. Therefore, by tracking the variation of loss of very sample at the different stages, it is possible to detect noise labels.
12. A Novel Self-Supervised Re-labeling: self-supervision to generate robust features; using a parallel network to re-label the noise data.
13. SELF: Learning to filter noisy labels with self-ensemble. Filter noise samples with moving average networks (the average prediction is not consistent with the given label BEFORE over-fitting to the dataset), the filtered samples will be used as unlabeled samples in semi-supervised learning.
14. DivideMix: Learning with noisy labels as semi-supervised learning. Use GMM on per-sample loss to filter noise data as unlabeled data. Next, use MixMatch/co-refinement/co-guessing to train network.  
15. Confident Learning: Estimating Uncertainty in Dataset Labels  
16. Can gradient clipping mitigate label noise.
    Informally, clipping controls the dynamics of iterates, thus enhancing the rate of convergence to a local minimum. (Capping the global gradient norm at a specified threshold). 
    From optimisation lens: By ensuring that the gradient norm is not too large, the dynamics of iterates are more well-behaved.
    
    From privacy lens: it prevents any single instance from overly influencing parameter updates.
    
    In this paper, we study gradient clipping from *robustness* lens. Intuitively, clipping the gradient prevents over-confident descent steps.
    
    a. gradient clipping alone does NOT endow label noise robustness.
    
    b. The composite loss-based gradient clippling, a variant that have label noise robustness. ONLY clip the contribution of the *base loss*, being partially Huberised loss.
    
17. CURRICULUM LOSS: ROBUST LEARNING AND GENERALIZATION AGAINST LABEL CORRUPTION
    
18. SIMPLE AND EFFECTIVE REGULARIZATION METHODS FOR TRAINING ON NOISILY LABELED DATA WITH GENERALIZATION GUARANTEE  (many useful ideas in the related works) [code](https://link.zhihu.com/?target=https%3A//drive.google.com/drive/folders/1TDlUuL0I-EzIybjz2pMAgyaYP5F6dq6o)
    
    This paper proposes and analyzes two simple and intuitive regularization methods, the generalization bound is independent of the network size, and the comparable to the bound one can tget when there is no label noise.
    
    a. regularization by the distance between the network parameters to initialization--L2SP. THE ROLE OF OVER-PARAMETRIZATION IN GENERALIZATION OF NEURAL NETWORKS.
    
    b. adding a trainable auxiliary variable to the network output for each training example: tries to fit y_i = f(x_i) + \lambda\*b_i. b_i is a trainable variable. The effectiveness of early stopping indicates that clean labels are somewhat easier to fit than wrong labels; therefore, adding an auxiliary variable could help “absorb” the noise in the labels.
    
    The ability to over-fit the entire training dataset is undesirable for generalization when noisy labels are present. In order to prevent over-fitting to mislabeled data, some form of regularization is necessary, i.e., reduce the *effective size* of the model or *early stopping* (but the validation error may up-down many times, it is trade-off between model precision and training time;)
    
    Gradient Descent with Early Stopping is Provably Robust to Label Noise for Neural Networks: shows under a clustering assumption on data, *gradient descent fits the correct labels before starting to over-fit wrong labels*.
    
    In this line of work (wide neural net and neural tangent kernel), parameters in a wide NN are shown to stay close to their initialization during SGD.
    
19. Learning to Learn from Noisy Labeled Data.     
    A meta-learning update is performed prior to conventional gradient update. Meta-learning based noise-tolerant training to learn from noisy labeled data without human supervision or access to any clean labels. Rather than designing a specific model, we propose a model-agnositc training algorithm.     
    The key idea is *a noise-tolerant model should be able to consistently learn the underlying knowledge from data despite different label noise.*     
    Meta-learning: learning update rule of a learner; finding weight initializations that can be easily fine-tuned or transferred. MAML trains model parameters that can learn well based on a few examples and a few gradient descent steps.     
    
    For SGD, some network parameters are more tolerant to label noise than others.    
    a. for each mini-batch, we generate a variety of synthetic noisy labels on the same images, we update the network parameters using one gradient update.     
    b. Then, we enforce the updated network to give consistent predictions with a teacher model unaffected by the synthetic noise.
    
20. Learning from Noisy Labels with Distillation     
    Use an auxiliary model (trained on small clean dataset as guidance), or use distillation guided by knowledge graph.    
    
21. Learning from noisy large-scale datasets with minimal supervision
22. Learning with Bad Training Data via Iterative Trimmed Loss Minimization
    
    It is useful in poorly curated datasets, irrelevant samples (out-of-distribution), as well as backdoor attacks.
    
    Based on the observations: the evolution of training accuracy is different for clean and bad samples. Repeat (a) selecting samples with lowest current loss. (b) retraining a model on only these samples.

23. Gradient regularization improves accuracy of discrimative models
24. Robust learning with jacobian regularization
25. CleanNet: Transfer learning for scalable image classification training with label noise: use one 'prototype' or 'representative sample' to represent one class, use this prototype to identify if it is noise or not.    
26. Symmetric Cross Entropy for Robust Learning with Noisy Labels     
    DNN with cross entropy exhibits overfitting to noisy labels on some 'easy' classes, while suffers on some other 'hard' classes. **class-biased: some classes overfitting, some class underfitting**. In addition, a low test accuracy (underfitting) on hard classes is a barrier to high overall accuracy (poor performance is not only caused by overfitting).       
    Dimensionality-driven: DNN first learns simple representation via dimensionality compression and then overfit to noise labels via dimensionality expansion.       
    CE by itself is not sufficient for learning of hard classes, especially under the noisy label scenario. Underfitting on hard classes is a major cause for overall performance degradation, due to the fact that the accuracy drop caused by overfitting is relatively small.     
    Symmetric Cross Entropy (SCE) = H(q, p) + H(p, q). Use the predictions as labels: Reverse Cross Entropy.
    
27. Robust training with ensemble consensus (sampling trainable examples from a noisy dataset by relying on small-loss criteria might be impractical)

    *DNN can not generalize to __neighborhoods__ of memorized features*, we hypothesize that noisy examples do not consistently incur small losses on the network under a certain perturbation. Training losses of noisy examples would increase by injecting certain perturbation to network parameters, while those of clean examples would not.
    
    Multiple consensus: M networks generated by adding perturbations
    
28. Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization.

    'Decoupling' and 'Co-teaching+' claim that the 'disagreement' strategy is crucial for alleviating the problem of learning with noisy labels.     
    Differences between 'co-teaching' and 'co-training': co-training uses two different views; co-training did not use the memorization property of DNN (the small loss for one network may be large for the other network).   
    
    a. Decoupling: select not consistent samples to update networks. Update by disagreement: S={(x,y): h1(x) != h2(x)}, update h1 with S and update h2 with S. Similar to hard-mining? Only hard mining is not enough.    
    b. Coteaching: use one network to select 'small-loss' samples to update the other network, and vice versa.    
    c. Coteaching+: 'Coteaching+Decoupling' since coteaching will make two prediction closer with the epochs increasing. (How does Disagreement Help Generalization against Label Corruption?)   

29. Using Pre-Training Can Improve Model Robustness and Uncertainty    
    Pre-training may not improve performance on traditional classification metrics, it improves model robustness and uncertainty estimates. Some researchers claims that the pre-training only imporves wall-clock time.    
    a. Pre-training does tremendously improve the model's adversarial robustness.     
    b. Training longer on a corrupted dataset leads to model deterioration, while for pre-trained model, this is not the case.    
    c. Pre-training can reduce overfitting in case of model calibration, rather than model accuracy.     
    
    Uncertainty estimates need to be useful for detecting out-of-distribution samples.    

30. Label-Noise Robust Generative Adversarial Networks. Noisy learning in the generative model.   
31. Robustness of Conditional GANs to Noisy Labels    
32. Self-Adaptive Training Bridging the Supervised and Self-Supervised learning (EMA for both the target prediction and the network parameters)


    
    
