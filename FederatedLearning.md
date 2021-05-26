# Towards Personalized Federated Learning

Pioneering FL: Federated Averaging (FedAvg). The training data is stored locally and is not shared during the trainig process. 
The goal is to train a model that performs well on most FL clients.

Most models are designed to fit the "average client", which may not perform well in the presence of statistical data heterogenity. 
Personalized federated learning (PFL) can be viewed as an intermediate paradigm between *server-based FL* (for global model) and the 
*local model training paradigm*. 

The challenge of PFL is to strike a careful balance between local task-specific knowledge and shared knowledge useful for the generalization.
In this paper, we bridge this gap by offering a unique data-based vs. model-based perspective for reviewing the PFL literature.

**Limitation of the Prevailing FL:** central parameter server-based FL is not suitable in the persence of statistical data heterogeneity. 
Performance is unstable and degraded. To build PFL models, alternatives to the central parameter server-based FL architecture are emerging.

**Privacy-Preservation Constraints:** privacy concerns are not adequately addressed. Without explicit data sharing, it is challenging to understand
 the extent of heterogeneity among the clients' datasets. Some works relax the privacy by allowing some data or metadata shared with the parameter server.
 
 Data-based approaches aim to smooth the statistical heterogeneity of data residing at participating clients. 
 These include data augmentation and client selection. 
 Client selection proposes to select a subset of participating clients for each training round to mitigate the bias.
 
 Model-based approaches:   
 **Single-model PFL**: a global FL model is trained and adapted on local data. Personalization is achieved by finding a more optimal initialization
 of the global model for local personalization, or by learning more desirable task-specific local representations. 
 _These approaches are best suited when the local data distributions do not deviate significantly from the global data._
 
 Meta-learning: optimization-based meta-learning (i.e., Model-Agnostic Meta-Learning) are known for their good generalization and fast adaptation on new heterogeneous tasks.
 Meta-learning runs in two phases: meta-training and meta-testing. Meta-training used for FL global model training, meta-testing use for FL personalization. 
 _instead of training global model that performs well on most clients, Meta-learning based FL learns a good initial global model that performs well on a new heterogenerous task_. 
 
 Regularization (tackle the weight divergence problem in FL setting): proximal term to the local subproblem; parameter importance (Elastic weight consolidation) is estimated and penalization is used
 to perserve important parameters. In addition, for personalization, modeling an attentive collaboration to enforce stronger collaboration amongest FL clients with similar data distributions.
 
 Parameter Decoupling (Private Embedding): decoupling the local private model parameters from the global FL model. Private parameters are trained locally and not shared with parameter server.
 
 
 **Multi-model PFL**: a single model is not effective when data distributions are significantly different.  
 clustering: there exists a natural grouping of clients' local datasets.  Hierarchical clustering framework for FL.
 
 
 **N-Model FPL**: a model is learnt for each individual client in the federation.  
 *Multi-task Learning*: training each FL client as a task in MTL.  
 *Model Interpolation*: mixture of global and local models. Each FL client learns an individual local models, a penalty parameter is used to discourage the local model from beding too dissimilar from the mean model.  
 *Transfer Learning*: training local models; training a global model via FL; training personalized models by integrating the global and local models via transfer learning.   
 *Parameter Decoupling*: base + personalized layers design for DNN. Personalized layers are kept private at the clients, base layers are shared with FL server.
 
 
 
 
