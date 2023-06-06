# KMIC
This is our Pytorch implementation for the paper:
KMIC: A Knowledge-aware Recommendation with Multivariate Intentions Contrastive Learning, (underReview).

# Introduction
Knowledge graph (KG) has been widely applied for recommendation systems (RS) due to its rich structured semantic information. A recent technical trend is to develop graph neural networks with self-supervised Learning (SSL) founded on KG to relieve the data sparsity and noise issues. However, these models fail to (i) integrate the multivariate intention feature and (ii) capture the multi-intent self-supervised signals.

In this paper, we propose a novel knowledge-aware recommendation with multivariate intentions contrastive learning (KMIC) to alleviate the above weakness. Our approach aims to fuse multivariate intention features behind the intrinsic interaction behaviors and enhance the representation ability of RS through multi-intent selfsupervised signals. Technically, we devise a multivariate intentions awareness module to learn history intent, contextual intent, and future latent intent. It effectively models the multivariate intent as attentive learning of interaction relations, which encourages the personalization awareness of different intents for better model capability. Besides, we design a multi-intents contrastive learning module, which mines comprehensive intention information among user, item, and interaction sequence. Extensive experiments conducted on three real-world datasets demonstrate the superiority of our model over existing state-of-the-art methods. 

# Requirement
pytorch==1.10.1

numpy==1.21.6

scikit-learn==1.0.2

# Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes.

- Train and Test

```
 python kmic_main.py 
```

# Dataset
We provide three processed datasets: Book-Crossing, MIND, and Last.FM.

|                       |               | Last.FM | MIND      | Book-Crossing |
| --------------------- | ------------- | ------- | --------- | ------------- |
| User-Item Interaction | #Users1,872   | 1,872   | 299,995   | 17,860        |
|                       | #Items        | 3,846   | 47,034    | 14,967        |
|                       | #Interactions | 42,346  | 5,090,652 | 139,746       |
| Knowledge Graph       | #Entities     | 9,366   | 57,434    | 77,903        |
|                       | #Relations    | 60      | 62        | 25            |
|                       | #Triplets     | 15,518  | 746,270   | 151,500       |
