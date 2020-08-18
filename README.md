# EAE
An Experimental Study of State-of-the-Art Entity Alignment Approaches


### Surveyed methods
The repos of the methods discussed in this paper can be found in the following. 

1. [MTransE](https://github.com/muhaochen/MTransE-tf) (IJCAI 2017): Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment
2. [JAPE-Stru/JAPE](https://github.com/nju-websoft/JAPE) (ISWC 2017): Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding <!--: The results of JAPE-Stru are marked with "SE"-->
3. [GCN/GCN-Align](https://github.com/1049451037/GCN-Align) (EMNLP 2018): Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks <!--: The first results in each file denote GCN, and the second one denote GCN-Align-->
4. [RSNs](https://github.com/nju-websoft/RSN) (ICML 2019): Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs
5. [MuGNN](https://github.com/thunlp/MuGNN) (ACL 2019): Multi-Channel Graph Neural Network for Entity Alignment
6. [KECG](https://github.com/THU-KEG/KECG) (EMNLP 2019): Semi-supervised Entity Alignment via Joint Knowledge Embedding Model and Cross-graph Model
7. [ItransE](https://github.com/thunlp/IEAJKE) (IJCAI 2017): Iterative Entity Alignment via Joint Knowledge Embeddings
8. [BootEA](https://github.com/nju-websoft/BootEA) (IJCAI 2018): Bootstrapping Entity Alignment with Knowledge Graph Embedding
9. NAEA (IJCAI 2019): Neighborhood-Aware Attentional Representation for Multilingual Knowledge Graphs. (The codes are not publicly available yet, but one could ask the authors for a preliminary version)
10. [TransEdge](https://github.com/nju-websoft/TransEdge) (ISWC 2019) : TransEdge: Translating Relation-Contextualized Embeddings for Knowledge Graphs
11. [HMAN](https://github.com/h324yang/HMAN) (EMNLP 2019): Aligning Cross-lingual Entities with Multi-Aspect Information
12. [GM-Align](https://github.com/syxu828/Crosslingula-KG-Matching) (ACL 2019): Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network
13. [RDGCN](https://github.com/StephanieWyt/RDGCN) (IJCAI 2019): Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs
14. [HGCN](https://github.com/StephanieWyt/HGCN-JE-JR) (EMNLP 2019): Jointly Learning Entity and Relation Representations for Entity Alignment
15. [CEA](https://github.com/DexterZeng/CEA)  (ICDE 2020): Collective Entity Alignment via Adaptive Features

The log files of our implementatin can be found in the **logs** directory. 

### Dataset
The new mono-lingual dataset can be found in the **dataset** directory

### A simple approach combining exsiting modules
We also offer a solution that combines the modules in exsiting methods, which can achieve competitive performance.


### Aknowledgement
We thank the authors of aforementioned papers for their great works and for making the source codes publicly available.