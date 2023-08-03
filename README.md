# Awesome-List-Protein-Binding-Site-Prediction
List of papers on protein binding site prediction

## GO and EC prediction

### Structure-based protein function prediction using graph convolutional networks
This paper presents DeepFRI, a novel method for predicting protein function using graph convolutional networks. The authors demonstrate that DeepFRI outperforms existing methods for protein function prediction, and is able to predict functions for a large number of proteins with high accuracy and resolution. The method leverages sequence features from pretrained language model and protein structures, and is able to expand the number of predictable functions using homology models. Additionally, DeepFRI enables site-specific annotations at the residue-level using grad-CAM, and has been shown to be useful in annotating structures from the PDB and SWISS-MODEL.
The data used in this study consists of protein structures from the Protein Data Bank (PDB) and homology models from the SWISS-MODEL repository. The authors created a non-redundant set of PDB chains by clustering all PDB chains with blastclust at 95% sequence identity, and selecting a representative PDB chain that is annotated and of high quality from each cluster. They also included annotated SWISS-MODEL chains in their training procedure, and removed similar SWISS-MODEL sequences at 95% sequence identity. Including SWISS-MODEL models led to a 5-fold increase in the number of training samples and a larger coverage of more specific Gene Ontology (GO) terms. 

**Highlight: Use grad-CAM for functional site prediction, which is unsupervised**

### Discovering functionally important sites in proteins
This paper presents a machine learning method that combines statistical models for protein sequences with biophysical models of stability to predict functionally important sites in proteins. The model is trained using experimental data on variant effects and can be used to discover active sites, regulatory sites, and binding sites. The utility of the model is demonstrated through prospective prediction and experimental validation on the functional consequences of missense variants in HPRT1, which may cause Lesch-Nyhan syndrome. The paper addresses the challenge of distinguishing the effects of amino acid substitutions on intrinsic function from their effects on stability and cellular abundance by using multiplexed assays of variant effects. The goal is to pinpoint the molecular mechanisms underlying perturbed function and identify residues that directly contribute to the function.
the data used in this study includes experimental data on variant effects, specifically missense variants in the HPRT1 protein. The authors used multiplexed assays of variant effects (MAVEs) to probe the consequences of individual substitutions on both function and abundance. The data used for training the machine learning model and the predictions made in this study are available at the GitHub repository: https://github.com/KULL-Centre/_2022_functional-sites-cagiada.


## Protein catalytic site prediction

### Database

#### Mechanism and Catalytic Site Atlas
M-CSA is a database of enzyme reaction mechanisms. It provides annotation on the protein, catalytic residues, cofactors, and the reaction mechanisms of hundreds of enzymes.
M-CSA contains 1003 hand-curated entries.

### EXIA2: Web Server of Accurate and Rapid Protein Catalytic Residue Prediction
The paper describes the EXIA2 Web Server, a method for predicting catalytic residues in proteins based on their special side chain orientation. The paper uses six benchmark datasets to evaluate the performance of the EXIA2 Web Server for predicting catalytic residues in proteins. The datasets include PW79, POOL160, EF fold, EF superfamily, EF family, and P100, which include over 1,200 proteins and 861,404 residues (3,664 catalytic residues and 857,740 noncatalytic residues). The definition of catalytic residues is based on Catalytic Site Atlas version 2.2.12. The paper compares the prediction performance of EXIA2 with that of three state-of-the-art prediction methods, including CRpred, POOL, ConSurf, and ResBoost, on the PW79 and POOL160 datasets. The performance is evaluated using recall (R), precision (P), and area under ROC curve (AUCROC). 

### Protein structure based prediction of catalytic residues

The methods described in the paper Protein structure based prediction of catalytic residues aim to predict functional residues in protein structures using a neural network approach. The authors explore a range of features based on protein structure, including distance to the centroid and amino acid type, combined with sequence conservation. The neural network is trained using a supervised, feed-forward approach with one hidden layer of ten units. The performance of the method is evaluated using various performance measures, including sensitivity, precision, and the F1-measure. The results show that the proposed method achieves comparable performance to other state-of-the-art methods. The benchmarks and baselines used in the paper are not explicitly mentioned in the provided information. However, the authors mention comparing their method to other existing methods, such as CRpred and the method of Youn et al., which rely on sequence conservation and structural information. The performance of the proposed method is evaluated using different datasets, including a training set and an independent test set.




## Protein binding site prediction
Protein–protein interaction sites prediction by ensemble random forests with synthetic minority oversampling technique 


