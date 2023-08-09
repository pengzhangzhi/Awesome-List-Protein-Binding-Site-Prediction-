# Awesome-List-Protein-Binding-Site-Prediction
List of papers on protein binding site prediction


### Database

#### Mechanism and Catalytic Site Atlas

<details>
<summary>summary</summary>
M-CSA is a database of enzyme reaction mechanisms. It annotates the protein, catalytic residues, cofactors, and the reaction mechanisms of hundreds of enzymes.
M-CSA contains 1003 hand-curated entries.
</details>


#### PDBbind database
<details>
<summary>summary</summary>

Contains ~13000 complex strcuctures formed between protein-small molecule ligand, protein-protein, protein-nucleic acid and nucleic acid-small molecule ligand. 
Binding affinity data and structural information for a total of 12,995 biomolecular complexes, including protein-ligand (10656), nucleic acid-ligand (87), protein-nucleic acid (660), and protein-protein complexes (1592), which is the largest collection of this kind so far.
</details>

#### BioLiP database
<details>
<summary>summary</summary>
BioLiP is a semi-manually curated database for high-quality, biologically relevant ligand-protein binding interactions. The structure data are collected primarily from the Protein Data Bank (PDB), with biological insights mined from literature and other specific databases. BioLiP aims to construct the most comprehensive and accurate database for serving the needs of ligand-protein docking, virtual ligand screening and protein function annotation. Questions about the BioLiP Database can be posted at the Service System Discussion Board. Since ligand molecules (e.g., Glycerol, Ethylene glycol) are often used as additives (i.e., false positives) for solving the protein structures, not all ligands present in the PDB database are biologically relevant. BioLiP uses a composite automated and manual procedure for examining the biological relevance of ligands in the PDB database. Each entry in BioLiP contains a comprehensive list of annotations on: ligand-binding residues; ligand binding affinity (from the original literature, plus Binding MOAD, PDBbind-CN, BindingDB); catalytic site residues (mapped from Mechanism and Catalytic Site Atlas); Enzyme Commission (EC) numbers and Gene Ontology (GO) terms mapped by the SIFTS database; crosslinks to external databases, including RCSB PDB, PDBe, PDBj, PDBsum, Binding MOAD, PDBbind-CN, Mechanism and Catalytic Site Atlas, QuickGO, ExPASy ENZYME, ChEMBL, DrugBank, ZINC, UniProt, PubMed.
</details>


## General binding site prediction
#### Krapp, L.F., Abriata, L.A., Cortés Rodriguez, F. et al. PeSTo: parameter-free geometric deep learning for accurate prediction of protein binding interfaces. Nat Commun 14, 2175 (2023). https://doi.org/10.1038/s41467-023-37701-8
<details>
<summary>summary</summary>
Given the protein structure, the model predicts the binding site of nucleic acids, lipids, ions, and small molecules.
The dataset is composed of all the biological assemblies from the Protein Data Bank.
The method was compared with ScanNet, MaSIF-site, SPPIDER35 and PSIVER on protein-binding site prediction. 
The testing set is composed of the clusters containing any of the 53 subunits from the MaSIF-site benchmark dataset or 230 structures from the Protein-Protein Docking Benchmark 5.038 (PPDB5) dataset. Additionally, we extracted a subset 417 structures common in the benchmark dataset of ScanNet15 and the testing dataset of PeSTo. 
</details>

## EC prediction

#### Zhenkun Shi Rui Deng Qianqian Yuan Zhitao Mao Ruoyu Wang Haoran Li Xiaoping Liao Hongwu Ma. . Enzyme Commission Number Prediction and Benchmarking with Hierarchical Dual-core Multitask Learning Framework. Research. 2023:6;0153. DOI:10.34133/research.0153
<details>
<summary>summary</summary>
propose three enzyme-related tasks.
</details>

#### Tianhao Yu et al. ,Enzyme function prediction using contrastive learning.Science379,1358-1363(2023).DOI:10.1126/science.adf2465
<details>
<summary>summary</summary>

</details>


## GO and EC prediction

#### Gligorijević, V., Renfrew, P. D., Kosciolek, T., Leman, J. K., Berenberg, D., Vatanen, T., Chandler, C., Taylor, B. C., Fisk, I. M., Vlamakis, H., Xavier, R. J., Knight, R., Cho, K., & Bonneau, R. (2021). Structure-based protein function prediction using graph convolutional networks. Nature Communications, 12(1), 1-14. https://doi.org/10.1038/s41467-021-23303-9
<details>
<summary>summary</summary>

The paper introduces DeepFRI, a Graph Convolutional Network for predicting protein functions by leveraging sequence features extracted from a protein language model and protein structures. It outperforms current leading methods and sequence-based Convolutional Neural Networks and scales to the size of current sequence repositories. DeepFRI has significant de-noising capability, with only a minor drop in performance when protein models replace experimental structures. Class activation mapping allows function predictions at an unprecedented resolution, allowing site-specific annotations at the residue level in an automated manner. The authors show their method's utility and high performance by annotating structures from the PDB and SWISS-MODEL, making several new confident function predictions.
The paper compares DeepFRI with the following baselines:

- BLAST: a sequence alignment tool that uses the E-value as a measure of similarity1.
PSI-BLAST: an iterative version of BLAST that builds a position-specific scoring matrix (PSSM) from the initial hits and searches again1.
- ProtFun: a method that predicts protein functions from sequence-derived features such as amino acid composition, predicted secondary structure and solvent accessibility1.
- GOtcha: a method that predicts protein functions by transferring annotations from homologous proteins using a score that reflects the reliability of the match1.
- PFP: a method that predicts protein functions by using a k-nearest neighbor algorithm on a large database of annotated proteins1.
- ESG: a method that predicts protein functions by using an ensemble of SVM classifiers trained on different sequence features1.
- DeepGO: a deep convolutional neural network (CNN) that predicts protein functions from sequence features extracted by a bidirectional recurrent neural network (RNN)2.
- DeepText2GO: a deep CNN that predicts protein functions from text features extracted by a natural language processing model2.
- GOLabeler: a deep CNN that predicts protein functions from both sequence and text features2.
The authors used the following evaluation metrics:

- Fmax: the maximum F1 score over all possible thresholds.
- AUC-ROC: the area under the receiver operating characteristic (ROC) curve.
- AUC-PR: the area under the precision-recall (PR) curve.

The authors used two benchmark datasets to evaluate the performance of DeepFRI:
- The CAFA3 dataset: a dataset of 368 proteins with 1,000 GO terms from the third round of the Critical Assessment of Function -Annotation (CAFA) experiment.
- The PDB-SWISS-MODEL dataset: a dataset of 1,000 proteins with 2,000 GO terms from the Protein Data Bank (PDB) and SWISS-MODEL.

**Highlight: Use grad-CAM for functional site prediction, which is unsupervised**

</details>
  

#### Cagiada, M., Bottaro, S., Lindemose, S., Schenstrøm, S. M., & Stein, A. (2023). Discovering functionally important sites in proteins. Nature Communications, 14(1), 1-13. https://doi.org/10.1038/s41467-023-39909-0

<details>
<summary>summary</summary>

This paper presents a machine learning method that combines statistical models for protein sequences with biophysical models of stability to predict functionally important sites in proteins. The model is trained using experimental data on variant effects and can be used to discover active sites, regulatory sites, and binding sites. The utility of the model is demonstrated through prospective prediction and experimental validation on the functional consequences of missense variants in HPRT1, which may cause Lesch-Nyhan syndrome. The paper addresses the challenge of distinguishing the effects of amino acid substitutions on intrinsic function from their effects on stability and cellular abundance by using multiplexed assays of variant effects. The goal is to pinpoint the molecular mechanisms underlying perturbed function and identify residues that directly contribute to the function.
the data used in this study includes experimental data on variant effects, specifically missense variants in the HPRT1 protein. The authors used multiplexed assays of variant effects (MAVEs) to probe the consequences of individual substitutions on both function and abundance. The data used for training the machine learning model and the predictions made in this study are available at the GitHub repository: https://github.com/KULL-Centre/_2022_functional-sites-cagiada.

</details>



## Protein catalytic site prediction

#### Das S, Scholes HM, Sen N, Orengo C. CATH functional families predict functional sites in proteins. Bioinformatics. 2021 May 23;37(8):1099-1106. doi: 10.1093/bioinformatics/btaa937. PMID: 33135053; PMCID: PMC8150129.

<details>
<summary>summary</summary>


The paper presents a new method for predicting functional sites in proteins using features derived from protein sequence, structure and CATH functional families (FunFams)12. The main contributions of the paper are:

The paper introduces FunSite, a machine learning predictor that identifies catalytic, ligand-binding and protein–protein interaction functional sites using features derived from protein sequence and structure, and evolutionary data from CATH FunFams1.
The paper shows that FunSite outperforms other publicly available functional site prediction methods on a comprehensive benchmark dataset3. The paper also shows that the FunFam-based features are more informative and discriminating than the commonly used PSI-BLAST-based features for functional site prediction.
The paper demonstrates that FunSite can predict multiple types of functional sites for a given query protein and account for the possibility that a residue may have more than one functional role. The paper also provides insights into the characteristics and importance of different features for different types of functional sites.
The paper compares its proposed method, called FunSite, with several baselines that are also designed to identify catalytic, ligand-binding and protein–protein interaction functional sites. The baselines are:

S-SITE: A method that uses sequence profiles and structural information to predict functional sites1.
ConCavity: A method that combines evolutionary conservation with geometric properties of protein structures to identify binding pockets2.
COFACTOR: A method that uses a combination of sequence, structure and evolutionary information to predict enzyme active sites and ligand-binding sites3.
FINDSITE: A method that uses a threading-based approach to transfer functional annotations from known protein structures to unknown ones4.
3DLigandSite: A method that uses structural alignments of homologous proteins to predict ligand-binding sites.
These baselines represent different types of approaches for functional site prediction, such as sequence-based, structure-based, evolution-based and threading-based methods. The paper shows that FunSite outperforms all these baselines on a comprehensive benchmark dataset.

The paper uses two benchmark datasets to evaluate the performance of FunSite and the baselines. The first dataset is called FunFam-40, which contains 40 protein families with known functional sites from the CATH database. The second dataset is called FunSite-40, which contains 40 proteins with known functional sites from the PDB database. The paper shows that FunSite outperforms all the baselines on both datasets.

The paper uses several evaluation metrics to compare the performance of FunSite and the baselines. These metrics include sensitivity, specificity, precision, F1 score and area under the receiver operating characteristic curve (AUC-ROC). The paper also uses Matthews correlation coefficient (MCC) as an additional metric to evaluate the overall performance of FunSite and the baselines.


</details>


#### Lu CH, Yu CS, Chien YT, Huang SW. EXIA2: web server of accurate and rapid protein catalytic residue prediction. Biomed Res Int. 2014;2014:807839. doi: 10.1155/2014/807839. Epub 2014 Sep 11. PMID: 25295274; PMCID: PMC4177735.

<details>
<summary>summary</summary>
The paper describes the EXIA2 Web Server, a method for predicting catalytic residues in proteins based on their special side chain orientation. The paper uses six benchmark datasets to evaluate the performance of the EXIA2 Web Server for predicting catalytic residues in proteins. The datasets include PW79, POOL160, EF fold, EF superfamily, EF family, and P100, which include over 1,200 proteins and 861,404 residues (3,664 catalytic residues and 857,740 noncatalytic residues). The definition of catalytic residues is based on Catalytic Site Atlas version 2.2.12. The paper compares the prediction performance of EXIA2 with that of three state-of-the-art prediction methods, including CRpred, POOL, ConSurf, and ResBoost, on the PW79 and POOL160 datasets. The performance is evaluated using recall (R), precision (P), and area under ROC curve (AUCROC). 
</details>


#### Fajardo, J.E., Fiser, A. Protein structure based prediction of catalytic residues. BMC Bioinformatics 14, 63 (2013). https://doi.org/10.1186/1471-2105-14-63

<details>
<summary>summary</summary>
The methods described in the paper Protein structure based prediction of catalytic residues aim to predict functional residues in protein structures using a neural network approach. The authors explore a range of features based on protein structure, including distance to the centroid and amino acid type, combined with sequence conservation. The neural network is trained using a supervised, feed-forward approach with one hidden layer of ten units. The performance of the method is evaluated using various performance measures, including sensitivity, precision, and the F1-measure. The results show that the proposed method achieves comparable performance to other state-of-the-art methods. The benchmarks and baselines used in the paper are not explicitly mentioned in the provided information. However, the authors mention comparing their method to other existing methods, such as CRpred and the method of Youn et al., which rely on sequence conservation and structural information. The performance of the proposed method is evaluated using different datasets, including a training set and an independent test set.

</details>





## DNA binding site prediction

#### Qianmu Yuan and others, AlphaFold2-aware protein–DNA binding site prediction using graph transformer, Briefings in Bioinformatics, Volume 23, Issue 2, March 2022, bbab564, https://doi.org/10.1093/bib/bbab564
<details>
<summary>summary</summary>
This paper proposes a novel method, GraphSite, that utilizes the AlphaFold2 protein structure prediction and a graph transformer model to predict protein-DNA binding sites accurately. The method outperforms existing sequence-based and structure-based methods, demonstrating significant improvements in performance.
The baseline methods of this paper are:
BiLSTM: a sequence-based method that uses a two-layer bidirectional long short-term memory network and a multilayer perceptron to predict DNA-binding residues from amino acid features1.
TargetS, TargetDNA, SVMnuc, and DNAPred: four sequence-based methods that use different machine learning algorithms and sequence-derived features to learn local patterns of DNA-binding characteristics2.
COACH-D, NucBind, DNABind, and GraphBind: four structure-based methods that use protein structures as input and employ different techniques to identify DNA-binding sites, such as template-based methods, machine learning methods or hybrid methods.

Train and test sets: Train_573 and Test_129, Test_196, which are named by the numbers of proteins in the datasets. These datasets were collected from the BioLiP database. The authors used several evaluation metrics such as accuracy, sensitivity, specificity, precision, F1-score, Matthews correlation coefficient (MCC), and area under the receiver operating characteristic curve (AUC-ROC) to evaluate the performance of their method.

</details>


#### Yi-Heng Zhu, Dong-Jun Yu, ULDNA: Integrating Unsupervised Multi-Source Language Models with LSTM-Attention Network for Protein-DNA Binding Site Prediction, bioRxiv 2023.05.30.542787; doi: https://doi.org/10.1101/2023.05.30.542787
<details>
<summary>Summary</summary>


</details>

## RNA binding site prediction

#### Song, Y., Yuan, Q., Zhao, H., & Yang, Y. (2023). Accurately identifying nucleic-acid-binding sites through geometric graph learning on language model predicted structures. bioRxiv, 2023-07.


#### Xia, Y., Xia, C., Pan, X., & Shen, H. (2021). GraphBind: Protein structural context embedded rules learned by hierarchical graph neural networks for recognizing nucleic-acid-binding residues. Nucleic Acids Research, 49(9), e51. https://doi.org/10.1093/nar/gkab044
<details>
<summary>Summary</summary>

The paper presents GraphBind, a novel method for nucleic-acid-binding residue prediction based on an end-to-end graph neural network. The main contributions of the paper are:
It proposes a structural-context-based graph representation to capture protein residues' local geometric and bio-physicochemical characteristics and their interactions with nucleic acids.
It designs a hierarchical graph neural network for embedding the graph into a fixed-size latent representation for downstream prediction.
It demonstrates the superior performance of GraphBind over eight state-of-the-art methods on two benchmark datasets for DNA- and RNA-binding residue prediction.
The benchmark datasets are constructed from the BioLiP database, which is a collection of biologically relevant ligand-protein interactions. The datasets are divided into training and test sets according to the release date. The datasets are also augmented by transferring binding annotations from similar protein chains to increase the number of binding residues in the training sets. The details of the datasets are summarized in Table 11.

The evaluation metrics are recall, precision, F1-score, Matthews correlation coefficient (MCC), and area under the receiver operator characteristic (ROC) curve (AUC). These metrics are used to assess the performance of GraphBind and other methods on binary classification of binding and non-binding residues.

The baselines are eight state-of-the-art methods for nucleic-acid-binding residue prediction, including deep-learning-based methods, shallow-machine-learning-based methods, template-based methods, and consensus methods. 
These methods are:
- TargetDNA: a sequence-based method that uses evolutionary information and predicted secondary structure profiles as input and employs multiple SVMs with boosting as the classifier.
- TargetS: a sequence-based method that uses evolutionary information, predicted secondary structure profiles and ligand-specific propensity as input and applies the AdaBoost algorithm as the classifier.
- NucBind: a consensus method that fuses a sequence-based SVM classifier and a template-based method 1.
- DNAPred: a sequence-based method that proposes a two-stage imbalanced learning algorithm with an ensemble technique 2.
- RNABindRPlus: a consensus method that combines outputs from a sequence homology-based method and a SVM classifier 3.
- NucleicNet: a structure-based deep learning method that analyzes physicochemical properties of grid points on protein surface and takes a deep residual network as the classifier 45.
- aaRNA: a sequence- and structure-based artificial neural network classifier that uses a structural descriptor Laplacian norm to measure surface convexity/concavity 6.
- DNABind: a consensus method that integrates a sequence-based SVM classifier, a structure-based SVM classifier and a template-based method.
</details>

## Protein binding site prediction

#### Tubiana, J., & Wolfson, H. J. (2022). ScanNet: An interpretable geometric deep learning model for structure-based protein binding site prediction. Nature Methods, 19(6), 730-739. https://doi.org/10.1038/s41592-022-01490-7

<details>
<summary>Summary</summary>


ScanNet: a new model for protein binding site prediction. ScanNet is a geometric deep learning model that can learn the spatio-chemical arrangements of atoms and amino acids that characterize the functional sites of proteins12. It can predict the binding sites of small molecules, other proteins, and antibodies from the 3D structure of a protein. It outperforms existing methods based on handcrafted features, structural homology, and surface-based geometric deep learning3.
ScanNet architecture and components. ScanNet consists of four main stages: atomic neighborhood embedding, atom to amino acid pooling, amino acid neighborhood embedding, and neighborhood attention. It uses trainable, linear filters to detect specific spatio-chemical patterns in the local neighborhoods of atoms and amino acids4. It also uses multi-headed attention to pool information from relevant atoms and neighbors. It outputs a residue-wise label probability for each amino acid.
ScanNet applications and results. ScanNet is applied to two related tasks: prediction of protein–protein binding sites (PPBS) and B cell epitopes (BCE)5. It is trained and evaluated on newly compiled datasets of annotated PPBSs and BCEs derived from the Dockground and SabDab databases, respectively. It achieves high accuracy and precision for both tasks, and can generalize to unseen protein folds. It also predicts BCEs for the SARS-CoV-2 spike protein, validating known antigenic regions and suggesting new ones67.
We constructed a nonredundant dataset of 20K representative protein chains with annotated binding sites derived from the 
Dockground database of protein complexes45. The PPBS dataset 
covers a wide range of complex sizes, types, organism taxonomies, 
protein lengths (Extended Data Fig. 3a–d) and contains around 
5M amino acids, of which 22.7% are PPBS.
The baseline methods used for comparison in the paper are based on handcrafted features, structural homology, and surface-based geometric deep learning1. The evaluation metrics used in the paper are precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC-ROC)1. The precision is the number of true positives divided by the number of true positives plus false positives. The recall is the number of true positives divided by the number of true positives plus false negatives. The F1 score is the harmonic mean of precision and recall. The AUC-ROC is a measure of the trade-off between true positive rate and false positive rate.

</details>


#### Khan, S. H., Tayara, H., & Chong, K. T. (2022). ProB-Site: Protein Binding Site Prediction Using Local Features. Cells, 11(13). https://doi.org/10.3390/cells11132117

<details>
<summary>Summary</summary>


</details>

#### Zhang, J., & Kurgan, L. (2018). Review and comparative assessment of sequence-based predictors of protein-binding residues. Briefings in Bioinformatics, 19(5), 821-837. https://doi.org/10.1093/bib/bbx022
<details>
<summary>Summary</summary>


</details>

#### Sorzano, C. O., Carazo, J. M., & Segura, J. (2019). BIPSPI: A method for the prediction of partner-specific protein–protein interfaces. Bioinformatics, 35(3), 470-

477. https://doi.org/10.1093/bioinformatics/bty647
<details>
<summary>Summary</summary>


</details>

#### Chen, R., Li, X., Yang, Y., Song, X., Wang, C., & Qiao, D. (2022). Prediction of protein-protein interaction sites in intrinsically disordered proteins. Frontiers in Molecular Biosciences, 9, 985022. https://doi.org/10.3389/fmolb.2022.985022
<details>
<summary>Summary</summary>


</details>

#### Khan, S. H., Tayara, H., & Chong, K. T. (2022). ProB-Site: Protein Binding Site Prediction Using Local Features. Cells, 11(13), 2117. https://doi.org/10.3390/cells11132117
<details>
<summary>Summary</summary>


</details>

#### Sunny, S., Prakash, P.B., Gopakumar, G. et al. DeepBindPPI: Protein–Protein Binding Site Prediction Using Attention Based Graph Convolutional Network. Protein J 42, 276–287 (2023). https://doi.org/10.1007/s10930-023-10121-9
<details>
<summary>Summary</summary>


</details>

## Ligand binding site prediction

#### Jake E McGreig and others, 3DLigandSite: structure-based prediction of protein–ligand binding sites, Nucleic Acids Research, Volume 50, Issue W1, 5 July 2022, Pages W13–W20, https://doi.org/10.1093/nar/gkac250
<details>
<summary>Summary</summary>


</details>

#### Zhao J, Cao Y, Zhang L. Exploring the computational methods for protein-ligand binding site prediction. Comput Struct Biotechnol J. 2020 Feb 17;18:417-426. doi: 10.1016/j.csbj.2020.02.008. PMID: 32140203; PMCID: PMC7049599.
<details>
<summary>Summary</summary>


</details>

#### Roche, D.B.; Brackenridge, D.A.; McGuffin, L.J. Proteins and Their Interacting Partners: An Introduction to Protein–Ligand Binding Site Prediction Methods. Int. J. Mol. Sci. 2015, 16, 29829-29842. https://doi.org/10.3390/ijms161226202
<details>
<summary>Summary</summary>


</details>


#### Santana, C. A., Izidoro, S. C., C, R., Tyzack, J. D., Ribeiro, A. J., Pires, D. E., & Thornton, J. M. (2022). GRaSP-web: A machine learning strategy to predict binding sites based on residue neighborhood graphs. Nucleic Acids Research, 50(W1), W392-W397. https://doi.org/10.1093/nar/gkac323
<details>
<summary>Summary</summary>


</details>



## Metal-binding site prediction

#### Cheng, Y., Wang, H., Xu, H., Liu, Y., Ma, B., Chen, X., Zeng, X., Wang, X., Wang, B., Shiau, C., Ovchinnikov, S., Su, X., & Wang, C. (2023). Co-evolution-based prediction of metal-binding sites in proteomes by machine learning. Nature Chemical Biology, 19(5), 548-555. https://doi.org/10.1038/s41589-022-01223-z
<details>
<summary>Summary</summary>


</details>

#### Aditi Shenoy et al., M-Ionic: Prediction of metal ion binding sites from sequence using residue embeddings, bioRxiv 2023.04.06.535847; doi: https://doi.org/10.1101/2023.04.06.535847
<details>
<summary>Summary</summary>


</details>

#### Dürr, S. L., Levy, A., & Rothlisberger, U. (2023). Metal3D: A general deep learning framework for accurate metal ion location prediction in proteins. Nature Communications, 14(1), 1-14. https://doi.org/10.1038/s41467-023-37870-6
<details>
<summary>Summary</summary>


</details>

### Review

#### Özçelik, R., van Tilborg, D., Jiménez-Luna, J., Grisoni, F., Structure-Based Drug Discovery with Deep Learning, ChemBioChem 2023, 24, e202200776.
<details>
<summary>summary</summary>
The paper discusses the importance of identifying ‘druggable’ binding sites in proteins in structure-based drug discovery (SBDD). The authors describe various methods for binding site detection such as interatomic gap volumes and regions of buried pocket surfaces. They also discuss how deep learning methods have gained traction to detect binding sites. These approaches can be grouped by the molecular representations they rely on, i.e., protein sequence, 3D structure, and surface. The authors describe sequence-based models and 3D structure-based models that use the spatial information of proteins to detect likely binding sites. Early approaches represented the protein structure with voxels featurized with pharmacophore-like properties, along with convolutional neural networks. Subsequent works have refined structure-based binding site detection with additional techniques from the computer vision domain, e.g., image segmentation. BiteNet extended ‘static’ CNN-based approaches by incorporating conformational ensembles of proteins. The approach was later adapted to predict protein-peptide binding sites.
</details>

#### Yan J, Friedrich S, Kurgan L. A comprehensive comparative review of sequence-based predictors of DNA- and RNA-binding residues. Brief Bioinform. 2016 Jan;17(1):88-105. doi: 10.1093/bib/bbv023. Epub 2015 May 1. PMID: 25935161.

<details>
<summary>summary</summary>
A review of sequence-based predictors of DNA- and RNA-binding residues.12 The authors summarize 30 computational methods that use protein sequences to predict which amino acids interact with DNA or RNA molecules. They compare the features, models, outputs and availability of these methods.
A new benchmark data set for evaluation of predictive performance. The authors collect high-quality structures of protein–DNA and protein–RNA complexes from the Protein Data Bank and define binding residues based on two cutoff distances (3.5 A˚ and 5 A˚ ). They also transfer binding annotations between similar proteins to obtain more complete data. They split the data into training and test sets based on release date and sequence similarity3.
A selection of methods for empirical assessment. The authors choose 10 sequence-only methods that are available as web servers and are runtime-efficient. They include four predictors of DNA-binding residues, three predictors of RNA-binding residues, and three consensus-based approaches that combine multiple methods46.
An evaluation of predictive quality and cross-prediction ability. The authors use sensitivity, specificity, MCC, AUC and a new measure called Ratio to assess the predictions of the selected methods on the test data sets. They also investigate how well the methods can discriminate between DNA- and RNA-binding residues7.
A design and assessment of consensus predictors.8 The authors propose several types of consensuses that integrate predictions from different methods using logic-based, majority-vote-based or machine learning-based approaches. They also introduce a novel method for combined prediction of DNA- and RNA-binding residues with four outcomes65. They show that some consensuses can improve predictive performance and reduce cross-prediction errors9.
</details>


#### Ashwin Dhakal and others, Artificial intelligence in the prediction of protein–ligand interactions: recent advances and future directions, Briefings in Bioinformatics, Volume 23, Issue 1, January 2022, bbab476, https://doi.org/10.1093/bib/bbab476
<details>
<summary>Summary</summary>


</details>

