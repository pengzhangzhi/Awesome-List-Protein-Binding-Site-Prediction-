# Awesome-List-Protein-Binding-Site-Prediction
List of papers on protein binding site prediction


### Database

#### Mechanism and Catalytic Site Atlas
M-CSA is a database of enzyme reaction mechanisms. It provides annotation on the protein, catalytic residues, cofactors, and the reaction mechanisms of hundreds of enzymes.
M-CSA contains 1003 hand-curated entries.

#### PDBbind database
Contains ~13000 complex strcuctures formed between protein-small molecule ligand, protein-protein, protein-nucleic acid and nucleic acid-small molecule ligand. 
Binding affinity data and structural information for a total of 12,995 biomolecular complexes, including protein-ligand (10656), nucleic acid-ligand (87), protein-nucleic acid (660), and protein-protein complexes (1592), which is the largest collection of this kind so far.

#### BioLiP database
BioLiP is a semi-manually curated database for high-quality, biologically relevant ligand-protein binding interactions. The structure data are collected primarily from the Protein Data Bank (PDB), with biological insights mined from literature and other specific databases. BioLiP aims to construct the most comprehensive and accurate database for serving the needs of ligand-protein docking, virtual ligand screening and protein function annotation. Questions about the BioLiP Database can be posted at the Service System Discussion Board. Since ligand molecules (e.g., Glycerol, Ethylene glycol) are often used as additives (i.e., false positives) for solving the protein structures, not all ligands present in the PDB database are biologically relevant. BioLiP uses a composite automated and manual procedure for examining the biological relevance of ligands in the PDB database. Each entry in BioLiP contains a comprehensive list of annotations on: ligand-binding residues; ligand binding affinity (from the original literature, plus Binding MOAD, PDBbind-CN, BindingDB); catalytic site residues (mapped from Mechanism and Catalytic Site Atlas); Enzyme Commission (EC) numbers and Gene Ontology (GO) terms mapped by the SIFTS database; crosslinks to external databases, including RCSB PDB, PDBe, PDBj, PDBsum, Binding MOAD, PDBbind-CN, Mechanism and Catalytic Site Atlas, QuickGO, ExPASy ENZYME, ChEMBL, DrugBank, ZINC, UniProt, PubMed.

## GO and EC prediction

#### Gligorijević, V., Renfrew, P. D., Kosciolek, T., Leman, J. K., Berenberg, D., Vatanen, T., Chandler, C., Taylor, B. C., Fisk, I. M., Vlamakis, H., Xavier, R. J., Knight, R., Cho, K., & Bonneau, R. (2021). Structure-based protein function prediction using graph convolutional networks. Nature Communications, 12(1), 1-14. https://doi.org/10.1038/s41467-021-23303-9
This paper presents DeepFRI, a novel method for predicting protein function using graph convolutional networks. The authors demonstrate that DeepFRI outperforms existing methods for protein function prediction, and is able to predict functions for a large number of proteins with high accuracy and resolution. The method leverages sequence features from pretrained language model and protein structures, and is able to expand the number of predictable functions using homology models. Additionally, DeepFRI enables site-specific annotations at the residue-level using grad-CAM, and has been shown to be useful in annotating structures from the PDB and SWISS-MODEL.
The data used in this study consists of protein structures from the Protein Data Bank (PDB) and homology models from the SWISS-MODEL repository. The authors created a non-redundant set of PDB chains by clustering all PDB chains with blastclust at 95% sequence identity, and selecting a representative PDB chain that is annotated and of high quality from each cluster. They also included annotated SWISS-MODEL chains in their training procedure, and removed similar SWISS-MODEL sequences at 95% sequence identity. Including SWISS-MODEL models led to a 5-fold increase in the number of training samples and a larger coverage of more specific Gene Ontology (GO) terms. 

**Highlight: Use grad-CAM for functional site prediction, which is unsupervised**

#### Cagiada, M., Bottaro, S., Lindemose, S., Schenstrøm, S. M., & Stein, A. (2023). Discovering functionally important sites in proteins. Nature Communications, 14(1), 1-13. https://doi.org/10.1038/s41467-023-39909-0
This paper presents a machine learning method that combines statistical models for protein sequences with biophysical models of stability to predict functionally important sites in proteins. The model is trained using experimental data on variant effects and can be used to discover active sites, regulatory sites, and binding sites. The utility of the model is demonstrated through prospective prediction and experimental validation on the functional consequences of missense variants in HPRT1, which may cause Lesch-Nyhan syndrome. The paper addresses the challenge of distinguishing the effects of amino acid substitutions on intrinsic function from their effects on stability and cellular abundance by using multiplexed assays of variant effects. The goal is to pinpoint the molecular mechanisms underlying perturbed function and identify residues that directly contribute to the function.
the data used in this study includes experimental data on variant effects, specifically missense variants in the HPRT1 protein. The authors used multiplexed assays of variant effects (MAVEs) to probe the consequences of individual substitutions on both function and abundance. The data used for training the machine learning model and the predictions made in this study are available at the GitHub repository: https://github.com/KULL-Centre/_2022_functional-sites-cagiada.


## Protein catalytic site prediction

#### Das S, Scholes HM, Sen N, Orengo C. CATH functional families predict functional sites in proteins. Bioinformatics. 2021 May 23;37(8):1099-1106. doi: 10.1093/bioinformatics/btaa937. PMID: 33135053; PMCID: PMC8150129.

#### Lu CH, Yu CS, Chien YT, Huang SW. EXIA2: web server of accurate and rapid protein catalytic residue prediction. Biomed Res Int. 2014;2014:807839. doi: 10.1155/2014/807839. Epub 2014 Sep 11. PMID: 25295274; PMCID: PMC4177735.
The paper describes the EXIA2 Web Server, a method for predicting catalytic residues in proteins based on their special side chain orientation. The paper uses six benchmark datasets to evaluate the performance of the EXIA2 Web Server for predicting catalytic residues in proteins. The datasets include PW79, POOL160, EF fold, EF superfamily, EF family, and P100, which include over 1,200 proteins and 861,404 residues (3,664 catalytic residues and 857,740 noncatalytic residues). The definition of catalytic residues is based on Catalytic Site Atlas version 2.2.12. The paper compares the prediction performance of EXIA2 with that of three state-of-the-art prediction methods, including CRpred, POOL, ConSurf, and ResBoost, on the PW79 and POOL160 datasets. The performance is evaluated using recall (R), precision (P), and area under ROC curve (AUCROC). 

#### Fajardo, J.E., Fiser, A. Protein structure based prediction of catalytic residues. BMC Bioinformatics 14, 63 (2013). https://doi.org/10.1186/1471-2105-14-63

The methods described in the paper Protein structure based prediction of catalytic residues aim to predict functional residues in protein structures using a neural network approach. The authors explore a range of features based on protein structure, including distance to the centroid and amino acid type, combined with sequence conservation. The neural network is trained using a supervised, feed-forward approach with one hidden layer of ten units. The performance of the method is evaluated using various performance measures, including sensitivity, precision, and the F1-measure. The results show that the proposed method achieves comparable performance to other state-of-the-art methods. The benchmarks and baselines used in the paper are not explicitly mentioned in the provided information. However, the authors mention comparing their method to other existing methods, such as CRpred and the method of Youn et al., which rely on sequence conservation and structural information. The performance of the proposed method is evaluated using different datasets, including a training set and an independent test set.




## DNA binding site prediction

#### Qianmu Yuan and others, AlphaFold2-aware protein–DNA binding site prediction using graph transformer, Briefings in Bioinformatics, Volume 23, Issue 2, March 2022, bbab564, https://doi.org/10.1093/bib/bbab564
This paper proposes a novel method, GraphSite, that utilizes the AlphaFold2 protein structure prediction and a graph transformer model to accurately predict protein-DNA binding sites. The method outperforms existing sequence-based and structure-based methods, demonstrating significant improvements in performance.


#### Yan J, Friedrich S, Kurgan L. A comprehensive comparative review of sequence-based predictors of DNA- and RNA-binding residues. Brief Bioinform. 2016 Jan;17(1):88-105. doi: 10.1093/bib/bbv023. Epub 2015 May 1. PMID: 25935161.

#### Özçelik, R., van Tilborg, D., Jiménez-Luna, J., Grisoni, F., Structure-Based Drug Discovery with Deep Learning, ChemBioChem 2023, 24, e202200776.


## RNA binding site prediction

#### Song, Y., Yuan, Q., Zhao, H., & Yang, Y. (2023). Accurately identifying nucleic-acid-binding sites through geometric graph learning on language model predicted structures. bioRxiv, 2023-07.
#### Xia, Y., Xia, C., Pan, X., & Shen, H. (2021). GraphBind: Protein structural context embedded rules learned by hierarchical graph neural networks for recognizing nucleic-acid-binding residues. Nucleic Acids Research, 49(9), e51. https://doi.org/10.1093/nar/gkab044

## Protein binding site prediction

#### Tubiana, J., & Wolfson, H. J. (2022). ScanNet: An interpretable geometric deep learning model for structure-based protein binding site prediction. Nature Methods, 19(6), 730-739. https://doi.org/10.1038/s41592-022-01490-7

#### Khan, S. H., Tayara, H., & Chong, K. T. (2022). ProB-Site: Protein Binding Site Prediction Using Local Features. Cells, 11(13). https://doi.org/10.3390/cells11132117

#### Zhang, J., & Kurgan, L. (2018). Review and comparative assessment of sequence-based predictors of protein-binding residues. Briefings in Bioinformatics, 19(5), 821-837. https://doi.org/10.1093/bib/bbx022


#### Sorzano, C. O., Carazo, J. M., & Segura, J. (2019). BIPSPI: A method for the prediction of partner-specific protein–protein interfaces. Bioinformatics, 35(3), 470-477. https://doi.org/10.1093/bioinformatics/bty647

#### Chen, R., Li, X., Yang, Y., Song, X., Wang, C., & Qiao, D. (2022). Prediction of protein-protein interaction sites in intrinsically disordered proteins. Frontiers in Molecular Biosciences, 9, 985022. https://doi.org/10.3389/fmolb.2022.985022

#### Khan, S. H., Tayara, H., & Chong, K. T. (2022). ProB-Site: Protein Binding Site Prediction Using Local Features. Cells, 11(13), 2117. https://doi.org/10.3390/cells11132117

#### Sunny, S., Prakash, P.B., Gopakumar, G. et al. DeepBindPPI: Protein–Protein Binding Site Prediction Using Attention Based Graph Convolutional Network. Protein J 42, 276–287 (2023). https://doi.org/10.1007/s10930-023-10121-9


## Ligand (e.g., compound/zinc) binding site prediction

#### Jake E McGreig and others, 3DLigandSite: structure-based prediction of protein–ligand binding sites, Nucleic Acids Research, Volume 50, Issue W1, 5 July 2022, Pages W13–W20, https://doi.org/10.1093/nar/gkac250

#### Zhao J, Cao Y, Zhang L. Exploring the computational methods for protein-ligand binding site prediction. Comput Struct Biotechnol J. 2020 Feb 17;18:417-426. doi: 10.1016/j.csbj.2020.02.008. PMID: 32140203; PMCID: PMC7049599.

#### Roche, D.B.; Brackenridge, D.A.; McGuffin, L.J. Proteins and Their Interacting Partners: An Introduction to Protein–Ligand Binding Site Prediction Methods. Int. J. Mol. Sci. 2015, 16, 29829-29842. https://doi.org/10.3390/ijms161226202

#### Ashwin Dhakal and others, Artificial intelligence in the prediction of protein–ligand interactions: recent advances and future directions, Briefings in Bioinformatics, Volume 23, Issue 1, January 2022, bbab476, https://doi.org/10.1093/bib/bbab476

#### Santana, C. A., Izidoro, S. C., C, R., Tyzack, J. D., Ribeiro, A. J., Pires, D. E., & Thornton, J. M. (2022). GRaSP-web: A machine learning strategy to predict binding sites based on residue neighborhood graphs. Nucleic Acids Research, 50(W1), W392-W397. https://doi.org/10.1093/nar/gkac323


#### Aditi Shenoy et al., M-Ionic: Prediction of metal ion binding sites from sequence using residue embeddings, bioRxiv 2023.04.06.535847; doi: https://doi.org/10.1101/2023.04.06.535847

#### Yi-Heng Zhu, Dong-Jun Yu, ULDNA: Integrating Unsupervised Multi-Source Language Models with LSTM-Attention Network for Protein-DNA Binding Site Prediction, bioRxiv 2023.05.30.542787; doi: https://doi.org/10.1101/2023.05.30.542787
