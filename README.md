### [Identification of stellar clumps in cosmological simulations of galaxy formation 星系形成宇宙学模拟中恒星团的识别](https://www.overleaf.com/read/hzkbsfppnyks#08bf3f)

### 0.Abstract

We **develop a new algorithm to identify stellar clumps based on particle data in cosmological simulations**. The algorithm is based on the hierarchical density-based spatial clustering of applications with noise (HDBSCAN) method that is widely used in unsupervised classification. (Our improvement here ...) Among all tests on **synthesized particle data**, this new algorithm performs better than the classical FoF and DBSCAN algorithms with ... We apply the new clump finder to two cosmological hydrodynamical zoom-in simulations, The san-Zoom and FIRE-2 high-redshift suite. For both simulations, the algorithm can robustly identify stellar clumps with masses and sizes that can vary for orders of magnitudes and converge against the mass threshold. We find that ..

我们根据**宇宙学模拟中的粒子数据开发了一种识别恒星团块的新算法**。该算法基于**广泛应用于无监督分类的基于密度的分层空间聚类（HDBSCAN）方法**。(我们在这里的改进是......）在对合成粒子数据进行的所有测试中，这种新算法的性能都优于经典的 FoF 算法和 DBSCAN 算法，而且...... 我们在两个宇宙学流体力学放大模拟（The san-Zoom and FIRE-2 high-redshift suite）中应用了新的团块发现算法。对于这两种模拟，**该算法都能稳健地识别出质量和大小可变化几个数量级的恒星团块**，并与质量阈值趋同。我们发现.

### 1. Introduction

In this paper, **we introduce a tailored version of HDBSCAN for clump finding based on particle/cell data in cosmological hydro dynamical simulations.** We evaluate the **effectiveness** of this method and compare it with various other clump-finding algorithms, focus-ing on how well these **algorithms identify and characterize** stellar clumps in complex simulated environments. The paper is organized as follows: (1) In Section ??, we ... (4) In Section ??, we discuss the strengths and limitations of each method using both synthetic and real simulation data.

在本文中，我们**介绍了一种基于宇宙学流体力学模拟中的粒子/细胞数据来寻找星团的 HDBSCAN 定制版本**。我们评估了该方法的**有效性**，并将其与其他各种星团寻找算法进行了比较，重点关注这些算法在复杂模拟环境中识别和描述恒星团的能力。本文的结构如下 (1) 在第......节，我们讨论了每种方法的优势和局限性。



> **引入了改进的HDBSCAN对宇宙模拟数据进行聚类**

Q：通用的HDBSCAN是怎么进行的，有什么优势和局限性

Q：我们是怎么改进的？

> **评估了方法的effectiveness，并和其他聚类算法进行比较**，算法的identify and characterize的能力

Q：怎么评估的，有哪些指标，这些指标哪些优势。

Q：一个通用的聚类算法的评估指标有哪些？



### 2 CLUMP-FINDING ALGORITHMS

#### 2.1 FoF

The Friend-of-Friend (FoF) algorithm is a widely employed methodfor identifying groups of particles in cosmological simulations, par-ticularly in studies involving dark matter haloes and large-scale struc-ture formation (e.g., Davis et al. 1985; Springel et al. 2005). Thisalgorithm connects particles based on their proximity, defined by aEuclidean distance smaller than a specified linking length, 𝑙. Groupsare then formed by linking together particles that are within this dis-tance of each other. In cosmological contexts, the linking length 𝑙plays a critical role and is often chosen to correspond to the expectedscale of spherical collapse overdensities in the simulated universe.This parameter determines the spatial extent over which particles areconsidered to belong to the same structure or halo.To refine the grouping process and remove spurious structuresthat may arise due to chance alignments or numerical artifacts insimulations, an additional parameter is introduced: 𝑀min or 𝑁min.This parameter sets a threshold for the minimum number of particles(or mass) required to qualify as a valid group. Structures identifiedby the FoF algorithm that do not meet this criterion are typicallydiscarded as noise or background fluctuations.The FoF algorithm’s simplicity and efficiency make it suitablefor analyzing large-scale datasets in cosmology, where identifyingand characterizing dark matter haloes and other large structures areessential tasks. However, like any clustering algorithm, its perfor-mance and the interpretability of results can depend on the specificvalues chosen for 𝑙 and 𝑀min, as well as the underlying density anddistribution of particles in the simulation.

在宇宙学模拟中，特别是在涉及暗物质光环和大尺度结构形成的研究中，朋友（FoF）算法是一种广泛使用的识别粒子群的方法（如 Davis 等人，1985 年；Springel 等人，2005 年）。这种算法根据粒子间的距离（由小于指定连接长度的欧几里得距离定义）将粒子连接起来。然后，将彼此距离在此范围内的粒子连接起来，就形成了组。在宇宙学背景下，连接长度𝑙起着至关重要的作用，通常被选择为与模拟宇宙中球形坍缩过密的预期规模相对应。 为了完善分组过程，并去除可能因偶然排列或模拟中的数字伪影而产生的虚假结构，还引入了一个附加参数：𝑀min 或 𝑁min。FoF 算法的简单性和高效性使其适用于分析宇宙学中的大规模数据集，其中识别和描述暗物质光环和其他大型结构是至关重要的任务。然而，与任何聚类算法一样，它的性能和结果的可解释性可能取决于为𝑙和𝑀min选择的具体值，以及模拟中粒子的基本密度和分布。

#### 2.2 DBSCAN

DBSCAN, short for Density-Based Spatial Clustering of Applica-tions with Noise, is a popular clustering algorithm in data miningand machine learning. It is designed to identify clusters of arbitraryshapes in a spatial dataset, which may contain noise and outliers.The fundamental idea behind DBSCAN is to group together closelypacked points in high-density regions. It categorizes points into threemain types based on their relationship with nearby points:• Core Points: A point is classified as a core point if within thelinking length, 𝑙, there are at least 𝑁core neighboring points (includingitself). These core points are indicative of dense regions.• Reachable Points: A point that is not a core point but lies withinthe distance 𝑙 of a core point is classified as a reachable point. Thismeans the point lies within the vicinity of a cluster but may not bedensely surrounded itself.• Outliers: A point that is neither a core point nor reachable froma core point within the distance 𝑙 is considered an outlier. Thesepoints typically reside in low-density regions or are isolated fromany cluster.DBSCAN operates by iteratively expanding clusters from seedcore points. Similar to FoF, it requires at least two free parameters,𝑁core and the linking length 𝑙. In practice, one can also apply a mini-mum mass for identified clusters similar to the FoF case. DBSCAN’sperformance can vary based on the density and distribution of datapoints, making it suitable for datasets where clusters have differentdensities or where noise is present

DBSCAN 是 Density-Based Spatial Clustering of Applica-tions with Noise 的缩写，是数据挖掘和机器学习中一种流行的聚类算法。DBSCAN 背后的基本思想是将高密度区域中紧密堆积的点归为一类。它根据点与附近点的关系将点分为三大类型： - 核心点： 如果在链接长度（𝑙）范围内，至少有𝑁个核心邻近点（包括它自己），那么这个点就被归类为核心点。这些核心点是密集区域的标志： 非核心点但与核心点的距离在 𝑙 范围内的点被归类为可到达点。这意味着该点位于一个群集的附近，但可能本身并不被密集包围： 离群点：既不是核心点，又无法在𝑙距离内从核心点到达的点被视为离群点。

DBSCAN 通过从种子核心点迭代扩展簇来运行。与 FoF 类似，它至少需要两个自由参数：𝑁core 和连接长度 𝑙。在实践中，我们也可以对识别出的簇应用类似于 FoF 案例的最小质量（mini-mum mass）。DBSCAN 的性能可以根据数据点的密度和分布而变化，因此适用于数据集群密度不同或存在噪声的数据集。

#### 2.3 HDBSCAN

One limitation of DBSCAN is its difficulty in **effectively handling clusters that exhibit varying densities throughout the dataset**, as il-lustrated in Figure 3. This issue arises because DBSCAN relies on a fixed linking length 𝑙 to determine the neighborhood of points. In such cases, where traditional density-based clustering methods may falter, the hierarchial version of DBSCAN (HDBSCAN) offers a ro-bust alternative. HDBSCAN addresses the challenge of clusters with varying densities by employing a hierarchical clustering method and generating an optimized clustering based on the stability of clusters.

The steps of HDBSCAN are as follows:

如图 3 所示，**DBSCAN 的一个局限是难以有效处理在整个数据集中呈现不同密度的聚类。**出现这一问题的原因是 DBSCAN 依赖于固定的链接长度 𝑙 来确定点的邻域。在这种情况下，传统的基于密度的聚类方法可能会失效，而分层版 DBSCAN（HDBSCAN）则提供了一种有效的替代方法。HDBSCAN 采用分层聚类方法，并根据聚类的稳定性生成优化聚类，从而解决了密度不一的聚类难题：

**图 3 图示说明了类似 FoF 的算法为何会在聚类层次结构中失效（当只提供空间信息时），原则上这需要不同的链接长度来检测。**

Mutual reachability distance extends the notion of reachabilitydistance used in DBSCAN by considering the mutual accessibil-ity between points within a neighborhood defined by a core point.In practical terms, computing mutual reachability distances allowsHDBSCAN to build a connectivity graph that reflects both the geo-metric proximity and the density-based relationships between points

相互可达性距离扩展了 DBSCAN 中使用的可达性距离概念，考虑了核心点定义的邻域内各点之间的相互可达性。

在实际应用中，通过计算相互可达距离，HDBSCAN 可以构建一个连通图，该图既能反映各点之间的地理距离，也能反映各点之间基于密度的关系。

(2) After obtaining 𝑂 (𝑁2) edges of mutual reachability distances forevery pair of particles, we use the Prim’s algorithm Prim (1957) tobuild a minimum spanning tree (MST). This is illustrated in theleft panel of Figure 1. Then a single linkage tree is constructed by **iteratively merging the nearest clusters based on mutual reachability distances derived from the MST**. This hierarchical structure begins with each point as its own cluster and proceeds to merge clustersthat are most tightly interconnected until all points are unified into a single cluster, as depicted in the middle panel of Figure 1. This treeillustrates the evolving relationships between clusters at differentdensity levels, aiding in the identification of meaningful clustersacross varying scales of density in the dataset

**Q:MST**

(2)在得到每对粒子的相互可达距离的𝑂 (𝑁2)条边后，我们使用普里莫算法 Prim（1957）来构建最小生成树（MST）。

如图 1 左侧所示。然后，根据 MST 得出的相互可达性距离，通过迭代合并最近的簇来构建一棵单链树。这种分层结构从每个点作为自己的簇开始，然后合并相互联系最紧密的簇，直到所有点统一为一个簇，如图 1 中间面板所示。这棵树展示了不同密度水平的聚类之间不断演化的关系，有助于识别数据集中不同密度范围内有意义的聚类。

(3) Based on the single linkage tree, we will do optimizations tomerge or partition different nodes following the procedure describedin Figure 2. Each node in the single linkage tree contains the massof all particles that are associated with substructures below it. Wefirst traverse the single linkage tree and label nodes whose mass isless than 𝑀min as light nodes and the rest of the nodes are labelledas heavy nodes.(Jacob: clean the description here. the definitions are unclear.) Wedefine the persistence of a cluster as when the linking length (definedin DBSCAN) decreases from ∞ to 0, the cluster firstly separatesand becomes a independent cluster, and it loses some points until itsplits into two separate clusters. Instead of measuring linking lengthin distance, we adopt 𝜆 = 1distance . For any given cluster, we define𝜆birth and 𝜆death as the 𝜆 values when the cluster first separates andwhen the cluster split into smaller clusters, respectively. Furthermore,for each point 𝑝 within a cluster, we define 𝜆p as the 𝜆 value at whichthe point separates from the cluster. This value falls between 𝜆birthand 𝜆death, as the point either exits during the cluster’s lifespan orupon its division into smaller clusters. The stability for each clusteris defined as

(3) 在单链树的基础上，我们将按照图 2 所描述的步骤对不同节点进行优化合并或分割。单链树中的每个节点都包含与其下子结构相关的所有粒子的质量。我们首先遍历单链树，将质量小于𝑀min 的节点标为轻节点，其余节点标为重节点（雅各布：此处描述不清，定义不明）。我们将聚类的持续性定义为：当链接长度（DBSCAN 中的定义）从∞减小到 0 时，聚类首先分离并成为一个独立的聚类，然后会丢失一些点，直至分裂成两个独立的聚类。我们采用𝜆 = 1distance，而不是用距离来衡量连接长度。对于任何给定的簇，我们将𝜆birth 和𝜆death 分别定义为簇第一次分离和簇分裂成小簇时的𝜆值。此外，对于聚类中的每个点𝑝，我们将𝜆p定义为该点脱离聚类时的𝜆值。𝜆p值介于 “出生 ”和 “死亡 ”之间，因为该点要么在聚类的生命周期中退出，要么在聚类分裂成更小的聚类时退出。每个聚类的稳定性定义为

We iterate over all leaf heavy nodes and consider the merging of leafnodes into a larger cluster as in step two of fig2. If the father node has two heavy leaf nodes, discard the two leaf nodes from the heavy treeand maintain the father node as an integrated bigger cluster whensum of stability of two child nodes is less father’s stability. If thefather node has one heavy leaf node and a light node, discard theleaf node from the heavy tree and maintain the father node as anintegrated bigger cluster if stability of son is less father’s stability.There are already some benchmark hdbscan libraries. Therefore, forimplementation of the algorithm, building of single linkage tree isimported from the "hdbscan" library. Our optimization on the treeis performed linearly and different from the benchmark one. Theprimary modification in our version of HDBSCAN compared to thebenchmark is that our approach stops merging parent nodes oncetheir child nodes have ceased merging. In contrast, the benchmarkHDBSCAN may continue merging parent nodes even if their chil-dren have already stopped, depending on the stability test results.This adjustment is made because the stability test indicates that thepoints are well-separated and should not be merged further at higherlevels. By halting the merging process for parent nodes when theirchildren are stable, we preserve this separation and avoid unnecessaryconsolidation.

我们遍历所有重叶节点，并考虑将叶节点合并成一个更大的集群，如图 2 的第二步。如果父节点有两个重叶节点，当两个子节点的稳定度之和小于父节点的稳定度时，从重树中丢弃这两个叶节点，保留父节点作为一个完整的更大的簇。如果父节点有一个重叶节点和一个轻节点，如果子节点的稳定性小于父节点的稳定性，则舍弃重树上的叶节点，保留父节点作为一个完整的更大的簇。因此，为了实现该算法，我们从 “hdbscan ”库中导入了单链树的构建。我们对该树的优化是线性进行的，与基准优化不同。与基准算法相比，我们的 HDBSCAN 版本的主要改进是，当父节点的子节点停止合并时，我们的方法就停止合并父节点。之所以做出这样的调整，是因为稳定性测试表明父节点和子节点已经很好地分离，不应该在更高层次上继续合并。通过在父节点的子节点稳定时停止合并进程，我们可以保持这种分离，避免不必要的合并。

### 3 APPLICATIONS 

#### 3.1 Metrics for the performance of clustering 聚类性能指标

#### 3.2 Silhouette Coefficient 轮廓系数

The silhouette score is a metric used to evaluate the quality of clustersin clustering algorithms. It measures how an object is cohesive to its own cluster compared to how it’s separated from other clusters. Foreach point 𝑖, 𝑎(𝑖) is the average distance to all other points in thesame cluster; 𝑏(𝑖) is the average distance to all points in the nearestcluster (the one that the data point is not a part of). The silhouettescore for 𝑖 is

剪影得分是聚类算法中用来评估聚类质量的一个指标。它衡量的是一个物体与其他簇的分离程度，以及它在自己簇中的凝聚力。每个点 𝑖、𝑎(𝑖) 是与同一聚类中所有其他点的平均距离；𝑏(𝑖) 是与最近聚类（数据点不属于的聚类）中所有点的平均距离。𝑖的剪影分数为

Then computing the average silhouette score of all data points in thedataset produces an overall measure of clustering quality. Silhouette score ranges in [−1, 1], where −1 indicates poor fitting into the clusterwhile 1 indicates that the data point is well-clustered and far fromother clusters

然后计算数据集中所有数据点的平均剪影得分，得出聚类质量的总体衡量标准。剪影得分范围为 [-1, 1]，其中 -1 表示数据点与聚类的拟合程度较差，而 1 则表示数据点聚类良好，远离其他聚类。

#### 3.3 Calinski-Harabasz Index

Calinski-Harabasz Index measures the ratio of the sum of between-cluster dispersion to within-cluster dispersion for a given clusteringsolution. Between-cluster dispersion measures how far the cluster centers are from each other. A higher between-cluster dispersion in-dicates that the clusters are well-separated. Within-cluster dispersionmeasures how compact the clusters themselves are, typically using ameasure like the sum of squared distances from cluster members totheir cluster center. A lower within-cluster dispersion indicates thatthe points within each cluster are tightly grouped around their center.A larger CH index suggests a better clustering solution.

卡林斯基-哈拉巴什指数（Calinski-Harabasz Index）用于测量给定聚类方案的聚类间离散度之和与聚类内离散度之比。簇间离散度衡量簇中心之间的距离。簇间离散度越高，说明簇之间的距离越远。簇内离散度衡量簇本身的紧凑程度，通常使用簇成员到簇中心的距离平方和等指标。簇内离散度越小，说明每个簇内的点都紧密地围绕着簇中心。

#### 3.4 Davies-Bouldin Index 戴维斯-博尔丁指数

The Davies-Bouldin Index serves as a validation measure for assess-ing clustering models. It is computed by averaging the similaritymeasure of each cluster with the cluster that is most alike to it. Sim-ilarity is defined as the ratio of dispersion distances and distancesbetween cluster centroids.
戴维斯-博尔丁指数（Davies-Bouldin Index）是评估聚类模型的一种验证方法。它是通过平均每个聚类与与其最相似的聚类的相似度来计算的。相似度被定义为分散距离与聚类中心点之间距离的比值。

#### 3.5 Performance tests on synthetic data 对合成数据的性能测试

#### 3.6 Applications for cosmological simulations 宇宙学模拟应用

To apply our algorithm to simulation data, one should load in 2D or3D coordinates of stars and 𝑀min. 𝑁core is decided based on the input𝑀min and is in a linear relationship with it, so it is set by default.The algorithm will generate a list, where each element is itself a listcontaining the coordinates of a cluster. We apply HDBSCAN to Fire-2 and thesan data. Both are zoomed in within the range of +/−5𝑘 𝑝𝑐and projected onto the 𝑥 − 𝑦𝑎𝑥𝑖𝑠. The FIRE data has 2222636 pointswhile thesan data has 189563 points. We will analyze our clusteringresult based on the following perspectives

要将我们的算法应用于模拟数据，需要载入恒星的二维或三维坐标以及𝑀min。算法将生成一个列表，其中每个元素本身就是一个包含星团坐标的列表。我们将 HDBSCAN 应用于 Fire-2 和 san 数据。两者都在 +/-5𝑘 𝑝𝑐 范围内放大，并投影到 𝑥 - 𝑦𝑎𝑥𝑖𝑠 上。FIRE 数据有 2222636 个点，而 SAN 数据有 189563 个点。我们将从以下角度分析聚类结果

#### 3.7 image of clusters 聚类图像

The images of clustering algorithms applied to FIRE and Thesan arerespectively shown in figure 5 and 6. The algorithms we applied in-clude FoF, DBSCAN, benchmark HDBSCAN and our HDBSCAN.We use 𝑀𝑚𝑖𝑛 = 1𝑒6, 5𝑒6, 1𝑒7M⊙ for benchmark HDBSCAN andour HDBSCAN. For FIRE data, FoF and DBSCAN can’t distin-guish different groups well in the center of the image. For bench-mark HDBSCAN, while it can distinguish more detailed clusters,change of 𝑀min can’t affect the cluster result well so that we can’t visualize the clusters based on different levels. For our HDBSCAN,𝑀min = 5𝑒6, 1𝑒7M⊙ both look better because they can have detailedrecognization of clusters but the noise is not too much. For The-san data, FoF, DBSCAN, and benchmark HDBSCAN all have thesame problem of not distinguishing the central part in the image.Our HDBSCAN with 𝑀min = 5𝑒6M⊙ has a reasonable clusteringbecause the clusters identified in the central part corresponds withthe lightened places density map

应用于 FIRE 和 Thesan 的聚类算法图像分别如图 5 和图 6 所示。我们使用𝑀𝑚𝑖𝑛 = 1𝑒6, 5𝑒6, 1𝑒7M⊙对基准 HDBSCAN 和我们的 HDBSCAN 进行聚类。对于 FIRE 数据，FoF 和 DBSCAN 无法很好地区分图像中心的不同组。对于基准 HDBSCAN，𝑀min 的变化虽然能区分出更细致的聚类，但并不能很好地影响聚类结果，因此我们无法直观地看到基于不同层次的聚类。对于我们的 HDBSCAN，𝑀min = 5𝑒6、1𝑒7M⊙的效果都比较好，因为它们都能对聚类进行详细识别，而且噪声也不大。对于 The-san 数据，FoF、DBSCAN 和基准 HDBSCAN 都存在同样的问题，即无法区分图像的中心部分。

#### 3.8 index performance 指数表现

For FIRE and Thesan, index statistics are shown in Tables 2 and3, with the top three scores for each index highlighted in color. OurHDBSCAN, with a mass bound parameter of 5·106M⊙ , ranks withinthe top three for all indices and notably achieves the top rank for allindices in the Thesan dataset. Specifically, among the three indicesevaluated, our method ranks in the top three for seven out of ninecases in the FIRE data and for six out of nine cases in the Thesan data,whereas the benchmark method ranks in the top three only twice forFIRE data and three times for Thesan data
表 2 和表 3 显示了 FIRE 和 Thesan 的指数统计，每个指数的前三名用颜色标出。在质量边界参数为 5-106M⊙ 的情况下，我们的 HDBSCAN 在所有指数中均排名前三，在 Thesan 数据集中的所有指数中均名列前茅。具体地说，在评估的三个指标中，我们的方法在 FIRE 数据的九个案例中有七个进入前三名，在 Thesan 数据的九个案例中有六个进入前三名，而基准方法在 FIRE 数据中只有两次进入前三名，在 Thesan 数据中只有三次进入前三名。

#### 3.9 attune with parameters

Based on the indexes performance, we find that around 5𝑒6 is ap-propriate parameter for 𝑀min. Also, interestingly, we have a findingthat, As 𝑁core changes, the regression slope for mass, Rhalf changes.We draw on degree regression and find that the slope and intercepthas a strict negative linear relationship. Based on former conclusionthat regression slope should be around 0.19 and intercept is -2.08 for𝑧 = 5.5. As in figure ??, on the left is the fire, some slopes are around0.19, but the intercept are around -2.4, a little different from -2.05.For the right, it’s Thesan data. There are still some slopes around0.19, but other slopes fall out of this number, though forming a neg-ative linear relationship. We choose the 𝑀min which makes slopenearest to 0.19.

根据指数的表现，我们发现 5𝑒6 左右是𝑀min 的合适参数。此外，有趣的是，我们发现随着𝑁core 的变化，质量、Rhalf 的回归斜率也会发生变化。根据之前的结论，当𝑧 = 5.5 时，回归斜率应为 0.19 左右，截距为-2.08。如图所示，左边是火灾数据，部分斜率在 0.19 左右，但截距在-2.4 左右，与-2.05 稍有出入。右侧是 Thesan 的数据，仍有一些斜率在 0.19 左右，但其他斜率超出了这一数字，虽然形成了负线性关系。我们选择的 𝑀min 使斜率与 0.19 最接近。









**Q：常见的聚类算法有哪些**

- K-means
- DBSCAN[解释链接](https://www.bilibili.com/video/BV1aA4y1o7UG/?spm_id_from=333.337.search-card.all.click&vd_source=bd4523c00efa2032e4b25b7202aca0ff) 
- 层次聚类(Hierarchical Clustering) 

相关密度OPTICS DENCLUE BIRCH ROCK AGNES(AGlomerative NESting)



HDBSCAN https://geeksforgeeks.org/hdbscan/

> **聚类**是一种**机器学习**技术，它根据相似性将数据分成组或簇。通过将相似的数据点放在一起，将不相似的点分成不同的簇，它试图揭示数据集中的底层结构。
>
> 在本文中，我们将重点介绍**HDBSCAN（基于密度的带噪声应用的层次化空间聚类）**技术。与其他聚类方法一样，HDBSCAN 首先确定数据点的接近度，区分高密度区域和稀疏区域。但 HDBSCAN 与其他方法的不同之处在于，它能够动态调整数据中不同密度和形式的聚类，从而产生更可靠、适应性更强的聚类结果。
>
> ## 什么是 HDBSCAN？
>
> HDBSCAN 是一种[聚类](https://www.geeksforgeeks.org/clustering-in-machine-learning/)算法，旨在根据数据点的密度分布发现数据集中的聚类。与其他一些聚类方法不同，它不需要预先指定聚类数量，使其更能适应不同的数据集。它使用高密度区域来识别聚类，并将孤立或低密度点视为噪声。HDBSCAN 对于结构复杂或密度不同的数据集特别有用，因为它创建了一个分层的聚类树，使用户能够以不同的粒度级别检查数据。
>
> ## HDBSCAN 如何工作？
>
> HDBSAN 检查数据集中数据点的密度。它首先计算基于密度的聚类层次结构，该层次结构从密集连接的数据点创建聚类。这种层次结构可以识别各种形状和大小的聚类。
>
> 然后，该算法从层次结构中提取聚类，同时考虑到层次结构不同层级之间的聚类分配的稳定性。它将稳定聚类识别为在多个层级上具有一致成员资格的聚类，从而确保聚类形成的稳健性。
>
> 此外，HDBSCAN通过考虑密度低的点或不属于任何簇的点来区分[噪声和有意义的簇。HDBSCAN 通过不断调整最小簇大小参数并添加最小生成树来捕获和消除噪声。](https://www.geeksforgeeks.org/noise-models-in-digital-image-processing/)
>
> ## HDBSCAN 的参数
>
> HDBSCAN 有许多参数可以调整，以修改特定数据集的聚类过程。以下是一些主要参数：
>
> - ***min_cluster_size***：此参数设置形成簇所需的最小点数。不满足此标准的点被视为噪声。调整此参数会影响算法找到的簇的粒度。
> - ***min_samples***：它设置一个点在邻域内被视为核心点的最小样本数量。
> - ***cluster_selection_epsilon***：此参数设置基于最小生成树选择聚类的[epsilon 值](https://www.geeksforgeeks.org/epsilon-naught-value/)。它确定在基于密度的聚类过程中，允许点之间的最大距离，以使它们被视为连通。
> - ***\*‘metric’：\***用于计算相互可达距离的距离度量。
> - ***\*'cluster_selection_method'\****：此方法用于从浓缩树中选择聚类。它可以是 'eom'（Excess of Mass'）、'leaf'（聚类树叶）、'Leaf-dm'（带有距离度量的树叶）或 'flat'（平面聚类）。
> - ***\*‘alpha’：\****影响聚类合并的链接标准的参数。
> - ***\*‘gen_min_span_tree’：\****如果参数为真，则生成[最小生成树](https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/)以供以后使用。
> - ***\*‘metric_params’：\****它是度量函数的附加关键字参数。
> - ***\*“algorithm”：\****用于相互[可达距离](https://www.geeksforgeeks.org/check-if-it-is-possible-to-reach-the-point-x-y-using-distances-given-in-an-array/)计算的算法。选项包括“best”、“generic”、“prime_kdtree”和“boruvka_kdtree”。
> - ***\*'core_distance_n_jobs'\****：为核心距离计算运行的并行作业数。
> - ***\*‘allow_single_cluster’：\****一个布尔值，指示是否允许单个集群输出。
>
> 
>
> ## HDBSCAN 聚类的优势
>
> HDBSCAN 聚类的一些优点包括：
>
> - ***\*自动聚类发现：\****它自动确定数据集中的聚类数量，而无需事先指定，使其适用于密度各异、结构复杂的数据集。
> - ***\*处理聚类形状：\****它可以识别不同形状和大小的聚类，包括非凸和形状不规则的聚类。
> - ***\*层次聚类：\**** HDBSCAN 构建[层次聚类](https://www.geeksforgeeks.org/hierarchical-clustering-in-data-mining/)结构，允许探索不同粒度级别的聚类，为数据的底层结构提供有价值的见解。
>
> ## HDBSCAN 聚类的缺点
>
> - ***\*计算密集型：\****由于最小生成树的构建和相互可达距离的计算，HDBSCAN 的计算成本可能很高，特别是对于大型数据集而言。
> - ***\*对距离度量敏感：\****在 HDBSCAN 中，使用的距离度量会影响聚类结果。某些距离度量可能无法准确捕捉数据的底层结构，从而导致[聚类](https://www.geeksforgeeks.org/difference-between-agglomerative-clustering-and-divisive-clustering/)结果不理想。
> - ***\*参数敏感性\****：尽管 HDBSCAN 对参数设置的敏感度比其他一些聚类算法较低，但它仍然需要参数调整，特别是最小聚类大小和最小样本参数，这会影响聚类结果。



### 你可以在meeting上讨论的问题

1. 模拟数据是否需要进行前处理？

2. 是否需要引入其他聚类指标

3. 需不需要考虑其他的聚类算法
4. 是否需要将多个聚类学习器进行集成来进行优化？
5. 目前选择了三组HDBSCAN参数进行模拟，是否需要选择更多的HDBSCAN参数，调整提高模型的性能？
6. 有没有完成星团分类的真实数据或图像，可以用于验证模型的有效性？