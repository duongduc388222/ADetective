Great. I’ll create a comprehensive, step-by-step implementation plan for your Lab Research Assistant Entry Test. This will include:

* Repository structure and naming conventions
* Data handling and filtering for Oligodendrocytes
* Label assignment based on ADNC status
* Donor-level train/test splitting
* Preprocessing checks
* Model development (MLP, Transformer, Foundation Model)
* Setup for Hugging Face Accelerate, bf16 precision, and Flash Attention
* Foundation model selection guidance
* Evaluation metrics and logging
* Optional metadata integration strategy

I’ll let you know once the complete roadmap is ready.


Plan for Building the Oligodendrocyte AD Pathology Classifier

This plan outlines each step to complete the Lab Research Assistant Entry Test. We will load and preprocess the SEA-AD single-cell dataset, filter for Oligodendrocytes, create a donor-level train/test split (High vs Not AD pathology), and implement multiple models (MLP, Transformer, and a fine-tuned foundation model) using PyTorch with Hugging Face Accelerate. Each step is detailed below with guidance and reasoning.

Step 1: Environment Setup and Data Access

Compute Environment: Since you have Google Colab, begin by ensuring it has a GPU runtime enabled (go to Runtime > Change runtime type > Hardware accelerator > GPU). Colab typically provides one GPU (e.g. Tesla T4 or P100); multi-GPU is not available in free Colab, but we will set up our code to be compatible with multiple GPUs in case you use a different platform later.

Install Dependencies: Use pip install for all necessary libraries:

AnnData/Scanpy: for loading the .h5ad file (scanpy will also install anndata).

PyTorch: the latest version (PyTorch 2.x to leverage built-in FlashAttention optimizations if available).

Hugging Face Accelerate: for easy multi-GPU training and mixed precision (bf16).

Foundation model libraries: depending on the chosen model (e.g. install scGPT via its GitHub or pip, or install scfoundation or CellFM if they have installation instructions).

Also install any other common packages you may need (numpy, pandas, scikit-learn for splitting, etc.).

Data Download: The dataset file SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad is available on Synapse. If you haven’t already, log in to Synapse and download this file to your Colab environment. (You mentioned you have it downloaded with columns available, so you might upload it to Colab or mount Google Drive to access it.)

Verify Data File: Ensure the .h5ad file is accessible in your environment (for example, by listing the directory or using !ls). Its size might be large, so be mindful of Colab’s memory limits. If it’s too large for Colab memory, you might consider using an environment with more RAM or only loading necessary parts.

Step 2: Load the AnnData and Inspect Metadata

Load with Scanpy: Use scanpy.read_h5ad("SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad") to load the AnnData object into memory. Store it as adata.

Check AnnData Structure: Inspect adata.shape to see the number of cells (observations) and genes (variables). Also examine adata.obs (cell metadata) and adata.var (gene metadata):

Use adata.obs.columns to list all metadata fields available. The dataset description indicates that adata.obs includes donor-level info like Donor ID, ADNC (Overall AD Neuropathologic Change), cell type annotations, and various pathology and demographic measures.

Print out a sample of adata.obs.head() to see how data is structured per cell. Confirm the presence of key columns:

Donor or Donor ID: identifies the donor each cell came from.

ADNC: The overall AD pathology rating of the donor (“Not AD”, “Low”, “Intermediate”, “High”).

Cell type annotations: likely a column indicating the cell type (possibly hierarchical, e.g., class/subclass/cluster). We will find the column that contains "Oligodendrocyte". Common naming could be cell_type or a taxonomy like cell_class/cell_subclass. (Look for any column with values like "Oligodendrocyte", possibly case-sensitive.)

Other metadata like age, sex, APOE genotype, Braak stage, etc. (We might use some later for optional enhancements.)

Donor-level vs Cell-level: Note that many fields (like ADNC, Braak, APOE) are donor-level attributes. All cells from the same donor will have identical values in those columns. This is important for splitting later (we must split by donor, not by cell).

Understanding ADNC categories: Not AD means the donor had no significant AD pathology, while High means high level of AD neuropathologic change. We will focus on these two extremes as our labels. (Donors labeled Low or Intermediate are to be excluded from this binary classification to avoid ambiguity.)

Verify Oligodendrocyte presence: Ensure that "Oligodendrocyte" is indeed present in the cell type annotations. For example, you can do adata.obs['cell_type'].unique() (using the actual column name) to see the list of cell types in the data. We expect Oligodendrocytes to be one of them.

Step 3: Define Donor Groups (High vs Not AD)

The first task is to identify which donors are “High” vs “Not AD” in terms of pathology:

Identify the Relevant Column: Based on the dataset info, the ADNC column in adata.obs indicates overall AD neuropathologic change. We will use this to define our target groups:

Label 1 (AD High): Donors with ADNC == "High".

Label 0 (Not AD): Donors with ADNC == "Not AD".

Exclude Other Categories: Donors labeled "Low" or "Intermediate" ADNC are neither of interest for this binary classification, so we will exclude cells from those donors entirely. Also exclude any cells with missing or ambiguous donor annotations.

Map Donors to Labels: Create a mapping (dictionary) from donor ID to label:

Extract a DataFrame of unique donors with their ADNC: e.g., donors_df = adata.obs[['Donor ID','ADNC']].drop_duplicates().

Filter this to include only rows where ADNC is "High" or "Not AD".

Then map each remaining donor ID to 1 (if ADNC High) or 0 (if ADNC Not AD).

Document Columns/Values: For clarity, note that we are using adata.obs["ADNC"] (Overall AD Neuropathologic Change) where values are exactly "High" and "Not AD" as the criteria for labels. All cells from donors meeting those criteria will inherit the donor’s label.

Step 4: Filter Cells to Oligodendrocytes Only

We next narrow the dataset to the cell type of interest:

Identify Cell Type Column: Based on the challenge description, the data has hierarchical cell annotations. Often, there might be columns like cell_class, cell_subclass, or cell_type. We need the one that specifically identifies Oligodendrocytes. Suppose the column is named "cell_type" for the main cell type class (if not, use the appropriate column from the earlier inspection).

Subset AnnData by Cell Type: Filter the AnnData to include only cells where cell_type == "Oligodendrocyte" (or the exact matching string from the data, e.g., it could be "Oligodendrocyte" or "Oligo" depending on how it’s abbreviated). In Scanpy, you can do:

adata_oligo = adata[ adata.obs['cell_type'] == 'Oligodendrocyte', : ].copy()


This creates a new AnnData with only Oligodendrocyte cells.

Filter by Donor Group: Further filter adata_oligo to keep only cells whose donors are in our High or Not AD groups:

You can use the donor-label mapping from Step 3. For example, if the donor ID column is Donor ID, do:

valid_donors = donor_to_label_map.keys()  # donors that are High or Not AD
adata_oligo = adata_oligo[ adata_oligo.obs['Donor ID'].isin(valid_donors), : ]


This drops any Oligodendrocyte cell that came from a donor not labeled as High or Not AD (i.e., drops those from Low/Intermediate ADNC donors).

Verify Filter Results: After filtering, it’s important to report the number of donors and cells in each group:

Calculate how many unique High donors and how many unique Not AD donors remain, and how many cells in total for each label.

For example, count adata_oligo.obs[adata_oligo.obs['ADNC']=="High"] cells vs "Not AD". Or simply use the donor mapping on this filtered data.

This ensures that (a) we have more than one donor in each category (so that a split is possible) and (b) no unwanted donors remain.

Maintain Labels in Cell Metadata: It’s helpful to add a new column in adata_oligo.obs for our binary label (so we don’t have to repeatedly look up donor info). For instance:

adata_oligo.obs['label'] = adata_oligo.obs['Donor ID'].map(donor_to_label_map)


This adds a label column of 0s and 1s corresponding to each cell’s donor group.

Step 5: Split Data into Train and Test Sets (Donor-Level Split)

To evaluate models fairly, we must split by donor, not by cell, to avoid any data leak across sets:

Gather Donor IDs: From the filtered Oligodendrocyte data, extract the list of unique donor IDs and their label. For instance, unique_donors = adata_oligo.obs[['Donor ID','label']].drop_duplicates(). Suppose this yields D donors (a subset of the original 84 donors, since we dropped intermediate/low AD cases).

Determine Split Size: Decide on a train/test split ratio. A common choice is 80% of donors for training, 20% for test. However, ensure both sets contain at least one High and one Not AD donor:

Use stratified splitting so that class proportions are preserved as much as possible. For example, if you have 10 High and 10 Not AD donors, you might put 8 of each in train and 2 of each in test.

If the number of donors is small, you might choose a slightly different ratio (e.g., 70/30) to ensure enough test samples.

Perform Donor Stratified Split: You can use sklearn.model_selection.train_test_split on the list of donors:

from sklearn.model_selection import train_test_split
train_donors, test_donors = train_test_split(
    unique_donors['Donor ID'], 
    test_size=0.2, 
    stratify=unique_donors['label'], 
    random_state=42
)


This returns two lists of donor IDs. (Make sure to pass the labels for stratification and set a random seed for reproducibility.)

Optional Validation Split: It’s often useful to carve out a validation set for hyperparameter tuning or early stopping. If desired, take the train donor list and further split it into (say) 80% train and 20% val (again stratified by label). This internal validation will not be used for final evaluation, but to monitor training.

Create Cell-Level Split: Now map these donor lists back to the cells:

Train cells: adata_train = adata_oligo[ adata_oligo.obs['Donor ID'].isin(train_donors) ]

Test cells: adata_test = adata_oligo[ adata_oligo.obs['Donor ID'].isin(test_donors) ]

(Likewise for a val set if made.)

Double-Check Partition: Verify that no donor appears in both sets (they shouldn’t, by construction). Also verify class distribution:

Count how many High/Not AD donors in train vs test, and how many cells from each class in train vs test. We will report these. For example:

Train: X High donors (Y cells), A Not AD donors (B cells)

Test: M High donors (N cells), P Not AD donors (Q cells)

Ensure each set has at least one of each class — if the stratified split was done correctly, this should hold.

Avoid Data Leakage: Since splitting is by donor, and all cells of a donor go to one set, there’s no overlap of donor-level features or pathology across train/test. This addresses the requirement of a strict donor-level split.

Step 6: Data Preprocessing and Feature Preparation

Before training models, we need to check what preprocessing is needed (or already provided) for the gene expression data:

Normalization Status: The dataset description indicates that expression data in the AnnData is log-normalized with raw counts also available
brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net
. This suggests that:

adata.X might already contain log<sub>e</sub>(counts+1) normalized values (which are often used for analysis).

There may be a layer or adata.raw that holds the original UMI counts.

Confirm this by inspecting adata_oligo.X statistics: e.g., check the range of values. If values are mostly around 0–5 or 0–10, that’s typical for log<sub>1p</sub> data (since raw counts could be in the hundreds or thousands, but log1p compresses them). Also, adata.uns might contain a 'log1p' entry indicating if log transform was applied.

Action: If data is already log-normalized, do not re-normalize or log-transform (to avoid doubling up). We will use the provided log-expression values as input features for our models. If it turned out to be raw counts, we would perform sc.pp.normalize_total and sc.pp.log1p, but likely this is unnecessary.

Highly Variable Genes (HVG): Determine if the AnnData is already subset to HVGs or if it contains all genes:

Check adata_oligo.n_vars (number of genes). If this number is very large (e.g., > 20,000 for human), then it likely includes all genes. If it’s around a few thousand, the data might have been filtered to HVGs or specific genes.

Also see if adata.var has an indicator like adata.var['highly_variable']. If so, perhaps the dataset creators flagged HVGs (often top 2000). In many single-cell workflows, one selects ~2000 HVGs for modeling to reduce noise and dimensionality.

If HVGs are not pre-selected: consider selecting a gene subset for the models we train from scratch (MLP and our custom transformer). Using all ~20k genes might be feasible for a simple model but could introduce a lot of noise and slow down training. A common approach is:

Use scanpy.pp.highly_variable_genes on the training set (so it’s unbiased by test) to find top 2000 HVGs
brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net
. We can then subset the data to those genes.

Note: For the foundation model fine-tuning, we should not drop genes that the foundation model expects. The foundation model likely has a fixed vocabulary (embedding) for a certain gene list, so we will handle that separately. For now, consider HVG selection mainly for the scratch models.

If HVGs are provided or already filtered: use those as is, to remain consistent with the challenge data preparation (and document that we used the provided HVGs).

Scaling: Decide if further scaling is needed:

Many models (especially neural nets) can work with log-normalized counts directly. The values are already roughly on a similar scale (log counts). We might not need to standardize each gene to zero mean/unit variance, but doing so could help the MLP converge (at the risk of losing some inherent meaning of zero = no expression).

A safe approach for MLP and transformer: z-score scale each feature gene based on the training set mean and std, then apply the same scaling to train/val/test. This centers the data and can improve optimization. You can achieve this by:

Calculating gene_means and gene_stds from adata_train.X (for each gene).

Subtracting and dividing for all sets. (Be careful to handle genes with zero std – though with many cells, most genes expressed in at least some cells will have non-zero variance.)

Alternatively, use sklearn.preprocessing.StandardScaler fitted on train data.

Do not apply scaling that mixes train and test data. Always compute any normalization parameters on train only.

For foundation model input, do not perform this kind of scaling unless the model’s instructions say so. The foundation model likely expects data in the same format as it was trained (which might be log-normalized counts or even raw counts binned – we’ll address that when preparing those inputs).

Feature Engineering: At this stage, our features for each cell are just the gene expression values (for selected genes). We are not yet including donor metadata features – that will be an optional extension later. So each cell is represented by a vector of length equal to number of genes (e.g., 2000 HVGs or full gene set).

Summary of Preprocessing Choices: We will document whether we ended up using all genes or HVGs, and whether we applied scaling:

Example: “We found the AnnData contained 24,000 genes. We selected the top 2,000 highly variable genes from the training set for modeling to reduce dimensionality
brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net
. The expression values were already library-size normalized and log<sub>1p</sub>-transformed, so we did not re-normalize. For the MLP and custom transformer, we further standardized each gene to zero-mean/unit-variance using training data statistics. These scaled log-expression values were used as input features.”

(Adjust this explanation to whatever decisions you make, and ensure to mention if you kept all genes for foundation model alignment.)

Step 7: Model Training Setup (Accelerate, Mixed Precision, and FlashAttention)

Before implementing individual models, set up the training framework with PyTorch and Hugging Face Accelerate for efficient training:

Hugging Face Accelerate: This library helps distribute training across multiple GPUs and manage mixed precision easily. You should:

Run accelerate config in a terminal (or use the Accelerate CLI) to create a default configuration. Choose multi-GPU if you expect to use more than one GPU (in Colab you likely have one GPU, but configure anyway for generality). You can specify device IDs or just accept defaults. Also specify that you want FP16 or BF16 mixed precision in the config (or you can override in code).

In the training scripts, use from accelerate import Accelerator. Then initialize an accelerator with accelerator = Accelerator(mixed_precision='bf16'). This will enable bfloat16 precision where supported (GPUs like A100 or V100 support bf16; if bf16 isn’t supported on your GPU, Accelerate might automatically fall back to FP16 or full precision).

Use the accelerator to wrap your model, optimizer, and data loaders. For example:

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)


This will move the model to GPU(s), split the data among GPUs, and set up for distributed training if applicable.

Multi-GPU Support: If you have access to multiple GPUs (e.g., on a cluster or cloud VM), you can launch the training with accelerate launch --multi_gpu script.py. This will spawn processes for each GPU and the Accelerator will coordinate them. In your code, after calling accelerator.prepare, the training loop generally looks similar to single-GPU but behind the scenes it averages gradients, etc. (On a single GPU, accelerator.prepare is still fine – it will just use one device.)

Mixed Precision (bf16): Using bfloat16 can speed up training and reduce memory usage without much loss in model accuracy. With mixed_precision='bf16', operations will use bf16 where possible (on TPUs or newer GPUs like Ampere architecture). Keep an eye on any operations that might not support bf16 – usually, Accelerate will handle casting automatically. If you encounter any NaNs or instability, you might try FP16 or full precision as a fallback.

Flash Attention for Transformers: Flash Attention is an optimized method to compute attention more efficiently (less memory, faster). In PyTorch 2.0+, the default scaled dot-product attention can automatically use flash attention under the hood when certain conditions are met (GPU with CUDA 11+ and sm_80 or higher, sequence length and head dimension constraints). For our custom transformer model:

We can use the PyTorch nn.TransformerEncoder or nn.MultiheadAttention modules which internally call the optimized kernels. Make sure you use PyTorch 2.x and set device='cuda' on the model.

Alternatively, PyTorch provides torch.nn.functional.scaled_dot_product_attention which uses flash attention if possible. If writing a custom attention loop, using this function would be beneficial.

You don’t need to do much beyond using these modern PyTorch features; however, you could verify by running on an A100 GPU and checking performance, or explicitly enabling it via environment flags if needed. If using Colab’s default (often a T4), flash attention might not be available (T4 is sm_75, which doesn’t support the newest flash kernels), but the code will still run using regular attention.

Optional library: There is a separate library flash-attn by Dao et al. If you choose, you could install flash-attn and use its ops for even more speed. This is advanced and not strictly necessary if PyTorch’s built-ins suffice. For the scope of this test, relying on PyTorch’s efficient implementation is acceptable.

Logging and Monitoring: Decide how to log training progress. You can simply print the loss and metrics each epoch (Accelerate has a .log method or you can use Python print on only one process using accelerator.is_local_main_process). You may also use libraries like TensorBoard or Weights & Biases, but for simplicity printing to console or storing to a list and outputting at the end is fine.

Deterministic Training: Set a random seed (e.g., np.random.seed, torch.manual_seed, and if multi-GPU, torch.cuda.manual_seed_all) for reproducibility. Also, you can configure torch.backends.cudnn.deterministic = True and torch.backends.cudnn.benchmark = False for deterministic behavior (with some trade-off in performance).

With the infrastructure in place, we can now train the different models.

Step 8: Implement and Train a Baseline MLP Model

First, establish a simple feed-forward neural network as a baseline:

Architecture Design: Construct a multi-layer perceptron (MLP) for binary classification:

Input size: equal to the number of features per cell (e.g., 2000 genes if using HVGs, or all genes if that’s the approach).

Hidden layers: 2–4 fully connected (Linear) layers. For example:

Layer1: 512 units, ReLU activation (if 2000 input features, 512 might be a reasonable first layer size).

Layer2: 256 units, ReLU.

Layer3: 128 units, ReLU (you can adjust number of layers and units depending on data size).

You may include BatchNorm layers after linear layers to stabilize training, and Dropout (e.g., 0.2–0.5 dropout rate) to reduce overfitting.

Output layer: 1 unit if using a sigmoid/BCE approach, or 2 units if using softmax. We will likely use a single output with sigmoid (since it’s binary).

Activation in the hidden layers: ReLU (Rectified Linear Unit) is standard. The final layer will output a logit (no activation, since we’ll apply the loss which expects a logit).

Loss Function: Use Binary Cross-Entropy with Logits (BCEWithLogitsLoss). This loss function is appropriate for a binary classification and expects logits. (It internally applies a sigmoid to the output and computes binary cross-entropy.) If we had chosen a 2-unit output with softmax, we would use CrossEntropyLoss, but a single-unit BCE is simpler here.

If the classes are imbalanced in the training data, consider using the pos_weight argument of BCEWithLogitsLoss to give more weight to the minority class. For example, if only 30% of training cells are label 1, you might set pos_weight = (number of 0 cells)/(number of 1 cells) to balance the effective loss.

Optimizer: Use Adam optimizer (a good default for neural nets). Set a learning rate (e.g., 1e-3 to start). You can also experiment with AdamW (Adam with weight decay) to regularize, especially if overfitting is observed.

Training Loop:

Create PyTorch DataLoader objects for train (and val, if applicable). Since each “example” is a cell, we’ll use cell-level batches. Ensure shuffling is True for training loader, and False for validation/test loader.

Decide batch size based on dataset size and memory; for instance, batch size 64 or 128 cells might be fine. (If the dataset is large, adjust accordingly. With ~ thousands of Oligodendrocyte cells, this is fine. If it’s tens of thousands, you can still use similar batch sizes and it will iterate more steps per epoch.)

For each epoch:

Set model to train mode, iterate over train DataLoader:

For each batch, use accelerator.forward or simply do:

inputs = batch_X  # shape [batch_size, num_features]
labels = batch_y  # shape [batch_size]
outputs = model(inputs)  # shape [batch_size, 1] logits
loss = loss_fn(outputs, labels.float())


Backpropagate: optimizer.zero_grad(), accelerator.backward(loss) (so that gradients are handled properly in multi-gpu), then optimizer.step().

If using a validation set, evaluate on val loader each epoch: set model to eval, loop through val data to compute loss and accuracy/F1, etc. Use torch.no_grad() context for this.

Print or log the epoch’s train loss and val loss/accuracy. This helps monitor for convergence or overfitting.

(Optionally, implement early stopping: if val loss doesn’t improve for N epochs, break out. Or save the best model on val metric.)

Continue for a reasonable number of epochs (e.g., 10, 20, or more). Since this is a small binary classification, 10-20 epochs might be sufficient if learning rate is appropriate. You can watch the loss; if it plateaus or starts increasing (val loss), you might stop.

Accelerate Usage: Remember to wrap the model, optimizer, and data loaders with accelerator.prepare(...) before the training loop. During training, use accelerator.backward(loss) instead of loss.backward(). Also, retrieve any metrics from GPU to CPU (e.g., preds = accelerator.gather(predictions) if you need to gather results from all GPUs).

After Training: Evaluate final performance on the test set (we’ll detail this in the evaluation step). Also, note down the final training accuracy vs test accuracy to see if there’s overfitting (e.g., if training accuracy is 99% but test is 70%, that indicates overfit).

Expected Results: The MLP gives us a baseline. It may achieve moderate accuracy/F1. If Oligodendrocyte gene expression carries a signal of AD pathology, the model should do better than random. Keep this result for comparison with the more complex models.

Step 9: Implement and Train a Transformer-Based Model

Next, we develop a custom transformer model to see if capturing gene interactions yields better performance. The key challenge is how to represent a cell’s gene expression as a sequence for the transformer:

Sequence Representation of a Cell: We will treat each gene as a position in a sequence and the gene’s expression value as the information at that position. For example, if we use 2000 genes, each cell becomes a sequence of length 2000 (plus maybe one special token). Directly feeding a length-2000 sequence into a transformer is feasible. Each “token” in this sequence isn’t a discrete word but rather a gene with an expression value. We need an embedding strategy:

Gene Embedding: Create a trainable embedding vector for each gene. Essentially, we’ll have an embedding matrix of shape [num_genes, embed_dim]. This will allow the model to learn a representation for each gene identity (capturing properties of that gene across all cells).

Encoding Expression Level: We can combine the expression value with the gene embedding in a couple of ways:

Multiplicative: Multiply the gene’s embedding vector by a scalar derived from the expression value. For instance, use expr_value (which is already log-normalized) directly as the scalar. So the token representation = expr_value * Embedding[gene]. This means if a gene is not expressed (0), it contributes a zero vector; higher expression scales up that gene’s embedding features.

Additive: Or learn a small linear projection for expression: e.g., have another embedding or MLP that takes the expression (possibly bucketed into bins or just as a continuous value) and produces an embedding vector, then add it to the gene’s base embedding. For simplicity, the multiplicative approach is straightforward and has the interpretation that the embedding provides gene-specific direction and the expression provides magnitude.

We should also consider normalizing or constraining the expression values when multiplying to avoid extremely large values. However, since our inputs are log counts, their range is not huge (e.g., often 0 to ~5 for many genes). We can still clip or normalize across genes if needed.

Special [CLS] Token: It’s common in classification transformers to prepend a special learnable token that represents the “sentence” (here, the cell). We will add a [CLS] token at the beginning of each sequence. This token’s embedding (initialized randomly) will be updated to aggregate information from all gene tokens via attention.

So, sequence length becomes num_genes + 1. Index 0 corresponds to CLS, indices 1...N correspond to each gene.

The final hidden state of the CLS token will be used as the cell’s representation for classification.

Transformer Architecture: Using PyTorch’s nn.TransformerEncoder:

Embedding Layer: Implement a module to generate embeddings for the whole sequence:

Create gene_embedding = nn.Parameter(torch.randn(num_genes, embed_dim)) (or use nn.Embedding(num_genes, embed_dim) which also gives a convenient interface).

Create cls_embedding = nn.Parameter(torch.randn(1, embed_dim)) for the CLS token.

For a given batch of cell expression data (shape [batch, num_genes]):

Expand cls_embedding to shape [batch, 1, embed_dim] (repeat for batch).

Lookup gene embeddings for all gene indices (0..N-1). Since each sequence always contains all genes in a fixed order, we can just use the embedding matrix (it’s effectively like positional encoding).

Multiply the gene embeddings by the expression values. E.g., if expr is [batch, num_genes], we can reshape it to [batch, num_genes, 1] and do expr * gene_embedding_matrix (broadcasting the scalar to the embed_dim). This yields [batch, num_genes, embed_dim].

Concatenate the CLS embedding at the start: final input to transformer of shape [batch, num_genes+1, embed_dim].

(Optionally, you might add a small positional encoding as well, although here each gene is essentially a “position” with a fixed identity. A positional encoding isn’t strictly necessary since the gene identity is already encoded and we do not want the model to think gene 1 and gene 2 have an order relationship beyond what their embeddings encode. We can skip explicit position encoding because each gene index can be considered its position encoding.)

Encoder Layers: Choose the number of layers and model dimension:

Set d_model = embed_dim (embedding size, e.g., 64 or 128). This must be divisible by the number of attention heads.

Number of heads: e.g. 4 or 8 heads. (If embed_dim=128, 8 heads of 16-dim each works).

Feed-forward dimension (dim_feedforward in TransformerEncoder): typically 2-4x d_model (e.g., 256 or 512 if d_model=128).

Number of layers: 2 or 3 layers can be sufficient for this task. (More layers increase complexity and risk overfitting given limited data.)

Include dropout in the TransformerEncoder (PyTorch’s default is no dropout unless specified; you can set dropout=0.1 for the MultiheadAttn and FFN sublayers).

Example construction:

encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=0.1, batch_first=True)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)


Here, batch_first=True allows input shape [batch, seq_len, d_model].

Classification Head: After the transformer encoder, we get an output of shape [batch, seq_len, d_model]. We care about the [CLS] token output which will be at index 0 of the sequence (if we prepended it). So extract cls_output = output[:, 0, :] which has shape [batch, d_model].

Feed this through a simple classifier layer: e.g., cls_output -> nn.Linear(d_model, 1) to get a logit for each cell.

We’ll train this end-to-end along with the transformer and embeddings.

Leverage Flash Attention: By using PyTorch 2’s implementation, the multi-head attention should automatically use optimized kernels. Ensure your code is using GPU and, if possible, test on an Ampere GPU (A100) for real flash attention. If using Colab’s T4, it will still run correctly, just without that specific optimization. No code changes are needed beyond using the built-in TransformerEncoderLayer. (If one wanted, they could also use ScaledDotProduct attention directly, but that’s not necessary here.)

Training the Transformer: Training loop is similar to the MLP:

Use the same train/test split of cells prepared earlier (train model on adata_train cells).

Use the same loss (BCEWithLogitsLoss) and optimizer (Adam). Possibly a lower initial LR like 1e-4 might be beneficial for the transformer since it has more parameters and a more complex architecture (to avoid overshooting minima). Monitor training; if it's learning slowly, you can increase LR a bit.

With Accelerate, prepare the model and dataloaders similarly. The major difference is the forward pass now involves the transformer sequence creation.

Memory considerations: A transformer with sequence length ~2000 and batch size 64 can be heavy on memory, especially with 3 layers and multi-heads. If you encounter CUDA OOM, you may need to:

Reduce batch size (e.g., to 32 or 16).

Reduce sequence length (e.g., use fewer genes if possible or consider splitting into smaller chunks – but splitting would lose global attention, so better to reduce gene count via HVG).

Use gradient accumulation to simulate larger batch if needed.

Ensure mixed precision is on, which saves memory.

Monitor Overfitting: The transformer has many more parameters than the MLP, so it might overfit training data. Keep an eye on training vs validation loss. If it overfits, you might:

Add more dropout in the model.

Stop early or use regularization like weight decay.

In extreme cases, reduce model size (fewer layers or smaller d_model).

Expected Performance: Ideally, the transformer might capture interactions between genes that a simple MLP might not. For example, combinations of gene expression patterns related to AD pathology could be picked up. However, transformers also need a lot of data to shine. If our dataset of Oligodendrocytes is not very large, the transformer might struggle to outperform the simpler MLP, or might need careful tuning. We will see the results in the evaluation step.

Step 10: Fine-Tune a Pretrained Foundation Model (scGPT/scFoundation/CellFM)

The challenge requires fine-tuning at least one pretrained single-cell foundation model on our task. These models (like scGPT, scFoundation, CellFM) have been pre-trained on large single-cell datasets to learn meaningful gene expression patterns
bmcgenomics.biomedcentral.com
. We will choose one (e.g., scGPT) to fine-tune for classifying High vs Not AD donor cells.

About scGPT: scGPT is a transformer-based model analogous to GPT, but for single-cell gene expression data
rna-seqblog.com
. It treats gene expression profiles like language, where genes (and possibly binned expression levels) are tokens. It was trained on millions of cells, learning both gene representations and cell representations in an unsupervised manner
bmcgenomics.biomedcentral.com
. Fine-tuning such a model can leverage this prior knowledge for specific tasks.

Set Up scGPT: Follow the instructions from the scGPT repository:

Install the scGPT package (if available via pip or by cloning the GitHub and adding to sys.path).

Download the pretrained weights. The scGPT GitHub or documentation should provide a link or script to get the pretrained model (possibly on HuggingFace model hub or as a checkpoint file). Ensure you get the weights that correspond to the pre-training on a large corpus (e.g., 33 million cells).

Also obtain the gene vocabulary used in pre-training. In scGPT, there is usually a list of gene names or IDs that the model knows (its “vocabulary” for gene tokens). This might be provided with the model or in their data preprocessing code.

Gene Vocabulary Alignment: This is a crucial step:

Compare our dataset’s gene identifiers to the model’s gene vocabulary. Our dataset likely uses gene symbols (e.g., “APP”, “MAPT” etc.) or Ensembl IDs. The foundation model likely uses gene symbols (perhaps uppercase HGNC symbols) for human genes.

Create a mapping from our gene index to the model’s token index:

For each gene in our adata_oligo.var_names, find if it exists in scGPT’s vocab list. If yes, note the index.

If there are genes in our data not in the vocab, decide what to do: we might skip those genes or map them to an “unknown gene” token if the model has one. Ideally, most common genes should be covered if the model was trained on broad data.

If the model expects a fixed input size (some scLLMs expect you to provide a sequence of gene tokens that match their vocab length or a subset), ensure we feed them correctly.

In scGPT’s pre-processing, often they bin expression values into categories (like gene is “off” vs “low” vs “high” expression, represented as separate tokens). For example, it might represent a cell as a sequence like “[CLS], GeneA_high, GeneB_off, GeneC_medium, ...” etc. Check the scGPT documentation for how to format input.

If such binning is required, implement the same binning: e.g., determine expression cutoffs for “low/medium/high” or use their provided function to tokenize a cell’s expression.

This effectively creates multiple tokens per gene (gene identity combined with expression level category).

Document the mapping: Clearly note how gene symbols from our data were mapped to gene tokens in the model, and if any genes were dropped or had to be renamed. For example, if our dataset uses gene aliases or older names that differ from the model’s, we might have to manually map those to the correct vocab entry.

Adapt the Model for Classification: Pretrained scGPT is a generative model (likely trained to reconstruct masked genes or similar). We need to adapt it to output a classification:

One approach: add a classification head on top of the scGPT model’s CLS token (if it has one) or on the pooled cell embedding. For instance, scGPT might output a representation for the whole cell (perhaps a special token or an averaged embedding). We attach a linear layer to that of size 1 (for our binary class).

Another approach: if scGPT was pre-trained with a CLS token and maybe even fine-tuned in examples for cell type classification, it might have some built-in way to do classification. (Check if the library has a sequence classification or cell classification example; perhaps their retina fine-tuning protocol is for cell types).

Either way, we’ll likely fine-tune all model weights (since the task specifically says to fine-tune, not just use it as feature extractor). Given the model’s size, consider using a smaller learning rate (e.g., 1e-5 or 1e-4) when updating a pre-trained model, to avoid destroying learned features too quickly.

We can also consider freezing some lower layers and only training the top layers or new head if we have very limited data, but the instructions didn’t explicitly mention that, and fine-tuning typically implies updating the model.

Training with Accelerate: Even though scGPT is a HuggingFace/transformer style model, we can still use our custom training loop with Accelerate:

Wrap the model and optimizer with accelerator.prepare as before. If using HF Transformers library, ensure the model is on cuda and set to train.

Use the same data (train cells) but we must convert each cell into the format expected by scGPT:

Write a collate function for DataLoader that takes a batch of cell gene vectors and converts them to token sequences (including adding CLS, etc., according to scGPT’s tokenizer).

Alternatively, if scGPT provides a ready Dataset class or tokenizer, use that. For example, they might have something like scgpt.data.Preprocessor to turn AnnData into token tensors.

The output will be something like a tensor of token IDs for each cell and possibly attention masks (if the sequence length varies or to mask padding, though in our case sequence might be fixed length of all genes or of all expressed genes).

Define the loss: since we have a binary label, use BCEWithLogitsLoss on the model’s output (assuming we have added a classification head that outputs a single logit for each sequence). If using an existing model head that outputs 2 logits (for two classes), use CrossEntropyLoss.

Train for a number of epochs similar to before (maybe fewer if the model is large and already somewhat specialized).

Monitor memory usage – foundation models can be large (scGPT might have hundreds of millions of parameters). On a single GPU with limited memory, you may need a small batch size (even batch size 8 or 16 might be needed). If multi-GPU is available, Accelerate will spread the batches.

Use mixed precision (bf16) to help with memory. Also consider gradient accumulation if batch size is constrained.

Performance Expectations: Fine-tuning a foundation model could potentially give a boost if the model’s learned gene patterns are relevant to AD pathology. For instance, it might already encode co-expression patterns or pathways that correlate with disease. However, it’s not guaranteed to outperform simpler models
bmcgenomics.biomedcentral.com
. In some cases, simple models with well-chosen features can rival large pre-trained models on specific tasks
bmcgenomics.biomedcentral.com
. We will compare results after evaluation.

Other Foundation Models: If time permits, you could try scFoundation (which is a graph neural network-based model using gene embeddings from pretraining) or CellFM. The procedure would be analogous: load pre-trained model, align gene inputs, add a classifier, fine-tune. For brevity, focusing on one (scGPT) is acceptable as it was explicitly suggested and is a transformer (making use of flash attention as well).

Step 11: Evaluation on the Test Set

After training each model (MLP, custom Transformer, and fine-tuned foundation model), evaluate them on the held-out test set of donors (all cells from those donors):

Prepare Test Data: We have adata_test which contains all test cells with their true labels. Depending on the model, ensure the data is in the correct format:

For MLP: a matrix of gene features (scaled as needed, just like train).

For custom Transformer: the gene expression sequences (we can reuse the same embedding logic inside the model for test predictions).

For scGPT: tokenized sequences as input to the model.

Run Predictions: For each model, get the output logits for test cells:

Put the model in eval() mode and disable gradients (torch.no_grad()).

If the model outputs logits, apply a sigmoid (for BCE setup) or softmax (for 2-class output) to get predicted probabilities. Typically, we’ll classify label 1 if sigmoid >= 0.5 (or argmax of softmax for 2-class).

Collect all predictions and true labels.

Compute Metrics:

Accuracy: straightforward to compute as (predicted_labels == true_labels).mean(). This gives the fraction of cells correctly classified.

F1 Score: Because this is binary, compute the F1 for the positive class (or we can give both class F1s, but usually the "High" class (label=1) F1 is of interest). Use sklearn.metrics.f1_score(y_true, y_pred) with pos_label=1.

F1 is useful if the classes are imbalanced, as it balances precision and recall.

ROC-AUC: We can also compute the Area Under the ROC Curve. Use sklearn.metrics.roc_auc_score(y_true, prob_scores) where prob_scores are the predicted probabilities for class 1. This gives a threshold-independent performance measure.

If classes are balanced and F1 is high, AUC will also be high. If data is imbalanced, AUC might give additional insight.

Confusion Matrix: It might help to see the confusion matrix (TP, FP, TN, FN counts) to know where errors are happening (e.g., are we mislabeling many High as Not AD or vice versa).

Ensure to calculate these metrics per model and perhaps aggregate them in a table for comparison.

Class Balance Note: Check the proportion of label 1 vs 0 in the test set. If, say, High donor cells are much fewer than Not AD cells (or vice versa), a high accuracy can be achieved by dominating the majority class. F1 and AUC help guard against being misled by accuracy in such cases. We should mention the test class distribution in our report. For instance, “Our test set contains 5 High donors (500 cells) and 4 Not AD donors (800 cells), so the classes are slightly imbalanced (~38% High cells).” In such a case, an accuracy of 62% could be achieved by always predicting "Not AD", so we rely on F1/AUC to gauge actual performance.

Compare Models:

Present the results of each model side by side. For example:

MLP: Accuracy X%, F1 Y (maybe also AUC).

Transformer: Accuracy A%, F1 B.

Foundation (scGPT): Accuracy M%, F1 N.

We expect some differences:

The MLP might serve as a baseline. Suppose it achieves moderate performance (just hypothetical, e.g., 75% accuracy, F1 around 0.7).

The Transformer could improve if the relationship between genes matters for classification. If AD pathology in Oligodendrocytes has a complex transcriptional signature, the transformer may capture it better. Look if its F1/accuracy surpasses the MLP. If it’s roughly the same or lower, possible reasons:

Not enough data for the transformer to generalize (overfit or underfit).

The signal might be mostly in individual gene levels rather than interactions, so an MLP (which is also quite expressive as it can learn non-linear combos) might suffice.

The scGPT fine-tuned model might do as well as or better than the custom models if the pretraining provided useful features. If scGPT performs better (say F1 improves a bit), we can attribute that to knowledge transfer from large-scale data (perhaps it recognized certain gene modules related to AD that generalize).

If it performs similarly or worse, we should discuss why: possibly the pretraining objective (e.g., generative modeling of expression) doesn’t directly translate to this classification task without more data or careful tuning. Recent research shows foundation models sometimes underperform simpler models in specialized tasks
bmcgenomics.biomedcentral.com
, especially if the task data is limited or the evaluation is strict.

90% F1? If none of the models reached ~0.90 F1, consider reasons:

The biology: Oligodendrocytes might not have a very distinct expression pattern distinguishing High AD vs Not AD. They are support cells, and AD pathology might primarily affect other cells (like neurons or microglia) more dramatically. So the signal in Oligodendrocytes could be subtle.

Data limitations: Only 84 donors and we used perhaps a subset of those (High vs Not AD might be, for example, 20 vs 20 donors). The sample size might not allow extremely high accuracy – especially if there's heterogeneity in gene expression unrelated to AD (e.g., age effects, other co-morbidities causing noise).

Model/generalization: It’s possible the model is slightly underfitting (if training F1 was also not high) or overfitting (if training F1 was high but test low). We should check overfitting signs. If overfitting, better regularization or more data could help; if underfitting, maybe the model complexity or features aren’t enough.

If by chance one model did get ~90% F1, that would be remarkable and indicate a very strong signal (or possibly a leak/overfitting if not careful). We’d then discuss that the model found a clear separating pattern (maybe a particular gene or set of genes that differ strongly between High and Not AD oligodendrocytes).

Summarize Findings: In the README or report, include a concise discussion:

Example: “The MLP achieved an accuracy of 78% (F1=0.75) on the test set. Our custom transformer model achieved similar accuracy ~80% (F1=0.78), suggesting that it captured some interactions but the gain was modest. The fine-tuned scGPT model performed slightly better, with accuracy 82% (F1=0.80). This indicates the pretrained knowledge in scGPT provided a small boost, though not massive – likely because oligodendrocyte-specific pathology signals are subtle. None of the models reached 90% F1; we suspect this is due to limited data and the fact that oligodendrocyte gene expression differences between high AD pathology and no AD are not extremely pronounced, unlike what might be seen in other cell types or larger datasets. Overfitting was observed in the transformer (train F1 0.90 vs test 0.78), implying we could benefit from more training data or stronger regularization.”

Include any observations about class imbalance (e.g., if the model tended to predict the majority class more often, how that affected precision/recall).

Also note which model you would pick as the final one. If the foundation model did best, it might be the top choice. If its complexity wasn’t worth the small gain, one might prefer the simpler model.

Step 12: (Optional) Incorporate Donor Metadata as Additional Features

As an extra step, we can enhance the model inputs with donor-level metadata (age, sex, APOE genotype, pathology measures, etc.) to see if it improves performance. These features are constant for all cells of a donor, but including them could help the model distinguish groups better:

Select Metadata Fields: Good candidates that might correlate with AD pathology:

Age at Death: Numeric (older age might correlate with higher pathology, though not strictly since Not AD could be old but without AD).

Sex: Categorical (possible slight effects on gene expression or AD incidence, encode as binary 0/1 for Male/Female).

APOE Genotype: Categorical (e.g., “3/4”, “4/4”, “3/3”). APOE4 is a risk factor for AD, so donors with one or two 4 alleles are more likely in High pathology. We can encode genotype in a simplified way: e.g., number of E4 alleles (0, 1, or 2) as an integer feature, and maybe a boolean for E2 allele (since E2 is protective). Alternatively, one-hot encode each genotype category.

Braak stage, Thal phase, CERAD score: These are pathology scores that actually contribute to the ADNC category. Including them would almost directly give away the label (e.g., Braak VI and Thal 5 strongly imply High ADNC). Using them might inflate performance in a trivial way, so it’s better to exclude those specific ones for a fair model (since our goal is to classify from gene expression primarily). However, they are allowed as features in principle, but it feels like leaking the answer. We’ll focus on non-ADNC metadata.

Percent amyloid (6E10) or tau (AT8) staining: These are quantitative pathology measures. Again, they correlate with ADNC by definition (High ADNC donors likely have high percent area of amyloid and tau). Including them would make the task too easy (the model could just learn a threshold on these). It’s better to stick with more independent features like age, sex, APOE, maybe education years or PMI (though those might be less directly relevant).

Cognitive status (Dementia/No dementia): This is also highly correlated with pathology (most High ADNC donors have dementia). Including it essentially gives a label-like signal. Probably avoid it to keep the task focused on transcriptomic prediction.

Encoding Metadata:

For each selected donor attribute, add it to each cell’s feature vector or provide it through another input pathway:

If using an MLP: simplest is to concatenate the metadata features to the gene expression vector. For example, if you have 3 metadata features, and 2000 gene features, the input size becomes 2003. We should scale these features appropriately:

Age: scale (e.g., z-score across donors, or divide by a max like 100 to get 0-1 roughly).

Sex: encode Male=0, Female=1 (or vice versa, and it’s already 0/1 so scale not needed).

APOE4 count: values 0,1,2 – you can one-hot encode into two binary features (or three, but one is redundant). Or simply use 0/1/2 as a numeric feature (treat carefully, it’s ordinal but not strictly linear effect; however 2 is definitely higher risk than 1 or 0).

If any feature has missing data for some donors, either fill with mean or have an indicator (but likely all these are filled).

For the transformer model: we can’t directly append to the sequence of gene tokens (that would confuse the model unless we create special tokens for metadata). Instead, we can combine at the classification head:

One approach: learn an embedding for each metadata feature and add it to the CLS token embedding at input. For example, add age as a scalar to CLS embedding somehow. This is tricky to do properly.

Another approach: process metadata through a small MLP to get a metadata-vector of same size as d_model, and concatenate or add it to the CLS output before the final linear layer. E.g., compute meta_features = MLP_meta(metadata) to get a [batch, d_model] vector, then do combined = cls_output + meta_features (if same dimension) and then classification layer on combined. Or concatenate and adjust the classifier accordingly.

A simpler hack: treat metadata as additional channels in the MLP that feeds the transformer: e.g., if you appended them to gene features and then made tokens, but that breaks the idea of sequence of genes (since metadata has no “gene index”). So better to incorporate at end.

We have to ensure whatever we do, we do it consistently for train and test.

For scGPT or foundation model: They are not really designed to take extra features. One could conceive encoding age as a special token like “Age_80” in the sequence if the model had such tokens (some research works do that by adding pseudo-genes or special markers). But without modifying the model architecture and embedding matrix (which would require adding new tokens and possibly updating embedding weights), it’s complicated. Likely skip adding metadata to the foundation model for now.

Integrate and Train: After adding metadata features, re-train the model (at least the MLP, possibly the transformer) to see if performance improves:

The training process remains the same, just the input data has extra dimensions (for MLP) or the CLS combination has extra inputs.

It might be wise to retrain from scratch rather than continuing from a previously trained model, because the input dimensionality changed. Start with the same initializations and train a new model with metadata included.

Evaluate Improvement: Compare metrics with and without metadata:

Did accuracy or F1 improve? For example, if age or APOE provided additional signal, the model might correct some misclassifications (e.g., an older donor with moderate gene signals might be correctly flagged as High due to age+APOE).

If improvement is marginal or none, perhaps the gene expression already captured the relevant effects of those covariates (or the model couldn’t utilize them effectively).

If improvement is significant, note which features likely contributed (APOE is a strong one for AD risk, so it might help distinguish borderline cases).

Document Usage: In your report, describe which fields were used and how encoded:

Example: “We incorporated donor age, sex, and APOE genotype as additional features. Age was standardized (mean ~80, sd ~10 years in our cohort). Sex was one-hot (Male=0, Female=1). APOE genotype was encoded as an ordinal feature for number of APOE4 alleles (0,1,2). These features were concatenated to each cell’s gene expression vector for the MLP model. The MLP with metadata achieved an F1 of 0.82 vs 0.75 without metadata, indicating a noticeable improvement, likely because APOE4 status and age helped differentiate High pathology donors. We did not add pathology scores as features, since those would directly overlap with the label (making the task trivial).”

Step 13: Prepare Deliverables (Code and Documentation)

Finally, ensure all required deliverables are ready to submit:

Code Repository: Organize your code into a GitHub repository. Avoid using Jupyter notebooks for final submission; instead, provide Python scripts and modules:

For example, have separate scripts for training each model:

train_mlp.py

train_transformer.py

train_scgpt.py

(and possibly train_mlp_with_metadata.py if you did the optional part, or you can integrate that via command-line flag in the same script).

A common structure: a src/ directory with modules (data loading, model definitions, etc.), and in the root, the training scripts that parse arguments and call into those modules.

Make sure any file paths (to the data, etc.) are either relative or given via command-line args for flexibility. The example commands given in the prompt show using --data-path and --cell-type arguments, so implement those in your scripts.

Include the usage of Accelerate in the scripts (either rely on accelerate launch externally or inside the script detect multiple GPUs via Accelerator).

README.md: This is extremely important for the evaluators to understand and run your code. It should include:

Environment Setup: how to install dependencies (requirements.txt should be provided, listing packages like torch, scanpy, accelerate, numpy, etc. Also list any specific versions if needed for compatibility).

Data Download instructions: since the data is from Synapse, mention that the user should download the .h5ad file and where to place it (or how to specify the path).

Donor Group Definition: explicitly state the column and values used (e.g., “We used AnnData.obs['ADNC'] to label donors: 'High' = 1, 'Not AD' = 0.”).

Oligodendrocyte Filtering: state how we filtered by cell type (mention the column name used, e.g., “using AnnData.obs['cell_type'], selecting only 'Oligodendrocyte' cells”). Report how many cells and donors remained in each class after filtering (this shows the dataset size going into modeling).

Train/Test split: describe that we did donor-level split (and if applicable, mention how many donors ended up in train vs test, with class breakdown).

Preprocessing: note whether data was already log-normalized (likely yes) and any feature selection or scaling applied. E.g., “Using 2000 HVGs (highly variable genes) as features, scaled to zero mean/unit var.” If no HVG filtering, state all genes were used and how you handled that in models.

Model Architectures & Hyperparameters: for each model (MLP, Transformer, scGPT), summarize the architecture and any key settings:

MLP: number of layers, hidden units, dropout, etc.

Transformer: number of layers, embedding dim, heads, how we represented input (this is important to explain since it’s a custom design), and that we used flash attention through PyTorch 2.0.

scGPT: which pre-trained model used, how gene vocabulary was aligned (possibly mention if any genes were dropped or how many matched out of total), any special adjustments (like learning rate, freezing).

Training details: batch size, number of epochs, optimizer (Adam), loss function (BCEWithLogits).

How we utilized Accelerate: e.g., “Training was done with HuggingFace Accelerate to seamlessly support data parallelism. We used accelerate launch to run on 2×A100 GPUs with mixed precision bf16.” (Adjust based on what you actually used; if you only had one GPU, you can still mention accelerate was used for future scalability.)

How we enabled bf16: e.g., through Accelerator(mixed_precision='bf16') or via the CLI flag --mixed_precision=bf16.

How flash attention was enabled: e.g., “Our Transformer model leverages PyTorch 2.0’s scaled_dot_product_attention, which utilizes Flash Attention on supported GPUs automatically.” If you installed any specific flash-attn library or xFormers, mention that.

Any other implementation details (seed for reproducibility, etc.).

Results Summary: present the evaluation results on test set for each model. A small table or bullet list works:

MLP: Accuracy, F1, (AUC).

Transformer: same metrics.

scGPT fine-tune: metrics.

If metadata features were tried, mention the effect (e.g., “Adding age and APOE improved MLP F1 from 0.75 to 0.80.”).

Briefly interpret the results (similar to what we outlined in the evaluation step).

How to Reproduce: provide example commands (as given in the prompt) to run your training scripts. For instance, include those accelerate launch examples:

accelerate launch --multi_gpu train_mlp.py --data-path /path/to/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad --cell-type Oligodendrocyte --precision bf16


(And similarly for the transformer and scGPT scripts, including any options like --use-flash-attn if you have a flag for that.)

Make sure the README is clear and concise so that another person can follow and replicate your steps.

Verify Everything: Before submission, run through a fresh instance (if possible) of setting up environment, loading data, training one model, to ensure no dependency or path issues. This catches any missing imports or minor bugs.

Submission: Once all is working and documented, push the code to GitHub. Double-check that no large data files are included in the repo (we only share code, not the actual dataset). Provide the GitHub link as the final answer for the test.