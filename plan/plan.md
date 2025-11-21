# 📋 ADetective Development Plan Overview

**Project**: Oligodendrocyte AD Pathology Classifier
**Duration**: ~12-14 hours
**Objective**: Binary classification of AD pathology (High vs Not AD) using single-cell RNA-seq data

---

## ⚠️ CRITICAL: Large Dataset Notice

**Data File**: `/Users/duonghongduc/Downloads/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad`
**Size**: 35GB

**Implications**:
- Memory constraints during data loading & preprocessing
- Expect slower I/O operations for all file reads/writes
- Intermediate outputs (processed data, checkpoints) require significant disk space
- Consider streaming/chunked processing where applicable
- Test on smaller subsets before full-scale runs

**Recommendations**:
- Monitor RAM usage during Phase 2 (Data Preprocessing)
- Use memory-efficient data structures (sparse matrices, selective loading)
- Save intermediate results regularly to avoid re-processing
- Plan for adequate free disk space (~100GB+ for safety)

---

## 🎯 Project Milestones

### ✅ Planning Phase
- [x] Create multishot development plan
- [x] Design project architecture
- [x] Document implementation phases
- [x] Prepare code templates

---

## 📊 Overall Progress Tracker

```
Phase 1: Environment Setup     [⬜⬜⬜⬜⬜] 0%
Phase 2: Data Preprocessing    [⬜⬜⬜⬜⬜] 0%
Phase 3: MLP Baseline          [⬜⬜⬜⬜⬜] 0%
Phase 4: Transformer Model     [⬜⬜⬜⬜⬜] 0%
Phase 5: scGPT Fine-tuning     [⬜⬜⬜⬜⬜] 0%
Phase 6: Testing & Polish      [⬜⬜⬜⬜⬜] 0%
```

---

## 📝 Detailed Task Checklist

### Phase 01: Environment Setup & Data Acquisition (0.5 hours)
**Progress Tracking**: [phase-01.md](phase-01.md)

- [ ] **Project Structure**
  - [ ] Create directory structure (src/, scripts/, configs/, etc.)
  - [ ] Initialize git repository
  - [ ] Create .gitignore file
- [ ] **Dependencies**
  - [ ] Create requirements.txt
  - [ ] Create requirements_scgpt.txt
  - [ ] Create setup.py
- [ ] **Configuration**
  - [ ] Implement Config class in src/utils/config.py
  - [ ] Set up dual environment support (local + Colab)
- [ ] **Colab Setup**
  - [ ] Create setup_colab.sh script
  - [ ] Test Google Drive mounting
- [ ] **Verification**
  - [ ] Push initial structure to GitHub
  - [ ] Clone and test in Google Colab
  - [ ] Verify GPU access

### Phase 02: Data Exploration & Preprocessing (2-3 hours)
**Progress Tracking**: [phase-02.md](phase-02.md) | **⚠️ LARGE DATA**: See data warning at top of plan

- [ ] **Data Loading**
  - [ ] Implement SEAADDataLoader class
  - [ ] Load SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad
  - [ ] Document data dimensions (cells × genes)
- [ ] **Metadata Exploration**
  - [ ] Identify donor ID column
  - [ ] Identify ADNC column
  - [ ] Identify cell type column
  - [ ] Document all metadata fields
- [ ] **Oligodendrocyte Filtering**
  - [ ] Filter for Oligodendrocyte cells only
  - [ ] Report number of cells retained
- [ ] **Label Creation**
  - [ ] Map ADNC "High" → Label 1
  - [ ] Map ADNC "Not AD" → Label 0
  - [ ] Exclude "Low" and "Intermediate" donors
  - [ ] Report donor counts per class
- [ ] **Data Splitting**
  - [ ] Implement donor-level stratified split
  - [ ] Create train (70%), val (10%), test (20%) sets
  - [ ] Verify no donor overlap between splits
  - [ ] Document split statistics
- [ ] **Feature Processing**
  - [ ] Select highly variable genes (2000 HVGs)
  - [ ] Check normalization status
  - [ ] Apply scaling if needed
  - [ ] Save processed AnnData

### Phase 03: MLP Baseline Implementation (1-2 hours)
**Progress Tracking**: [phase-03.md](phase-03.md)

- [ ] **Model Architecture**
  - [ ] Implement MLPClassifier class
  - [ ] Add batch normalization layers
  - [ ] Add dropout regularization
  - [ ] Initialize weights properly
- [ ] **Training Pipeline**
  - [ ] Implement ModelTrainer class
  - [ ] Set up Accelerate integration
  - [ ] Configure bf16 mixed precision
  - [ ] Add early stopping
- [ ] **Evaluation**
  - [ ] Implement ModelEvaluator class
  - [ ] Add metrics: accuracy, F1, ROC-AUC
  - [ ] Create confusion matrix
- [ ] **Configuration**
  - [ ] Create mlp_config.yaml
  - [ ] Set hyperparameters
- [ ] **Training Script**
  - [ ] Create train_mlp.py
  - [ ] Test with sample data
  - [ ] Run full training
  - [ ] Save model checkpoint
- [ ] **Results**
  - [ ] Record baseline performance
  - [ ] Generate evaluation plots

### Phase 04: Custom Transformer Implementation (2-3 hours)
**Progress Tracking**: [phase-04.md](phase-04.md)

- [ ] **Model Architecture**
  - [ ] Implement GeneTransformer class
  - [ ] Create gene embeddings
  - [ ] Add expression value scaling
  - [ ] Implement [CLS] token
  - [ ] Add positional encoding
- [ ] **Transformer Components**
  - [ ] Configure multi-head attention
  - [ ] Set up encoder layers
  - [ ] Add classification head
- [ ] **Flash Attention**
  - [ ] Check PyTorch 2.0+ availability
  - [ ] Verify GPU compute capability
  - [ ] Enable scaled_dot_product_attention
- [ ] **Training Script**
  - [ ] Create train_transformer.py
  - [ ] Add Flash Attention verification
  - [ ] Handle memory constraints
- [ ] **Configuration**
  - [ ] Create transformer_config.yaml
  - [ ] Optimize for memory usage
- [ ] **Training**
  - [ ] Run training with gradient accumulation
  - [ ] Monitor memory usage
  - [ ] Save best model
- [ ] **Results**
  - [ ] Compare with MLP baseline
  - [ ] Analyze attention weights

### Phase 05: Foundation Model Fine-tuning (3-4 hours)
**Progress Tracking**: [phase-05.md](phase-05.md)

- [ ] **scGPT Setup**
  - [ ] Install scGPT package
  - [ ] Download pretrained weights
  - [ ] Get gene vocabulary
- [ ] **Model Wrapper**
  - [ ] Implement scGPTWrapper class
  - [ ] Add tokenization logic
  - [ ] Create expression binning
- [ ] **Gene Alignment**
  - [ ] Load model gene vocabulary
  - [ ] Map dataset genes to model genes
  - [ ] Report matching statistics
  - [ ] Handle unmatched genes
- [ ] **Fine-tuning Setup**
  - [ ] Implement scGPTFineTuner class
  - [ ] Freeze bottom layers
  - [ ] Add classification head
  - [ ] Configure optimizer with warmup
- [ ] **Training Script**
  - [ ] Create train_scgpt.py
  - [ ] Handle large model memory
  - [ ] Use gradient accumulation
- [ ] **Configuration**
  - [ ] Create scgpt_config.yaml
  - [ ] Set fine-tuning parameters
- [ ] **Training**
  - [ ] Run fine-tuning
  - [ ] Monitor convergence
  - [ ] Save checkpoint
- [ ] **Results**
  - [ ] Evaluate performance
  - [ ] Compare with scratch models

### Phase 06: Testing & Polish (1-2 hours)
**Progress Tracking**: [phase-06.md](phase-06.md)

- [ ] **Comprehensive Evaluation**
  - [ ] Run all models on test set
  - [ ] Generate confusion matrices
  - [ ] Plot ROC curves
  - [ ] Create classification reports
- [ ] **Model Comparison**
  - [ ] Create compare_models.py script
  - [ ] Generate comparison table
  - [ ] Create visualization plots
  - [ ] Identify best model
- [ ] **Optional Enhancements**
  - [ ] Add donor metadata features (age, sex, APOE)
  - [ ] Test performance improvement
  - [ ] Document feature importance
- [ ] **Documentation**
  - [ ] Update main README.md
  - [ ] Add usage examples
  - [ ] Include performance metrics
  - [ ] Add citation information
- [ ] **Google Colab Integration**
  - [ ] Create colab_runner.ipynb
  - [ ] Test complete pipeline
  - [ ] Add visualization cells
  - [ ] Ensure reproducibility
- [ ] **Final Verification**
  - [ ] Test fresh clone from GitHub
  - [ ] Run in clean Colab environment
  - [ ] Verify all dependencies install
  - [ ] Check results reproducibility

---

## 📈 Performance Targets

| Metric | Target | Minimum | Achieved |
|--------|--------|---------|----------|
| Accuracy | >85% | >70% | ⬜ |
| F1 Score | >0.80 | >0.65 | ⬜ |
| ROC-AUC | >0.85 | >0.75 | ⬜ |

---

## 🚀 Deployment Checklist

### GitHub Repository
- [ ] Initialize repository
- [ ] Add comprehensive README
- [ ] Include all source code
- [ ] Add example notebooks
- [ ] Create release tag

### Google Colab
- [ ] Test runner notebook
- [ ] Verify data loading
- [ ] Check GPU utilization
- [ ] Ensure results save to Drive
- [ ] Add Colab badge to README

### Documentation
- [ ] Installation instructions
- [ ] Data preparation guide
- [ ] Model training commands
- [ ] Troubleshooting section
- [ ] Results interpretation

---

## 📅 Timeline Tracking

| Phase | Planned | Actual | Status |
|-------|---------|--------|--------|
| Phase 1 | 0.5h | ⬜ | Not Started |
| Phase 2 | 2-3h | ⬜ | Not Started |
| Phase 3 | 1-2h | ⬜ | Not Started |
| Phase 4 | 2-3h | ⬜ | Not Started |
| Phase 5 | 3-4h | ⬜ | Not Started |
| Phase 6 | 1-2h | ⬜ | Not Started |
| **Total** | **12-14h** | **⬜** | **0% Complete** |

---

## 🔍 Quality Assurance

### Code Quality
- [ ] All functions documented
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Logging configured

### Testing
- [ ] Unit tests for data loading
- [ ] Model initialization tests
- [ ] Training convergence test
- [ ] Evaluation metrics test

### Reproducibility
- [ ] Random seeds set
- [ ] Dependencies versioned
- [ ] Data paths configurable
- [ ] Results logged

---

## 📝 Notes & Issues

### Blockers
- ⬜ Data access from Synapse
- ⬜ GPU memory constraints
- ⬜ scGPT weights availability

### Decisions
- ⬜ Final HVG count (2000 vs all genes)
- ⬜ Metadata features inclusion
- ⬜ Best model selection criteria

### Improvements
- ⬜ Add cross-validation
- ⬜ Implement ensemble methods
- ⬜ Add interpretability analysis

---

## ✅ Sign-off

- [ ] **Technical Review**: Code quality and correctness
- [ ] **Performance Review**: Meets F1 score requirements
- [ ] **Documentation Review**: Clear and complete
- [ ] **Reproducibility Test**: Works in fresh environment
- [ ] **Final Approval**: Ready for submission

---

*Last Updated: [Date]*
*Progress: 0/100 tasks completed*