# SDGNN

This repository contains the source code for **SDGNN (Structural-Diversity Graph Neural Network)** experiments.

---

## 📂 Datasets

Due to GitHub’s file size limit (100 MB per file), the full datasets are **not included** in this repository.  
You can download them from the following link:

- **Baidu Netdisk**: [Download Link](https://pan.baidu.com/s/1TZ_U9PQmMMORPnehdY1QgA)  
- **Extraction Code**: `1234`

After downloading, please extract the datasets into the `datasets/` directory.  
The expected structure should be:

```
SDGNN/
│── datasets/
│   ├── cora/
│   ├── citeseer/
│   ├── pubmed/
│   └── ...
│── README.md
│── requirements.txt
```

---

## 🚀 Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/mingyue15694/SDGNN.git
   cd SDGNN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run an example**
   ```bash
   python Classification_prediction/OURS6_DGS0.py
   ```

---

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{kong2025parameterfree,
  title={Parameter-Free Structural-Diversity Message Passing for Graph Neural Networks},
  author={Kong, Mingyue and Zhang, Yinglong and Xu, Chengda and Xia, Xuewen and Xu, Xing},
  journal={arXiv preprint arXiv:2508.19884},
  year={2025},
  doi={10.48550/arXiv.2508.19884},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## ✉️ Contact

For questions about the code or dataset, please contact:  

**Yinglong Zhang** (Corresponding Author)  
📧 Email: [zhang_yinglong@126.com](mailto:zhang_yinglong@126.com)

**Mingyue Kong**
📧 Email: [3282682984@qq.com](mailto:3282682984@qq.com)
