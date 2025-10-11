# 架构对比：SE-ST-Combined vs STATE

## ✅ **已采用 STATE 的核心设计**

| 特性 | 实现状态 | 说明 |
|-----|---------|------|
| **按需数据加载** | ✅ | `__getitem__` 时读取 H5，不在 `__init__` 加载全部数据 |
| **Cell Sentences** | ✅ | 每个样本包含 128 cells（而非单个 cell） |
| **Sample with Replacement** | ✅ | 少于 128 cells 的扰动可以重复采样 |
| **Metadata Cache** | ✅ | 使用 `H5MetadataCache` 缓存类别编码 |
| **TOML Config** | ✅ | 使用 TOML 文件管理数据集配置 |
| **Zeroshot Split** | ✅ | 支持 cell type 级别的 train/val/test 划分 |

---

## ⚠️ **关键区别（可能影响性能）**

### 1. **H5 文件管理**
| | STATE | SE-ST-Combined |
|---|-------|----------------|
| **打开方式** | `__init__` 中打开一次，保持打开 | 每次 `__getitem__` 重新打开/关闭 |
| **访问方式** | `self.h5_file[...]` | `with h5py.File(...) as f: f[...]` |
| **性能影响** | ⚡ 快（避免重复打开） | 🐢 慢（每次重新打开文件） |
| **多进程** | ❌ 不支持（文件句柄无法 pickle） | ✅ 支持（每个 worker 独立打开） |

**STATE 代码：**
```python
# __init__
self.h5_file = h5py.File(self.h5_path, "r")

# __getitem__
row_data = self.h5_file[f"/obsm/{key}"][idx]
```

**我们的代码：**
```python
# __getitem__
with h5py.File(spec['h5_path'], 'r') as h5_file:
    pert_cells = h5_file[data_key][pert_indices, :]
```

---

### 2. **稀疏矩阵处理**
| | STATE | SE-ST-Combined |
|---|-------|----------------|
| **CSR 支持** | ✅ 完整支持（`fetch_gene_expression`） | ❌ 跳过 CSR 文件（`NotImplementedError`） |
| **Dense 支持** | ✅ | ✅ |
| **影响** | 可以处理所有 H5 文件 | 只能处理 dense 或 obsm 数据 |

**STATE 的 CSR 处理：**
```python
if attrs["encoding-type"] == "csr_matrix":
    indptr = self.h5_file["/X/indptr"]
    start_ptr = indptr[idx]
    end_ptr = indptr[idx + 1]
    sub_data = torch.tensor(
        self.h5_file["/X/data"][start_ptr:end_ptr], dtype=torch.float32
    )
    # ... 构建 sparse tensor
```

**我们的处理：**
```python
if "encoding-type" in h5_file["X"].attrs:
    if h5_file["X"].attrs["encoding-type"] == b"csr_matrix":
        raise NotImplementedError(
            "CSR sparse matrix detected - skipping this file"
        )
```

---

### 3. **LRU 缓存**
| | STATE | SE-ST-Combined |
|---|-------|----------------|
| **单细胞读取缓存** | ✅ `@lru_cache(maxsize=10000)` | ❌ 无缓存 |
| **影响** | 重复读取同一细胞时从缓存返回 | 每次都重新读取 H5 |

**STATE 代码：**
```python
@lru_cache(maxsize=10000)
def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
    row_data = self.h5_file[f"/obsm/{key}"][idx]
    return torch.tensor(row_data, dtype=torch.float32)
```

---

### 4. **配对策略**
| | STATE | SE-ST-Combined |
|---|-------|----------------|
| **配对逻辑** | 外部 `MappingStrategy`（可插拔） | 内部硬编码 |
| **灵活性** | 可以切换不同的配对策略 | 固定的 batch-matched 配对 |

---

### 5. **模型架构**
| | STATE | SE-ST-Combined |
|---|-------|----------------|
| **输入** | 直接使用 gene expression | **SE 编码器（600M 参数）** → embeddings |
| **State Transition** | ✅ `StateTransitionPerturbationModel` | ✅ `StateTransitionPerturbationModel` |
| **输出解码** | ✅ gene expression prediction | ✅ gene expression prediction |

**这是核心区别**：我们在 STATE 前面加了 SE（Single-cell Encoder）！

```
STATE:
  gene_expr → StateTransition → predicted_gene_expr

SE-ST-Combined:
  gene_expr → SE(600M) → embeddings → StateTransition → predicted_gene_expr
```

---

## 🎯 **建议优化（按优先级）**

### Priority 1: **保持 H5 文件打开**
**问题**：每次 `__getitem__` 重新打开文件很慢  
**解决方案**：
```python
# __init__
self.h5_file = h5py.File(self.h5_path, "r")

# But: 需要处理多进程（DataLoader num_workers > 0）
# Option A: 只在 num_workers=0 时保持打开
# Option B: 使用 worker_init_fn 在每个 worker 中打开文件
```

**Trade-off**：
- 单进程（`num_workers=0`）：✅ 保持打开 → 快
- 多进程（`num_workers>0`）：❌ 文件句柄无法 pickle → 必须每次重开

---

### Priority 2: **添加 LRU 缓存**
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def _fetch_cells(self, h5_path: str, data_key: str, idx: int) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        return f[data_key][idx, :]
```

---

### Priority 3: **支持 CSR 稀疏矩阵**
目前跳过了 `competition_train.h5`（CSR 格式），应该实现 STATE 的 CSR 读取逻辑。

---

### Priority 4: **可插拔配对策略**
将配对逻辑抽象为 `MappingStrategy` 类，支持：
- Batch-matched pairing（当前实现）
- Random pairing
- Cross-batch pairing

---

## 📊 **性能预期**

| 场景 | 当前性能 | 优化后性能 |
|-----|---------|-----------|
| **单进程训练** (`num_workers=0`) | 🐢 慢（重复打开文件） | ⚡ 快（保持文件打开 + LRU 缓存） |
| **多进程训练** (`num_workers>0`) | 🐢 中等（每个 worker 独立打开） | ⚡ 快（LRU 缓存） |
| **CSR 数据** | ❌ 无法使用 | ✅ 可用 |

---

## ✅ **结论**

**已实现核心架构**：
- ✅ 按需数据加载（不 OOM）
- ✅ Cell sentences（128 cells）
- ✅ Sample with replacement

**关键区别**：
1. 🔵 **模型架构**：SE-ST 是 STATE 的增强版（加了 SE 编码器）
2. 🟡 **文件管理**：我们每次重开（支持多进程），STATE 保持打开（单进程）
3. 🟡 **CSR 支持**：我们跳过，STATE 完整支持
4. 🟡 **缓存优化**：我们无缓存，STATE 有 LRU 缓存

**现在可以训练吗？** ✅ **是的！** 数据加载已经工作了，只是还有性能优化空间。

