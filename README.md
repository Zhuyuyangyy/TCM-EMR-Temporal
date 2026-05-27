# TCM-EMR-Temporal

中医电子病历证候演化时序模型与复诊结局预测

## 核心创新点

1. **证候演化建模** — 基于 Transformer/GRU 的证候时序演变建模
2. **复诊结局预测** — 基于历史就诊记录预测下次就诊结局
3. **治疗效果评估** — 估计不同治疗方案对证候演变的因果效应
4. **时序可解释性** — 识别关键就诊节点和证候转折点

## 快速开始

```bash
pip install -e .
uvicorn backend.main:app --port 8021 --reload
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/predict/outcome` | POST | 复诊结局预测 |
| `/api/predict/trajectory` | POST | 证候轨迹预测 |
| `/api/explain/temporal` | POST | 时序重要性解释 |

## License

MIT
