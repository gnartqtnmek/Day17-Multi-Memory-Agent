# Benchmark Report - Multi-Memory Agent (Lab 17)

- Generated at (UTC): 2026-04-24T13:11:25.839784+00:00
- Conversations: 10
- Total turns: 30

## KPI Comparison

| Metric | With Memory | Without Memory | Delta |
|---|---:|---:|---:|
| Response relevance | 54.35% | 3.26% | 51.09% |
| Context utilization | 95.65% | 0.00% | 95.65% |
| Token efficiency (keyword/token) | 0.0013 | 0.0019 | -0.0006 |
| Memory hit rate | 100.00% | 0.00% | 100.00% |

## Token Budget Breakdown (With Memory)

| Bucket | Tokens | Share |
|---|---:|---:|
| priority_1_kept | 4500 | 11.91% |
| priority_2_kept | 4561 | 12.07% |
| priority_3_kept | 1468 | 3.89% |
| priority_4_kept | 2956 | 7.82% |
| dropped_tokens | 2694 | 7.13% |

## Token Usage

- With memory: prompt=14944, response=1028, total=15972
- Without memory: prompt=331, response=711, total=1042