慎独 V5.2 
适配 JoinQuant 平台的小盘股多因子策略

10w 资金，2020/2/2~2026/2/3，JoinQuant 平台回测结果如下：
<img width="1601" height="739" alt="image" src="https://github.com/user-attachments/assets/98353509-438f-4722-98b9-8b09063b83f2" />

# Strategy V5.2 策略全景说明

## 全盘逻辑图

市场数据与持仓状态
-> 盘前行业资金流计算（找资金最活跃的行业）
-> 周调仓日触发选股引擎
-> 四因子共振打分（涨停基因 + 资金流 + EPS性价比 + 行业分散）
-> 形成目标持仓列表
-> 执行调仓（先卖后买，补齐目标仓位）
-> 盘中风控巡检（炸板 / 巨量 / 换手异常 / 止损）
-> 若触发风险则减仓或清仓
-> 盘中检查空余仓位并回补
-> 进入下一交易日循环


## 1. 策略目标与定位

这是一套 **小盘股多因子共振 + 盘中风控** 策略，目标是：

- 在小盘风格阶段获取超额收益；
- 遇到短线风险信号时快速降风险；
- 保持策略逻辑清晰，可向非技术同事解释。


## 2. 参数总览（策略“旋钮”）

代码位置：`Strategy 5/strategy_v5_2_smallcap_multifactor.py`

- **标的与持仓**
  - `MARKET_INDEX='399101.XSHE'`：中小板指作为核心市场观察对象
  - `TARGET_STOCK_NUM=4`：目标持仓数量
  - `POOL_TOP_N=320`：小盘候选池规模
  - `REBALANCE_DAY=1`：每周一主调仓

- **选股过滤**
  - `NEW_STOCK_DAYS=375`：过滤次新股
  - `MIN_DAILY_MONEY=1.2e7`：过滤低流动性股票
  - `LIMIT_DAYS_WINDOW=3*250`：涨停基因长周期观察窗口
  - `INIT_STOCK_COUNT=1000`：涨停基因初筛样本上限

- **四因子权重（固定）**
  - 涨停基因：`1.5`
  - 资金流：`1.4`
  - EPS性价比：`1.2`
  - 行业分散：`1.0`

- **交易与风险**
  - `STOP_LOSS_LIMIT=0.92`：个股止损线
  - `MARKET_STOP_THRESHOLD=0.95`：市场级减仓阈值
  - `SINGLE_STOCK_MAX=0.25`：单票仓位上限
  - `MIN_ORDER_VALUE=3000`：最小下单金额
  - `CLOSE_TAX=0.0005`：卖出印花税


## 3. 每日执行节奏（时间维度）

### 09:00 - `reset_daily_state`

- 清空“当日不再买入名单”
- 清理过期缓存

业务意义：避免止损后同日重复买回；保证缓存不失真。


### 09:01 - `before_market_open`

- 计算行业资金流前十行业（并缓存）

业务意义：先判断“钱流向哪几个行业”，后续资金流因子只在这些方向里选股。


### 09:05 - `prepare_hold_state`

- 记录当前持仓
- 记录“昨日涨停持仓”列表

业务意义：后续盘中检查“炸板”（昨日涨停、今日开板）用。


### 09:30 - `check_market_divergence`

- 检测指数顶背离（MACD逻辑）
- 若触发：直接卖出未封板持仓

业务意义：当市场出现“上涨动能衰减”迹象时，优先保命。


### 10:15（每周一）- `weekly_rebalance`

- 若近5日出现顶背离信号，暂停周调仓
- 否则运行选股引擎，执行主调仓

业务意义：把“风控优先级”放在“开新仓”前面。


### 10:30 / 14:30 - `afternoon_risk_control`

依次执行：

- `check_limit_break`：炸板检查
- `check_high_volume`：巨量异常检查
- `check_abnormal_turnover`：换手异常检查
- `check_remain_amount`：如仓位不足则回补

业务意义：盘中动态管理风险与仓位完整性。


### 11:25 - `sell_stocks`

- 个股止损
- 市场级大跌触发持仓减半

业务意义：中午前做一次强制风险检查，防止下午扩大损失。


## 4. 选股引擎：四因子共振是怎么工作的

主函数：`select_multifactor_smallcap`

流程：
候选池 -> 因子打分 -> 汇总排序 -> 行业去重 -> 输出目标列表


### 因子1：涨停基因（权重 1.5）

函数链：

- `get_limit_gene_candidates_full_chain`
  - `get_history_highlimit`：先找长周期内涨停基因强的股票
  - `get_start_point_candidates`：找“起爆点偏离较小”的股票（不追太远）
  - `filter_recent_extreme_movements`：剔除近期异常极端行为

投资逻辑：
“既要有历史爆发基因，又要避免末端追高和异常风险。”


### 因子2：资金流（权重 1.4）

函数：`get_moneyflow_candidates_reference_style`

- 仅在“行业资金流前十”的行业中选股票
- 提升组合与主流资金方向的一致性

投资逻辑：
“跟着资金走，而不是只看形态。”


### 因子3：EPS性价比（权重 1.2）

函数：`get_eps_value_candidates`

- 要求 `EPS > 0`
- 用“市值排名 + EPS排名”构建综合分

投资逻辑：
“小而不差，偏向基本面更健康的小票。”


### 因子4：行业分散（权重 1.0）

函数：`get_industry_candidates_reference_style`

- 从行业维度增加分散加分

投资逻辑：
“不把鸡蛋放在同一个行业篮子里。”


## 5. 调仓与买入逻辑（怎么下单）

函数：`execute_rebalance` + `buy_missing_positions`

- 先卖：不在新目标列表且非“昨日涨停保护”的持仓，先卖出
- 再买：按剩余仓位槽位分配资金买入
- 单票金额限制：不超过 `total_value * SINGLE_STOCK_MAX`
- 小单过滤：低于 `MIN_ORDER_VALUE` 不下单

投资逻辑：
“先清旧仓、再补新仓，控制集中度和交易噪音。”


## 6. 风控逻辑（怎么止损止盈）

### A. 个股止损

函数：`sell_stocks`

- 当 `现价 < 成本 * STOP_LOSS_LIMIT` 时清仓


### B. 市场级减仓

函数：`sell_stocks`

- 当市场开收比低于 `MARKET_STOP_THRESHOLD` 时，所有持仓减半


### C. 盘中三道检查

- `check_limit_break`：昨日涨停今日开板则卖出
- `check_high_volume`：出现极端放量则卖出
- `check_abnormal_turnover`：换手异常放大则卖出


### D. 顶背离防守

函数：`check_market_divergence`

- 检测到顶背离信号后，直接卖出未封板持仓


## 7. 代码层与投资层的映射

- **市场观察层**：行业资金流、顶背离检测
- **选股层**：四因子共振（基因 + 资金 + 基本面 + 分散）
- **执行层**：周调仓 + 盘中回补
- **风控层**：个股止损 + 市场减仓 + 盘中异常出清

这四层共同作用，形成“进攻与防守并行”的闭环。

## 8. 使用建议

- 先固定参数跑完整区间，再做单变量调参（避免误判）
- 优先观察：年化、最大回撤、Sharpe、Calmar、换手率
- 避免一次同时改多个核心参数（会看不出真正驱动因素）
