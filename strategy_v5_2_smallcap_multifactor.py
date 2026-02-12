"""
Strategy V5.2 - 小盘股多因子共振策略（展示版）

设计目标
1) 保留 V5 的整体交易框架（周调仓 + 盘中风控 + 盘中补仓）
2) 仅按修改 4 个关键点：
   - 顶背离触发时，直接卖出部分持仓
   - 涨停基因改为“完整事件链路”
   - 资金流因子改为“行业资金流优先构建”方案
   - 明确采用 1.5/1.4/1.2/1.0 因子权重

说明：本文件可直接用于 JoinQuant 回测。
"""

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta

try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False


class Config:
    # ==================== 基础参数 ====================
    MARKET_INDEX = '399101.XSHE'          # 中小板指
    TARGET_STOCK_NUM = 4                  # 目标持仓数量（保持 V5 风格）
    POOL_TOP_N = 320                      # 小市值基础池规模
    REBALANCE_DAY = 1                     # 每周一调仓

    # ==================== 选股过滤参数 ====================
    NEW_STOCK_DAYS = 375                  # 次新过滤天数
    MIN_DAILY_MONEY = 1.2e7               # 日成交额门槛
    LIMIT_DAYS_WINDOW = 3 * 250           # 涨停基因长周期窗口
    INIT_STOCK_COUNT = 1000               # 涨停基因初筛样本上限

    # ==================== 因子权重（显式固定） ====================
    # 要求点 #4：明确使用固定权重
    WEIGHT_LIMIT_GENE = 1.5
    WEIGHT_MONEYFLOW = 1.4
    WEIGHT_EPS_VALUE = 1.2
    WEIGHT_INDUSTRY = 1.0

    # ==================== 交易与风控参数 ====================
    STOP_LOSS_LIMIT = 0.92                # 个股止损线
    MARKET_STOP_THRESHOLD = 0.95          # 市场大跌阈值（开收比）
    SINGLE_STOCK_MAX = 0.25               # 单票仓位上限
    MIN_ORDER_VALUE = 3000                # 最小下单金额
    CLOSE_TAX = 0.0005                    # 卖出印花税

    # 盘中异常监控参数
    HIGH_VOLUME_LOOKBACK = 120
    HIGH_VOLUME_RATIO = 0.9
    TURNOVER_RATIO_THRESHOLD = 2.0
    TURNOVER_ABS_THRESHOLD = 0.10

    # 缓存
    CACHE_TIMEOUT_DAYS = 3

    # 展示日志
    LOG_FACTOR_STATS = True


def initialize(context):
    """
    初始化：设置交易环境、全局状态、定时任务
    """
    set_option('avoid_future_data', True)
    set_option('use_real_price', True)
    set_benchmark(Config.MARKET_INDEX)
    set_slippage(FixedSlippage(0.0003))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=Config.CLOSE_TAX,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0,
        ),
        type='stock',
    )

    log.set_level('order', 'error')

    # 策略运行期状态
    g.cache = {}
    g.top_sectors = []
    g.target_list = []
    g.hold_list = []
    g.yesterday_limit_up = []
    g.not_buy_again = set()
    g.divergence_flags = []

    # 定时任务（保持 V5 结构）
    run_daily(reset_daily_state, time='09:00')
    run_daily(before_market_open, time='09:01')
    run_daily(prepare_hold_state, time='09:05')
    run_daily(check_market_divergence, time='09:30')
    run_daily(sell_stocks, time='11:25')
    run_daily(afternoon_risk_control, time='10:30')
    run_daily(afternoon_risk_control, time='14:30')
    run_weekly(weekly_rebalance, Config.REBALANCE_DAY, '10:15')


def reset_daily_state(context):
    """
    每日开盘前重置“当日不再买入名单”，避免当日止损后反复买回。
    """
    g.not_buy_again = set()
    clean_expired_cache(context)


def clean_expired_cache(context):
    """
    清理过期缓存，控制数据新鲜度。
    """
    if not g.cache:
        return
    cur_date = context.previous_date
    expired = []
    for k, (cache_date, _) in g.cache.items():
        if (cur_date - cache_date).days >= Config.CACHE_TIMEOUT_DAYS:
            expired.append(k)
    for k in expired:
        del g.cache[k]


def before_market_open(context):
    """
    盘前准备行业资金流排名。

    要求点 #3：采用“先构建行业资金流”方案。
    """
    g.top_sectors = get_top_moneyflow_sectors_reference_style(context)


def prepare_hold_state(context):
    """
    记录当前持仓与“昨日涨停列表”。
    用于盘中炸板检查与卖出保护。
    """
    g.hold_list = [pos.security for pos in context.portfolio.positions.values()]
    if not g.hold_list:
        g.yesterday_limit_up = []
        return

    df = get_price(
        g.hold_list,
        end_date=context.previous_date,
        frequency='daily',
        fields=['close', 'high_limit'],
        count=1,
        panel=False,
    )
    if df is None or df.empty:
        g.yesterday_limit_up = []
        return

    g.yesterday_limit_up = df[df['close'] == df['high_limit']]['code'].tolist()


def get_top_moneyflow_sectors_reference_style(context):
    """
    行业资金流构建：
    1) 取指数成分股
    2) 映射申万二级行业
    3) 批量获取主力净流入
    4) 汇总行业资金流，取前10行业
    """
    cache_key = 'top_sectors_' + str(context.current_dt.date())
    if cache_key in g.cache:
        return g.cache[cache_key][1]

    try:
        sw_level2 = get_industries(name='sw_l2', date=context.current_dt.date())
        if not isinstance(sw_level2, pd.DataFrame):
            return []
        sw_code_to_name = pd.Series(sw_level2['name'].values, index=sw_level2.index).to_dict()

        index_stocks = get_index_stocks(Config.MARKET_INDEX)
        if not index_stocks:
            return []

        previous_trade_day = get_trade_days(end_date=context.current_dt.date(), count=2)[0]
        ind_results = get_industry(index_stocks, date=previous_trade_day)

        stock_sector_map = {}
        for stock in index_stocks:
            if stock not in ind_results:
                continue
            info = ind_results[stock].get('sw_l2', {})
            code = info.get('industry_code')
            name = info.get('industry_name')
            if code and code in sw_code_to_name:
                stock_sector_map[stock] = sw_code_to_name[code]
            elif name and name in sw_code_to_name.values():
                stock_sector_map[stock] = name

        if not stock_sector_map:
            return []

        all_stocks = list(stock_sector_map.keys())
        prev_trade_days = get_trade_days(end_date=context.current_dt.date(), count=2)
        if len(prev_trade_days) < 2:
            return []
        end_date = prev_trade_days[-2]

        money_flow_batches = []
        batch_size = 500
        for i in range(0, len(all_stocks), batch_size):
            batch_stocks = all_stocks[i: i + batch_size]
            batch_df = get_money_flow(
                security_list=batch_stocks,
                end_date=end_date,
                count=1,
                fields=['sec_code', 'net_amount_main'],
            )
            if isinstance(batch_df, pd.DataFrame) and not batch_df.empty:
                money_flow_batches.append(batch_df)

        if not money_flow_batches:
            return []

        money_flow_df = pd.concat(money_flow_batches, ignore_index=True)
        sector_mapping_df = pd.DataFrame(list(stock_sector_map.items()), columns=['sec_code', 'sector_name'])
        merged_df = pd.merge(money_flow_df, sector_mapping_df, on='sec_code', how='inner')
        sector_fund_flow = merged_df.groupby('sector_name')['net_amount_main'].sum().sort_values(ascending=False)
        top_sectors = sector_fund_flow.index[:10].tolist()

        g.cache[cache_key] = (context.current_dt.date(), top_sectors)
        return top_sectors
    except Exception as e:
        log.info('获取行业资金流失败: %s' % e)
        return []


def weekly_rebalance(context):
    """
    每周主调仓：
    - 若近5日出现指数顶背离信号，则暂停周调仓（防守）
    - 否则按多因子重新选股并调仓
    """
    if any(g.divergence_flags[-5:]):
        log.info('近5日指数出现顶背离，暂停本周调仓')
        return

    selected = select_multifactor_smallcap(context, Config.TARGET_STOCK_NUM)
    g.target_list = selected
    if not selected:
        log.info('候选为空，跳过调仓')
        return

    execute_rebalance(context, selected)


def select_multifactor_smallcap(context, target_num=4):
    """
    多因子共振选股主函数。
    """
    pool = get_comprehensive_smallcap_pool(context, Config.POOL_TOP_N)
    if not pool:
        return []

    scores = defaultdict(float)

    # 因子1：涨停基因（完整链路）
    limit_gene = get_limit_gene_candidates_full_chain(context, pool)
    for s in limit_gene[: target_num * 2]:
        scores[s] += Config.WEIGHT_LIMIT_GENE

    # 因子2：资金流（行业先行）
    moneyflow = get_moneyflow_candidates_reference_style(context, pool)
    for s in moneyflow[: target_num * 2]:
        scores[s] += Config.WEIGHT_MONEYFLOW

    # 因子3：EPS/性价比
    eps_value = get_eps_value_candidates(context, pool)
    for s in eps_value[: target_num * 2]:
        scores[s] += Config.WEIGHT_EPS_VALUE

    # 因子4：行业分散投票
    industry_list = get_industry_candidates_reference_style(context, pool)
    for s in industry_list[: target_num * 2]:
        scores[s] += Config.WEIGHT_INDUSTRY

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    candidate = [s for s, _ in ranked]

    if Config.LOG_FACTOR_STATS:
        log.info(
            '因子样本数 pool=%d limit=%d moneyflow=%d eps=%d industry=%d scored=%d' %
            (len(pool), len(limit_gene), len(moneyflow), len(eps_value), len(industry_list), len(candidate))
        )

    # 最终行业去重，降低组合同质化
    final_list = []
    used_industry = set()
    ind_map = get_industry(candidate, date=context.previous_date)
    for s in candidate:
        ind = ind_map.get(s, {}).get('sw_l1', {}).get('industry_code', 'unknown')
        if ind not in used_industry:
            final_list.append(s)
            used_industry.add(ind)
        if len(final_list) >= target_num:
            break

    log.info('V5.2 小盘多因子候选: %s' % final_list)
    return final_list


def get_comprehensive_smallcap_pool(context, top_n=300):
    """
    母池构建（保持 V5 方案）：
    - 指数成分股起步
    - 过滤停牌/ST/退市风险/次新/涨跌停
    - 按流通市值取小盘
    - 成交额过滤
    """
    base = get_index_stocks(Config.MARKET_INDEX)
    current_data = get_current_data()

    valid = []
    for s in base:
        try:
            sec = current_data[s]
            if sec.paused or sec.is_st:
                continue
            if '退' in sec.name:
                continue
            if (context.current_dt.date() - get_security_info(s).start_date).days < Config.NEW_STOCK_DAYS:
                continue
            if sec.last_price >= sec.high_limit or sec.last_price <= sec.low_limit:
                continue
            valid.append(s)
        except Exception:
            continue

    if not valid:
        return []

    q = query(valuation.code).filter(valuation.code.in_(valid)).order_by(valuation.circulating_market_cap.asc()).limit(top_n * 2)
    df = get_fundamentals(q, date=context.previous_date)
    if df is None or df.empty:
        return []

    pool = []
    for s in df['code'].tolist():
        try:
            money = attribute_history(s, 1, '1d', ['money'])['money'][-1]
            if money > Config.MIN_DAILY_MONEY:
                pool.append(s)
                if len(pool) >= top_n:
                    break
        except Exception:
            continue
    return pool


# ======= 要求点 #2：涨停基因改为“完整事件链路” =======
def get_limit_gene_candidates_full_chain(context, base_pool):
    pool = base_pool[: min(Config.INIT_STOCK_COUNT, len(base_pool))]
    pool = get_history_highlimit(context, pool, Config.LIMIT_DAYS_WINDOW)
    pool = get_start_point_candidates(context, pool, Config.LIMIT_DAYS_WINDOW)
    pool = filter_recent_extreme_movements(context, pool)
    return pool


def get_history_highlimit(context, stock_list, days):
    df = get_price(stock_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit'], count=days, panel=False)
    if df is None or df.empty:
        return []
    df = df[df['close'] == df['high_limit']]
    grouped = df.groupby('code').size().reset_index(name='count').sort_values(by=['count'], ascending=False)
    # 取涨停基因最显著的前10%
    top_k = max(1, int(len(grouped) * 0.10))
    return grouped['code'].tolist()[:top_k]


def get_start_point_candidates(context, stock_list, days):
    df = get_price(stock_list, end_date=context.previous_date, frequency='daily', fields=['open', 'low', 'close', 'high_limit'], count=days, panel=False)
    if df is None or df.empty:
        return stock_list

    stock_start_point = {}
    current_data = get_current_data()

    for code, group in df.groupby('code'):
        group = group.sort_values('time')
        limit_rows = group[group['close'] == group['high_limit']]
        if limit_rows.empty:
            continue
        latest_idx = limit_rows.iloc[-1].name
        previous_rows = group[group.index <= latest_idx].iloc[::-1]
        for _, row in previous_rows.iterrows():
            if row['close'] < row['open']:
                stock_start_point[code] = row['low']
                break

    stock_price_bias = {}
    for code, start_point in stock_start_point.items():
        if code in current_data and start_point > 0:
            stock_price_bias[code] = current_data[code].last_price / start_point

    # 偏离越低越好（更接近“起爆点”）
    sorted_list = sorted(stock_price_bias.items(), key=lambda x: x[1], reverse=False)
    result = [i[0] for i in sorted_list]
    return result if result else stock_list


def filter_recent_extreme_movements(context, stock_list):
    if not stock_list:
        return []

    df = get_price(stock_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit', 'low_limit', 'volume'], count=3, panel=False, fill_paused=False)
    if df is None or df.empty:
        return stock_list

    exclude = set()
    for stock in stock_list:
        stock_data = df[df['code'] == stock]
        if len(stock_data) < 3:
            exclude.add(stock)
            continue

        has_limit = (stock_data['close'] == stock_data['high_limit']).any() or (stock_data['close'] == stock_data['low_limit']).any()
        y_not_limit = stock_data.iloc[-1]['close'] != stock_data.iloc[-1]['high_limit']
        if has_limit and y_not_limit:
            exclude.add(stock)
            continue

        if y_not_limit:
            vol_bars = get_bars(stock, count=120, unit='1d', fields=['volume'], include_now=False, df=True)
            if vol_bars is not None and len(vol_bars) > 20:
                max_vol = vol_bars['volume'].max()
                avg_vol = vol_bars['volume'].iloc[-20:].mean()
                y_vol = stock_data.iloc[-1]['volume']
                cond_huge = y_vol > 0.9 * max_vol
                cond_abnormal = y_vol > 2.0 * avg_vol
                if cond_huge or cond_abnormal:
                    exclude.add(stock)

    return [s for s in stock_list if s not in exclude]


# ======= 要求点 #3：资金流因子采用“行业资金流优先构建”方案 =======
def get_moneyflow_candidates_reference_style(context, base_pool):
    if not g.top_sectors:
        return []
    try:
        ind_results = get_industry(base_pool, date=context.current_dt.date())
        filtered = []
        for stock in base_pool:
            if stock in ind_results:
                sw_l2_name = ind_results[stock].get('sw_l2', {}).get('industry_name')
                if sw_l2_name and sw_l2_name in g.top_sectors:
                    filtered.append(stock)
        return filtered[: Config.TARGET_STOCK_NUM * 2]
    except Exception:
        return []


def get_industry_candidates_reference_style(context, base_pool):
    try:
        ind_results = get_industry(base_pool, date=context.current_dt.date())
        industry_map = {}
        for stock in base_pool:
            if stock in ind_results:
                name = ind_results[stock].get('sw_l2', {}).get('industry_name')
                if name and name not in industry_map:
                    industry_map[name] = stock
        return list(industry_map.values())[: Config.TARGET_STOCK_NUM * 2]
    except Exception:
        return base_pool[: Config.TARGET_STOCK_NUM * 2]


def get_eps_value_candidates(context, pool):
    q = query(valuation.code, valuation.circulating_market_cap, indicator.eps).filter(
        valuation.code.in_(pool), indicator.eps > 0
    ).limit(200)
    df = get_fundamentals(q, date=context.previous_date)
    if df is None or df.empty:
        return []

    df['cap_rank'] = df['circulating_market_cap'].rank(ascending=True)
    df['eps_rank'] = df['eps'].rank(ascending=False)
    df['score'] = df['cap_rank'] * 0.6 + df['eps_rank'] * 0.4
    df = df.sort_values('score', ascending=True)
    return df['code'].tolist()


def execute_rebalance(context, target_list):
    target_set = set(target_list)
    hold_set = set(context.portfolio.positions.keys())

    for s in list(hold_set):
        if s not in target_set and s not in g.yesterday_limit_up:
            order_target_value(s, 0)
            g.not_buy_again.add(s)

    buy_missing_positions(context, target_list)


def buy_missing_positions(context, target_list):
    current_positions = context.portfolio.positions
    slots = max(0, Config.TARGET_STOCK_NUM - len(current_positions))
    if slots <= 0:
        return

    cash = context.portfolio.available_cash
    total = context.portfolio.total_value
    per_value = min(cash / max(1, slots), total * Config.SINGLE_STOCK_MAX)
    if per_value < Config.MIN_ORDER_VALUE:
        return

    for s in target_list:
        if s in g.not_buy_again:
            continue
        if s in current_positions and current_positions[s].total_amount > 0:
            continue
        order_target_value(s, per_value)
        if len(context.portfolio.positions) >= Config.TARGET_STOCK_NUM:
            break


def sell_stocks(context):
    # 个股止损
    for s in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[s]
        if pos.avg_cost <= 0:
            continue
        if pos.price < pos.avg_cost * Config.STOP_LOSS_LIMIT:
            order_target_value(s, 0)
            g.not_buy_again.add(s)

    # 市场级减仓
    mkt = get_price(
        Config.MARKET_INDEX,
        end_date=context.previous_date,
        frequency='daily',
        fields=['close', 'open'],
        count=1,
    )
    if mkt is not None and not mkt.empty:
        down_ratio = mkt['close'].iloc[0] / max(1e-6, mkt['open'].iloc[0])
        if down_ratio <= Config.MARKET_STOP_THRESHOLD:
            for s in list(context.portfolio.positions.keys()):
                pos = context.portfolio.positions[s]
                order_target_value(s, pos.value * 0.5)


def afternoon_risk_control(context):
    check_limit_break(context)
    check_high_volume(context)
    check_abnormal_turnover(context)
    check_remain_amount(context)


def check_limit_break(context):
    now = context.current_dt
    for s in g.yesterday_limit_up:
        if s not in context.portfolio.positions:
            continue
        pos = context.portfolio.positions[s]
        if pos.closeable_amount <= 0:
            continue
        data = get_price(s, end_date=now, frequency='1m', fields=['close', 'high_limit'], count=1, panel=False, fill_paused=True)
        if data is not None and not data.empty and data.iloc[0]['close'] < data.iloc[0]['high_limit']:
            order_target_value(s, 0)
            g.not_buy_again.add(s)


def check_high_volume(context):
    current_data = get_current_data()
    for s in list(context.portfolio.positions.keys()):
        if current_data[s].paused:
            continue
        if current_data[s].last_price == current_data[s].high_limit:
            continue
        if context.portfolio.positions[s].closeable_amount == 0:
            continue

        df = get_bars(s, count=Config.HIGH_VOLUME_LOOKBACK, unit='1d', fields=['volume'], include_now=True, df=True)
        if df is None or df.empty:
            continue
        if df['volume'].values[-1] > Config.HIGH_VOLUME_RATIO * df['volume'].max():
            order_target_value(s, 0)
            g.not_buy_again.add(s)


def check_abnormal_turnover(context):
    current_data = get_current_data()
    for s in list(context.portfolio.positions.keys()):
        if current_data[s].paused:
            continue
        if context.portfolio.positions[s].closeable_amount == 0:
            continue
        rt = get_turnover_ratio(context, s, is_avg=False)
        avg = get_turnover_ratio(context, s, is_avg=True)
        if avg <= 0:
            continue
        if rt > Config.TURNOVER_ABS_THRESHOLD and rt / avg > Config.TURNOVER_RATIO_THRESHOLD:
            order_target_value(s, 0)
            g.not_buy_again.add(s)


def get_turnover_ratio(context, stock, is_avg=False):
    if is_avg:
        vol_df = get_price(stock, end_date=context.previous_date, frequency='daily', fields=['volume'], count=20)
        cap_df = get_valuation(stock, end_date=context.previous_date, fields=['circulating_cap'], count=1)
        if vol_df is None or vol_df.empty or cap_df is None or cap_df.empty:
            return 0.0
        cap = cap_df['circulating_cap'].iloc[0]
        if cap <= 0:
            return 0.0
        return (vol_df['volume'] / (cap * 10000)).mean()

    vol_df = get_price(stock, start_date=context.current_dt.date(), end_date=context.current_dt, frequency='1m', fields=['volume'], panel=True, fill_paused=False)
    cap_df = get_valuation(stock, end_date=context.previous_date, fields=['circulating_cap'], count=1)
    if vol_df is None or vol_df.empty or cap_df is None or cap_df.empty:
        return 0.0
    cap = cap_df['circulating_cap'].iloc[0]
    if cap <= 0:
        return 0.0
    return vol_df['volume'].sum() / (cap * 10000)


def close_position(position):
    """
    对某只持仓直接清仓。
    """
    security = position.security
    order = order_target_value(security, 0)
    return order is not None


def detect_divergence(stock, context):
    """
    顶背离检测（MACD）
    """
    if not TALIB_AVAILABLE:
        return False

    fast, slow, sign = 12, 26, 9
    rows = (fast + slow + sign) * 5
    grid = attribute_history(stock, rows, fields=['close']).dropna()
    if len(grid) < rows:
        return False

    try:
        diff, dea, macd_hist = talib.MACD(grid['close'].values, fastperiod=fast, slowperiod=slow, signalperiod=sign)
        grid = pd.DataFrame({'close': grid['close'].values, 'dif': diff, 'macd': macd_hist * 2})

        mask = (grid['macd'] < 0) & (grid['macd'].shift(1) >= 0)
        if mask.sum() < 2:
            return False

        idx2 = mask[mask].index[-2]
        idx1 = mask[mask].index[-1]
        price_cond = grid['close'].iloc[idx2] < grid['close'].iloc[idx1]
        dif_cond = grid['dif'].iloc[idx2] > grid['dif'].iloc[idx1] > 0
        macd_cond = grid['macd'].iloc[-2] > 0 > grid['macd'].iloc[-1]

        if len(grid['dif']) > 20:
            recent_avg = grid['dif'].iloc[-10:].mean()
            prev_avg = grid['dif'].iloc[-20:-10].mean()
            trend_cond = recent_avg < prev_avg
        else:
            trend_cond = False

        return bool(price_cond and dif_cond and macd_cond and trend_cond)
    except Exception:
        return False


def check_market_divergence(context):
    """
    要求点 #1：顶背离触发时直接减仓处理
    - 以前：主要用于“暂停周调仓”
    - 现在：检测到顶背离时，直接卖出部分持仓（未封板持仓清仓）
    """
    signal = detect_divergence(Config.MARKET_INDEX, context)
    g.divergence_flags.append(signal)
    if not signal:
        return

    log.info('检测到指数顶背离，执行防守卖出（未封板持仓清仓）')
    current_data = get_current_data()
    for stock in list(g.hold_list):
        if stock not in context.portfolio.positions:
            continue
        if current_data[stock].last_price < current_data[stock].high_limit:
            close_position(context.portfolio.positions[stock])
            g.not_buy_again.add(stock)


def check_remain_amount(context):
    if len(context.portfolio.positions) >= Config.TARGET_STOCK_NUM:
        return
    target = g.target_list if g.target_list else select_multifactor_smallcap(context, Config.TARGET_STOCK_NUM)
    buy_missing_positions(context, target)
