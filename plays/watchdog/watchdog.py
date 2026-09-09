"""
盯盘助手
========

基于 jvQuant WebSocket 实时数据 + limit_up 已验证因子，持续监控标的。
通过飞书推送买卖信号，状态持久化到本地 JSON。

指令:
  盯 000001.SZ    → 开始盯盘
  停 000001.SZ    → 停止盯盘
  盯盘列表        → 查看监控列表
  清盯盘          → 全部停止
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from plays.watchdog.indicators import price_features, realtime_row, sma, atr
from plays.watchdog.signals import check_entry, check_abnormal, compute_factor_scores
from plays.watchdog.confirm import (STAND_TOL, check_buy_confirm,
                                     check_sell_confirm, trend_up_trigger,
                                     fund_accumulate_confirm, FA_N)
from scripts.tu_share import call_tushare  # noqa: E402

# 模块级加载 .env（模块级常量依赖 env，原 load_dotenv 在 main() 内晚于常量定义）
_env_file = PROJECT_DIR / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)

# WS 数据通过 ws_daemon 共享内存读取
SHM_DIR = Path("/dev/shm")
WS_SUB = SHM_DIR / "ws_sub.json"
WS_SNAP = SHM_DIR / "ws_snap.json"


def _read_ws_snap(code: str) -> dict:
    try:
        snap = json.loads(WS_SNAP.read_text())
        return snap.get(code, {})
    except Exception:
        return {}


def _write_ws_sub(shorts: list[str], l2_shorts: list[str] | None = None):
    try:
        WS_SUB.write_text(json.dumps({"shorts": shorts, "l2_shorts": l2_shorts or []}))
    except Exception:
        pass


def _norm(code: str) -> str:
    if "." in code:
        return code
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"


def _short(code: str) -> str:
    return code.replace(".SH", "").replace(".SZ", "")


logger = logging.getLogger(__name__)
STATE_FILE = PROJECT_DIR / "plays" / "watchdog" / "data" / "state.json"
SCAN_INTERVAL = 60
# L1 行情熔断（2026-08-13 新增）：交易时段内 ws_snap 快照冻结 ≥ 此秒数 → 
# 浮亏持仓市价卖出（CTP 交易不依赖 jvQuant 行情，ticket 独立，能卖）。
# 背景：8/13 13:02 起 jvQuant WS 网关 502 超 30 分钟，持仓无行情监控，
# 断链期间若票下跌，卖出确认器无法止损 → 损失放大。熔断主动止损防放大。
FUSE_TRIGGER_SEC = int(os.getenv("WATCHDOG_FUSE_SECONDS", "300"))
CONSECUTIVE_ENTRY_ROUNDS = int(os.getenv("CONSECUTIVE_ENTRY_ROUNDS", "3"))  # 连续 N 轮满足入场条件才触发 alerted
MAX_WATCH = 20      # 手动盯盘上限（surge 通道不设上限，2026-07-26 用户拍板）
# 买入股数（2026-08-11 用户拍板：全部买入 2 手=200 股，放大收益/亏损额度。
# 之前固定 1 手 100 股，单票盈亏太小；放大到 2 手后单票波动 ×2。）
BUY_VOL = int(os.getenv("WATCHDOG_BUY_VOL", "100"))  # 2026-08-18 用户拍板: 200→100 (1手), 等胜率回升稳定再改回
# 2026-08-20 买入三重风控（0820 实测 52 笔买光资金/57 只持仓=次日开门止损潮根源）：
# ①持仓上限（surge 通道此前不设限=07-26 拍板，但 52 笔证明必须限）
# ②单日买入笔数上限（强市日确认器批量触发）
# ③可用资金下限（资金买光仍继续买=裸奔）
MAX_POSITIONS = int(os.getenv("WATCHDOG_MAX_POSITIONS", "20"))
DAILY_BUY_LIMIT = int(os.getenv("WATCHDOG_DAILY_BUY_LIMIT", "15"))
MIN_FREE_CASH = float(os.getenv("WATCHDOG_MIN_FREE_CASH", "5000"))
# 2026-08-20 用户拍板：暂停 surge 打板玩法（连续亏损 1.2万，找新办法再开启）。
# 2026-08-21 新办法已就绪（回调低吸确认器，回测 30min+0.10%/胜率48% vs 追拉升-1.09%/16%），
# 用户拍板"打开试试" → PAUSE_BUY=0 恢复买入；surge 仅负责选票入盯（模型高分），
# 买入逻辑走回调低吸（不再追拉升）。
PAUSE_BUY = int(os.getenv("WATCHDOG_PAUSE_BUY", "0"))
# 2026-08-19 卖出量独立于 BUY_VOL：BUY_VOL 改 100 后卖出也卖 100 → 0818 前
# 200 股持仓只卖一半（09:30 实测 8 笔剩 100 股挂浮亏）。SELL_VOL 固定 200，
# 100 股持仓走 -2 降档逻辑（L936-940）。
SELL_VOL = int(os.getenv("WATCHDOG_SELL_VOL", "200"))
ABNORMAL_COOLDOWN_SECONDS = 300  # 异常推送冷却：同一 level 5 分钟内不重复推送
# 2026-09-09 用户拍板：暂停真实交易转模拟（PAPER_TRADING=1 时不发真单）。
# 配合 scripts/jvquant_trade_client.py 的 PAPER_TRADING：buy/sale 假设成交，
# 交割单写 {date}.paper.json（不污染真实 {date}.json），推送加【模拟】标记。
PAPER_TRADING = int(os.getenv("PAPER_TRADING", "0"))

# 候选池来源：limit_up pipeline 产出的 analysis
ANALYSIS_DIRS = [
    PROJECT_DIR / "wiki" / "raw" / "limit-up" / "analysis",
    PROJECT_DIR / "plays" / "limit_up" / "data" / "analysis",
]


# ---- 飞书推送 ----

def _push_feishu(text: str):
    """推送盯盘信号到飞书"""
    if PAPER_TRADING:
        text = f"🧪【模拟】{text}"
    try:
        env_file = PROJECT_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        app_id = os.getenv("FEISHU_APP_ID", "")
        app_secret = os.getenv("FEISHU_APP_SECRET", "")
        chat_id = os.getenv("FEISHU_CHAT_ID_SIGNAL", os.getenv("FEISHU_BOT_CHAT_ID", ""))

        if not app_id or not app_secret:
            logger.warning("飞书未配置 app_id/app_secret，跳过推送")
            return
        if not chat_id:
            logger.warning("飞书未配置 chat_id，跳过推送")
            return

        resp = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret}, timeout=10
        )
        token = resp.json().get("tenant_access_token", "")
        if not token:
            return

        send_resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}),
            }, timeout=10
        )
        result = send_resp.json()
        if result.get("code") == 0:
            logger.info(f"飞书推送成功: {text[:60]}...")
        else:
            logger.error(f"飞书推送失败: code={result.get('code')} msg={result.get('msg')}")
    except Exception as e:
        logger.error(f"飞书推送异常: {e}")


def _log_trade_journal(code: str, name: str, direction: str, price: float,
                       shares: int, entry_price: float | None = None,
                       entry_at: str = "", reason: str = "") -> None:
    """写一条交割单到 plays/trading/data/reports/{date}.json（格式对齐旧 trader）。"""
    try:
        today = datetime.now().strftime("%Y%m%d")
        report_dir = PROJECT_DIR / "plays" / "trading" / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        # 2026-09-09 模拟模式：写独立 .paper.json，不污染真实交割单
        #（18:00 复盘 / compile 读 {date}.json 算真实盈亏）。
        if PAPER_TRADING:
            report_file = report_dir / f"{today}.paper.json"
        else:
            report_file = report_dir / f"{today}.json"

        if direction == "买入":
            pnl, pnl_pct = 0.0, 0.0
            amount = round(price * shares, 2)
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            pnl = round((price - (entry_price or 0)) * shares, 2)
            pnl_pct = round((price / (entry_price or price) - 1) * 100, 2)
            amount = round(price * shares, 2)
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "code": code, "name": name, "direction": direction,
            "price": price, "shares": shares, "amount": amount,
            "time": t, "reason": reason, "pnl": pnl, "pnl_pct": pnl_pct,
            "paper": True if PAPER_TRADING else False,
        }

        existing = []
        if report_file.exists():
            try:
                existing = json.loads(report_file.read_text())
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.append(entry)
        report_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
        logger.info(f"交割单写入: {direction} {code} {name} {price}x{shares} -> {report_file}")
    except Exception as e:
        logger.error(f"交割单写入失败 {code}: {e}")


# ---- 盯盘状态 ----

class WatchState:
    """单只股票的盯盘状态"""

    def __init__(self, code: str, name: str = ""):
        self.code = code
        self.name = name
        self.added_at = datetime.now().isoformat()
        self.status = "watching"  # watching | alerted | entered | exited
        self.source = "manual"    # manual | surge —— surge 票只发入场信号，盘后零信号汰换
        self.entry_pushed_date = ""  # 入场信号推送日期 YYYYMMDD（汰换依据）
        self.t1_blocked_date = ""  # T+1 当日不可卖标记 YYYYMMDD（次日自动恢复出场监控）
        self.entry_price: float = 0.0
        self.entry_at: str = ""
        self.highest_since_entry: float = 0.0
        self.bars_held: int = 0
        self.signal_type: str = ""
        self.signal_reason: str = ""
        self.signal_at: str = ""
        self.last_alert_at: str = ""
        # 异常状态推送去重
        self.last_abnormal_level: str = ""
        self.last_abnormal_pushed_at: float = 0.0
        # 挂起买单委托复查（2026-08-06 修复：委托已报未成交但实际成交的竞态）
        self.pending_buy_order_id: str = ""
        self.pending_buy_since = None  # type: ignore[assignment]
        # 今日卖出受阻静默（2026-08-07：砸盘卖不出/挂单不成交的票，
        # 不再反复推送异常状态噪音；次日自动恢复）
        self.sell_blocked_date: str = ""
        # 卖出挂单时间（2026-08-10：挂单超时撤单重挂，防"挂死"）
        self.pending_sell_order_id: str = ""
        self.pending_sell_since: float = 0.0
        # 撤单后强制重卖标记（2026-08-10：即使 check_exit 未再触发也重挂）
        self.force_resell: bool = False
        # 资金流向历史（最近 N 轮扫描）
        self.netflow_history: list[float] = []
        # 日线数据缓存
        self.daily_rows: list[dict] = []
        self.daily_basic: dict = {}
        self.dim_scores: dict = {}
        self.last_daily_update: str = ""
        # 上一轮扫描快照（用于诱空检测）
        self.prev_last: float = 0.0
        self.prev_vwap: float = 0.0
        self.prev_vol_ratio: float = 0.0
        # 2026-08-21 回调低吸买入所需（用户拍板方向2）：
        self.day_high: float = 0.0     # 当日最高价（每轮更新，watching 阶段）
        self.prev_close: float = 0.0   # 昨收（ws_snap pre_close）
        # 确认期基准价（2026-08-12：连续上涨确认的起点）
        # watching 首次触发信号时记录，后续轮次要求 last >= base×1.005
        #（净上涨 0.5% 才算"趋势向上"，横盘/缓慢爬升不确认——
        #  原"last>=prev_last"只防急跌，横盘高位票照买（共达电声/武汉凡谷实测））
        self.entry_base_price: float = 0.0
        # ── 确认器状态（2026-08-12 简化蓝图，plays/watchdog/confirm.py）──
        # 买入确认器：创新高+放量触发 → 站稳 3 轮 → 买入
        self.confirm_base: float = 0.0
        self.confirm_count: int = 0
        # 卖出确认器：最高点回撤 4% 连续 2 轮 → 卖出（防插针/趋势恢复重置）
        self.pullback_count: int = 0
        # 确认器已站稳 3 轮（alerted 分支直接买入，跳过 30s 二次确认；不持久化）
        self._confirm_ready: bool = False
        # 2026-08-14 日线更新失败标记（内存字段）：当天失败不重试，
        # 避免失败票每轮串行重试 Tushare 拖慢确认器判定（钝化根因之一）。
        self._daily_fail_today: bool = False

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name, "added_at": self.added_at,
            "status": self.status, "entry_price": self.entry_price,
            "source": self.source, "entry_pushed_date": self.entry_pushed_date,
            "t1_blocked_date": self.t1_blocked_date,
            "entry_at": self.entry_at, "highest_since_entry": self.highest_since_entry,
            "bars_held": self.bars_held, "signal_type": self.signal_type,
            "signal_reason": self.signal_reason, "signal_at": self.signal_at,
            "last_alert_at": self.last_alert_at,
            "last_abnormal_level": self.last_abnormal_level,
            "last_abnormal_pushed_at": self.last_abnormal_pushed_at,
            "pending_buy_order_id": self.pending_buy_order_id,
            # 2026-08-10：datetime 不能直接 json 序列化（10:00 实测
            # "Object of type datetime is not JSON serializable" → state 保存失败）
            "pending_buy_since": self.pending_buy_since.isoformat()
            if self.pending_buy_since else None,
            "sell_blocked_date": self.sell_blocked_date,
            "pending_sell_order_id": self.pending_sell_order_id,
            "pending_sell_since": self.pending_sell_since,
            "force_resell": self.force_resell,
            "netflow_history": self.netflow_history,
            "daily_basic": self.daily_basic,
            "dim_scores": self.dim_scores,
            "last_daily_update": self.last_daily_update,
            "entry_base_price": self.entry_base_price,
            "confirm_base": self.confirm_base,
            "confirm_count": self.confirm_count,
            "pullback_count": self.pullback_count,
            "day_high": self.day_high,
            "prev_close": self.prev_close,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WatchState":
        s = cls(d["code"], d.get("name", ""))
        s.added_at = d.get("added_at", "")
        s.status = d.get("status", "watching")
        s.source = d.get("source", "manual")
        s.entry_pushed_date = d.get("entry_pushed_date", "")
        s.t1_blocked_date = d.get("t1_blocked_date", "")
        s.entry_price = d.get("entry_price", 0.0)
        s.entry_at = d.get("entry_at", "")
        s.highest_since_entry = d.get("highest_since_entry", 0.0)
        s.bars_held = d.get("bars_held", 0)
        s.signal_type = d.get("signal_type", "")
        s.signal_reason = d.get("signal_reason", "")
        s.signal_at = d.get("signal_at", "")
        s.last_alert_at = d.get("last_alert_at", "")
        s.last_abnormal_level = d.get("last_abnormal_level", "")
        s.last_abnormal_pushed_at = d.get("last_abnormal_pushed_at", 0.0)
        s.pending_buy_order_id = d.get("pending_buy_order_id", "")
        _pbs = d.get("pending_buy_since")
        s.pending_buy_since = datetime.fromisoformat(_pbs) if isinstance(_pbs, str) else None
        s.sell_blocked_date = d.get("sell_blocked_date", "")
        s.pending_sell_order_id = d.get("pending_sell_order_id", "")
        s.pending_sell_since = d.get("pending_sell_since", 0.0)
        s.force_resell = d.get("force_resell", False)
        s.netflow_history = d.get("netflow_history", [])
        s.daily_basic = d.get("daily_basic", {})
        s.dim_scores = d.get("dim_scores", {})
        s.last_daily_update = d.get("last_daily_update", "")
        s.entry_base_price = d.get("entry_base_price", 0.0)
        s.confirm_base = d.get("confirm_base", 0.0)
        s.confirm_count = d.get("confirm_count", 0)
        s.pullback_count = d.get("pullback_count", 0)
        s.day_high = d.get("day_high", 0.0)
        s.prev_close = d.get("prev_close", 0.0)
        return s


# ---- 引擎 ----

class WatchdogEngine:
    """盯盘引擎（单例）"""

    def __init__(self):
        self._lock = threading.Lock()
        self._states: dict[str, WatchState] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # ── 确认器历史缓存（2026-08-12 简化蓝图）──
        # 每只票最近 15 轮 last / 分钟量（60s 轮询差分），供买入确认器判定
        self._price_hist: dict[str, list[float]] = {}
        self._vol_hist: dict[str, list[float]] = {}
        self._prev_tv: dict[str, float] = {}
        # 2026-08-27 主力吸筹影子信号：每只票 big_net+super_net 当日累计值序列
        self._bignet_hist: dict[str, list[float]] = {}
        # 2026-08-27 用户拍板"直接换不用等"：主力吸筹替代追拉升。
        # 1=fund_accumulate 触发即正式下单；0=回退到下方追拉升确认器。
        self._use_fund_accumulate = int(os.getenv("CONFIRM_USE_FUND_ACCUMULATE", "1")) == 1
        self._scan_count = 0
        self._subscribed: set[str] = set()
        self._snap_fail_count = 0
        # 已确认过的买单委托号（2026-08-10 幂等防重：state 保存失败时
        # pending 不清会导致同一委托重复落账，用集合兜底）
        self._confirmed_buy_orders: set[str] = set()
        # 2026-08-17 本引擎发出的卖出委托 order_id（-2/-5 分支区分
        # "自己上一轮委托已成交" vs "用户手动卖出"——古井贡酒 13:00 误报
        # 假出场+假交割单案例：手动卖出的已成委托被当成系统自己刚卖）
        self._sold_orders: set[str] = set()
        # 2026-08-20 单日买入计数（风控）：启动时从交割单恢复当日已买笔数，
        # 防止重启后归零导致继续爆量（0820 实测 52 笔）。
        self._daily_buys: dict[str, int] = {}
        try:
            from pathlib import Path
            _rep = Path(PROJECT_DIR) / "plays/trading/data/reports" / \
                f"{datetime.now().strftime('%Y%m%d')}.json"
            if _rep.exists():
                import json as _json
                _recs = _json.loads(_rep.read_text())
                self._daily_buys[datetime.now().strftime("%Y%m%d")] = sum(
                    1 for x in _recs if x.get("direction") == "买入")
        except Exception:
            pass
        self._snap_fail_by_code: dict[str, int] = {}  # code -> 无行情连续轮数（单票失效检测）
        # 2026-08-17 确认器拒绝原因分布（转化率诊断黑盒）：reason 类别 -> 次数
        self._confirm_reject_counts: dict[str, int] = {}
        self._netflow_cache: dict[str, tuple[float, float]] = {}  # code -> (ts, netflow)
        self._netflow_hist_ts: dict[str, float] = {}  # code -> 上次 netflow 采样时间（连续轮询下按真实时间采样）
        self._alert_ts: dict[str, float] = {}         # code -> 入场信号触发时间（连续轮询下按真实时间确认）
        self._state_mtime: float = 0.0
        # L1 行情熔断标志（2026-08-13）：当天只熔断一次
        self._fuse_triggered: bool = False
        # 进程启动：清空残留 pending（防跨重启幽灵单）；运行中 reload 保留
        self._load_state(clear_pending=True)
        # 2026-08-21 启动强制清零确认状态：重启后旧 trend_up 触发状态（confirm_base
        # 残留）会继续站稳→ready→追拉升买入（金一文化 10:34 回调仅-0.4%却买入实锤，
        # 不符合回调低吸 -3% 条件）。新引擎必须从回调低吸重新触发。
        for _s in self._states.values():
            if _s.status == "watching":
                _s.confirm_base = 0.0
                _s.confirm_count = 0
                _s.signal_type = ""
                _s.signal_reason = ""
        self._save_state()

    # ---- 生命周期 ----

    def start(self):
        if self._running:
            return
        self._running = True
        # 同步历史标的订阅到 ws_daemon
        with self._lock:
            codes = list(self._states.keys())
        if codes:
            self._subscribe(codes)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("盯盘引擎已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._save_state()
        logger.info("盯盘引擎已停止")

    def _subscribe(self, codes: list[str]):
        """订阅标的到 ws_daemon。"""
        shorts = [_short(c) for c in codes]
        self._subscribed.update(shorts)
        _write_ws_sub(list(self._subscribed), list(self._subscribed))

    def _unsubscribe(self, codes: list[str]):
        """从 ws_daemon 取消订阅。"""
        shorts = [_short(c) for c in codes]
        self._subscribed.difference_update(shorts)
        _write_ws_sub(list(self._subscribed), list(self._subscribed))

    # ---- 指令 ----

    def _resolve_name(self, code: str) -> str:
        try:
            resp = call_tushare("stock_basic", {"ts_code": code}, "ts_code,name")
            items = resp.get("data", {}).get("items", [])
            if items and len(items[0]) > 1:
                return items[0][1]
        except Exception:
            pass
        return code

    def add(self, codes: list[str]) -> str:
        codes = [_norm(c) for c in codes]
        msgs = []
        with self._lock:
            current = sum(1 for st in self._states.values() if st.source not in ("surge", "panel"))
            for code in codes:
                if code in self._states:
                    msgs.append(f"{code} 已在盯盘中")
                    continue
                if current >= MAX_WATCH:
                    msgs.append(f"盯盘已达上限({MAX_WATCH}只)，无法添加 {code}")
                    continue
                name = self._resolve_name(code)
                st = WatchState(code, name)
                self._update_daily(st)
                self._states[code] = st
                current += 1
                msgs.append(f"开始盯盘 {name}({code})")
            self._save_state()

        self._subscribe(codes)

        # 添加成功不推送飞书（避免干扰）；指令响应由飞书 bot 直接回复
        for code in codes:
            if code in self._states:
                st = self._states[code]
                logger.info(f"开始盯盘 {st.name}({code}) 日线数据: {'OK' if st.daily_rows else '加载中'}")

        return "\n".join(msgs)

    def remove(self, codes: list[str]) -> str:
        codes = [_norm(c) for c in codes]
        msgs = []
        with self._lock:
            for code in codes:
                if code in self._states:
                    st = self._states.pop(code)
                    pnl = ""
                    if st.status == "entered" and st.entry_price > 0:
                        pnl_pct = (self._last_price(code) / st.entry_price - 1) * 100
                        pnl = f" 盈亏{pnl_pct:+.2f}%"
                    msgs.append(f"停止盯盘 {st.name}({code}){pnl}")
                else:
                    msgs.append(f"{code} 未在盯盘中")
            self._save_state()
        self._unsubscribe(codes)
        return "\n".join(msgs)

    def clear_all(self) -> str:
        with self._lock:
            count = len(self._states)
            codes = list(self._states.keys())
            self._states.clear()
            self._save_state()
        self._unsubscribe(codes)
        return f"已清空{count}只盯盘标的"

    def list_all(self) -> str:
        with self._lock:
            if not self._states:
                return "当前无盯盘标的"
            lines = ["📋 盯盘列表:"]
            for code, st in self._states.items():
                icon = {"watching": "👁", "alerted": "⏳", "entered": "📈", "exited": "🔚"}.get(st.status, "❓")
                lines.append(f"  {icon} {st.name}({st.code}) [{st.status}]")
            return "\n".join(lines)

    # ---- 指令 ----

    # ---- 循环 ----

    def _loop(self):
        logger.info("盯盘循环启动")
        trading_day_logged = False
        self._state_mtime = self._get_state_mtime()
        # 对账节流：每 30 分钟一次，且只在交易时段
        self._reconcile_last = 0.0
        while self._running:
            try:
                if _is_trading_time():
                    if not trading_day_logged:
                        logger.info("进入交易时段，开始盯盘扫描")
                        trading_day_logged = True

                    # 持仓对账兜底（2026-08-06：珍宝岛/昭衍新药失管事故）——
                    # 把 CTP 实际持仓中不在 entered 的票自动加回盯盘，
                    # 防止买入竞态/状态丢失导致持仓脱离监控。
                    if time.time() - self._reconcile_last >= 1800:
                        self._reconcile_last = time.time()
                        self._reconcile_holds()

                    # 日切检测：交易日变更时重新订阅
                    _today_str = datetime.now().strftime("%Y%m%d")
                    if getattr(self, "_trade_date", "") != _today_str:
                        self._trade_date = _today_str
                        with self._lock:
                            _codes = list(self._states.keys())
                        if _codes:
                            self._subscribe(_codes)
                            logger.info(f"新交易日 {_today_str}，重新订阅 {len(_codes)} 只")

                    # 共享内存数据检测
                    snap_ok = WS_SNAP.exists()
                    if not snap_ok:
                        logger.warning(f"ws_daemon 共享内存未就绪")
                        time.sleep(5)
                        continue
                    # L1 行情熔断（2026-08-13）：交易时段内快照冻结超阈值 →
                    # 浮亏持仓市价卖出，避免断链期间损失放大。
                    # 持久化 marker（当天只熔断一次，重启不重复触发——
                    # V1 实测：重启后 _fuse_triggered 重置又触发一轮，反复发卖单）。
                    _fuse_marker = STATE_FILE.parent / f"fuse_{datetime.now().strftime('%Y%m%d')}.marker"
                    if not _fuse_marker.exists():
                        try:
                            _mt = WS_SNAP.stat().st_mtime
                            _frozen = time.time() - _mt
                            if _frozen >= FUSE_TRIGGER_SEC:
                                _fuse_marker.write_text(
                                    datetime.now().isoformat())
                                self._fuse_sell_losers(_frozen)
                        except Exception as _fe:
                            logger.warning(f"熔断检查异常: {_fe}")
                    # 动态加载外部 state.json 变更（用户通过 client 添加/删除）
                    self._reload_state_if_changed()
                    with self._lock:
                        codes = list(self._states.keys())
                    if codes:
                        self._scan_round(codes)
                    time.sleep(SCAN_INTERVAL)
                else:
                    # 只有真正收盘（>=15:00）或非交易日才触发盘后汰换。
                    # 2026-08-03 事故：午休 11:30-13:00 也是非交易时段，
                    # 原逻辑午休一到就 _eod_purge()，把当天已买入的持仓票
                    # （状态 watching/alerted）当"零信号票"删掉，07-31 因此
                    # 丢失 600986/603226/603679/605598 四只持仓盯盘。
                    _now_eod = datetime.now()
                    _is_really_eod = _now_eod.hour >= 15 or _now_eod.weekday() >= 5
                    if trading_day_logged and _is_really_eod:
                        logger.info("交易时段结束，暂停盯盘扫描")
                        trading_day_logged = False
                        self._eod_purge()
                    wait = _next_trading_start()
                    sleep_for = min(wait, 300.0)
                    for _ in range(int(sleep_for)):
                        if not self._running:
                            break
                        time.sleep(1)
            except Exception:
                import traceback
                logger.error(f"盯盘循环异常:\n{traceback.format_exc()}")
                time.sleep(SCAN_INTERVAL)
        logger.info("盯盘循环退出")

    def _eod_purge(self):
        """盘后汰换：surge 票当天一次入场信号都没触发（仍 watching/alerted）→ 移出盯盘列表。

        entered 状态的票保留（可能有持仓，继续走出场管理）。
        """
        with self._lock:
            doomed = [c for c, st in self._states.items()
                      if st.source in ("surge", "panel") and st.status != "entered"]
        if doomed:
            logger.info(f"盘后汰换 surge 零信号 {len(doomed)} 只: {doomed}")
            self.remove(doomed)

    def _get_state_mtime(self) -> float:
        try:
            return STATE_FILE.stat().st_mtime if STATE_FILE.exists() else 0.0
        except Exception:
            return 0.0

    def _reload_state_if_changed(self):
        """检测 state.json 是否被外部修改（watchdog_client 写入），有变化则重新加载。"""
        mtime = self._get_state_mtime()
        if mtime > self._state_mtime:
            logger.info("检测到 state.json 变更，重新加载")
            old_codes = set(self._states.keys())
            self._load_state()
            new_codes = set(self._states.keys())
            added = new_codes - old_codes
            removed = old_codes - new_codes
            if added:
                logger.info(f"新增盯盘: {added}")
                self._subscribe(list(added))
            if removed:
                logger.info(f"移除盯盘: {removed}")
                self._unsubscribe(list(removed))
            self._state_mtime = mtime

    def _execute_buy(self, code: str, st: WatchState, last: float, vwap: float, now: datetime):
        """买入执行：置 entered → 大盘弱势闸 → buy() → 成交确认/挂起复查。

        2026-08-12 提取：确认器 ready 与旧 30s 确认两条路径共用，
        避免两处 ~70 行重复（之前两处不同步会埋 bug）。
        大盘弱势闸拦下时回退 watching 并重置确认状态；委托未成交挂起复查。
        2026-08-20 买入三重风控前置检查（0820 实测 52 笔买光资金 57 只持仓）。
        """
        # 2026-08-20 暂停买入总开关（surge 玩法暂停，只做离场）
        if PAUSE_BUY:
            logger.info(f"{code} 暂停买入(PAUSE_BUY)，仅离场模式")
            st.status = "watching"
            st.signal_type = ""
            st.confirm_base = 0.0
            st.confirm_count = 0
            self._save_state()
            return
        # ① 持仓上限：entered 数 >= MAX_POSITIONS 不再买入
        try:
            _entered = sum(1 for s in self._states.values()
                           if s.status == "entered")
            if _entered >= MAX_POSITIONS:
                logger.warning(
                    f"{code} 跳过买入: 持仓已达上限({_entered}/{MAX_POSITIONS})")
                st.status = "watching"
                st.signal_type = ""
                st.confirm_base = 0.0
                st.confirm_count = 0
                self._save_state()
                return
            # ② 单日买入笔数上限
            _today = now.strftime("%Y%m%d")
            if self._daily_buys.get(_today, 0) >= DAILY_BUY_LIMIT:
                logger.warning(
                    f"{code} 跳过买入: 今日已买{self._daily_buys.get(_today,0)}笔"
                    f"(上限{DAILY_BUY_LIMIT})")
                st.status = "watching"
                st.signal_type = ""
                st.confirm_base = 0.0
                st.confirm_count = 0
                self._save_state()
                return
            # ③ 可用资金下限
            try:
                from scripts.jvquant_trade_client import check_hold
                _avail = float((check_hold() or {}).get("usable") or 0)
                if _avail < MIN_FREE_CASH:
                    logger.warning(
                        f"{code} 跳过买入: 可用资金不足({_avail:.0f}<"
                        f"{MIN_FREE_CASH:.0f})")
                    st.status = "watching"
                    st.signal_type = ""
                    st.confirm_base = 0.0
                    st.confirm_count = 0
                    self._save_state()
                    return
            except Exception:
                pass  # 查资金失败不阻塞（继续原逻辑）
        except Exception:
            pass
        st.status = "entered"
        st.entry_price = last
        st.entry_at = now.strftime("%Y-%m-%d %H:%M:%S")
        st.highest_since_entry = last
        st.bars_held = 0
        st.entry_pushed_date = now.strftime("%Y%m%d")
        self._save_state()
        _surge_tag = "【盯盘】" if st.source in ("surge", "panel") else ""
        # 2026-08-19 移除大盘弱势禁买（用户拍板）：0819 实测上证 -1.66% 但
        # 主闸票 100% 上涨（结构性行情），弱势闸一刀切禁买误杀主闸机会。
        # 现策略：弱势只关排雷池（surge 侧已做），主闸买入交给确认器把关
        #（D 变体"触发轮仍在涨"拦截冲高回落，弱市自然少触发）。
        # 下单 + 飞书通知
        try:
            from scripts.jvquant_trade_client import buy
            short = _short(code)
            r = buy(short, st.name, vol=BUY_VOL)
            code_r = r.get("code", "?")
            if code_r in ("-2", "-3"):
                # 风控拒绝：已持仓 / 资金不足，回退状态继续盯盘
                logger.warning(f"{code} 跳过买入: {r.get('message', r)}")
                st.status = "watching"
                st.signal_type = ""
                st.signal_reason = ""
                st.signal_at = ""
                self._save_state()
            elif code_r == "0":
                order_id = r.get('order_id', '?')
                # 2026-08-03 修复：jvquant buy() 返回 code=0 只是"委托已报"，
                # 不是成交（杰克科技事故：status=已报但系统误判入场）。
                # 下单后查 check_order 确认"已成"才置 entered+推送；
                # 挂单未成交（已报/部分成交）→ 保持 watching 等待，
                # 避免资金冻结期间误判持仓、浪费盯盘。
                _deal_ok = False
                try:
                    from scripts.jvquant_trade_client import get_trade_client
                    _ord = get_trade_client().check_order()
                    _lst = (_ord or {}).get("list") or []
                    for _o in _lst:
                        if str(_o.get("order_id", "")) == str(order_id):
                            _deal_ok = _o.get("status") == "已成"
                            break
                except Exception as _e:
                    logger.warning(f"{code} 查成交状态异常: {_e}")
                if _deal_ok:
                    _today_key = now.strftime("%Y%m%d")
                    self._daily_buys[_today_key] = self._daily_buys.get(_today_key, 0) + 1
                    _push_feishu(
                        f"📈 {st.name}({code}) 入场{_surge_tag}\n"
                        f"入场价: {last:.2f}\n"
                        f"信号: {st.signal_reason}\n"
                        f"VWAP: {vwap:.2f}\n"
                        f"order_id: {order_id}"
                    )
                    _log_trade_journal(
                        code, st.name, "买入", last, BUY_VOL,
                        reason=f"信号: {st.signal_type or st.signal_reason}",
                    )
                else:
                    # 委托已报未成交：不置 entered，保持 watching
                    #（信号清掉，避免下一轮重复触发买入）
                    logger.warning(
                        f"{code} 委托{order_id}未成交(status=已报)，"
                        f"保持 watching 复查委托{order_id}")
                    st.status = "watching"
                    st.signal_type = ""
                    st.signal_reason = ""
                    st.signal_at = ""
                    st.pending_buy_order_id = str(order_id)
                    st.pending_buy_since = datetime.now()
                    self._save_state()
            else:
                logger.warning(f"{code} 下单返回异常: code={code_r} msg={r.get('message', r)}")
                st.status = "watching"
                st.signal_type = ""
                st.signal_reason = ""
                st.signal_at = ""
                self._save_state()
        except Exception as e:
            logger.error(f"{code} 下单异常: {e}")
            st.status = "watching"
            st.signal_type = ""
            st.signal_reason = ""
            st.signal_at = ""
            self._save_state()

    def _scan_round(self, codes: list[str]):
        today = datetime.now().strftime("%Y%m%d")
        now = datetime.now()
        self._scan_count += 1

        # 2026-08-14 并行日线更新：需要更新的票一次线程池批量(8 并发)。
        # 原实现：每只票在循环内串行调 Tushare(daily+daily_basic 2 个请求)，
        # 100 只票一轮 3-10 分钟，确认器 60s 轮询被拖成几分钟 → 买入/卖出
        # 全部钝化(8/14 09:30 实测 29 只串行花了 90s)。并行后一轮几十秒。
        try:
            from concurrent.futures import ThreadPoolExecutor
            _need = []
            for _c in codes:
                with self._lock:
                    _st = self._states.get(_c)
                if _st and not _st._daily_fail_today \
                        and (_st.last_daily_update != today or not _st.daily_rows):
                    _need.append(_st)
            if _need:
                with ThreadPoolExecutor(max_workers=8) as _ex:
                    list(_ex.map(self._update_daily, _need))
        except Exception as _de:
            logger.warning(f"并行日线更新异常: {_de}")

        if self._scan_count % 20 == 1:
            with self._lock:
                summary = ", ".join(f"{st.name}({st.code})[{st.status}]" for st in self._states.values())
            logger.info(f"盯盘心跳 #{self._scan_count}: {len(codes)}只 [{summary}]")
            # 2026-08-17 转化率诊断：确认器拒绝原因分布（每20轮打印一次）
            if self._confirm_reject_counts:
                _top = sorted(self._confirm_reject_counts.items(),
                              key=lambda x: -x[1])[:8]
                logger.info("确认器拒绝分布: " + ", ".join(
                    f"{k}={v}" for k, v in _top))

        for code in codes:
            with self._lock:
                st = self._states.get(code)
                if not st:
                    continue

            # ── ① 行情读取(快, 共享内存) ──
            market = _read_ws_snap(_short(code))
            if not market or not market.get("last"):
                self._snap_fail_count += 1
                # 2026-08-06：按 code 统计无行情轮次（原引擎级计数互相污染），
                # entered 持仓连续 10 轮无行情（≈10分钟）→ 飞书告警。
                # 只告警不重启 ws_daemon——2026-08-05 教训：贸然重启/重订
                # 会把好的订阅也搞崩（单票静默失效 ≠ 全量失效）。
                fail_n = self._snap_fail_by_code.get(code, 0) + 1
                self._snap_fail_by_code[code] = fail_n
                if fail_n == 10 and st.status == "entered":
                    _push_feishu(
                        f"⚠️ {code} {st.name} 持仓无实时行情 {fail_n} 轮\n"
                        f"（ws_daemon 单票订阅可能静默失效，离场信号暂时无法判定）"
                    )
                if self._scan_count % 10 == 0:
                    logger.warning(f"{code} 无行情数据（共享内存无数据{self._snap_fail_count}次）")
                continue
            self._snap_fail_count = 0  # 有数据了，重置
            self._snap_fail_by_code[code] = 0  # 有数据重置单票计数

            vwap_data = _read_ws_snap(f"{_short(code)}_vwap")
            vwap = float(vwap_data) if isinstance(vwap_data, (int, float)) else 0.0
            # vwap 兜底：jvQuant 对盘中新订阅票无 tick 历史 → get_vwap 返回空。
            # 快照自带 trade_amount/trade_volume，vwap = 成交额/成交量（精确当日均价）
            if vwap <= 0:
                _ta = float(market.get("trade_amount") or 0)
                _tv = float(market.get("trade_volume") or 0)
                if _ta > 0 and _tv > 0:
                    vwap = _ta / _tv
            last = float(market.get("last") or 0)
            # 2026-08-21 回调低吸状态：当日最高 + 昨收（ws_snap pre_close）
            if last > 0:
                st.day_high = max(st.day_high, last)
            if not st.prev_close:
                try:
                    st.prev_close = float(market.get("pre_close") or 0)
                except (TypeError, ValueError):
                    pass

            # ── ② 确认器历史维护（2026-08-12：最近 15 轮 last + 分钟量差分）──
            _ph = self._price_hist.setdefault(code, [])
            _ph.append(last)
            if len(_ph) > 15:
                self._price_hist[code] = _ph[-15:]
            _tv = float(market.get("trade_volume") or 0)
            # 首次遇到该 code 只记录基线，不产生"全天累计量"假差分
            #（否则盘中重启后第一轮 _dv=全天累计，巨量污染 vol_hist 前 10 轮，
            #  放量判定被抬高 → 确认器 10 分钟内难触发）
            if code in self._prev_tv:
                _dv = max(0.0, _tv - self._prev_tv[code])
            else:
                _dv = 0.0
            self._prev_tv[code] = _tv
            _vh = self._vol_hist.setdefault(code, [])
            _vh.append(_dv)
            if len(_vh) > 15:
                self._vol_hist[code] = _vh[-15:]
            # 2026-08-27 主力吸筹：维护 big_net+super_net 当日累计值序列
            #（差分后得每轮 60s 增量，供 fund_accumulate_confirm）
            _bn = float(market.get("big_net_amount") or 0) + float(market.get("super_net_amount") or 0)
            _bh = self._bignet_hist.setdefault(code, [])
            _bh.append(_bn)
            if len(_bh) > FA_N + 2:
                self._bignet_hist[code] = _bh[-(FA_N + 2):]

            # prev_last 快照：确认器(卖出)用 st.prev_last，异常检测用旧值——
            # 2026-08-14 重构后先保存旧值再更新（确认器提前不再依赖慢操作）
            _prev_last_old = st.prev_last
            st.prev_last = last

            # ── ③ 确认器判定（2026-08-14 提前：纯内存/行情，不受日线/因子/
            #     资金流等慢操作阻塞——日线失败票不再被 continue 跳过信号）──
            if st.status == "watching":
                # 2026-08-06：先复查挂起买单委托（委托已报但未确认成交）
                # 成交 → 补置 entered；未成交 → 本轮跳过入场（避免重复下单）
                if st.pending_buy_order_id:
                    if self._check_pending_buy(code, st):
                        continue
                    # 未成交仍在挂起：本轮不触发新买入，等下一轮复查
                    time.sleep(0.1)
                    continue
                # ── 2026-08-27 主力吸筹买入（正式切换，替代追拉升）──
                # 用户拍板"直接换不用等"：fund_accumulate 触发即下单。
                # CONFIRM_USE_FUND_ACCUMULATE=0 时回退到下方追拉升确认器。
                if self._use_fund_accumulate:
                    try:
                        _bb = float(market.get("big_buy_amount") or 0)
                        _bs = float(market.get("big_sell_amount") or 0)
                        _fa_ok, _fa_rs = fund_accumulate_confirm(
                            self._bignet_hist.get(code, []),
                            last, st.day_high, st.prev_close, _bb, _bs)
                        if _fa_ok:
                            st.confirm_base = 0.0
                            st.confirm_count = 0
                            st.signal_type = "fund_accumulate"
                            st.signal_reason = f"主力吸筹: {_fa_rs}"
                            st.signal_at = now.strftime("%H:%M:%S")
                            logger.info(f"{code} 主力吸筹触发买入: {_fa_rs}")
                            self._execute_buy(code, st, last, vwap, now)
                    except Exception as _e:
                        logger.error(f"{code} 主力吸筹判定异常: {_e}")
                    continue  # 用 fund_accumulate 时跳过下方追拉升确认器
                # ── 买入确认器（2026-08-12 简化蓝图，替代 check_entry/3轮确认/净上涨门禁）──
                # 创新高+窗口放量触发 → 站稳 2 轮 → ready 买入；
                # 防诱多由站稳天然覆盖（冲高回落=站不稳=重置）。
                _ph2 = self._price_hist.get(code, [])
                _vh2 = self._vol_hist.get(code, [])
                # 重启恢复：价格历史清空但 confirm_base 持久化残留 → 若价格已涨离
                # 基准(如 +5%)，站稳后会在远离触发价的位置追高买入。价格历史
                # 为空即确认状态失效，重置重新触发。
                if st.confirm_base > 0 and not _ph2:
                    logger.warning(f"{code} 重启后确认状态失效, 重置重新触发")
                    st.confirm_base = 0.0
                    st.confirm_count = 0
                _action, _base, _cnt = check_buy_confirm(
                    _ph2, _vh2, st.confirm_base, st.confirm_count,
                    st.day_high, st.prev_close)
                if _action == "wait":
                    # 2026-08-17 转化率诊断：记录拒绝原因分布（黑盒打破）。
                    # check_buy_confirm 丢弃 trend_up_trigger 的 reason，这里
                    # 单独再算一次（纯内存，便宜）。类别取主因（括号前）。
                    _ok2, _rs = trend_up_trigger(_ph2, _vh2)
                    if not _ok2:
                        _k = _rs.split("(")[0]
                        self._confirm_reject_counts[_k] = \
                            self._confirm_reject_counts.get(_k, 0) + 1
                elif _action == "trigger":
                    # 2026-08-14 当日形态过滤：高位衰落(high_decay)不触发。
                    # 1 周 117 笔验证：high_decay 35%胜率 -0.86% 唯一明显负收益；
                    # rising/low_rise/flat 不能拦(rising 58%+0.39% 最好)。
                    if self._shape_high_decay(code):
                        logger.info(
                            f"{code} 当日形态高位衰落, 不触发(跳过买入)")
                        st.signal_reason = "形态高位衰落, 不触发"
                    else:
                        st.confirm_base = _base
                        st.confirm_count = 0
                        st.signal_type = "trend_up"
                        st.signal_reason = f"趋势确认: 创新高+放量({_base:.2f})"
                elif _action == "stand":
                    st.confirm_count = _cnt
                    st.signal_reason = f"趋势确认: 站稳{_cnt}轮"
                elif _action == "ready":
                    # 保留 confirm_base（触发价）供 alerted 分支做"触发后不回落"闸
                    #（2026-08-17：立新 14.47触发→14.35成交 / 紫光 40.92→40.66，
                    #  站稳 2 轮窗口内回落 0.6-0.8% 仍放行，买在回落位）
                    st.confirm_count = 0
                    st.status = "alerted"
                    st._confirm_ready = True
                    st.signal_type = "trend_up"
                    st.signal_reason = "趋势确认: 站稳2轮"
                    st.signal_at = now.strftime("%H:%M:%S")
                    st.last_alert_at = now.strftime("%H:%M")
                    self._alert_ts[code] = time.time()
                    self._save_state()
                elif _action == "reset":
                    # 站不稳/跌破站稳线 → 重新等触发
                    st.confirm_base = 0.0
                    st.confirm_count = 0

            elif st.status == "alerted":
                # 2026-08-12 简化蓝图：确认器 ready（创新高+站稳2轮）→ 直接买入，
                # 跳过 30s 二次确认 + check_entry（站稳本身就是确认）。
                if getattr(st, "_confirm_ready", False):
                    st._confirm_ready = False
                    # ★ 2026-08-17 触发后回落闸（买趋势持续，不追回落）：
                    # ready 后价格跌回触发价 0.5% 以下 → 放弃买入，回退 watching
                    # 重新等。根因：确认器判定"过去10分钟拉升形态"，触发点常是
                    # 短期尖峰，站稳 2 轮窗口内回落 0.6-0.8% 仍放行（立新 14.47→
                    # 14.35 / 紫光 40.92→40.66 实盘买在回落位）。
                    if st.confirm_base > 0 and last < st.confirm_base * STAND_TOL:
                        logger.warning(
                            f"{code} 触发价{st.confirm_base:.2f}回落至{last:.2f}, "
                            f"放弃买入重新等")
                        st.confirm_base = 0.0
                        st.confirm_count = 0
                        st.status = "watching"
                        st.signal_type = ""
                        st.signal_reason = ""
                        st.signal_at = ""
                        self._save_state()
                        continue
                    st.confirm_base = 0.0
                    self._execute_buy(code, st, last, vwap, now)
                    continue
                # 30s 确认路径(check_entry 需要 row/scores)在慢操作后处理(见 ⑤)

            elif st.status == "entered":
                # 2026-08-06：reconcile 加回的持仓可能 entry_price=0（对账时快照未就绪）。
                # 首个有行情的轮次用 last 初始化入场基准，避免 check_exit 除零。
                if st.entry_price <= 0 and last > 0:
                    st.entry_price = last
                    st.highest_since_entry = max(st.highest_since_entry, last)
                    logger.info(f"{code} 对账持仓入场基准初始化: {last:.2f}")
                if last > st.highest_since_entry:
                    st.highest_since_entry = last
                st.bars_held += 1

                # ── 卖出确认器（2026-08-12 简化蓝图 + 2026-08-18 组合离场）──
                # 高位出场（隔夜低开开盘卖）/固定止损（入场价-4%兜底）/
                # 回撤 2%（最高点，原逻辑）/防插针/趋势恢复
                _mk = _read_ws_snap(_short(code))  # 短码键（L1281 _last_price 同款；全码查不到→prev_close=0 高位出场永不触发 0819 实测）
                _prev_close = 0.0
                try:
                    _prev_close = float((_mk or {}).get("pre_close", 0) or 0)
                except (TypeError, ValueError):
                    _prev_close = 0.0
                _overnight = st.entry_at[:10] < now.strftime("%Y-%m-%d") \
                    if st.entry_at and len(st.entry_at) >= 10 else False
                _sell_ok, _sell_reason, st.pullback_count = check_sell_confirm(
                    st.highest_since_entry, last, st.prev_last, st.pullback_count,
                    entry_price=st.entry_price, prev_close=_prev_close,
                    is_overnight=_overnight, now=now)
                exit_triggered, exit_reason = _sell_ok, _sell_reason
                # 2026-08-10：撤单后强制重卖（即使确认器未触发，
                # 也要按最新盘口追价重挂——否则挂单撤了就没人卖了）
                if st.force_resell and not exit_triggered:
                    exit_triggered = True
                    exit_reason = f"挂单超时撤单重挂(现价{last:.2f})"
                if exit_triggered:
                    # T+1 当日不可卖已确认：跳过卖出尝试，保留盯盘次日恢复
                    # （避免每轮调 CTP 重复拿 -5）
                    if st.t1_blocked_date == now.strftime("%Y%m%d"):
                        logger.debug(f"{code} T+1冻结中(标记{st.t1_blocked_date}),跳过卖出尝试")
                        self._save_state()
                        continue
                    pnl_pct = (last / st.entry_price - 1) * 100
                    _surge_tag = "【盯盘】" if st.source in ("surge", "panel") else ""
                    is_profit = "止盈" in exit_reason
                    # 下单卖出 + 飞书通知
                    # 安全规则(2026-07-31):
                    #   - sale 成功(code=0)才移除盯盘
                    #   - 失败/异常保留 entered,下轮重试(防仓位失管)
                    #   - 持仓不足(-2)解析可用股数,确为 0 才移除
                    _exit_ok = False
                    _sell_vol = SELL_VOL  # 实际下单股数（降档后更新，交割单用它）
                    try:
                        from scripts.jvquant_trade_client import sale
                        short = _short(code)
                        r = sale(short, st.name, vol=SELL_VOL)
                        # 2026-08-11：存量 1 手持仓（100股，BUY_VOL 改 200 前买入）
                        # 卖 200 股会返回 -2 持仓不足 → 降档 100 股重试，避免存量
                        # 持仓永远卖不出（离场信号触发但 sale 失败）。
                        # 2026-08-12：降档后 _sell_vol 同步 100——否则交割单记 200
                        # 股但实际成交 100 股，盈亏统计翻倍（13 笔卖出实测全错）。
                        # 2026-08-19：SELL_VOL 固定 200 不随 BUY_VOL(100)——
                        # 0818 前 200 股持仓必须一次卖清，100 股持仓走降档。
                        if str(r.get("code")) == "-2":
                            _sell_vol = 100
                            r = sale(short, st.name, vol=100)
                            logger.warning(
                                f"{code} 持仓不足(2手)，降档 100 股重试: {r.get('message', r)}")
                        code_r = r.get("code", "?")
                        if code_r == "-5":
                            # T+1 判定（2026-08-06 修复）：-5 有三种可能——
                            #   a) 真·当日买入不可卖（保留盯盘，次日恢复）
                            #   b) 自己的卖出委托已成交（可用0被误判成 T+1）
                            #   c) 自己的卖出委托挂单中（已报未成交）冻结可用股
                            # 600650 实测：委托已成 → 误判 T+1 被当持仓盯一天。
                            # 002319 实测：委托挂单中(已报) → 可用0 被误判 T+1，
                            #   标 t1_blocked_date 后当日不再尝试卖出 → 挂单永不成交、
                            #   持仓永远冻结（2026-08-07）。
                            # 修复：查 check_order 区分 b/c——已成→离场移除；
                            #   已报挂单→保持 entered 复查（不标 T+1）；
                            #   无卖出委托→真 T+1。
                            msg = r.get("message", "")
                            _sold = False
                            _pending_sell = False
                            try:
                                from scripts.jvquant_trade_client import get_trade_client
                                _ord3 = get_trade_client().check_order()
                                for _o in (_ord3 or {}).get("list") or []:
                                    if str(_o.get("code", "")) == _short(code) \
                                            and "卖出" in str(_o.get("type", "")):
                                        if _o.get("status") == "已成" \
                                                and str(_o.get("order_id", "")) in self._sold_orders:
                                            # 2026-08-17 只认本引擎发出的卖出委托
                                            _sold = True
                                            break
                                        if _o.get("status") in ("已报", "待报") \
                                                and str(_o.get("order_id", "")) in self._sold_orders:
                                            _pending_sell = True
                            except Exception:
                                pass
                            if _sold:
                                # 自己已卖出成交：按正常离场推送 + 移除
                                _push_feishu(
                                    f"{'💰' if is_profit else '🛑'} {st.name}({code}) "
                                    f"{'止盈' if is_profit else '出场'}{_surge_tag}\n"
                                    f"{exit_reason}\n"
                                    f"入场: {st.entry_price:.2f} → 现价: {last:.2f}\n"
                                    f"盈亏: {pnl_pct:+.2f}%"
                                )
                                logger.warning(f"{code} 卖出委托已成交(-5误判T+1),正常离场移除")
                                _exit_ok = True
                            elif _pending_sell:
                                # 自己挂的卖单还在盘口（已报未成交）：可用0是挂单冻结，
                                # 不是 T+1。保持 entered 复查，等挂单成交/撤单。
                                # 2026-08-07：标记当日卖出受阻——砸盘卖不出的票
                                # 不再反复推送异常状态噪音（大晟文化 12 次/小时）。
                                st.sell_blocked_date = datetime.now().strftime("%Y%m%d")
                                # 2026-08-10：挂单超时（>180s）→ 撤单追价重挂
                                #（大晟文化 09:30 挂 4.09，价格跌到 3.80 永不成交）
                                if st.pending_sell_since and \
                                        time.time() - st.pending_sell_since > 180:
                                    try:
                                        from scripts.jvquant_trade_client import cancel
                                        cancel(st.pending_sell_order_id)
                                        logger.warning(
                                            f"{code} 卖出挂单{st.pending_sell_order_id} "
                                            f"超时{time.time()-st.pending_sell_since:.0f}s "
                                            f"未成交，撤单追价重挂")
                                    except Exception as _ce:
                                        logger.warning(f"{code} 撤单失败: {_ce}")
                                    finally:
                                        st.pending_sell_order_id = ""
                                        st.pending_sell_since = 0.0
                                        st.force_resell = True
                                logger.warning(
                                    f"{code} 卖出委托挂单中(已报)冻结可用股,"
                                    f"保持 entered 复查(当日静默): {msg}")
                                _exit_ok = False
                            else:
                                # 真 T+1：当日买入可用0，不可卖。保留 entered 继续盯盘，
                                # 记录 t1_blocked_date，次日可用后自动恢复出场监控
                                st.t1_blocked_date = datetime.now().strftime("%Y%m%d")
                                logger.warning(f"{code} T+1当日不可卖,保留盯盘次日恢复: {msg}")
                                _exit_ok = False
                        elif code_r == "-2":
                            msg = r.get("message", "")
                            # message 形如 "持仓不足: xxx 可用{n}股,需{m}股"
                            m = re.search(r"可用(\d+)股", str(msg))
                            usable = int(m.group(1)) if m else 0
                            if usable == 0:
                                # 可用0股：可能是「确实无仓（手动卖出）」也可能是
                                # 「自己上一轮委托已成交」。2026-08-05 实测：
                                # 14:22 sale 返回 code=0 委托已报 → 14:23 再次 sale
                                # 返回 -2 可用0（委托已成），走此分支静默移除，
                                # 导致离场飞书通知/交割单丢失（天娱数科案例）。
                                # 修复：先查 check_order，若该票有"已成"卖出委托
                                # → 按正常离场推送+写交割单；无成交记录才静默移除。
                                _already_dealt = False
                                try:
                                    from scripts.jvquant_trade_client import get_trade_client
                                    _ord2 = get_trade_client().check_order()
                                    for _o in (_ord2 or {}).get("list") or []:
                                        if str(_o.get("code", "")) == _short(code) \
                                                and "卖出" in str(_o.get("type", "")) \
                                                and _o.get("status") == "已成" \
                                                and str(_o.get("order_id", "")) in self._sold_orders:
                                            # 2026-08-17 只认本引擎发出的卖出委托
                                            #（否则用户手动卖出的已成委托被误判成
                                            #  系统自己刚卖 → 假出场+假交割单，
                                            #  古井贡酒 13:00 实测 93.92x100）
                                            _already_dealt = True
                                            break
                                except Exception:
                                    pass
                                if _already_dealt:
                                    # 系统自己已卖出成交：补离场推送 + 交割单
                                    _push_feishu(
                                        f"{'💰' if is_profit else '🛑'} {st.name}({code}) "
                                        f"{'止盈' if is_profit else '出场'}{_surge_tag}\n"
                                        f"{exit_reason}\n"
                                        f"入场: {st.entry_price:.2f} → 现价: {last:.2f}\n"
                                        f"盈亏: {pnl_pct:+.2f}%"
                                    )
                                    _log_trade_journal(
                                        code, st.name, "卖出", last, _sell_vol,
                                        entry_price=st.entry_price, entry_at=st.entry_at,
                                        reason=exit_reason,
                                    )
                                    logger.warning(f"{code} 已成交卖出(可用0),补发离场推送+交割单")
                                else:
                                    # 确实无仓(可能已手动卖出),移除盯盘
                                    logger.warning(f"{code} 跳过卖出(无持仓): {msg}")
                                _exit_ok = True
                            else:
                                # 部分持仓不足1手,保留 entered 下轮重试
                                logger.warning(f"{code} 卖出被拒(可用{usable}股),保留盯盘重试: {msg}")
                        elif code_r == "0":
                            order_id = r.get('order_id', '?')
                            # 2026-08-17 记录本引擎发出的卖出委托（-2/-5 分支
                            # 据此区分"自己刚卖" vs "用户手动卖"）
                            self._sold_orders.add(str(order_id))
                            # 2026-08-03 修复：sale() 返回 code=0 只是"委托已报"，
                            # 未成交就移除盯盘=持仓失管（与买入侧同源 bug）。
                            # 查 check_order 确认"已成"才推送+移除；挂单中保留 entered 重试。
                            # 2026-08-10：新委托发出 → 清撤单重卖标记（避免重复卖）
                            st.force_resell = False
                            _deal_ok = False
                            try:
                                from scripts.jvquant_trade_client import get_trade_client
                                _ord = get_trade_client().check_order()
                                _lst = (_ord or {}).get("list") or []
                                for _o in _lst:
                                    if str(_o.get("order_id", "")) == str(order_id):
                                        _deal_ok = _o.get("status") == "已成"
                                        break
                            except Exception as _e:
                                logger.warning(f"{code} 卖出查成交状态异常: {_e}")
                            if _deal_ok:
                                _push_feishu(
                                    f"{'💰' if is_profit else '🛑'} {st.name}({code}) {'止盈' if is_profit else '出场'}{_surge_tag}\n"
                                    f"{exit_reason}\n"
                                    f"入场: {st.entry_price:.2f} → 现价: {last:.2f}\n"
                                    f"盈亏: {pnl_pct:+.2f}%\n"
                                    f"order_id: {order_id}"
                                )
                                _log_trade_journal(
                                    code, st.name, "卖出", last, _sell_vol,
                                    entry_price=st.entry_price, entry_at=st.entry_at,
                                    reason=exit_reason,
                                )
                                _exit_ok = True
                            else:
                                # 卖出委托已报未成交：保留 entered 下轮复查，
                                # 避免未成交就移除盯盘导致持仓失管
                                # 2026-08-10：记录挂单时间，超时自动撤单重挂
                                #（大晟文化 09:30 挂单 4.09 到 10:03 未成交，
                                #  价格跌到 3.80，挂单永不成交——必须追价重挂）
                                st.pending_sell_order_id = str(order_id)
                                st.pending_sell_since = time.time()
                                logger.warning(
                                    f"{code} 卖出委托{order_id}未成交(status=已报)，"
                                    f"保留 entered 复查")
                                _exit_ok = False
                        else:
                            # 其他异常返回(如 -1 无价格/-4 重登录失败),保留盯盘下轮重试
                            logger.warning(f"{code} 卖出返回异常,保留盯盘重试: code={code_r} msg={r.get('message', r)}")
                    except Exception as e:
                        logger.error(f"{code} 卖出异常,保留盯盘重试: {e}")
                    if _exit_ok:
                        self.remove([code])

            # ── ④ 慢操作（日线/因子/资金流/异常检测——确认器判定已在 ③ 完成，
            #      这些不再阻塞买卖信号）──
            if st.last_daily_update != today or not st.daily_rows:
                # 2026-08-14: 日线更新已在 _scan_round 开头并行批量完成；
                # 此处兜底只做单只重试(并行更新失败/新加入票)
                self._update_daily(st)

            if not st.daily_rows:
                # 日线失败/缺失：只跳过依赖日线的慢操作(因子/异常)，确认器已判定
                self._save_state()
                continue

            bid_price = market.get("bid_price", [None] * 10)
            ask_price = market.get("ask_price", [None] * 10)
            bid1 = float(bid_price[0]) if bid_price and bid_price[0] else 0
            ask1 = float(ask_price[0]) if ask_price and ask_price[0] else 0
            ask_bid = bid1 / ask1 if ask1 > 0 else 1.0
            daily_features = price_features(st.daily_rows)
            row = realtime_row(code, market, vwap, daily_features, st.daily_basic, st.dim_scores, st.daily_rows)
            scores = compute_factor_scores(row)

            # 补充 L1/L2 实时信号到 row（供 check_entry 入场确认）
            row["bid1"] = bid1
            row["ask1"] = ask1
            row["vwap"] = vwap
            row["inner_vol"] = market.get("inner_vol", 0)
            row["outer_vol"] = market.get("outer_vol", 0)
            row["last"] = float(market.get("last") or 0)
            # 10档盘口深度（诱多/空识别用）
            row["bid_qty"] = market.get("bid_qty", [])
            row["ask_qty"] = market.get("ask_qty", [])

            # 资金流向历史：按真实时间采样（≥60s 一个点，10点≈10分钟窗口；
            # 连续轮询下若按轮追加，窗口会从5分钟塌缩到几十秒）
            _now_ts = time.time()
            if _now_ts - self._netflow_hist_ts.get(code, 0.0) >= 60:
                netflow = self._get_netflow(code)
                st.netflow_history.append(netflow)
                if len(st.netflow_history) > 10:
                    st.netflow_history = st.netflow_history[-10:]
                self._netflow_hist_ts[code] = _now_ts
            else:
                netflow = (st.netflow_history[-1] if st.netflow_history
                           else self._get_netflow(code))

            # ── 异常状态检测（资金离场 / 抛压 / 诱空）──
            abnormal, level, abnormal_reason = check_abnormal(
                row, scores, netflow, ask_bid,
                entry_price=st.entry_price if st.status == "entered" else 0.0,
                netflow_history=st.netflow_history,
                prev_last=_prev_last_old,
                prev_vwap=st.prev_vwap,
                prev_vol_ratio=st.prev_vol_ratio,
            )

            # 保存本轮快照供下一轮诱空检测
            st.prev_vwap = vwap
            st.prev_vol_ratio = row.get("vol_ratio_proxy", 1.0)

            if abnormal:
                if level == "bear_trap":
                    # 诱空不移出盯盘,不推送
                    st.last_abnormal_level = level
                    st.last_abnormal_pushed_at = time.time()
                elif st.source in ("surge", "panel"):
                    # surge/panel 票保持静默（只发入场信号）
                    st.last_abnormal_level = level
                    st.last_abnormal_pushed_at = time.time()
                    # 2026-08-13 修复：critical 只移除未持仓的 watching/alerted 票；
                    # entered(持仓)不移除——否则持仓脱离离场监控（永杉锂业 08-13
                    # 实测：critical 移除后 CTP 200 股失管，回撤 2.8% 无人管）。
                    if level == "critical" and st.status != "entered":
                        self.remove([code])
                        continue
                else:
                    # 当日卖出受阻的票（砸盘卖不出）：异常状态只记日志不推送，
                    # 避免每 5 分钟刷一条噪音（2026-08-07 大晟文化 12 次/小时）
                    if st.sell_blocked_date == now.strftime("%Y%m%d"):
                        if level != st.last_abnormal_level or \
                                time.time() - st.last_abnormal_pushed_at >= ABNORMAL_COOLDOWN_SECONDS:
                            logger.warning(
                                f"{code} 异常状态 [{level}] {abnormal_reason} "
                                f"(当日卖出受阻，静默不推送)")
                            st.last_abnormal_level = level
                            st.last_abnormal_pushed_at = time.time()
                    else:
                        icon = "🚨" if level == "critical" else "⚠️"
                        msg = (
                            f"{icon} {st.name}({code}) 异常状态 [{level}]\n"
                            f"{abnormal_reason}\n"
                            f"现价: {last:.2f} | VWAP: {vwap:.2f}"
                        )
                        # 冷却期内不重复推送
                        since_last = time.time() - st.last_abnormal_pushed_at
                        if level != st.last_abnormal_level or since_last >= ABNORMAL_COOLDOWN_SECONDS:
                            _push_feishu(msg)
                            st.last_abnormal_level = level
                            st.last_abnormal_pushed_at = time.time()
                    if level == "critical":
                        self.remove([code])
                        continue
            else:
                # 状态恢复正常，清空冷却记录（下次异常立即推送）
                # bear_trap 不移出盯盘，不清冷却，防止高频重复
                if st.last_abnormal_level != "bear_trap":
                    st.last_abnormal_level = ""

            # ── ⑤ alerted 30s 确认路径（需要 row/scores，2026-08-14 重构后移此）──
            if st.status == "alerted" and not getattr(st, "_confirm_ready", False):
                # 信号确认：触发满 30s 后仍满足条件才入场
                # （连续轮询下按真实时间判断，保持原 30s 间隔的确认语义）
                if time.time() - self._alert_ts.get(code, 0.0) >= 30:
                    triggered, _, _ = check_entry(row, scores)
                    # 2026-08-12：alerted 最终确认也要查价格不跌——原逻辑只
                    # 查 check_entry 信号（价格还在 VWAP 上即可），但票可能已
                    # 从高点冲高回落（恒为科技 -3.9%、天娱数科距高点 -5.7%
                    # 实测仍可入）。alerted 最终防线：30s 确认期内价格不跌
                    #（watching 已用基准价净上涨确认，此处兜底防止确认后急跌）
                    _price_ok = (_prev_last_old <= 0) or (last >= _prev_last_old)
                    if triggered and _price_ok:
                        self._execute_buy(code, st, last, vwap, now)
                    else:
                        st.status = "watching"
                        st.signal_type = ""
                        st.signal_reason = ""
                        st.signal_at = ""
                        self._save_state()

            self._save_state()

    def _shape_high_decay(self, code: str) -> bool:
        """拉 THS 当日分时判断高位衰落形态（2026-08-14）。

        返回 True = high_decay（高点回落+高位反抽=脉冲，不买）。
        数据/接口失败一律放行(False)——形态过滤是锦上添花，不因
        数据缺失阻断买入（THS 反爬/限流时不误伤正常信号）。
        """
        try:
            from plays.watchdog.confirm import classify_intraday_shape
            ths = self._ths_singleton()
            r = ths.get_index_intraday(_short(code))
            if not r or not r.get("points") or len(r["points"]) < 10:
                return False
            prices = [p for _, p in r["points"]]
            return classify_intraday_shape(prices) == "high_decay"
        except Exception as e:
            logger.warning(f"{code} 形态判定失败(放行): {e}")
            return False

    def _ths_singleton(self):
        """THSClient 懒加载单例（形态判定复用连接，避免每票新建触发反爬）。"""
        if getattr(self, "_ths_inst", None) is None:
            from scripts.ths_client import THSClient
            self._ths_inst = THSClient()
        return self._ths_inst

    def _last_price(self, code: str) -> float:
        market = _read_ws_snap(_short(code))
        return float(market.get("last") or 0) if market else 0.0

    def _get_netflow(self, code: str) -> float:
        """从 jvQuant 获取当日大单+中单净流向（元）。

        节流：每只股票 300s 缓存。资金流向是当日累计值，30s 粒度无意义，
        不节流时每股每轮一次 REST，30 只票一轮 30-60s 会拖垮 30s 节奏。
        """
        now = time.time()
        cached = self._netflow_cache.get(code)
        if cached and now - cached[0] < 300:
            return cached[1]
        val = 0.0
        try:
            from scripts.jvquant_client import get_jvquant_client
            client = get_jvquant_client()
            data = client.get_fundflow_single(_short(code))
            if data:
                # main_net/big_net/mid_net 单位万元，转元
                val = (
                    float(data.get("main_net", 0) or 0)
                    + float(data.get("big_net", 0) or 0)
                    + float(data.get("mid_net", 0) or 0)
                ) * 10000
        except Exception as e:
            logger.debug(f"获取资金流向失败 {code}: {e}")
        self._netflow_cache[code] = (now, val)
        return val

    # ---- 日线数据 ----

    def _update_daily(self, st: WatchState) -> bool:
        try:
            resp = call_tushare("daily", {"ts_code": st.code, "limit": 120},
                               "trade_date,open,high,low,close,pre_close,vol,amount,pct_chg")
            items = resp.get("data", {}).get("items", [])
            if not items or len(items) < 30:
                logger.warning(f"{st.code} 日线数据不足({len(items)}条)")
                return False

            fields = resp["data"]["fields"]
            rows = sorted(items, key=lambda x: x[0])
            st.daily_rows = [dict(zip(fields, row)) for row in rows]
            # 2026-08-21 昨收修复：jvquant L1 不推 pre_close（ws_snap 恒 0），
            # 高位出场/回调低吸的 prev_close 从 tushare 日线最新收盘取。
            if st.daily_rows and st.daily_rows[-1].get("close"):
                st.prev_close = float(st.daily_rows[-1]["close"])

            # daily_basic
            basic_resp = call_tushare("daily_basic", {"ts_code": st.code, "limit": 1},
                                      "ts_code,trade_date,pe,pb,circ_mv,turnover_rate,volume_ratio")
            basic_items = basic_resp.get("data", {}).get("items", [])
            if basic_items:
                basic_fields = basic_resp["data"]["fields"]
                st.daily_basic = dict(zip(basic_fields, basic_items[0]))

            # 五维度分：从最新 analysis 文件读取
            st.dim_scores = self._load_dim_scores(st.code)
            st.last_daily_update = datetime.now().strftime("%Y%m%d")
            st._daily_fail_today = False
            logger.info(f"{st.code} 日线数据更新完成({len(rows)}条)")
            return True
        except Exception as e:
            st._daily_fail_today = True  # 2026-08-14: 当天失败不重试(防每轮串行阻塞)
            logger.error(f"{st.code} 日线更新失败: {e}")
            return False

    def _load_dim_scores(self, code: str) -> dict:
        """从最新 analysis 文件读取五维度分"""
        files = []
        for base in ANALYSIS_DIRS:
            if base.exists():
                files.extend(base.glob("*.json"))
        if not files:
            return {}
        latest = sorted(files, key=lambda f: f.name, reverse=True)[0]
        try:
            data = json.loads(latest.read_text())
            if isinstance(data, list):
                for item in data:
                    if item.get("code") == code:
                        return item.get("scores", {})
        except Exception:
            pass
        return {}

    # ---- 候选池 ----

    def load_candidates(self) -> list[tuple[str, str]]:
        """从最新 limit_up analysis 加载候选股池。返回 [(code, name), ...]。"""
        files = []
        for base in ANALYSIS_DIRS:
            if base.exists():
                files.extend(base.glob("*.json"))
        if not files:
            return []
        latest = sorted(files, key=lambda f: f.name, reverse=True)[0]
        try:
            data = json.loads(latest.read_text())
        except Exception:
            return []

        candidates = []
        if isinstance(data, list):
            for item in data:
                code = item.get("code")
                name = item.get("name", "")
                total_score = float(item.get("total_score") or item.get("model_score", 0) or 0)
                scores = item.get("scores", {})
                if code and total_score >= 85:
                    candidates.append((code, name, total_score, scores))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(c[0], c[1]) for c in candidates[:MAX_WATCH]]

    # ---- 状态持久化 ----

    def _fuse_sell_losers(self, frozen_sec: float):
        """L1 行情熔断：快照冻结超阈值 → 浮亏持仓市价卖出。

        2026-08-13 新增（用户拍板）：jvQuant WS 行情丢失时（如 8/13 502 超
        30 分钟），持仓无实时监控，断链期间票若下跌确认器无法止损 → 损失
        放大。熔断主动卖出浮亏持仓（CTP 交易走 ticket 独立通道，不依赖
        jvQuant 行情，能卖）。浮盈持仓保留（平台恢复后继续监控）。
        """
        logger.warning(f"L1 行情熔断: 快照冻结 {frozen_sec:.0f}s，浮亏持仓止损")
        with self._lock:
            codes = [c for c, v in self._states.items() if v.status == "entered"]
        sold, kept, failed = [], [], []
        for code in codes:
            st = self._states[code]
            _mkt = _read_ws_snap(_short(code))
            last = float(_mkt.get("last") or 0) if _mkt else 0.0
            if st.entry_price <= 0 or last <= 0:
                failed.append(f"{st.name}({code} 无价格)")
                continue
            if last >= st.entry_price:
                kept.append(f"{st.name}({code})")
                continue
            try:
                from scripts.jvquant_trade_client import sale, get_trade_client
                short = _short(code)
                r = sale(short, st.name, vol=SELL_VOL)  # 2026-08-19 不随 BUY_VOL
                if str(r.get("code")) == "-2":
                    r = sale(short, st.name, vol=100)
                # 2026-08-13 熔断 V2：sale 返回 0 只是"委托已报"，必须查
                # check_order 确认"已成"才写交割单+移除（V1 实测：5 只卖单
                # 仅盈方微成交，翔鹭/宇环/久其/南华已报未成 → 写幽灵交割单）。
                if str(r.get("code")) == "0":
                    _oid = r.get("order_id", "")
                    _deal_ok = False
                    try:
                        _ord = get_trade_client().check_order()
                        for _o in (_ord or {}).get("list") or []:
                            if str(_o.get("order_id", "")) == str(_oid):
                                _deal_ok = _o.get("status") == "已成"
                                break
                    except Exception as _ce:
                        logger.warning(f"{code} 熔断查成交异常: {_ce}")
                    if _deal_ok:
                        _log_trade_journal(
                            code, st.name, "卖出", last, BUY_VOL,
                            reason="熔断: 行情丢失")
                        with self._lock:
                            self._states.pop(code, None)
                        sold.append(f"{st.name}({code}@{last:.2f})")
                    else:
                        # 已报未成：保持 entered，等待（行情恢复后确认器接管）
                        failed.append(f"{st.name}({code} 卖单已报未成{_oid})")
                else:
                    failed.append(f"{st.name}({code} {str(r.get('message', r))[:30]})")
            except Exception as e:
                failed.append(f"{st.name}({code} {str(e)[:30]})")
            finally:
                if sold or failed:
                    self._save_state()
        msg = (f"🚨 L1 行情熔断: 快照冻结 {frozen_sec:.0f}s\n"
               f"浮亏止损卖出 {len(sold)} 只: {sold}\n"
               f"浮盈保留 {len(kept)} 只: {kept[:8]}\n")
        if failed:
            msg += f"失败 {len(failed)} 只: {failed[:5]}"
        logger.warning(msg)
        try:
            _push_feishu(msg)
        except Exception as _e:
            logger.warning(f"熔断飞书推送失败: {_e}")

    def _reconcile_holds(self):
        """持仓对账兜底：CTP 实际持仓 vs 盯盘 entered，失管持仓自动加回。

        2026-08-06 修复（珍宝岛/昭衍新药/神剑股份失管事故）：
        买入委托"已报"后实际成交，但 watchdog 未确认 → 状态未进 entered →
        盘后零信号汰换 → 持仓脱离监控（止损/止盈永不检查）。
        这里每 30 分钟把 CTP 有仓但不在 entered 的票加回盯盘（entered 状态），
        保证任何原因导致的失管持仓都能被重新接管。
        """
        try:
            from scripts.jvquant_trade_client import check_hold
            r = check_hold()
            hl = (r or {}).get("hold_list") or []
            if not hl:
                return
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            today = datetime.now().strftime("%Y%m%d")
            with self._lock:
                changed = False
                for h in hl:
                    vol = int(h.get("hold_vol", 0) or 0)
                    if vol <= 0:
                        continue
                    short = str(h.get("code", ""))
                    if len(short) != 6:
                        continue
                    code = f"{short}.SH" if short.startswith("6") else f"{short}.SZ"
                    st = self._states.get(code)
                    if st and st.status == "entered":
                        continue
                    name = h.get("name", "") or short
                    if st is None:
                        st = WatchState(code, name)
                        st.added_at = now_str
                        self._states[code] = st
                    # 置 entered，从当前价/持仓信息恢复入场基准
                    st.status = "entered"
                    changed = True
                    st.source = "reconcile"
                    st.entry_at = st.entry_at or now_str
                    st.entry_pushed_date = today
                    st.name = name
                    if st.entry_price <= 0:
                        # CTP 无成本价字段，用快照当前价近似（比失管好，
                        # 止损/止盈从该价起算；真实成本价可后续手工修正）
                        _mkt = _read_ws_snap(short)
                        _last = float(_mkt.get("last") or 0) if _mkt else 0.0
                        st.entry_price = _last or 0.0
                    if st.highest_since_entry <= 0:
                        st.highest_since_entry = st.entry_price
                    logger.warning(
                        f"[对账] {code} {name} CTP持仓{vol}股不在entered盯盘，"
                        f"自动加回盯盘 (entry={st.entry_price:.2f})")
                if changed:
                    self._save_state()
            # 新加入的票订阅 WS
            with self._lock:
                codes = [c for c, s in self._states.items() if s.status == "entered"]
            self._subscribe(codes)
        except Exception as e:
            logger.warning(f"[对账] 持仓对账异常: {e}")

    def _check_pending_buy(self, code: str, st: WatchState) -> bool:
        """复查挂起的买单委托：已成则补置 entered + 推送 + 落账。

        2026-08-06 修复（珍宝岛/昭衍新药失管事故）：
        买入返回 code=0 委托已报时，check_order 那一瞬间可能未成交，
        实际却成交了（封板打开瞬间/排队成交）。原逻辑清信号保持 watching，
        之后不再触发买入 → 盘后零信号汰换 → 持仓失管。
        修复：委托已报后记录 order_id，每轮复查直到确认成交。
        返回 True 表示本轮已处理（进入 entered 或仍等待）。
        """
        oid = st.pending_buy_order_id
        if not oid:
            return False
        # 2026-08-10 幂等保护：同一委托号已被确认过 → 不再重复写交割单。
        #（珍宝岛实测：08-07 3笔 + 08-10 2笔"挂单成交"买入，同一 order_id
        #  被重复确认——根因是 state 保存失败后 pending 未清，下轮又复查
        #  同一委托。用已处理集合兜底，即使 state 再丢也能防重复落账。）
        if oid in self._confirmed_buy_orders:
            logger.warning(f"{code} 买单委托{oid} 已确认过，跳过重复落账")
            st.pending_buy_order_id = ""
            st.pending_buy_since = None
            return True
        now = datetime.now()
        # 挂起超 10 分钟仍未成交 → 放弃（可能撤单/废单），清 pending 防无限复查
        if st.pending_buy_since and (now - st.pending_buy_since).total_seconds() > 600:
            logger.warning(f"{code} 买单委托{oid} 超10分钟未成交，放弃复查")
            st.pending_buy_order_id = ""
            st.pending_buy_since = None
            return False
        try:
            from scripts.jvquant_trade_client import get_trade_client
            _ord = get_trade_client().check_order()
            for _o in (_ord or {}).get("list") or []:
                # 2026-08-10 交叉匹配 bug 修复：必须 order_id + code 双匹配。
                # 原逻辑只比 order_id——state 残留的 pending_buy_order_id 若
                # 恰好等于其他票的委托号（委托号复用），会把别人的成交误判
                # 成该票成交，写幽灵交割单（珍宝岛 5 笔幽灵单实测：08-07 3笔
                # + 08-10 2笔，CTP 实际从未买入）。
                if str(_o.get("order_id", "")) == str(oid) \
                        and str(_o.get("code", "")) == _short(code):
                    if _o.get("status") == "已成":
                        # 2026-08-18 幽灵单根治：写单前先验证 CTP 实际持仓。
                        # 8/17 5 笔 + 8/18 4 笔珍宝岛"挂单成交"假单——check_order
                        # 显示委托"已成"但 CTP 从未持仓（残留引擎/历史委托复查
                        # 写假单）。成交必须由持仓佐证：CTP 无该票持仓 → 不落账。
                        try:
                            from scripts.jvquant_trade_client import check_hold
                            _hl = (check_hold() or {}).get("hold_list") or []
                            _held = any(
                                str(h.get("code", "")) == _short(code) for h in _hl)
                        except Exception:
                            _held = False
                        if not _held:
                            logger.warning(
                                f"{code} 委托{oid}显示已成但 CTP 无持仓，"
                                f"疑似幽灵单，不落账并清 pending")
                            st.pending_buy_order_id = ""
                            st.pending_buy_since = None
                            self._save_state()
                            return True
                        # 确认成交：补置 entered + 推送 + 落账
                        last = float(st.prev_last or 0) or st.entry_price or 0
                        _push_feishu(
                            f"📈 {st.name}({code}) 入场【surge】\n"
                            f"入场价: {last:.2f} (委托{oid}已成)\n"
                            f"信号: {st.signal_reason or '挂单成交'}\n"
                            f"order_id: {oid}"
                        )
                        _log_trade_journal(
                            code, st.name, "买入", last, BUY_VOL,
                            reason=f"信号: {st.signal_type or '挂单成交'}",
                        )
                        st.status = "entered"
                        st.entry_price = last if last > 0 else st.entry_price
                        st.entry_at = now.strftime("%Y-%m-%d %H:%M:%S")
                        st.highest_since_entry = max(st.highest_since_entry, last)
                        st.pending_buy_order_id = ""
                        st.pending_buy_since = None
                        self._confirmed_buy_orders.add(oid)  # 幂等：已确认过
                        self._save_state()
                        logger.info(f"{code} 挂单委托{oid}确认成交，补置 entered")
                        return True
                    break
        except Exception as e:
            logger.warning(f"{code} 复查买单委托异常: {e}")
        return False

    def _save_state(self):
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {code: st.to_dict() for code, st in self._states.items()}
            # 2026-08-06：原子写（tempfile + rename）——state.json 被
            # surge_scanner 和 watchdog 多进程共写，非原子 write_text/open("w")
            # 会互相截断（实测 10:31 损坏一次）。原子写保证读取方永远看到
            # 完整 JSON（旧内容或新内容，绝不会半截）。
            # 2026-08-10：tmp 名必须带 pid——三个进程（watchdog/surge/
            # restore_positions）原先共用 state.json.tmp，并发时 A 写 tmp
            # 被 B 覆盖 → A rename 报 FileNotFoundError（13:03 实测）。
            _tmp = STATE_FILE.with_name(f"state.json.tmp.{os.getpid()}")
            with open(_tmp, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            _tmp.rename(STATE_FILE)
        except Exception as e:
            logger.error(f"状态保存失败: {e}")

    def _load_state(self, clear_pending: bool = False):
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    data = json.load(f)
                # 重建 _states：state.json 中没有的 → 删除
                new_states = {}
                for code, d in data.items():
                    new_states[code] = WatchState.from_dict(d)
                # 2026-08-10：进程启动（clear_pending=True）后清空所有
                # pending_buy_order_id——重启时挂单可能已成交也可能已废，
                # 跨重启复查会把历史委托误判成新成交写幽灵交割单（13:41/15:54
                # 珍宝岛 5 笔幽灵单全是我部署重启触发）。
                # ★ 2026-08-17 修复：只有真重启才清。盘中 state.json 变更
                # 触发的 reload（surge_scanner 每 60s 写 state → _reload_state_
                # if_changed → 本函数）若也清空，会把"已报未成交"的在途买单
                # 复查放弃——委托随后成交无人确认 → 漏账 + 延迟入账
                # （0817 立新能源 754483 / 0814 江海 528948 实测）。
                if clear_pending:
                    for st in new_states.values():
                        if st.pending_buy_order_id:
                            logger.warning(
                                f"{st.code} 重启清空残留挂单复查(pending={st.pending_buy_order_id})")
                            st.pending_buy_order_id = ""
                            st.pending_buy_since = None
                self._states.clear()
                self._states.update(new_states)
                logger.info(f"加载盯盘状态: {len(self._states)} 只")
        except Exception as e:
            logger.error(f"状态加载失败: {e}")


# ---- 全局单例 ----

_engine: Optional[WatchdogEngine] = None


def get_engine() -> WatchdogEngine:
    global _engine
    if _engine is None:
        _engine = WatchdogEngine()
    return _engine


# ---- 交易时段 ----

def _is_trading_time() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    if h < 9 or (h == 9 and m < 30):
        return False
    if h >= 15:
        return False
    if h == 11 and m >= 30:
        return False
    if h == 12:
        return False
    return True


def _next_trading_start() -> float:
    from datetime import timedelta
    now = datetime.now()
    wd = now.weekday()

    if wd == 4 and now.hour >= 15:
        next_day = now + timedelta(days=3)
    elif wd >= 5:
        next_day = now + timedelta(days=(7 - wd))
    elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
        next_day = now.replace(hour=9, minute=30, second=0, microsecond=0)
    elif (now.hour == 11 and now.minute >= 30) or now.hour == 12:
        next_day = now.replace(hour=13, minute=0, second=0, microsecond=0)
    elif now.hour >= 15:
        next_day = now + timedelta(days=1)
        next_day = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
    else:
        return 0.0

    return max((next_day - now).total_seconds(), 0.0)


# ---- 入口 ----

def main():
    """cron 每日拉起模式（2026-07-26）：09:20 hermes cron 启动 → 内部 30s 循环
    → 15:05 自动退出（EOD 汰换 15:00 已执行）。非交易日直接退出。
    pid 守卫防多实例（cron 与手动启动撞车）。代码改动次日自然生效。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # pid 防多实例
    pid_file = PROJECT_DIR / "plays" / "watchdog" / "data" / "health" / "watchdog.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            os.kill(old_pid, 0)
            logger.info(f"已有实例在跑 (PID {old_pid})，退出")
            return
        except (ValueError, PermissionError):
            pass
        except OSError:
            pass  # 旧进程不存在
    pid_file.write_text(str(os.getpid()))
    import atexit
    atexit.register(lambda: pid_file.unlink() if pid_file.exists() else None)

    # 非交易日不启动（cron 按工作日触发，节假日在此兜底）
    from plays.limit_up.utils import _is_trade_day, _today_str
    if not _is_trade_day(_today_str()):
        logger.info("非交易日，退出")
        return

    engine = get_engine()
    engine.start()

    logger.info("=" * 50)
    logger.info("盯盘引擎已启动！")
    logger.info("状态文件: %s", STATE_FILE)
    logger.info("扫描间隔: %ds", SCAN_INTERVAL)
    logger.info("盯盘上限: %d 只", MAX_WATCH)
    logger.info("=" * 50)
    logger.info("通过飞书发送指令: 盯/停/盯盘列表/清盯盘")

    try:
        while True:
            time.sleep(60)
            # 15:05 自动退出（EOD 汰换在 15:00 收盘轮已执行，留 5 分钟余量）
            if datetime.now().hour >= 15 and datetime.now().minute >= 5:
                logger.info("收盘退出（15:05），明日由 cron 重新拉起")
                engine.stop()
                return
            with engine._lock:
                count = len(engine._states)
            logger.debug("心跳: 引擎%s, 盯盘%d只", "运行中" if engine._running else "已停止", count)
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭...")
        engine.stop()
        logger.info("盯盘引擎已关闭")


if __name__ == "__main__":
    main()
