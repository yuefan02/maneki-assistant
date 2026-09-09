#!/usr/bin/env python3
"""jvQuant CTP 券商交易客户端封装

用法:
    from scripts.jvquant_trade_client import get_trade_client
    client = get_trade_client()
    client.buy("600519", "贵州茅台")      # 自动从10档算最优价
    client.buy("600519", "贵州茅台", 1572.12, 100)  # 指定价

安全策略:
    - 下单前自动检查可用资金是否够、是否已持仓
    - 登录失效自动重登录重试
    - 密码从 ~/.ctp_pwd 读取（chmod 600）:
        echo "CTP_PWD=你的密码" > ~/.ctp_pwd
        chmod 600 ~/.ctp_pwd

    .env 只存账号:
        CTP_ACC=资金账号

首次设置:
    python3 scripts/jvquant_trade_client.py setup
    → 读取 ~/.ctp_pwd 登录柜台 → 生成 ctp_{账号}_ticket.json 缓存
    → 后续运行直接用 ticket，auto_relogin 自动刷新

下单价格策略（自动从 ws_snap 10档盘口取）:
    buy:  卖一量够→卖一价(立即成交); 不够→卖二价(吃透两层)
    sale: 买一量够→买一价; 不够→买二价
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

env_file = PROJECT_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file, override=False)

# ── 模拟交易开关（PAPER_TRADING=1 时不发真单，假设成交）──
# 用户 2026-09-09 拍板：暂停真实交易转模拟，系统不再碰真实 CTP 账户。
# 开启后 buy/sale/cancel/check_order/check_hold 全部返回模拟数据，
# watchdog 的成交确认链（code=0 → check_order 已成）完整走通但不发真单。
PAPER_TRADING = os.getenv("PAPER_TRADING", "0") == "1"


class _PaperClient:
    """模拟交易客户端：所有下单假设成交，绝不连 CTP 发真单。

    维护一个进程内委托列表供 check_order 返回（status 恒"已成"），
    check_hold 返回虚拟资金 + 空持仓（watchdog 的 state.json 才是
    模拟持仓账本，这里只保证"不拦截 + 不真下单"）。
    """

    def __init__(self):
        self._orders: list[dict] = []
        self._seq = 0

    def _next_oid(self) -> str:
        self._seq += 1
        return f"PAPER{int(time.time())}{self._seq}"

    def _record(self, code, otype, price, vol) -> str:
        oid = self._next_oid()
        short = (code or "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        self._orders.append({
            "order_id": oid,
            "code": short,
            "type": otype,
            "status": "已成",
            "deal_volume": str(vol or 100),
            "order_price": str(price or 0),
        })
        return oid

    def buy(self, code=None, name=None, price=None, vol=None, **kw) -> dict:
        oid = self._record(code, "证券买入", price, vol or 100)
        return {"code": "0", "order_id": oid,
                "message": f"模拟买入成交 {code} {name}"}

    def sale(self, code=None, name=None, price=None, vol=None, **kw) -> dict:
        oid = self._record(code, "证券卖出", price, vol or 100)
        return {"code": "0", "order_id": oid,
                "message": f"模拟卖出成交 {code} {name}"}

    def cancel(self, order_id=None, **kw) -> dict:
        return {"code": "0", "message": "模拟撤单成功"}

    def check_order(self, **kw) -> dict:
        return {"list": list(self._orders)}

    def check_hold(self, **kw) -> dict:
        # 虚拟资金设大额，避免 watchdog 三重风控的 MIN_FREE_CASH 拦截；
        # hold_list 空 → _reconcile_holds 不动作（模拟持仓由 state.json 管理）
        return {"total": "1000000.00", "usable": "1000000.00",
                "hold_list": [], "day_earn": "0", "hold_earn": "0"}

    def login(self, **kw) -> dict:
        return {"code": "0"}


_paper_client = None


def _get_paper_client():
    global _paper_client
    if _paper_client is None:
        _paper_client = _PaperClient()
    return _paper_client


# ── 密码安全读取 ──

_CTP_PWD_FILE = Path.home() / ".ctp_pwd"


def _get_password() -> str:
    pwd = os.environ.get("CTP_PWD", "")
    if pwd:
        return pwd
    if _CTP_PWD_FILE.exists():
        try:
            for line in _CTP_PWD_FILE.read_text().strip().split("\n"):
                line = line.strip()
                if line.startswith("CTP_PWD="):
                    return line.split("=", 1)[1].strip().strip("\"'")
        except Exception:
            pass
    return ""


# ── 持仓缓存（防频繁 API 调用）──

_HOLD_CACHE: dict = {}          # 最后查询的 hold 结果
_HOLD_CACHE_TS: float = 0.0    # 缓存更新时间
_HOLD_CACHE_TTL = 5.0           # 缓存有效期 5s


def _get_hold(client) -> dict:
    """带缓存的 check_hold，5s 内不重复请求"""
    global _HOLD_CACHE, _HOLD_CACHE_TS
    now = time.time()
    if now - _HOLD_CACHE_TS < _HOLD_CACHE_TTL and _HOLD_CACHE:
        return _HOLD_CACHE
    try:
        _HOLD_CACHE = client.check_hold()
        _HOLD_CACHE_TS = now
    except Exception as e:
        logger.warning("查询持仓异常: %s", e)
        _HOLD_CACHE = {}
    return _HOLD_CACHE


def _invalidate_hold_cache():
    """下单后主动失效缓存（下次查询强制刷新）"""
    global _HOLD_CACHE_TS
    _HOLD_CACHE_TS = 0.0


# ── 全局单例 ──

_client = None


def get_trade_client(log_level=logging.WARNING):
    """获取 CTP 交易客户端（单例，auto_relogin=True）"""
    global _client
    if PAPER_TRADING:
        return _get_paper_client()
    if _client is not None:
        return _client

    import jvQuant.ctp_client as ctp

    token = os.getenv("JVQUANT_TOKEN", "")
    acc = os.getenv("CTP_ACC", "")
    pwd = _get_password()

    if not token:
        raise ValueError("JVQUANT_TOKEN 未配置")
    if not acc:
        raise ValueError("CTP_ACC 未配置，请在 .env 中设置 CTP_ACC=你的资金账号")
    if not pwd:
        raise ValueError(
            "CTP_PWD 未找到。请执行:\n"
            f'  echo "CTP_PWD=你的密码" > {_CTP_PWD_FILE}\n'
            f"  chmod 600 {_CTP_PWD_FILE}\n"
            "然后重试。"
        )

    logging.getLogger("ctp_client").setLevel(logging.WARNING)  # 防密码泄漏到日志
    _client = ctp.Construct(token=token, ctp_acc=acc, ctp_pwd=pwd,
                            log_level=log_level, auto_relogin=True)
    return _client


# ── 盘口定价（10档 → 最优下单价格）──

_SNAP_FILE = Path("/dev/shm/ws_snap.json")


def _short(code: str) -> str:
    return code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")


def _snap_of(code: str) -> dict:
    try:
        snap = json.loads(_SNAP_FILE.read_text())
        return snap.get(_short(code), {})
    except Exception:
        return {}


def best_buy_price(code: str, vol: int = 100) -> tuple[float, str]:
    """10档盘口 → 最优买入价（追涨场景，优先成交）"""
    d = _snap_of(code)
    ask_p = d.get("ask_price", [])
    ask_q = d.get("ask_qty", [])
    last = float(d.get("last") or 0)

    if ask_p and ask_q and len(ask_p) >= 2:
        a1 = float(ask_p[0]) if str(ask_p[0]).strip() else 0
        a1q = float(ask_q[0]) if str(ask_q[0]).strip() else 0
        a2 = float(ask_p[1]) if str(ask_p[1]).strip() else 0

        if a1 > 0 and a1q >= vol:
            return a1, f"卖一{a1}量{a1q:.0f}"
        if a1 > 0 and a2 > 0:
            return a2, f"卖一量{a1q:.0f}→卖二{a2}"
    if last > 0:
        return last, f"最新价{last}"
    return 0.0, "无数据"


def best_sell_price(code: str, vol: int = 100) -> tuple[float, str]:
    """10档盘口 → 最优卖出价（止损/止盈场景，优先成交）"""
    d = _snap_of(code)
    bid_p = d.get("bid_price", [])
    bid_q = d.get("bid_qty", [])
    last = float(d.get("last") or 0)

    if bid_p and bid_q and len(bid_p) >= 2:
        b1 = float(bid_p[0]) if str(bid_p[0]).strip() else 0
        b1q = float(bid_q[0]) if str(bid_q[0]).strip() else 0
        b2 = float(bid_p[1]) if str(bid_p[1]).strip() else 0

        if b1 > 0 and b1q >= vol:
            return b1, f"买一{b1}量{b1q:.0f}"
        if b1 > 0 and b2 > 0:
            return b2, f"买一量{b1q:.0f}→买二{b2}"
    if last > 0:
        return last, f"最新价{last}"
    return 0.0, "无数据"


# ── 交易接口（含余额/持仓检查 + 自动重登录）──

def buy(code: str, name: str, price: float | str = None, vol: int | str = 100) -> dict:
    """买入证券，自动检查可用资金和已持仓情况。

    price=None 时从10档算最优价。
    登录失效自动调 login() 重试一次。
    """
    if PAPER_TRADING:
        return _get_paper_client().buy(code=code, name=name, price=price, vol=vol)
    vol_i = int(vol)
    # 自动定价
    if price is None:
        price_f, _reason = best_buy_price(code, vol_i)
        print(f"  [📡] 买入 {code} {name} {vol_i}股 最优价={price_f} ({_reason})")
    else:
        price_f = float(price)

    if price_f <= 0:
        return {"code": "-1", "message": f"无法获取{code}有效价格"}

    # 2026-08-03 修复：涨停一字板检查——ask1 为空（无卖盘）+ 涨幅≥9.9% = 封死涨停，
    # 此时按涨停价挂单只会排队买不进、冻结资金（08-03 实盘 4 笔挂单未成交即此因：
    # 605488/002421/600468/603337 全是涨停封板，资金被挂单冻结到可用仅 2.3 元）。
    # 2026-08-04 修正：ws_snap 无 pct_chg 字段（L1 只有价格+盘口），原判断 _pct>=9.9
    # 永远不成立 → 涨停票没拦住。改用盘口特征：ask1 空（无卖盘）+ bid1 巨额封单
    # （买一量 ≥ 500万手级 或 ≥ 卖一量的 50 倍）= 涨停封板。
    _snap = _snap_of(code)
    try:
        _ask_p = _snap.get("ask_price") or []
        _bid_p = _snap.get("bid_price") or []
        _ask_q = _snap.get("ask_qty") or []
        _bid_q = _snap.get("bid_qty") or []
        _ask1 = float(_ask_p[0]) if _ask_p and str(_ask_p[0]).strip() else 0
        _bid1q = float(_bid_q[0]) if _bid_q and str(_bid_q[0]).strip() else 0
        _ask1q = float(_ask_q[0]) if _ask_q and str(_ask_q[0]).strip() else 0
    except Exception:
        _ask1, _bid1q, _ask1q = 0.0, 0.0, 0.0
    # 涨停封板特征：卖一档无挂单（ask1 空）+ 买一档巨额堆积（封单 ≥ 100万股 或 ≥ 卖一量 50 倍）
    _limit_up = (_ask1 <= 0) and (_bid1q >= 1000000 or (_ask1q > 0 and _bid1q >= _ask1q * 50))
    if _limit_up:
        return {"code": "-6", "message": f"{code}({name}) 涨停封板(ask1空 封单{_bid1q:.0f}股)，挂单买不进，跳过"}

    client = get_trade_client()

    # ── 前置检查：可用资金 + 是否已持仓 ──
    hold = _get_hold(client)
    hold_code = _short(code)

    # 是否已持仓
    if hold.get("hold_list"):
        for h in hold["hold_list"]:
            if h.get("code") == hold_code:
                return {"code": "-2", "message": f"已持仓 {hold_code}({name})，禁止重复买入"}

    # 可用资金是否够
    usable = float(hold.get("usable", "0"))
    estimated_cost = price_f * vol_i
    if usable < estimated_cost:
        return {"code": "-3", "message": f"可用资金不足: 需{estimated_cost:.0f} 可用{usable:.0f}"}

    # ── 发单 + 失效缓存 ──
    try:
        r = client.buy(code=code, name=name, price=str(price_f), vol=str(vol_i))
    except Exception as e:
        logger.warning("买入异常(尝试重登录): %s", e)
        # 重登录重试
        try:
            client.login()
            time.sleep(0.5)
            r = client.buy(code=code, name=name, price=str(price_f), vol=str(vol_i))
        except Exception as e2:
            logger.error("重登录后买入仍失败: %s", e2)
            return {"code": "-4", "message": f"买入失败(重登录后): {e2}"}

    # 下单失败也尝试重登录
    if not r or r.get("code", "") != "0":
        logger.warning("买入返回异常(code=%s), 尝试重登录重试", r.get("code", "?") if r else "空")
        try:
            client.login()
            time.sleep(0.5)
            r = client.buy(code=code, name=name, price=str(price_f), vol=str(vol_i))
        except Exception as e2:
            logger.error("重登录后买入仍异常: %s", e2)

    _invalidate_hold_cache()
    return r


def sale(code: str, name: str, price: float | str = None, vol: int | str = 100) -> dict:
    """卖出证券，自动检查持仓数量。

    price=None 时从10档算最优价。
    登录失效自动调 login() 重试一次。
    """
    if PAPER_TRADING:
        return _get_paper_client().sale(code=code, name=name, price=price, vol=vol)
    vol_i = int(vol)
    # 自动定价
    if price is None:
        price_f, _reason = best_sell_price(code, vol_i)
        print(f"  [📡] 卖出 {code} {name} {vol_i}股 最优价={price_f} ({_reason})")
    else:
        price_f = float(price)

    if price_f <= 0:
        return {"code": "-1", "message": f"无法获取{code}有效价格"}

    client = get_trade_client()

    # ── 前置检查：确认持仓数量 ──
    hold = _get_hold(client)
    hold_code = _short(code)
    hold_vol = 0
    usable_vol = 0
    if hold.get("hold_list"):
        for h in hold["hold_list"]:
            if h.get("code") == hold_code:
                hold_vol = int(h.get("hold_vol", "0"))
                usable_vol = int(h.get("usable_vol", "0"))
                break

    if hold_vol < vol_i:
        return {"code": "-2", "message": f"持仓不足: {hold_code}({name}) 可用{usable_vol}股，需{vol_i}股"}

    # T+1：总持仓够但可用为 0（当日买入不可卖），区别于"无持仓"
    if usable_vol < vol_i:
        return {"code": "-5", "message": f"T+1当日不可卖: {hold_code}({name}) 持仓{hold_vol}股 可用{usable_vol}股，需{vol_i}股"}

    # ── 发单 + 失效缓存 ──
    try:
        r = client.sale(code=code, name=name, price=str(price_f), vol=str(vol_i))
    except Exception as e:
        logger.warning("卖出异常(尝试重登录): %s", e)
        try:
            client.login()
            time.sleep(0.5)
            r = client.sale(code=code, name=name, price=str(price_f), vol=str(vol_i))
        except Exception as e2:
            logger.error("重登录后卖出仍失败: %s", e2)
            return {"code": "-4", "message": f"卖出失败(重登录后): {e2}"}

    # 下单失败也尝试重登录
    if not r or r.get("code", "") != "0":
        logger.warning("卖出返回异常(code=%s), 尝试重登录重试", r.get("code", "?") if r else "空")
        try:
            client.login()
            time.sleep(0.5)
            r = client.sale(code=code, name=name, price=str(price_f), vol=str(vol_i))
        except Exception as e2:
            logger.error("重登录后卖出仍异常: %s", e2)

    _invalidate_hold_cache()
    return r


def cancel(order_id: str) -> dict:
    if PAPER_TRADING:
        return _get_paper_client().cancel(order_id=order_id)
    client = get_trade_client()
    return client.cancel(order_id=order_id)


def check_order() -> dict:
    if PAPER_TRADING:
        return _get_paper_client().check_order()
    client = get_trade_client()
    return client.check_order()


def check_hold() -> dict:
    if PAPER_TRADING:
        return _get_paper_client().check_hold()
    client = get_trade_client()
    r = _get_hold(client)
    return r


# ── CLI ──

def cli():
    import argparse

    parser = argparse.ArgumentParser(description="jvQuant CTP 交易客户端")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("setup", help="首次登录生成 ticket 缓存")
    bp = sub.add_parser("buy", help="买入（不传 price 自动定价）")
    bp.add_argument("code")
    bp.add_argument("name")
    bp.add_argument("price", nargs="?", type=float, default=None, help="不传则自动定价")
    bp.add_argument("vol", nargs="?", type=int, default=100)
    sp = sub.add_parser("sale", help="卖出（不传 price 自动定价）")
    sp.add_argument("code")
    sp.add_argument("name")
    sp.add_argument("price", nargs="?", type=float, default=None, help="不传则自动定价")
    sp.add_argument("vol", nargs="?", type=int, default=100)
    cp = sub.add_parser("cancel", help="撤单")
    cp.add_argument("order_id")
    sub.add_parser("check_order", help="查询委托")
    sub.add_parser("check_hold", help="查询持仓")
    sub.add_parser("balance", help="查询可用资金")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        if args.cmd == "setup":
            pwd = _get_password()
            if not pwd:
                print(f"错误: ~/.ctp_pwd 未配置。请先执行:\n"
                      f'  echo "CTP_PWD=你的密码" > ~/.ctp_pwd\n'
                      f"  chmod 600 ~/.ctp_pwd", file=sys.stderr)
                return
            import jvQuant.ctp_client as ctp
            token = os.getenv("JVQUANT_TOKEN", "")
            acc = os.getenv("CTP_ACC", "")
            client = ctp.Construct(token=token, ctp_acc=acc, ctp_pwd=pwd,
                                   auto_relogin=False)
            fpath = Path.cwd() / f"ctp_{acc}_ticket.json"
            print(f"✅ ticket 已生成: {fpath}")
            return

        if args.cmd == "buy":
            r = buy(args.code, args.name, args.price, args.vol)
            print(f"买入结果: {r}")
        elif args.cmd == "sale":
            r = sale(args.code, args.name, args.price, args.vol)
            print(f"卖出结果: {r}")
        elif args.cmd == "cancel":
            r = cancel(args.order_id)
            print(f"撤单结果: {r}")
        elif args.cmd == "check_order":
            r = check_order()
            print(f"委托: {r}")
        elif args.cmd == "check_hold":
            r = check_hold()
            print(f"持仓: {r}")
        elif args.cmd == "balance":
            r = check_hold()
            print(f"总资产: {r.get('total', '?')}")
            print(f"可用资金: {r.get('usable', '?')}")
            print(f"当日盈亏: {r.get('day_earn', '?')}")
            print(f"持仓盈亏: {r.get('hold_earn', '?')}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)


if __name__ == "__main__":
    cli()
