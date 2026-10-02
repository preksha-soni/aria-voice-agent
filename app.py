"""Aria - AI voice CX agent for Aura Skincare. FastAPI + OpenAI-compatible tool calling (Groq free tier), streamed."""
import json, os, re, uuid
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or "missing",
                base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")  # free + fast on Groq
# gpt-oss models "think" first; keep that short so replies start fast
EXTRA = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}
app = FastAPI(title="Aria - Aura Skincare Voice Agent")
BASE = Path(__file__).parent

# ---------------- Mock order database ----------------
ORDERS = {
    "ORD-101": dict(customer="Priya Sharma", product="Vitamin C Serum (30ml)", value=699,
                    status="Out for Delivery", courier="BlueDart", tracking="BD-982103",
                    notes="Expected by 6 PM today"),
    "ORD-102": dict(customer="Rahul Verma", product="Hydrating Sunscreen SPF 50", value=499,
                    status="Delivered", courier="Delhivery", tracking="DL-441029",
                    notes="Delivered 14 days ago", delivered_days_ago=14),
    "ORD-103": dict(customer="Ananya Patel", product="Green Tea Face Wash + Toner", value=850,
                    status="Processing", notes="Ordered 3 hours ago"),
}


def normalize_id(raw: str) -> str | None:
    m = re.search(r"(\d{3,})", raw or "")
    return f"ORD-{m.group(1)}" if m else None


def get_order_details(order_id: str = "", cancelled=()) -> dict:
    """Deterministic policy flags are computed HERE in code, so the LLM never has to guess."""
    oid = normalize_id(order_id)
    if not oid:
        return {"found": False, "error": "No valid order ID provided. Ask the customer for it."}
    o = ORDERS.get(oid)
    if not o:
        return {"found": False, "order_id": oid,
                "error": "No order with this ID exists. Ask the customer to repeat or verify."}
    if oid in cancelled:  # cancelled earlier in THIS call only (mock DB itself never changes)
        o = {**o, "status": "Cancelled", "notes": "Cancelled during this call"}
    days = o.get("delivered_days_ago")
    return {"found": True, "order_id": oid, **o,
            "cancellable": o["status"] == "Processing",
            "return_window_open": days is not None and days <= 7,  # still must be unopened/unused
            "damage_report_window_open": days is not None and days <= 2,
            "shipping_fee": 0 if o["value"] > 499 else 50}


def cancel_order(order_id: str = "", cancelled=None) -> dict:
    """Mock cancellation. Eligibility is enforced in code, never by the LLM."""
    cancelled = cancelled if cancelled is not None else set()
    d = get_order_details(order_id, cancelled)
    if not d["found"]:
        return d
    if not d["cancellable"]:
        return {"success": False, "order_id": d["order_id"], "status": d["status"],
                "reason": "Only orders in Processing status can be cancelled."}
    cancelled.add(d["order_id"])
    return {"success": True, "order_id": d["order_id"], "message": "Order cancelled."}


TOOLS = [{"type": "function", "function": {
    "name": "get_order_details",
    "description": "Look up a customer's order (status, product, value, tracking, cancellation and "
                   "return eligibility). Call whenever the customer asks about an order or wants to "
                   "cancel/return one and has given an order ID.",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string", "description": "e.g. ORD-101"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
    "name": "cancel_order",
    "description": "Cancel an order. Call ONLY after the customer has clearly confirmed they want this "
                   "specific order cancelled. The system checks eligibility itself.",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string", "description": "e.g. ORD-103"}}, "required": ["order_id"]}}}]

SYSTEM = """You are Aria, a friendly, professional, concise Indian customer support specialist for Aura Skincare, a premium organic Indian skincare brand. You are on a LIVE VOICE CALL.

STYLE: Reply in 1-2 short spoken sentences. No markdown, lists, emojis or symbols. Say rupees as "rupees". Warm, natural Indian English; if the customer speaks Hinglish, reply in simple Hinglish. Never repeat what was already said.

SPEECH-TO-TEXT MAY BE WRONG: if the message is garbled, cut off or makes no sense, politely ask them to repeat. When you mention an order, say it as "order one zero one" (digits one by one, never the letters ORD or a dash).

POLICIES (the only facts you may state; never invent anything else):
- Shipping: free above Rs 499; below Rs 499 a Rs 50 fee. Standard delivery 3-5 business days.
- Returns: within 7 days of delivery, only unopened, unused products in original packaging. Damaged/defective items: report within 48 hours of delivery with photos, for a replacement.
- Cancellation: only while status is Processing. Once Shipped or Out for Delivery it cannot be cancelled, but the customer may refuse delivery at the doorstep.
- COD: available up to Rs 2,500; pay by cash or UPI at the doorstep.

RULES:
- Never promise refunds, exceptions, discounts or timelines outside these policies. If a request falls outside policy, politely explain why and offer what IS possible.
- For order questions, ALWAYS call get_order_details; never guess order data. If no order ID is given, ask for it. If not found, say you couldn't locate it and ask them to verify. Trust the tool's cancellable / return_window_open flags. For returns, also ask whether the product is unopened and unused. If the customer gives a full set of digits (like "one zero one"), treat it as that order ID and look it up. Understand "double" and "triple" (triple nine means 999). If the digits are incomplete or fewer than three, do NOT guess; ask them to repeat the full number.
- You can cancel an eligible order with cancel_order, but ONLY after you name the order and the customer clearly says yes. Never say an order is cancelled unless the tool returned success. You cannot process refunds or returns; explain the policy instead. Never offer actions you cannot do. Do not upsell or suggest new purchases.
- If you lack information, say so honestly. Never hallucinate.
- Only help with Aura Skincare topics. Politely decline anything else (e.g. flights) and steer back.
- Answer only what was asked. Mention cancellation, returns or refunds only if the customer raises them.
- If the customer wants to return an order that is not delivered yet, say returns apply only after delivery, and offer cancellation if its status is Processing.
- Do not share details of an order beyond what the customer asks."""

GREETING = "Hi, this is Aria from Aura Skincare. How can I help you today?"
SESSIONS: dict[str, dict] = {}


class ChatIn(BaseModel):
    session_id: str
    message: str


class SessionIn(BaseModel):
    session_id: str


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/start")
def start():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"history": [{"role": "user", "content": "(call connected)"},
                                 {"role": "assistant", "content": GREETING}],
                     "transcript": [("Aria", GREETING)], "cancelled": set()}
    return {"session_id": sid, "greeting": GREETING}


def sse(obj) -> str:
    return json.dumps(obj) + "\n"


@app.post("/chat")
def chat(body: ChatIn):
    s = SESSIONS.get(body.session_id)
    if s is None:
        return {"error": "unknown session"}
    s["history"].append({"role": "user", "content": body.message})
    s["transcript"].append(("Customer", body.message))

    def gen():
        spoken = ""
        try:
            for _ in range(4):  # tool loop
                stream = client.chat.completions.create(
                    model=MODEL, max_tokens=800, tools=TOOLS, stream=True, extra_body=EXTRA,
                    messages=[{"role": "system", "content": SYSTEM}] + s["history"])
                text, calls = "", {}
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    d = chunk.choices[0].delta
                    if d.content:
                        text += d.content
                        spoken += d.content
                        yield sse({"type": "text", "d": d.content})
                    for tc in d.tool_calls or []:
                        c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        c["id"] = tc.id or c["id"]
                        if tc.function:
                            c["name"] = tc.function.name or c["name"]
                            c["args"] += tc.function.arguments or ""
                msg = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                        "name": c["name"], "arguments": c["args"] or "{}"}} for c in calls.values()]
                s["history"].append(msg)
                if not calls:
                    break
                for c in calls.values():
                    args = json.loads(c["args"] or "{}")
                    if c["name"] == "get_order_details":
                        out = get_order_details(args.get("order_id", ""), s["cancelled"])
                    elif c["name"] == "cancel_order":
                        out = cancel_order(args.get("order_id", ""), s["cancelled"])
                    else:
                        out = {}
                    yield sse({"type": "tool", "name": c["name"], "input": args, "output": out})
                    s["history"].append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
        except Exception as e:  # graceful degradation, never crash the call
            msg = "Sorry, I'm having a little trouble right now. Could you please say that again?"
            spoken += msg
            yield sse({"type": "text", "d": msg})
            yield sse({"type": "error", "d": str(e)[:300]})
            print("chat error:", e)
        finally:
            if spoken:
                s["transcript"].append(("Aria", spoken))
        yield sse({"type": "done"})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


SUMMARY_PROMPT = """Summarise this customer support call. Return ONLY a JSON object with keys:
customer_intent: one of ORDER_TRACKING, ORDER_CANCELLATION, RETURN_REFUND, DAMAGED_PRODUCT, SHIPPING_INFO, COD_INFO, PRODUCT_INFO, OUT_OF_SCOPE, OTHER
order_id: string like "ORD-101", or null
resolution_status: one of RESOLVED, PARTIALLY_RESOLVED, UNRESOLVED, OUT_OF_POLICY, NO_INTERACTION
call_summary: 1-2 factual sentences."""


@app.post("/summary")
def summary(body: SessionIn):
    s = SESSIONS.get(body.session_id)
    if s is None:
        return {"error": "unknown session"}
    transcript = [{"speaker": a, "text": t} for a, t in s["transcript"]]
    text = "\n".join(f"{x['speaker']}: {x['text']}" for x in transcript)
    try:
        r = client.chat.completions.create(model=MODEL, max_tokens=800, extra_body=EXTRA,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": text}])
        outcome = json.loads(r.choices[0].message.content)
    except Exception as e:
        outcome = {"customer_intent": "OTHER", "order_id": None,
                   "resolution_status": "UNRESOLVED", "call_summary": f"Summary failed: {e}"}
    return {"transcript": transcript, "outcome": outcome}


@app.get("/orders")
def orders():
    return ORDERS


@app.get("/health")
def health():
    """Open http://127.0.0.1:8000/health to see why the AI call fails."""
    try:
        r = client.chat.completions.create(model=MODEL, max_tokens=200, extra_body=EXTRA,
                                           messages=[{"role": "user", "content": "hi"}])
        return {"ok": True, "model": MODEL, "reply": r.choices[0].message.content}
    except Exception as e:
        return {"ok": False, "model": MODEL, "key_set": bool(os.getenv("LLM_API_KEY")),
                "error": str(e)[:400]}