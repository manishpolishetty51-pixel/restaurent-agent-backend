
import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langsmith import traceable
from pydantic import BaseModel, Field

from tools import tools
import flow
from jev_decision import safe_interpret_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

Do not invent restaurant information.

For reservations, never guess the date or party size. If either is
missing, ask the user for it before calling the availability tool.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""

model = init_chat_model(
    os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

app = FastAPI(title="Restaurant Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)
    state: dict | None = None  # echo back the `state` from the last response


class ChatResponse(BaseModel):
    google: dict  # Gemini's answer: source, reply, tool_calls
    jev: dict     # Jev: source, available, understood, next_question
    state: dict   # send this back with the next request


def _text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content) if content is not None else ""


def _google_payload(result) -> dict:
    out = result["messages"]

    # Arguments the model passed to each tool, keyed by tool-call id.
    call_args = {
        c["id"]: c.get("args", {})
        for m in out
        for c in (getattr(m, "tool_calls", None) or [])
    }

    tool_calls = [
        {
            "name": m.name,
            "args": call_args.get(getattr(m, "tool_call_id", None), {}),
            "output": _text(m.content),
        }
        for m in out
        if getattr(m, "type", None) == "tool"
    ]

    reply = _text(out[-1].content) if out else ""
    return {"source": "Google Gemini", "reply": reply, "tool_calls": tool_calls}


@traceable(name="Gemini Restaurant Agent", run_type="chain")
async def invoke_gemini_agent(messages: list[dict]):
    return await agent.ainvoke({"messages": messages})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history]
    state = req.state or dict(flow.START)

    # Jev reads the user's answer to the question it asked last turn.
    asked = flow.question_for(state)
    jev_result = await safe_interpret_answer(asked, req.message, history)
    if jev_result["available"]:
        method = "jev"
        choice = jev_result["choice"]
    else:
        method = "fallback"
        choice = flow.match_label(asked, req.message)

    advance = flow.step(state, choice)

    # Google answers the resolved step (or the user's own words if unresolved).
    messages = history + [
        {"role": "user", "content": advance["gemini_message"] or req.message}
    ]

    try:
        result = await invoke_gemini_agent(messages)
    except Exception:
        logger.exception("Gemini agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    google = _google_payload(result)

    new_state = advance["state"]
    next_question = flow.question_for(new_state)
    if advance["notice"]:
        next_question["text"] = f"{advance['notice']} {next_question['text']}"

    return ChatResponse(
        google=google,
        jev={
            "source": "Jev",
            "available": jev_result["available"],
            "understood": {
                "question_id": asked["id"],
                "choice": choice,
                "confidence": jev_result["confidence"],
                "method": method,
            },
            "next_question": next_question,
        },
        state=new_state,
    )


def _line(event: dict) -> str:
    return json.dumps(event) + chr(10)


def _jev_payload(asked, jev_result, choice, method, new_state, notice) -> dict:
    next_question = flow.question_for(new_state)
    if notice:
        next_question["text"] = f"{notice} {next_question['text']}"
    return {
        "source": "Jev",
        "available": jev_result["available"],
        "understood": {
            "question_id": asked["id"],
            "choice": choice,
            "confidence": jev_result["confidence"],
            "method": method,
        },
        "next_question": next_question,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Same as /chat, but sends Jev's and Gemini's answers as separate
    newline-delimited JSON events, each as soon as it is ready.

    When the message is exactly one of the offered choices (a button click),
    Gemini starts immediately, in parallel with Jev; otherwise it waits for
    Jev to work out what the user meant."""
    history = [m.model_dump() for m in req.history]
    state = req.state or dict(flow.START)
    asked = flow.question_for(state)

    async def run_jev(choice_hint):
        jev_result = await safe_interpret_answer(asked, req.message, history)
        if jev_result["available"]:
            method, choice = "jev", jev_result["choice"]
        else:
            method, choice = "fallback", flow.match_label(asked, req.message)
        # The flow follows the same choice Gemini was started with.
        used = choice_hint if choice_hint is not None else choice
        advance = flow.step(state, used)
        return {"type": "jev", "advance": advance, "payload": _jev_payload(
            asked, jev_result, choice, method, advance["state"], advance["notice"])}

    async def run_gemini(advance):
        messages = history + [
            {"role": "user", "content": advance["gemini_message"] or req.message}
        ]
        try:
            result = await invoke_gemini_agent(messages)
        except Exception:
            logger.exception("Gemini agent invocation failed")
            return {"type": "error", "detail": "The restaurant assistant could not process your request."}
        return {"type": "google", "payload": _google_payload(result)}

    async def events():
        hint = flow.match_label(asked, req.message)
        jev_task = asyncio.create_task(run_jev(hint))
        pending = {jev_task}
        if hint is not None:
            pending.add(asyncio.create_task(run_gemini(flow.step(state, hint))))
        gemini_started = hint is not None

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                ev = t.result()
                if ev["type"] == "jev":
                    if not gemini_started:
                        gemini_started = True
                        pending.add(asyncio.create_task(run_gemini(ev["advance"])))
                    yield _line({"type": "jev", "jev": ev["payload"], "state": ev["advance"]["state"]})
                elif ev["type"] == "google":
                    yield _line({"type": "google", "google": ev["payload"]})
                else:
                    yield _line(ev)

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    return {"status": "ok"}
