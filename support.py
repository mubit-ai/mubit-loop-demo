"""Customer-support scenario for the chat UI.

Products are fictional, so the model cannot know their facts. Everything
the agent learns arrives through verifiable events, never from a hidden
answer script:

- helpdesk backend calls (lookup_account, refund, credit, escalate)
  with deterministic, policy-versioned verdicts
- tier-2 escalation notes (the scripted resolution a senior agent
  sends back — the agent decides when to escalate)
- scripted customer follow-ups whose triggers are deterministic facts
  (a refund that policy allows was not issued; a reply is missing the
  one true answer), not reactions to the agent's prose

Several ticket DATASETS are provided. Each is sequenced for a learning
curve: every learnable fact appears cold (the agent must escalate or
get denied) before it appears warm (recalled from Mubit and applied
first-touch), and the policy-change contradiction lands only after the
lesson it breaks has earned trust.

Lessons are stored in Mubit tagged [kb:*], [policy:*], [fix:*], one
entry per key. record_outcome closes the loop after every ticket; a
policy lesson contradicted by a verified backend event is retired and
replaced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from demo import llm_json, MODEL


# ---------------------------------------------------------------------------
# Dataset schema.
# ---------------------------------------------------------------------------

@dataclass
class Ticket:
    id: str
    kind: str                      # knowledge | fix | refund | compound | adhoc
    customer: str
    email: str
    opening: str
    verify_tokens: list[str] = field(default_factory=list)
    note_key: str = ""             # tier-2 note this ticket can unlock
    order_id: str = ""             # the order a refund ticket is about
    pushback: str = ""             # fires when a refund policy allows was not issued
    miss_reply: str = ""           # fires when the reply lacks the verify tokens
    confirm: str = ""              # customer close-out when resolved


@dataclass
class Dataset:
    id: str
    label: str
    product: str                   # one-line product description for the agent prompt
    windows: dict                  # {policy_version: {plan: days}}
    accounts: dict
    orders: dict
    tier2_notes: dict
    tickets: list
    policy_change_before: str
    policy_change_text: str


# ---------------------------------------------------------------------------
# Dataset A — Orbit, a project-analytics SaaS.
# Curve: T-01 kb cold (escalate) · T-02 policy cold (denied) · T-03 fix cold
# (escalate) · T-04 policy cold (accepted, annual) · T-05 fix warm ·
# T-06 kb warm · T-07 policy warm (declined via the learned window —
# reinforcement) · [policy change] T-08 contradiction · T-09 the replaced
# lesson wins first-touch · T-10 compound warm.
# ---------------------------------------------------------------------------

ORBIT = Dataset(
    id="orbit",
    label="Orbit — SaaS support",
    product="Orbit, a project-analytics SaaS",
    windows={1: {"monthly": 14, "annual": 30}, 2: {"monthly": 30, "annual": 30}},
    accounts={
        "priya@northwind.io": {"name": "Priya", "plan": "monthly", "role": "Admin", "orders": []},
        "sam@brightloop.co": {"name": "Sam", "plan": "monthly", "role": "Owner", "orders": ["ORD-7301"]},
        "dana@finchhq.com": {"name": "Dana", "plan": "monthly", "role": "Admin", "orders": []},
        "vera@atlasworks.com": {"name": "Vera", "plan": "annual", "role": "Owner", "orders": ["ORD-5512"]},
        "jonas@meridianlabs.de": {"name": "Jonas", "plan": "monthly", "role": "Member", "orders": []},
        "tom@caskandbarrel.uk": {"name": "Tom", "plan": "monthly", "role": "Owner", "orders": []},
        "leah@harborlight.co": {"name": "Leah", "plan": "monthly", "role": "Owner", "orders": ["ORD-7455"]},
        "noah@petalpos.com": {"name": "Noah", "plan": "monthly", "role": "Owner", "orders": ["ORD-8890"]},
        "omar@kestrelapps.com": {"name": "Omar", "plan": "monthly", "role": "Owner", "orders": ["ORD-8921"]},
        "ana@driftwood.studio": {"name": "Ana", "plan": "annual", "role": "Owner", "orders": ["ORD-9944"]},
    },
    orders={
        "ORD-7301": {"email": "sam@brightloop.co", "amount": 29.00, "age_days": 20, "plan": "monthly", "refunded": False},
        "ORD-5512": {"email": "vera@atlasworks.com", "amount": 290.00, "age_days": 21, "plan": "annual", "refunded": False},
        "ORD-7455": {"email": "leah@harborlight.co", "amount": 29.00, "age_days": 25, "plan": "monthly", "refunded": False},
        "ORD-8890": {"email": "noah@petalpos.com", "amount": 29.00, "age_days": 17, "plan": "monthly", "refunded": False},
        "ORD-8921": {"email": "omar@kestrelapps.com", "amount": 29.00, "age_days": 20, "plan": "monthly", "refunded": False},
        "ORD-9944": {"email": "ana@driftwood.studio", "amount": 290.00, "age_days": 6, "plan": "annual", "refunded": False},
    },
    tier2_notes={
        "kb:invoice-location": (
            "Invoices and VAT statements are under Billing > Statements. The Billing "
            "tab is visible to workspace Owners only — a non-Owner needs an Owner to "
            "download them or to grant the Owner role."
        ),
        "fix:error-1017": (
            "Error 1017 after a workspace rename: the old workspace slug stays in the "
            "local session cache. Fix: sign out, clear the app cache under Settings > "
            "Advanced > Clear cache, then sign in again using the new workspace URL."
        ),
    },
    tickets=[
        Ticket("T-01", "knowledge", "Priya", "priya@northwind.io",
               "Hi — finance is chasing me for VAT invoices for the last two months and I "
               "can't find them anywhere in the app. Where do I download them? "
               "— Priya (priya@northwind.io)",
               verify_tokens=["billing", "statements", "owner"],
               note_key="kb:invoice-location",
               miss_reply="I don't see anything like that on my screen. Can you check with "
                          "someone? Finance needs these by Friday.",
               confirm="Ah, that explains it — I'm not an Owner. I'll ask Rahul to pull them. Thanks!"),
        Ticket("T-02", "refund", "Sam", "sam@brightloop.co",
               "I meant to cancel after the trial and got charged on the 6th — that's about "
               "three weeks back now. Can you refund that charge? — sam@brightloop.co",
               order_id="ORD-7301",
               confirm="Okay, that's clear at least. Thanks for looking into it properly."),
        Ticket("T-03", "fix", "Dana", "dana@finchhq.com",
               "Since yesterday everyone on my team gets 'Error 1017: session invalid' when "
               "they try to log in. We renamed our workspace last week, if that matters. "
               "This is blocking our sprint reporting. — Dana (dana@finchhq.com)",
               verify_tokens=["advanced", "cache"],
               note_key="fix:error-1017",
               miss_reply="We tried that — everyone still hits error 1017. Renaming the "
                          "workspace is the only thing that changed on our side.",
               confirm="That did it — everyone's back in. Thank you!"),
        Ticket("T-04", "refund", "Vera", "vera@atlasworks.com",
               "Our team moved to another tool and auto-renew charged us for a full year "
               "three weeks ago. Can that renewal be refunded? — vera@atlasworks.com",
               order_id="ORD-5512",
               pushback="It was an accidental auto-renewal — can you at least check whether "
                        "annual plans are handled differently before saying no?",
               confirm="Refund received. Appreciate the quick turnaround."),
        Ticket("T-05", "fix", "Jonas", "jonas@meridianlabs.de",
               "hey — two of my colleagues get error 1017 at login since we changed our "
               "workspace name. any idea? jonas@meridianlabs.de",
               verify_tokens=["advanced", "cache"],
               note_key="fix:error-1017",
               miss_reply="still the same 1017 error for both of them.",
               confirm="works now, danke!"),
        Ticket("T-06", "knowledge", "Tom", "tom@caskandbarrel.uk",
               "Quick one: our auditors need last quarter's invoices and I can't find a "
               "billing section anywhere. — Tom (tom@caskandbarrel.uk)",
               verify_tokens=["billing", "statements"],
               note_key="kb:invoice-location",
               miss_reply="There's no menu with that name that I can see.",
               confirm="Got them, cheers."),
        Ticket("T-07", "refund", "Leah", "leah@harborlight.co",
               "Hi — I paused the project we bought Orbit for back in July. The last charge "
               "was about three and a half weeks ago; is that refundable? "
               "— leah@harborlight.co",
               order_id="ORD-7455",
               confirm="Ah well, worth asking. Thanks for the straight answer."),
        Ticket("T-08", "refund", "Noah", "noah@petalpos.com",
               "You charged me on the 9th and I stopped using Orbit the same week. I'd like "
               "that payment back please. — noah@petalpos.com",
               order_id="ORD-8890",
               pushback="Your pricing page literally says '30-day money-back guarantee' — I'm "
                        "looking at it right now. Please check again.",
               confirm="Refund confirmed on my end. Thanks."),
        Ticket("T-09", "refund", "Omar", "omar@kestrelapps.com",
               "We got charged on the 6th but only ever used the free features — could that "
               "payment be refunded? — omar@kestrelapps.com",
               order_id="ORD-8921",
               confirm="That was easy — refund received. Thanks!"),
        Ticket("T-10", "compound", "Ana", "ana@driftwood.studio",
               "We're consolidating tools at the studio. Two things: please refund last "
               "week's annual renewal, and where do I download our final invoices for "
               "bookkeeping? — Ana (ana@driftwood.studio)",
               verify_tokens=["billing", "statements"],
               note_key="kb:invoice-location",
               order_id="ORD-9944",
               pushback="And the refund for the renewal? That hasn't come through yet.",
               miss_reply="And the invoices? I still need those for our books.",
               confirm="Perfect — refund's in and I found the statements. That's everything."),
    ],
    policy_change_before="T-08",
    policy_change_text=("Orbit ships a spring policy update: 30-day money-back guarantee "
                        "on every plan. Support is not told."),
)


# ---------------------------------------------------------------------------
# Dataset B — Maple & Twine, an online home-goods store.
# Same curve shape, different domain and voices. Plans are membership
# tiers; "refund window" reads as the return window.
# ---------------------------------------------------------------------------

MAPLE = Dataset(
    id="maple",
    label="Maple & Twine — e-commerce",
    product="Maple & Twine, an online home-goods store",
    windows={1: {"standard": 14, "plus": 30}, 2: {"standard": 30, "plus": 30}},
    accounts={
        "rosa@casaverde.mx": {"name": "Rosa", "plan": "standard", "role": "Customer", "orders": ["MT-4410"]},
        "marcus@gmail.com": {"name": "Marcus", "plan": "standard", "role": "Customer", "orders": ["MT-4487"]},
        "keiko@plumfield.jp": {"name": "Keiko", "plan": "plus", "role": "Customer", "orders": ["MT-4433"]},
        "lena@outlook.com": {"name": "Lena", "plan": "standard", "role": "Customer", "orders": ["MT-4512"]},
        "dev@hearthandco.in": {"name": "Dev", "plan": "standard", "role": "Customer", "orders": ["MT-4530"]},
        "june@fernway.ca": {"name": "June", "plan": "standard", "role": "Customer", "orders": ["MT-4391"]},
        "hana@willowmere.co.nz": {"name": "Hana", "plan": "standard", "role": "Customer", "orders": ["MT-4602"]},
        "tomas@vinterhus.se": {"name": "Tomas", "plan": "standard", "role": "Customer", "orders": ["MT-4633"]},
        "oliver@yahoo.co.uk": {"name": "Oliver", "plan": "plus", "role": "Customer", "orders": ["MT-4550"]},
    },
    orders={
        "MT-4410": {"email": "rosa@casaverde.mx", "amount": 68.00, "age_days": 5, "plan": "standard", "refunded": False},
        "MT-4487": {"email": "marcus@gmail.com", "amount": 42.50, "age_days": 3, "plan": "standard", "refunded": False},
        "MT-4433": {"email": "keiko@plumfield.jp", "amount": 214.00, "age_days": 22, "plan": "plus", "refunded": False},
        "MT-4512": {"email": "lena@outlook.com", "amount": 31.00, "age_days": 2, "plan": "standard", "refunded": False},
        "MT-4530": {"email": "dev@hearthandco.in", "amount": 89.00, "age_days": 4, "plan": "standard", "refunded": False},
        "MT-4391": {"email": "june@fernway.ca", "amount": 57.25, "age_days": 20, "plan": "standard", "refunded": False},
        "MT-4602": {"email": "hana@willowmere.co.nz", "amount": 73.50, "age_days": 25, "plan": "standard", "refunded": False},
        "MT-4633": {"email": "tomas@vinterhus.se", "amount": 48.00, "age_days": 18, "plan": "standard", "refunded": False},
        "MT-4550": {"email": "oliver@yahoo.co.uk", "amount": 126.00, "age_days": 6, "plan": "plus", "refunded": False},
    },
    tier2_notes={
        "kb:return-label": (
            "Prepaid return labels: Orders > select the order > Return items. The "
            "button generates a QR code and PDF label, and it appears only after the "
            "carrier marks the order Delivered."
        ),
        "fix:missing-delivery": (
            "Marked-delivered-but-missing parcels: have the customer check the delivery "
            "photo under Orders > Tracking, wait 24 hours (carriers often scan early), "
            "then file a trace claim from Order details > Report an issue. A replacement "
            "ships as soon as the trace is open."
        ),
    },
    tickets=[
        Ticket("B-01", "knowledge", "Rosa", "rosa@casaverde.mx",
               "Hello! I want to send back the linen curtains from my last order — they're "
               "lovely but the colour is wrong for our walls. I can't find where to print a "
               "return label though? — Rosa (rosa@casaverde.mx)",
               verify_tokens=["return items", "delivered"],
               note_key="kb:return-label",
               miss_reply="I looked where you said and there's no such button on my order.",
               confirm="Found it — the Return items button was right there once I opened the "
                       "order. Label printed, gracias!"),
        Ticket("B-02", "refund", "June", "june@fernway.ca",
               "I bought a table runner almost three weeks ago and it's just been sitting in "
               "the closet — we redecorated. Any chance of a refund? — june@fernway.ca",
               order_id="MT-4391",
               confirm="Fair enough — I did leave it a while. Thanks for actually checking "
                       "the policy rather than guessing."),
        Ticket("B-03", "fix", "Marcus", "marcus@gmail.com",
               "Tracking says my order MT-4487 was DELIVERED yesterday but there's nothing "
               "on my porch, nothing at the neighbours. $42.50 down the drain?? "
               "— marcus@gmail.com",
               verify_tokens=["photo", "trace"],
               note_key="fix:missing-delivery",
               miss_reply="I've already walked around the whole building twice. What's the "
                          "actual process here?",
               confirm="Okay — the delivery photo shows a door that isn't mine, so I've filed "
                       "the claim like you said. Replacement on the way, thank you."),
        Ticket("B-04", "refund", "Keiko", "keiko@plumfield.jp",
               "The ceramic dinner set I ordered three weeks ago turned out to be the wrong "
               "glaze for our table. I would like to return it for a refund rather than an "
               "exchange, please. — Keiko (keiko@plumfield.jp)",
               order_id="MT-4433",
               pushback="I have been a Plus member for two years — could you check whether "
                        "Plus orders have a longer return window before refusing?",
               confirm="Refund confirmed. Thank you for handling it."),
        Ticket("B-05", "fix", "Lena", "lena@outlook.com",
               "my order says delivered but i never got it?? MT-4512. this was a birthday "
               "present. — lena@outlook.com",
               verify_tokens=["photo", "trace"],
               note_key="fix:missing-delivery",
               miss_reply="there's no parcel anywhere, i checked everywhere already",
               confirm="the photo shows it at my old address?! filing the claim now. thanks "
                       "for the quick help"),
        Ticket("B-06", "knowledge", "Dev", "dev@hearthandco.in",
               "Need to return the brass planters from order MT-4530 — wrong size for the "
               "shelf. How do I get a shipping label? — Dev (dev@hearthandco.in)",
               verify_tokens=["return items", "delivered"],
               note_key="kb:return-label",
               miss_reply="I don't see that option anywhere on the site.",
               confirm="Got the label, dropping it off tomorrow. Cheers."),
        Ticket("B-07", "refund", "Hana", "hana@willowmere.co.nz",
               "Kia ora — I ordered wool throws about three and a half weeks back and "
               "they're still unopened; we moved house and the colours don't work anymore. "
               "Can I return them for a refund? — hana@willowmere.co.nz",
               order_id="MT-4602",
               confirm="Fair enough — I did leave it too long. Thanks for checking."),
        Ticket("B-08", "refund", "June", "june@fernway.ca",
               "Me again, about my table runner. The store credit is still sitting unused — "
               "I'd honestly rather have the refund if there's any way at all. "
               "— june@fernway.ca",
               order_id="MT-4391",
               pushback="My sister returned curtains after three weeks just last night — the "
                        "banner on your homepage says 'Holiday returns: 30 days on "
                        "everything'. Please just try it.",
               confirm="There it is — refund received. Lovely, thank you!"),
        Ticket("B-09", "refund", "Tomas", "tomas@vinterhus.se",
               "Hej — I'd like to return the candlesticks from about two and a half weeks "
               "ago; they don't fit our table setting. Is a refund still possible? "
               "— tomas@vinterhus.se",
               order_id="MT-4633",
               confirm="Refund received. Tack!"),
        Ticket("B-10", "compound", "Oliver", "oliver@yahoo.co.uk",
               "Two things: the walnut bookends from last week arrived chipped so I'd like a "
               "refund, and I also need a return label for sending them back. "
               "— Oliver (oliver@yahoo.co.uk)",
               verify_tokens=["return items", "delivered"],
               note_key="kb:return-label",
               order_id="MT-4550",
               pushback="And the refund for the bookends? That part hasn't happened yet.",
               miss_reply="And how exactly do I print the label?",
               confirm="Refund's showing and the label printed. All sorted, thanks."),
    ],
    policy_change_before="B-08",
    policy_change_text=("Maple & Twine launches holiday returns: a 30-day window on every "
                        "order. Support is not told."),
)


# ---------------------------------------------------------------------------
# Dataset C — Orbit, day two. Same product as A, new customers and voices,
# with the two escalation arcs opening back-to-back so the curve is
# steeper: two cold failures first, then warm wins.
# ---------------------------------------------------------------------------

ORBIT2 = Dataset(
    id="orbit2",
    label="Orbit — day two",
    product="Orbit, a project-analytics SaaS",
    windows={1: {"monthly": 14, "annual": 30}, 2: {"monthly": 30, "annual": 30}},
    accounts={
        "mara@quillandgrain.com": {"name": "Mara", "plan": "monthly", "role": "Member", "orders": []},
        "yusuf@lanternworks.ae": {"name": "Yusuf", "plan": "monthly", "role": "Owner", "orders": ["ORD-9102"]},
        "chloe@statichaus.de": {"name": "Chloe", "plan": "monthly", "role": "Admin", "orders": []},
        "ravi@copperline.in": {"name": "Ravi", "plan": "annual", "role": "Owner", "orders": ["ORD-9155"]},
        "elin@fjordanalytics.no": {"name": "Elin", "plan": "monthly", "role": "Owner", "orders": []},
        "nadia@silverbirch.fi": {"name": "Nadia", "plan": "monthly", "role": "Owner", "orders": ["ORD-9210"]},
        "pat@brambleco.ie": {"name": "Pat", "plan": "monthly", "role": "Owner", "orders": ["ORD-9188"]},
        "bram@delftlogic.nl": {"name": "Bram", "plan": "monthly", "role": "Owner", "orders": ["ORD-9230"]},
        "sofia@tidepool.pt": {"name": "Sofia", "plan": "annual", "role": "Owner", "orders": ["ORD-9201"]},
    },
    orders={
        "ORD-9102": {"email": "yusuf@lanternworks.ae", "amount": 29.00, "age_days": 19, "plan": "monthly", "refunded": False},
        "ORD-9155": {"email": "ravi@copperline.in", "amount": 290.00, "age_days": 25, "plan": "annual", "refunded": False},
        "ORD-9210": {"email": "nadia@silverbirch.fi", "amount": 29.00, "age_days": 22, "plan": "monthly", "refunded": False},
        "ORD-9188": {"email": "pat@brambleco.ie", "amount": 29.00, "age_days": 17, "plan": "monthly", "refunded": False},
        "ORD-9230": {"email": "bram@delftlogic.nl", "amount": 29.00, "age_days": 21, "plan": "monthly", "refunded": False},
        "ORD-9201": {"email": "sofia@tidepool.pt", "amount": 290.00, "age_days": 4, "plan": "annual", "refunded": False},
    },
    tier2_notes={
        "kb:invoice-location": (
            "Invoices and VAT statements are under Billing > Statements. The Billing "
            "tab is visible to workspace Owners only — a non-Owner needs an Owner to "
            "download them or to grant the Owner role."
        ),
        "fix:error-1017": (
            "Error 1017 after a workspace rename: the old workspace slug stays in the "
            "local session cache. Fix: sign out, clear the app cache under Settings > "
            "Advanced > Clear cache, then sign in again using the new workspace URL."
        ),
    },
    tickets=[
        Ticket("C-01", "fix", "Mara", "mara@quillandgrain.com",
               "Our whole content team is locked out with 'Error 1017: session invalid'. "
               "The only recent change is that IT renamed our workspace on Monday. Please "
               "advise urgently. — Mara (mara@quillandgrain.com)",
               verify_tokens=["advanced", "cache"],
               note_key="fix:error-1017",
               miss_reply="Tried that. Everyone is still locked out with the same 1017 code.",
               confirm="That worked — the team is back in. Much appreciated."),
        Ticket("C-02", "knowledge", "Chloe", "chloe@statichaus.de",
               "where do i find invoices? tax season. — chloe@statichaus.de",
               verify_tokens=["billing", "statements", "owner"],
               note_key="kb:invoice-location",
               miss_reply="i see no billing anything in my sidebar",
               confirm="ah, need an owner. ok. thx"),
        Ticket("C-03", "refund", "Yusuf", "yusuf@lanternworks.ae",
               "Salaam — we were charged on the monthly plan just under three weeks ago but "
               "the team stopped using Orbit at the start of the month. Refund, please? "
               "— yusuf@lanternworks.ae",
               order_id="ORD-9102",
               confirm="Understood. Thank you for explaining it clearly."),
        Ticket("C-04", "fix", "Elin", "elin@fjordanalytics.no",
               "Hei — error 1017 for three users this morning. We merged two workspaces "
               "last week and renamed the surviving one. — elin@fjordanalytics.no",
               verify_tokens=["advanced", "cache"],
               note_key="fix:error-1017",
               miss_reply="Same error still, on all three machines.",
               confirm="All three are in. Takk!"),
        Ticket("C-05", "refund", "Ravi", "ravi@copperline.in",
               "Hi team — our annual renewal went through about three and a half weeks ago "
               "but we'd already decided to consolidate on another tool. Is a refund still "
               "possible this far out? — Ravi (ravi@copperline.in)",
               order_id="ORD-9155",
               pushback="Before you say no — could you check what the annual plan terms "
                        "actually allow? I'd rather not dispute the charge.",
               confirm="Refund received in full. That was painless, thank you."),
        Ticket("C-06", "knowledge", "Pat", "pat@brambleco.ie",
               "Accountant needs every invoice from this year and I've clicked every menu "
               "twice. Where are they hiding? — Pat (pat@brambleco.ie)",
               verify_tokens=["billing", "statements"],
               note_key="kb:invoice-location",
               miss_reply="Nothing by that name on my end.",
               confirm="Found them under Statements. Sound, thanks."),
        Ticket("C-07", "refund", "Nadia", "nadia@silverbirch.fi",
               "Moi — we were billed just over three weeks ago but the team has been on "
               "summer break since midsummer. Any chance of a refund? "
               "— nadia@silverbirch.fi",
               order_id="ORD-9210",
               confirm="Thought so, but I had to ask. Thanks for the quick answer."),
        Ticket("C-08", "refund", "Pat", "pat@brambleco.ie",
               "One more thing while I have you — that charge from just over two weeks ago: "
               "we're downsizing to the free tier, can I get it back? — pat@brambleco.ie",
               order_id="ORD-9188",
               pushback="A colleague swears the terms now say 30 days for everyone — new "
                        "policy this spring. Would you mind re-checking?",
               confirm="Refund's through. That's everything sorted, cheers."),
        Ticket("C-09", "refund", "Bram", "bram@delftlogic.nl",
               "Hoi — we were charged on the 5th but switched tools mid-month. Could that "
               "payment be refunded? — bram@delftlogic.nl",
               order_id="ORD-9230",
               confirm="Refund's in. Dank je wel!"),
        Ticket("C-10", "compound", "Sofia", "sofia@tidepool.pt",
               "Olá! Two things: please refund last week's annual charge — we picked the "
               "annual plan by mistake while updating our card — and point me to where I "
               "can download our past invoices. — Sofia (sofia@tidepool.pt)",
               verify_tokens=["billing", "statements"],
               note_key="kb:invoice-location",
               order_id="ORD-9201",
               pushback="And the annual charge — can that refund go through as well?",
               miss_reply="And the invoices? Where do I find those?",
               confirm="Refund confirmed and I can see the statements. Obrigada!"),
    ],
    policy_change_before="C-08",
    policy_change_text=("Orbit ships a spring policy update: 30-day money-back guarantee "
                        "on every plan. Support is not told."),
)


DATASETS = {d.id: d for d in (ORBIT, MAPLE, ORBIT2)}

# ---------------------------------------------------------------------------
# Active dataset + backend state.
# ---------------------------------------------------------------------------

CURRENT: Dataset = ORBIT
POLICY_VERSION = 1
CREDITS: list[dict] = []


def use_dataset(dataset_id: str) -> Dataset:
    global CURRENT
    CURRENT = DATASETS.get(dataset_id, ORBIT)
    reset_backend()
    return CURRENT


def set_policy(version: int) -> None:
    global POLICY_VERSION
    POLICY_VERSION = 2 if version == 2 else 1


def reset_backend() -> None:
    global POLICY_VERSION
    POLICY_VERSION = 1
    CREDITS.clear()
    for o in CURRENT.orders.values():
        o["refunded"] = False


def refund_window(plan: str) -> int:
    return CURRENT.windows[POLICY_VERSION][plan]


def would_refund(order_id: str) -> bool:
    o = CURRENT.orders.get(order_id)
    return bool(o) and not o["refunded"] and o["age_days"] <= refund_window(o["plan"])


def tool_call(tool: str, args: dict) -> dict:
    if tool == "lookup_account":
        email = (args.get("email") or "").lower().strip()
        acct = CURRENT.accounts.get(email)
        if not acct:
            return {"error": f"no account for {email}"}
        orders = [
            {"order_id": oid, "amount": CURRENT.orders[oid]["amount"],
             "age_days": CURRENT.orders[oid]["age_days"],
             "refunded": CURRENT.orders[oid]["refunded"]}
            for oid in acct["orders"]
        ]
        return {"email": email, "plan": acct["plan"], "role": acct["role"], "orders": orders}
    if tool == "refund":
        oid = (args.get("order_id") or "").upper().strip()
        o = CURRENT.orders.get(oid)
        if not o:
            return {"error": f"unknown order {oid}"}
        if o["refunded"]:
            return {"error": f"{oid} already refunded"}
        window = refund_window(o["plan"])
        if o["age_days"] > window:
            return {"denied": True, "reason": f"charge is {o['age_days']} days old; "
                    f"the {o['plan']} plan refund window is {window} days",
                    "plan": o["plan"], "age_days": o["age_days"], "window": window}
        o["refunded"] = True
        return {"ok": True, "refunded": o["amount"], "order_id": oid,
                "plan": o["plan"], "age_days": o["age_days"], "window": window}
    if tool == "credit":
        email = (args.get("email") or "").lower().strip()
        amount = float(args.get("amount") or 10)
        CREDITS.append({"email": email, "amount": amount})
        return {"ok": True, "credited": amount, "email": email}
    if tool == "escalate":
        return {"ok": True, "routed": "tier-2"}
    return {"error": f"unknown tool {tool}"}


# ---------------------------------------------------------------------------
# Mubit memory for this scenario.
# ---------------------------------------------------------------------------

TAG = re.compile(r"^\[([a-z]+:[a-z0-9_-]+)\]\s*(.*)$", re.S)
GATE = 0.5
QUERY = "support product knowledge refund policy fix procedure"


class SupportMemory:
    def __init__(self, client, run_id: str, emit) -> None:
        self.client = client
        self.run_id = run_id
        self.emit = emit  # emit(kind, text) -> a work line in the chat bubble
        client.set_run_id(run_id)

    def recall(self, quiet: bool = False) -> dict[str, dict]:
        out = self.client.recall(
            query=QUERY, limit=16, entry_types=["lesson"], evidence_only=True,
            mode="direct_bypass", prefer_current_run=True, include_working_memory=False,
        )
        lessons: dict[str, dict] = {}
        for e in out.get("evidence") or []:
            text = (e.get("content") or e.get("text") or "").strip()
            m = TAG.match(text)
            if not m:
                continue
            key, body = m.group(1), m.group(2).strip()
            try:
                meta = json.loads(e.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                meta = {}
            row = {
                "id": e.get("id"), "text": body,
                "confidence": float(meta.get("confidence", 0.5)),
                "fresh": "confidence" not in meta,
                "reinforcement": meta.get("reinforcement_count"),
                "stored_at": meta.get("ingested_at") or "",
                "window": meta.get("window"),
            }
            prev = lessons.get(key)
            if prev is None or row["stored_at"] > prev["stored_at"]:
                lessons[key] = row
        if not quiet:
            if lessons:
                names = "  ".join(f"[{k} {r['confidence']:.2f}]" for k, r in sorted(lessons.items()))
                self.emit("recall", f"recall -> {len(lessons)} stored lessons  {names}")
            else:
                self.emit("recall", "recall -> no stored lessons (cold start)")
        return lessons

    def store(self, key: str, text: str, extra_meta: dict | None = None) -> None:
        content = f"[{key}] {text}"[:800]
        self.emit("store", f'stored lesson [{key}] "{text[:110]}"')
        meta = {"key": key}
        if extra_meta:
            meta.update(extra_meta)
        self.client.remember(
            content=content, intent="lesson", lesson_type="success",
            lesson_scope="run", lesson_importance="high",
            upsert_key=key, metadata=meta, wait=True,
        )

    def outcome(self, key: str, lesson: dict, ok: bool, why: str) -> None:
        if not lesson.get("id"):
            return
        r = self.client.record_outcome(
            reference_id=lesson["id"], outcome="success" if ok else "failure",
            signal=0.9 if ok else -0.9, rationale=why,
        )
        conf = r.get("updated_confidence")
        if conf is not None:
            word = "reinforced" if ok else "contradicted"
            self.emit("outcome",
                      f"[{key}] {word}: confidence {lesson['confidence']:.2f} -> {conf:.2f}")

    def retire(self, key: str, lesson: dict) -> None:
        self.emit("retire", f"retired [{key}] — no longer matches a verified outcome")
        try:
            self.client.delete_lesson({"run_id": self.run_id, "lesson_id": lesson["id"]})
        except Exception as exc:
            self.emit("retire", f"retire failed ({exc})")

    def record_step(self, step_id: str, name: str, ok: bool, rationale: str) -> None:
        self.client.record_step_outcome(
            step_id=step_id, step_name=name,
            outcome="success" if ok else "failure",
            signal=0.9 if ok else 0.2, rationale=rationale, directive_hint=rationale,
        )


# ---------------------------------------------------------------------------
# The agent. One JSON-mode LLM call per turn.
# ---------------------------------------------------------------------------

def agent_system() -> str:
    return (
        f"You are the support agent for {CURRENT.product}. Reply in 2-4 warm, "
        "specific sentences. Hard rules: never invent product navigation, policy "
        "numbers, or troubleshooting steps — when the needed fact is not in your "
        "learned knowledge, tool results, or a tier-2 note, escalate instead of "
        "guessing. When your learned knowledge or a tier-2 note does contain the "
        "answer, give it to the customer directly — repeat exact menu paths and "
        "steps — and do not escalate. Take only the actions the customer asked "
        "for: a customer asking how to return an item or find a page wants "
        "instructions, not a refund.\n"
        "For refunds: look up the account first. If your learned knowledge gives "
        "the refund window for the customer's plan: when the charge is inside "
        "that window, process the refund with the refund tool; when it is "
        "outside, do not attempt the refund — offer one goodwill credit (10 "
        "percent of the charge, at most 30) and say why. If you do not know the "
        "window for that plan, attempt the refund: the billing system enforces "
        "policy and reports the reason when it declines. If the customer "
        "disputes your policy understanding with a concrete claim, call the "
        "refund tool to verify — the system is authoritative and its verdict "
        "overrides your learned knowledge.\n"
        "Tools: lookup_account(email), refund(order_id), credit(email, amount), "
        "escalate(summary). Never repeat a tool call whose result is already in "
        "the tool results. At most one goodwill credit per ticket.\n"
        'Return only JSON: {"reply": "<message to the customer>", '
        '"actions": [{"tool": "...", "args": {...}}], '
        '"resolution": "resolved" | "pending" | "escalate"}. Use "pending" when '
        'you need the results of your actions before you can answer; use '
        '"escalate" (or the escalate tool) to hand the ticket to tier 2.'
    )


def agent_turn(convo: list[dict], lessons: dict[str, dict], tool_log: list[dict],
               note: str | None, nudge: str | None = None) -> dict:
    system = agent_system()
    if lessons:
        lines = "\n".join(f"- [{k}] {r['text']}" for k, r in sorted(lessons.items()))
        system += f"\nLearned knowledge from earlier tickets (apply it):\n{lines}"
    parts = ["Conversation so far:"]
    for m in convo:
        parts.append(f"{m['who']}: {m['text']}")
    if tool_log:
        parts.append("Tool results this ticket: " + json.dumps(tool_log))
    if note:
        parts.append(f"Tier-2 note just arrived: {note}")
    if nudge:
        parts.append(f"Instruction from the support system: {nudge}")
    parts.append("Write your next turn.")
    out = llm_json(system, "\n".join(parts))
    if not isinstance(out, dict):
        out = {}
    out.setdefault("reply", "")
    out.setdefault("actions", [])
    out.setdefault("resolution", "pending")
    return out


DISTILL_SYSTEM = (
    "You keep the support team's lesson file. Write each lesson as one short line "
    "a support agent can apply on the next ticket. Keep exact menu paths, button "
    "names, error codes, plan or tier names, and every number verbatim from the "
    "evidence — the next agent repeats them to customers. Return JSON only."
)


def distill(events: list[dict]) -> dict[str, str]:
    """One LLM call: turn this ticket's verified events into per-key lessons."""
    want = {e["key"]: e["evidence"] for e in events}
    user = (
        "These events were verified on the ticket that just closed.\n"
        f"Evidence per lesson key: {json.dumps(want)}\n"
        'Write one lesson line per key. Return {"lessons": {"<key>": "<one line>"}}'
    )
    out = llm_json(DISTILL_SYSTEM, user)
    lessons = out.get("lessons") or {}
    result = {}
    for key, evidence in want.items():
        text = lessons.get(key)
        if not isinstance(text, str) or not text.strip():
            text = evidence
        result[key] = text.strip()
    return result


# ---------------------------------------------------------------------------
# Ticket orchestration.
# ---------------------------------------------------------------------------

MAX_TURNS = 6


def relevant_keys(ticket: Ticket) -> set[str]:
    """The lesson keys this ticket can fairly credit or blame. Other
    recalled lessons still ride in the prompt, but their confidence is
    not moved by a ticket they had no part in."""
    keys = set()
    if ticket.note_key:
        keys.add(ticket.note_key)
    if ticket.order_id and ticket.order_id in CURRENT.orders:
        keys.add(f"policy:refund-{CURRENT.orders[ticket.order_id]['plan']}")
    return keys


def run_ticket(ticket: Ticket, memory: SupportMemory, emit, stop=None) -> dict:
    """Plays one ticket. emit(evt: dict) streams UI events."""
    emit({"t": "ticket_start", "id": ticket.id, "title": ticket.opening[:60]})
    emit({"t": "customer", "name": ticket.customer, "text": ticket.opening})

    convo = [{"who": ticket.customer, "text": ticket.opening}]
    tool_log: list[dict] = []
    events: list[dict] = []        # verified learnings: {key, evidence, meta, contradicts}
    lessons = memory.recall()
    applied = {k: r for k, r in lessons.items() if r["confidence"] >= GATE}
    quarantined = sorted(set(lessons) - set(applied))
    if quarantined:
        memory.emit("recall", f"below the {GATE} gate, not applied: {', '.join(quarantined)}")
    relevant = relevant_keys(ticket)
    recalled_hits = sorted(set(applied) & relevant) if relevant else []
    if recalled_hits:
        emit({"t": "chips", "items": [{"kind": "recalled", "label": k} for k in recalled_hits]})

    note_sent = False
    pushback_sent = False
    miss_sent = False
    escalated = False
    final_reply = ""
    turns = 0
    replies = 0
    note: str | None = None

    def send_note() -> None:
        nonlocal note, note_sent, escalated, note_nudge
        escalated = True
        note_sent = True
        memory.emit("escalate", "escalated to tier 2")
        note = CURRENT.tier2_notes[ticket.note_key]
        # The note stays in the conversation so later turns keep the answer.
        convo.append({"who": "Tier-2", "text": note})
        note_nudge = ("Tier 2 has sent the resolution note above. Write the "
                      "reply to the customer now, using the note's exact "
                      "steps. Do not escalate again and do not call tools.")
        emit({"t": "note", "text": note})
        events.append({"key": ticket.note_key, "evidence": note})

    note_nudge: str | None = None
    dispute_nudge: str | None = None
    while turns < MAX_TURNS:
        if stop is not None and stop.is_set():
            break
        turns += 1
        memory.emit("llm", f"agent turn {turns} — {MODEL}")
        out = agent_turn(convo, applied, tool_log, note, dispute_nudge or note_nudge)
        note = None

        refund_result = None
        escalate_called = False
        for action in out.get("actions", [])[:4]:
            tool = str(action.get("tool") or "")
            args = action.get("args") or {}
            # Guardrails on repeat calls. A skipped call still lands in the
            # tool log so the model sees why nothing new came back. A refund
            # stays repeatable after a denial (a customer dispute legitimately
            # retries it) but not after a success.
            def skip(reason: str) -> None:
                memory.emit("tool", f"{tool} repeat skipped — {reason}")
                if not any(t["tool"] == tool and t["args"] == args
                           and "skipped" in t["result"] for t in tool_log):
                    tool_log.append({"tool": tool, "args": args,
                                     "result": {"skipped": reason}})

            if tool == "lookup_account" and any(
                    t["tool"] == tool and t["args"] == args
                    and "skipped" not in t["result"] for t in tool_log):
                skip("result already on the ticket; reuse it")
                continue
            if tool == "credit" and any(
                    t["tool"] == "credit" and "skipped" not in t["result"]
                    for t in tool_log):
                skip("one goodwill credit per ticket")
                continue
            if tool == "refund" and any(
                    t["tool"] == "refund" and t["result"].get("ok")
                    and t["args"].get("order_id") == args.get("order_id")
                    for t in tool_log):
                skip("this order is already refunded")
                continue
            if tool == "escalate" and note_sent:
                skip("the ticket is already with tier 2; answer from its note")
                continue
            result = tool_call(tool, args)
            tool_log.append({"tool": tool, "args": args, "result": result})
            memory.emit("tool", f"{tool}({json.dumps(args)}) -> {json.dumps(result)[:130]}")
            memory.record_step(f"{ticket.id}/t{turns}-{tool[:10]}", tool,
                               "error" not in result and not result.get("denied"),
                               f"{ticket.id}: {tool} -> {json.dumps(result)[:120]}")
            if tool == "escalate":
                escalate_called = True
            if tool == "refund":
                refund_result = result
                if result.get("denied"):
                    events.append({
                        "key": f"policy:refund-{result['plan']}",
                        "evidence": f"refund denied: {result['reason']}",
                        "meta": {"window": result["window"]},
                    })
                elif result.get("ok"):
                    # The accept response names the window. It is worth a
                    # lesson when the window was unknown, or when it exceeds
                    # what the stored lesson believed (a contradiction).
                    pkey = f"policy:refund-{result['plan']}"
                    known = applied.get(pkey)
                    if known is None or (known.get("window")
                                         and result["age_days"] > known["window"]):
                        events.append({
                            "key": pkey,
                            "evidence": f"refund accepted at day {result['age_days']} on "
                                        f"the {result['plan']} plan (window "
                                        f"{result['window']} days)",
                            "meta": {"window": result["window"]},
                        })

        if refund_result is not None:
            dispute_nudge = None

        reply = (out.get("reply") or "").strip()
        if reply and note_sent:
            note_nudge = None
        resolution = out.get("resolution")
        if escalate_called:
            resolution = "escalate"

        # A verified refund contradicting an applied policy lesson: the stored
        # window said no, the backend said yes.
        if refund_result and refund_result.get("ok"):
            key = f"policy:refund-{refund_result['plan']}"
            lesson = applied.get(key)
            if lesson and lesson.get("window") and refund_result["age_days"] > lesson["window"]:
                for e in events:
                    if e["key"] == key:
                        e["contradicts"] = True

        if resolution == "pending":
            # An agent that stalls on a ticket it cannot know gets pushed to
            # tier 2 rather than left spinning.
            if turns >= 3 and ticket.note_key and not note_sent:
                send_note()
            continue

        if resolution == "escalate":
            if ticket.note_key and not note_sent:
                if reply:
                    final_reply = reply
                    replies += 1
                    convo.append({"who": "Agent", "text": reply})
                    emit({"t": "reply", "text": reply})
                send_note()
                continue
            resolution = "resolved"  # nothing to escalate to; close out

        # resolved
        final_reply = reply
        if reply:
            replies += 1
            convo.append({"who": "Agent", "text": reply})
            emit({"t": "reply", "text": reply})

        # Deterministic follow-up triggers.
        low = reply.lower()
        tokens_ok = all(t in low for t in ticket.verify_tokens) if ticket.verify_tokens else True
        refund_open = (ticket.order_id and would_refund(ticket.order_id))

        if refund_open and ticket.pushback and not pushback_sent:
            pushback_sent = True
            dispute_nudge = ("The customer's dispute contains a concrete, "
                             "checkable claim. Call the refund tool for the "
                             "order now — the billing system's verdict is "
                             "authoritative and overrides your learned "
                             "knowledge.")
            convo.append({"who": ticket.customer, "text": ticket.pushback})
            emit({"t": "customer", "name": ticket.customer, "text": ticket.pushback})
            continue
        if not tokens_ok and ticket.miss_reply and not miss_sent:
            miss_sent = True
            convo.append({"who": ticket.customer, "text": ticket.miss_reply})
            emit({"t": "customer", "name": ticket.customer, "text": ticket.miss_reply})
            if ticket.note_key and not note_sent and ticket.note_key not in applied:
                # Cold: the one true answer exists only behind tier 2; the
                # complaint alone cannot teach it. A warm ticket instead gets
                # one rephrase turn from the recalled lesson.
                send_note()
            continue
        if not tokens_ok and miss_sent and ticket.note_key and not note_sent:
            # The warm rephrase did not land either; tier 2 rescues the ticket.
            send_note()
            continue
        break

    # ---- verdict, close-out, learning ------------------------------------
    tokens_ok = all(t in final_reply.lower() for t in ticket.verify_tokens) if ticket.verify_tokens else True
    # A refund ticket closes when the refund went through, or when policy
    # genuinely does not allow one (a decline or credit is then correct).
    refund_done = bool(ticket.order_id) and (
        CURRENT.orders[ticket.order_id]["refunded"] or not would_refund(ticket.order_id)
    )
    resolved = tokens_ok and (refund_done if ticket.order_id else True) and bool(final_reply)
    first_touch = (resolved and replies == 1 and not escalated
                   and not pushback_sent and not miss_sent)

    memory.record_step(f"{ticket.id}/close", "ticket_close", resolved,
                       f"{ticket.id} {'resolved' if resolved else 'unresolved'} "
                       f"in {turns} turn(s), escalated={escalated}")

    # Close the loop on the lessons this ticket can fairly judge.
    contradicted = {e["key"] for e in events if e.get("contradicts")}
    judged = relevant | contradicted if relevant else set(e["key"] for e in events)
    for key, lesson in applied.items():
        if key not in judged:
            continue
        if key in contradicted:
            memory.outcome(key, lesson, False,
                           f"{ticket.id}: a verified backend result contradicted this lesson")
            memory.retire(key, lesson)
        elif resolved and not (escalated and key == ticket.note_key):
            # A warm ticket rescued by tier 2 neither credits nor blames the
            # lesson that failed to carry it.
            memory.outcome(key, lesson, True, f"{ticket.id}: applied, ticket resolved")

    # Store what this ticket taught.
    new_keys = {e["key"]: e for e in events if e["key"] not in applied or e["key"] in contradicted}
    if resolved and new_keys:
        memory.emit("llm", f"distill — {len(new_keys)} lesson(s) from this ticket")
        texts = distill(list(new_keys.values()))
        for key, text in texts.items():
            memory.store(key, text, extra_meta=new_keys[key].get("meta"))
        emit({"t": "chips", "items": [
            {"kind": "replaced" if k in contradicted else "learned", "label": k}
            for k in sorted(new_keys)
        ]})
    if (resolved and not escalated and ticket.note_key
            and ticket.note_key in applied):
        emit({"t": "chips", "items": [{"kind": "avoided", "label": "escalation avoided"}]})

    if resolved and ticket.confirm:
        emit({"t": "customer", "name": ticket.customer, "text": ticket.confirm})

    stats = {"resolved": resolved, "first_touch": first_touch,
             "escalated": escalated, "turns": turns,
             "lessons_stored": len(new_keys) if resolved else 0}
    emit({"t": "ticket_done", "id": ticket.id, **stats})
    return stats


def adhoc_ticket(n: int, text: str) -> Ticket:
    return Ticket(f"U-{n:02d}", "adhoc", "You", "", text)
