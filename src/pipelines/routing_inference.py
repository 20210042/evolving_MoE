"""Phase-2 routing + one-step expert raw generation."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from agents.base import Agent
from pipelines.base_pipeline import BasePipeline
from prompts.coding import build_expert_prompt
from prompts.meta import (
    MANAGER_LEGAL_PROMPT,
    MANAGER_PROMPT,
    MANAGER_MATH_PROMPT,
    MANAGER_QASC_PROMPT,
    MANAGER_SNI_PROMPT,
)
from utils.domains import task_family
from utils.helpers import finalize_generation_output, extract_json_object


class GMRoutingPipeline(BasePipeline):
    """Manager routes to one expert; expert generates code in a single turn."""

    def __init__(
        self,
        agent: Agent,
        scouting_report_path: str,
        domain: str = "coding",
        routing_memory_path: str = "results/routing_memory.json",
        max_refine_iters: int = 2,  # deprecated: kept for CLI compat
        gen_enable_thinking: bool = True,
        router_use_description: bool = False,
        router_enable_thinking: bool = False,
        router_few_shot: bool = True,
        router_few_shot_retrieval: bool = False,
        router_retrieval_k: int = 5,
        router_top_k: int = 1,
    ):
        super().__init__(agent, domain)
        self.max_refine_iters = max_refine_iters
        self.gen_enable_thinking = gen_enable_thinking
        # Router info/behavior toggles (all default = current byte-identical behavior):
        #   description  → expose system_prompt instead of strengths keyword-list
        #   thinking     → let the router reason before emitting the JSON choice
        #   few_shot     → append past-routing examples
        #   few_shot_retrieval → replace random per-expert examples with the top-k
        #        exclusive_solves (each expert's UNIQUE niche) most lexically similar to
        #        the current problem (deterministic, discriminative, and leaner).
        self.router_use_description = router_use_description
        self.router_enable_thinking = router_enable_thinking
        self.router_few_shot = router_few_shot
        self.router_few_shot_retrieval = router_few_shot_retrieval
        self.router_retrieval_k = router_retrieval_k
        # top_k>1: router picks k experts; each generates; scoring = pass if ANY passes
        # (union over the k picks). Legit for coding since execution verifies the winner.
        self.router_top_k = max(1, int(router_top_k))
        self.scouting_report_path = scouting_report_path

        try:
            with open(self.scouting_report_path, "r", encoding="utf-8") as f:
                self.roster = json.load(f)
        except Exception as exc:
            logging.error("Failed to load scouting report at %s: %s", scouting_report_path, exc)
            self.roster = []

        # Retrieval few-shot index: pool of (expert_id, niche problem text) drawn from
        # each expert's exclusive_solves + tf-idf so we can rank by similarity per problem.
        self._retr_pool: List[Dict[str, Any]] = []
        self._retr_idf: Dict[str, float] = {}
        if self.router_few_shot_retrieval:
            self._build_retrieval_index()

        self.routing_memory: list = []
        mem = Path(routing_memory_path)
        if mem.is_file():
            try:
                self.routing_memory = json.loads(mem.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _roster_json(self) -> str:
        def summary(p):
            base = {
                "id": p.get("id", "default_id"),
                "name": p.get("name", p.get("persona_name", "Expert")),
            }
            # approach 모드: strengths 대신 identity(system_prompt)+approach를 라우터에 노출
            if p.get("approach"):
                base["identity"] = p.get("system_prompt", "")
                base["approach"] = p.get("approach", "")
            elif self.router_use_description:
                # description 모드: 콤마-키워드 strengths 대신 전문가 자기서술(system_prompt)을
                # 노출 → 라우터에 "무슨 문제를 어떻게 푸는가" 서술 신호를 준다.
                base["description"] = (
                    p.get("system_prompt") or p.get("strengths") or "Specialized coding expert"
                )
            else:
                # strengths 있으면 그대로(주제형 byte-identical). failure-mode 페르소나는
                # strengths가 없어 generic 기본값으로 떨어지면 라우터가 이름만 보고 눈 가림 →
                # system_prompt(실패유형 설명)로 fallback해 매칭 단서를 준다. (키는 유지)
                base["strengths"] = (
                    p.get("strengths") or p.get("system_prompt") or "Specialized coding expert"
                )
            return base

        return json.dumps([summary(p) for p in self.roster], indent=2)

    def _few_shot_suffix(self) -> str:
        examples: list = []
        roster_size = len(self.roster)
        agent_history_pools: Dict[str, list] = {}
        for p in self.roster:
            hist = p.get("routing_history", [])
            if hist:
                agent_history_pools[p["id"]] = list(hist)
                chosen = random.choice(agent_history_pools[p["id"]])
                examples.append({"best_expert_id": p["id"], "instruction": chosen["instruction"]})
                agent_history_pools[p["id"]].remove(chosen)

        if len(examples) < roster_size:
            extra_needed = roster_size - len(examples)
            available_agent_ids = [aid for aid, pool in agent_history_pools.items() if pool]
            while extra_needed > 0 and available_agent_ids:
                aid = random.choice(available_agent_ids)
                chosen = random.choice(agent_history_pools[aid])
                examples.append({"best_expert_id": aid, "instruction": chosen["instruction"]})
                agent_history_pools[aid].remove(chosen)
                if not agent_history_pools[aid]:
                    available_agent_ids.remove(aid)
                extra_needed -= 1

        if not examples:
            return ""
        random.shuffle(examples)
        few_shot_str = "\n\n### Past Successful Routing Examples:\n"
        for i, mem in enumerate(examples):
            few_shot_str += (
                f"Example {i+1}:\nProblem: {mem['instruction']}"
                f"\nOptimal Expert ID: {mem['best_expert_id']}\n\n"
            )
        return few_shot_str

    # --- Retrieval few-shot (strengths mode) -------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2]

    def _build_retrieval_index(self) -> None:
        """Pool each expert's exclusive_solves (problems only they solved = true niche)
        and compute idf so per-problem tf-idf cosine can rank the most similar niche."""
        pool: List[Dict[str, Any]] = []
        for p in self.roster:
            for s in (p.get("exclusive_solves") or []):
                if isinstance(s, str) and s.strip():
                    toks = self._tokenize(s)
                    if toks:
                        pool.append({"id": p["id"], "text": s, "tf": Counter(toks)})
        n = len(pool)
        df: Counter = Counter()
        for d in pool:
            for w in d["tf"]:
                df[w] += 1
        self._retr_idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        for d in pool:
            d["norm"] = math.sqrt(sum((tf * self._retr_idf.get(w, 0.0)) ** 2 for w, tf in d["tf"].items())) or 1.0
        self._retr_pool = pool
        logging.info("Router retrieval few-shot index: %d niche examples over %d experts.",
                     n, sum(1 for p in self.roster if p.get("exclusive_solves")))

    def _few_shot_retrieval(self, prompt: str) -> str:
        if not self._retr_pool:
            return ""
        q_tf = Counter(self._tokenize(prompt))
        if not q_tf:
            return ""
        q_norm = math.sqrt(sum((tf * self._retr_idf.get(w, 0.0)) ** 2 for w, tf in q_tf.items())) or 1.0
        scored = []
        for d in self._retr_pool:
            dot = sum(q_tf[w] * self._retr_idf.get(w, 0.0) * d["tf"][w] * self._retr_idf.get(w, 0.0)
                      for w in q_tf if w in d["tf"])
            if dot > 0:
                scored.append((dot / (q_norm * d["norm"]), d))
        if not scored:
            return ""
        scored.sort(key=lambda x: x[0], reverse=True)
        # ≤1 example per expert so a verbose persona with a large exclusive_solves pool
        # can't fill every slot (Polyglot/Linguistic magnet) — surfaces k DISTINCT experts.
        top = []
        seen = set()
        for sc, d in scored:
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            top.append(d)
            if len(top) >= self.router_retrieval_k:
                break
        out = "\n\n### Similar problems each expert uniquely solved:\n"
        for i, d in enumerate(top):
            out += f"Example {i+1}:\nProblem: {d['text']}\nOptimal Expert ID: {d['id']}\n\n"
        return out

    def _few_shot_for(self, prompt: str) -> str:
        """Per-problem few-shot: retrieval (if on) else shared random suffix."""
        if self.router_few_shot_retrieval:
            return self._few_shot_retrieval(prompt)
        return self._few_shot_suffix() if self.router_few_shot else ""

    def _parse_expert_id(self, router_res: str) -> str:
        default_id = self.roster[0]["id"] if self.roster else "default"
        data = extract_json_object(router_res)
        if not data:
            logging.warning("Failed to parse Manager JSON routing output.")
            return default_id
        expert_id = data.get("selected_expert_id") or data.get("selected_critic_id")
        if expert_id and any(p["id"] == expert_id for p in self.roster):
            return expert_id
        logging.warning("Router selected unknown expert id: %s", expert_id)
        return default_id

    def _topk_suffix(self, k: int) -> str:
        return (
            f"\n\nSelect the TOP {k} specialists most likely to solve it, best first."
            f'\nOutput only valid JSON: {{"selected_expert_ids": ["id1", ..., "id{k}"]}}'
        )

    def _manager_template(self):
        family = task_family(domain=self.domain)
        if family == "math":
            return MANAGER_MATH_PROMPT
        if family == "qasc":
            return MANAGER_QASC_PROMPT
        if family == "lbox":
            return MANAGER_LEGAL_PROMPT
        if family == "sni":
            return MANAGER_SNI_PROMPT
        return MANAGER_PROMPT

    def _parse_expert_ids(self, router_res: str, k: int) -> List[str]:
        """Parse an ordered list of up to k valid expert ids; pad with unused roster
        members so there are always k distinct candidates (never fewer)."""
        valid = {p["id"] for p in self.roster}
        ids: List[str] = []
        data = extract_json_object(router_res)
        if data:
            raw = data.get("selected_expert_ids") or data.get("selected_expert_id")
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                for x in raw:
                    if x in valid and x not in ids:
                        ids.append(x)
        if not ids:
            logging.warning("Failed to parse top-%d routing ids; falling back.", k)
        for p in self.roster:  # pad to k distinct
            if len(ids) >= k:
                break
            if p["id"] not in ids:
                ids.append(p["id"])
        return ids[:k]

    def _prepare_prompt(self, input_item: dict) -> str:
        prompt = input_item.get("instruction") or input_item.get("prompt") or input_item.get("problem")
        starter_code = input_item.get("starter_code")
        if starter_code:
            prompt = f"{prompt}\n\nStarter Code:\n```python\n{starter_code}\n```"
        return prompt or ""

    def _route_one(self, prompt: str, few_shot_str: str) -> str:
        roster_str = self._roster_json()
        mgr_template = self._manager_template()
        manager_prompt = (
            mgr_template.substitute(scouting_report=roster_str, problem_description=prompt)
            + few_shot_str
        )
        router_res = self.agent.chat(
            [
                {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
                {"role": "user", "content": manager_prompt},
            ],
            enable_thinking=self.router_enable_thinking,
        )
        return self._parse_expert_id(router_res)

    def run(self, input_item: dict) -> Dict[str, Any]:
        prompt = self._prepare_prompt(input_item)
        ds = input_item.get("dataset") or "mbpp"
        model_name = self.agent.llm.model_name
        few_shot_str = self._few_shot_for(prompt)

        selected_id = self._route_one(prompt, few_shot_str)
        history: list = [{"stage": "routing", "selected_expert": selected_id}]

        selected_player = next((p for p in self.roster if p["id"] == selected_id), None)
        if not selected_player:
            return {
                "id": input_item.get("id"),
                "initial_output": "",
                "final_output": "",
                "history": history,
            }

        expert_sys = selected_player.get("system_prompt", "You are an expert coder.")
        gen_msg = build_expert_prompt(
            prompt,
            expert_sys,
            dataset=ds,
            model_name=model_name,
            starter_code=None,
            approach=selected_player.get("approach"),
            domain=self.domain,
        )
        raw = self.agent.chat(gen_msg, enable_thinking=self.gen_enable_thinking)
        output = finalize_generation_output(raw, dataset=ds, domain=self.domain)
        history.append({"stage": "generation", "code": output})

        return {
            "id": input_item.get("id"),
            "initial_output": output,
            "final_output": output,
            "history": history,
        }

    def run_batch(self, items: List[dict]) -> List[Dict[str, Any]]:
        if not items:
            return []

        model_name = self.agent.llm.model_name
        # non-retrieval: shared random suffix computed once; retrieval: per-problem below
        shared_few_shot = "" if self.router_few_shot_retrieval else (
            self._few_shot_suffix() if self.router_few_shot else ""
        )
        roster_str = self._roster_json()

        k = self.router_top_k
        route_msgs = []
        prompts: List[str] = []
        mgr_template = self._manager_template()
        for item in items:
            prompt = self._prepare_prompt(item)
            prompts.append(prompt)
            few_shot_str = self._few_shot_retrieval(prompt) if self.router_few_shot_retrieval else shared_few_shot
            manager_prompt = (
                mgr_template.substitute(scouting_report=roster_str, problem_description=prompt)
                + few_shot_str
                + (self._topk_suffix(k) if k > 1 else "")
            )
            route_msgs.append(
                [
                    {"role": "system", "content": "You are a strict JSON API. Only output valid JSON."},
                    {"role": "user", "content": manager_prompt},
                ]
            )

        route_out = self.agent.chat_batch(route_msgs, enable_thinking=self.router_enable_thinking)

        def _expert_sys(sid: str):
            player = next((p for p in self.roster if p["id"] == sid), None)
            sys = player.get("system_prompt", "You are an expert coder.") if player else "You are an expert coder."
            appr = player.get("approach") if player else None
            return sys, appr

        # --- top-1: existing byte-identical path -------------------------------------
        if k == 1:
            selected_ids = [self._parse_expert_id(r) for r in route_out]
            gen_msgs = []
            for item, prompt, sid in zip(items, prompts, selected_ids):
                ds = item.get("dataset") or "mbpp"
                sys, appr = _expert_sys(sid)
                gen_msgs.append(build_expert_prompt(
                    prompt, sys, dataset=ds, model_name=model_name,
                    starter_code=None, approach=appr, domain=self.domain))
            gen_out = self.agent.chat_batch(gen_msgs, enable_thinking=self.gen_enable_thinking)
            results = []
            for item, sid, raw in zip(items, selected_ids, gen_out):
                ds = item.get("dataset") or "mbpp"
                code = finalize_generation_output(raw, dataset=ds, domain=self.domain)
                results.append({
                    "id": item.get("id"),
                    "initial_output": code, "final_output": code,
                    "history": [{"stage": "routing", "selected_expert": sid},
                                {"stage": "generation", "code": code}],
                })
            return results

        # --- top-k: route to k experts, generate each; scoring OR's the candidates ----
        selected_lists = [self._parse_expert_ids(r, k) for r in route_out]
        gen_msgs = []
        flat: List[tuple] = []  # (item_idx, expert_id)
        for i, (prompt, ids) in enumerate(zip(prompts, selected_lists)):
            ds = items[i].get("dataset") or "mbpp"
            for sid in ids:
                sys, appr = _expert_sys(sid)
                gen_msgs.append(build_expert_prompt(
                    prompt, sys, dataset=ds, model_name=model_name,
                    starter_code=None, approach=appr, domain=self.domain))
                flat.append((i, sid))
        gen_out = self.agent.chat_batch(gen_msgs, enable_thinking=self.gen_enable_thinking)

        per_item: Dict[int, List[Dict[str, str]]] = {}
        for (i, sid), raw in zip(flat, gen_out):
            ds = items[i].get("dataset") or "mbpp"
            code = finalize_generation_output(raw, dataset=ds, domain=self.domain)
            per_item.setdefault(i, []).append({"expert": sid, "code": code})

        results = []
        for i, item in enumerate(items):
            cands = per_item.get(i, [])
            results.append({
                "id": item.get("id"),
                # final_output = first pick's code (lets top-1 scorers still work);
                # candidates = all k picks for union (top-k) scoring.
                "initial_output": cands[0]["code"] if cands else "",
                "final_output": cands[0]["code"] if cands else "",
                "candidates": cands,
                "history": [{"stage": "routing", "selected_experts": selected_lists[i]},
                            {"stage": "generation", "candidates": [c["expert"] for c in cands]}],
            })
        return results
