import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.multimodal import (
    PreparedMultimodalPrompt,
    validate_multimodal,
    validate_text_tokens,
)


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self.model_runner = None
        self._closed = False
        try:
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
            if not config.is_multimodal:
                config.eos = self.tokenizer.eos_token_id
            self.scheduler = Scheduler(config)
            atexit.register(self.exit)
        except Exception as error:
            try:
                self.exit()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def exit(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        runner = self.model_runner
        self.model_runner = None
        stopped = False
        try:
            scheduler = getattr(self, "scheduler", None)
            if scheduler is not None:
                for seq in list(scheduler.waiting) + list(scheduler.running):
                    scheduler.abort(seq)
            if runner is not None:
                runner.call("exit")
                stopped = True
        finally:
            for process in self.ps:
                if not stopped and process.is_alive():
                    process.terminate()  # Only workers created by this engine.
                process.join()
            atexit.unregister(self.exit)

    def add_request(
        self,
        prompt: str | list[int] | PreparedMultimodalPrompt,
        sampling_params: SamplingParams,
    ):
        multimodal = None
        if isinstance(prompt, PreparedMultimodalPrompt):
            if not self.config.is_multimodal:
                raise ValueError("Prepared input requires a Qwen3-VL engine")
            validate_multimodal(
                prompt,
                self.config.hf_config,
                max_visual_tokens=self.config.max_visual_tokens,
            )
            multimodal, prompt = prompt, prompt.token_ids
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("Prompt must contain at least one token")
        if self.config.is_multimodal and multimodal is None:
            validate_text_tokens(prompt, self.config.hf_config)
        seq = Sequence(prompt, sampling_params, multimodal=multimodal)
        self.scheduler.add(seq)

    def step(self):
        queue = self.scheduler.running or self.scheduler.waiting
        active = queue[0] if queue else None
        seqs = []
        try:
            seqs, is_prefill = self.scheduler.schedule()
            num_tokens = (
                sum(seq.num_scheduled_tokens for seq in seqs)
                if is_prefill
                else -len(seqs)
            )
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            outputs = [
                (seq.seq_id, seq.completion_token_ids)
                for seq in seqs
                if seq.is_finished
            ]
            return outputs, num_tokens
        except Exception:
            for seq in seqs or ([active] if active is not None else []):
                self.scheduler.abort(seq)
            raise

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str | list[int] | PreparedMultimodalPrompt],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        return outputs
