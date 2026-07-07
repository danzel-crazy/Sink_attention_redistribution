from typing import Optional

import torch

from cross_attention_sink_redistribution.sink_tokens import sink_token_selector


class cross_attention_importants:
    def __init__(self, args, sink_selector: Optional[sink_token_selector] = None):

        # sink tokens: reuse a shared selector if the caller already owns one (e.g. LlamaModel),
        # so sink_tokens_ids/scores don't drift between independently-configured copies.
        self.sink_selector = sink_selector if sink_selector is not None else sink_token_selector(args)
        self.enable_sink_masked = getattr(args, "enable_sink_masked", True)
        self.important_tokens_scores = []

        self.text_tokens_start_index = getattr(args, "text_tokens_start_index", 0)
        self.text_tokens_length = getattr(args, "text_tokens_length", 0)

        self.visual_tokens_start_index = self.text_tokens_start_index + self.text_tokens_length
        self.visual_tokens_length = getattr(args, "visual_tokens_length", 0)

    def _extract_text_tokens(self, hidden_states, start=None, length=None):
        start = self.text_tokens_start_index if start is None else start
        length = self.text_tokens_length if length is None else length
        return hidden_states[:, start:start + length, :]

    def _extract_visual_tokens(self, hidden_states):
        return hidden_states[:, self.visual_tokens_start_index:self.visual_tokens_start_index + self.visual_tokens_length, :]

    def compute_cross_attention(
        self,
        hidden_states,
        text_tokens_start_index=None,
        text_tokens_length=None,
        sink_local_ids=None,
    ):
        """
        text_tokens_start_index/text_tokens_length: override the static system-prompt-prefix
            config for this call - e.g. pass the actual question span, since its length varies
            per example and the static config can't track that.
        sink_local_ids: sink-token indices *local to the visual-token block*, already computed
            by the caller (e.g. fastv_forward). If omitted, sink tokens are (re)detected here via
            self.sink_selector - but note that changes the token population the quantile in
            resolve_sink_score_range is computed over, so prefer passing this in explicitly when
            the caller already has it, to keep sink membership consistent across the pipeline.
        """
        # text tokens as queries, visual tokens as keys: how much each visual token matters to the text
        text_tokens = self._extract_text_tokens(hidden_states, text_tokens_start_index, text_tokens_length)[0]
        visual_tokens = self._extract_visual_tokens(hidden_states)[0]

        scale = visual_tokens.shape[-1] ** 0.5
        cross_attn_scores = torch.matmul(text_tokens, visual_tokens.transpose(0, 1)) / scale

        # unmasked reference - e.g. for feeding into sink_attention_redistributor.redistribute(),
        # which does its own sink-zeroing + budget redistribution and would see a zero budget if
        # handed the already-masked scores below
        raw_weights = torch.softmax(cross_attn_scores, dim=-1)
        visual_importance_raw = raw_weights.mean(dim=0)
        self.important_tokens_scores_raw = visual_importance_raw

        visual_importance = visual_importance_raw
        if self.enable_sink_masked:
            if sink_local_ids is None:
                sink_ids = self.sink_selector._select_sink_tokens(hidden_states[0])
                visual_sink_mask = (sink_ids >= self.visual_tokens_start_index) & (
                    sink_ids < self.visual_tokens_start_index + self.visual_tokens_length
                )
                sink_local_ids = sink_ids[visual_sink_mask] - self.visual_tokens_start_index

            if sink_local_ids.numel() > 0:
                # block sink tokens *before* softmax so they take no part in the normalization,
                # instead of zeroing their post-softmax weight (which left the rest un-renormalized)
                masked_scores = cross_attn_scores.clone()
                masked_scores[:, sink_local_ids] = float("-inf")
                masked_weights = torch.softmax(masked_scores, dim=-1)
                visual_importance = masked_weights.mean(dim=0)

        self.important_tokens_scores = visual_importance
        return visual_importance