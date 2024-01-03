import torch
from util import batched_index, MaskingMode
from more_itertools import chunked
from einops import reduce
from tqdm import tqdm


class BaseConditionalModel:
    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer

    def _obtain_logits_over_set(self, batch, target_position, other_position, target_set, other_set):
        """
        Compute target_position potentials (i.e., we leave target_position as [MASK] and cycle through choices of
        other_position, which can take all the values in token_set)
        """
        batch = batch.copy()
        input_ids = batch.pop("input_ids")

        # Ensure target positions are masked
        input_ids = torch.scatter(input_ids, index=target_position[:, None], dim=1, value=self.tokenizer.mask_token_id)

        potentials = []
        for replacements in other_set.split(dim=1, split_size=1):
            input_ids_masked_target = input_ids.scatter(index=other_position[:, None], src=replacements, dim=1)
            output_logits = batched_index(
                self.model(input_ids=input_ids_masked_target, **batch).logits, target_position)
            selected_output_logits = output_logits.gather(index=target_set, dim=1)
            potentials.append(selected_output_logits)

        potentials = torch.stack(potentials, dim=2)
        return potentials

    def pairwise_log_conditional(self, batch, position_A, position_B, set_A, set_B):
        """
        Assumes we have a set_A: List[int] and set_B: List[int] of tokens that we are normalizing over.
        """
        raise NotImplementedError()

    def pairwise_log_conditional_with_logits(self, batch, position_A, position_B, logits_A, logits_B):
        """
        Assumes we have a logits_A: tensor[B x V x V] and logits_B: torch.tensor[B x V x V] of the logits of BERT
        when masking out position_A (or B) and replacing position_B (or A) with ALL possible tokens.
        """
        raise NotImplementedError()


class MRFConditionalModel(BaseConditionalModel):
    MASKING_MODE = MaskingMode.UNARY

    def __init__(self, normalize_logits: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_logits = normalize_logits

    def _construct_joint(self, A_potentials, B_potentials):
        batch_size, candidates_A, candidates_B = A_potentials.shape

        joint_unnormalized = A_potentials + B_potentials
        log_joint = joint_unnormalized.reshape(batch_size, -1).log_softmax(dim=1)
        log_joint = log_joint.reshape(batch_size, candidates_A, candidates_B)
        return log_joint

    @torch.no_grad()
    def pairwise_log_conditional(self, batch, position_A, position_B, set_A, set_B):
        A_potentials = self._obtain_logits_over_set(batch, position_A, position_B, set_A, set_B)
        B_potentials = self._obtain_logits_over_set(batch, position_B, position_A, set_B, set_A).transpose(1, 2)

        if self.normalize_logits:
            A_potentials = A_potentials.log_softmax(dim=1)
            B_potentials = B_potentials.log_softmax(dim=2)

        log_joint = self._construct_joint(A_potentials, B_potentials)
        return log_joint

    @torch.no_grad()
    def pairwise_log_conditional_with_logits(self, batch, position_A, position_B, logits_A, logits_B):
        """
        Assumes we have a logits_A: tensor[B x V x V] and logits_B: torch.tensor[B x V x V] of the logits of BERT
        when masking out position_A (or B) and replacing position_B (or A) with ALL possible tokens.
        """
        if self.normalize_logits:
            logits_A = logits_A.log_softmax(dim=1)
            logits_B = logits_B.log_softmax(dim=2)

        log_joint = self._construct_joint(logits_A, logits_B)
        return log_joint


class NaiveConditionalModel(BaseConditionalModel):
    @torch.no_grad()
    def pairwise_log_conditional(self, batch, position_A, position_B, set_A, set_B):
        batch = batch.copy()

        # Mask both positions
        input_ids = batch.pop("input_ids")
        input_ids_masked = torch.scatter(input_ids, index=torch.stack([position_A, position_B], dim=1), dim=1,
                                         value=self.tokenizer.mask_token_id)

        # Joint is product of marginals (i.e., sum of log marginals), where the marginals are obtained by
        # normalizing BERT logits over the appropriate sets.
        # Note that if we really don't want to normalize over set_A and set_B, then we should probably be using
        # the --burn-in flag in the evaluate_block_gibbs.py file.
        logits = self.model(input_ids=input_ids_masked, **batch).logits

        log_probs_A = batched_index(logits, position_A).gather(dim=1, index=set_A)
        log_probs_A = log_probs_A.log_softmax(dim=1).unsqueeze(2)

        log_probs_B = batched_index(logits, position_B).gather(dim=1, index=set_B)
        log_probs_B = log_probs_B.log_softmax(dim=1).unsqueeze(1)

        return log_probs_A + log_probs_B

    @torch.no_grad()
    def pairwise_log_conditional_with_logits(self, batch, position_A, position_B, logits_A, logits_B):
        """
        Assumes we have a logits_A: tensor[B x V x V] and logits_B: torch.tensor[B x V x V] of the logits of BERT
        when masking out position_A (or B) and replacing position_B (or A) with ALL possible tokens.
        """
        batch = batch.copy()
        input_ids = batch.pop("input_ids")

        # Mask both positions
        input_ids = torch.scatter(input_ids, index=torch.stack([position_A, position_B], dim=1), dim=1,
                                  value=self.tokenizer.mask_token_id)

        # Joint is product of marginals (i.e., sum of log marginals)
        logits = self.model(input_ids=input_ids, **batch).logits
        log_probs_A = batched_index(logits, position_A).log_softmax(dim=1).unsqueeze(2)
        log_probs_B = batched_index(logits, position_B).log_softmax(dim=1).unsqueeze(1)
        return log_probs_A + log_probs_B


class HCBConditionalModel(BaseConditionalModel):
    MASKING_MODE = MaskingMode.UNARY

    def __init__(self, pivot_mode: str, *args, **kwargs):
        """
        There are three pivot modes:
          - gold: uses the actual tokens as the pivot. This assumes that the gold tokens are the first elements of the
                  sets (unsupported)
          - both: masks both tokens of interest (at the same time) and uses the top candidate of each position
          - one:  masks one token at a time (keeping the other one as the existing token) and uses the top
                  prediction at those positions
        """
        super().__init__(*args, **kwargs)
        assert pivot_mode in ["both", "one"], "We only support 'both' and 'one' pivot selection schemes."
        self.pivot_mode = pivot_mode

    def _obtain_adjusted_log_probs_over_set(self, batch, target_position, other_position, target_set, other_set,
                                            target_pivot):
        """
        Compute target_position log probs (i.e., we leave target_position as [MASK] and cycle through choices of
        other_position, which can take all the values in other_set), subtracting the log probability of the pivot.
        """
        batch = batch.copy()
        input_ids = batch.pop("input_ids")

        # Ensure target positions are masked
        input_ids = torch.scatter(input_ids, index=target_position[:, None], dim=1, value=self.tokenizer.mask_token_id)

        potentials = []
        for replacements in other_set.split(dim=1, split_size=1):
            input_ids_masked_target = input_ids.scatter(index=other_position[:, None], src=replacements, dim=1)
            output_logits = batched_index(
                self.model(input_ids=input_ids_masked_target, **batch).logits, target_position)
            output_log_probs = output_logits.log_softmax(dim=1)
            selected_output_log_probs = output_log_probs.gather(index=target_set, dim=1)
            selected_output_log_probs -= output_log_probs.gather(index=target_pivot.unsqueeze(1), dim=1)
            potentials.append(selected_output_log_probs)

        potentials = torch.stack(potentials, dim=2)
        return potentials

    def _construct_joint(self, A_log_probs, B_log_probs):
        batch_size, candidates_A, candidates_B = A_log_probs.shape
        log_joint_unnormalized = A_log_probs + B_log_probs
        log_Z = log_joint_unnormalized.reshape(batch_size, -1).logsumexp(dim=1)
        log_joint = log_joint_unnormalized - log_Z[:, None, None]
        return log_joint

    @torch.no_grad()
    def pairwise_log_conditional(self, batch, position_A, position_B, set_A, set_B):
        batch_original = batch.copy()
        batch = batch.copy()
        input_ids = batch.pop("input_ids")
        batch_size = input_ids.shape[0]  # noqa
        candidates_A, candidates_B = set_A.shape[1], set_B.shape[1]  # noqa

        # Generate pivots
        if self.pivot_mode == "gold":
            pivot_A = set_A[:, 0]
        elif self.pivot_mode == "both":
            pivot_input_ids = input_ids.scatter(index=torch.stack([position_A, position_B], dim=1), dim=1,
                                                value=self.tokenizer.mask_token_id)
            pivot_A = batched_index(self.model(input_ids=pivot_input_ids, **batch).logits, position_A).argmax(dim=1)
        elif self.pivot_mode == "one":
            pivot_input_ids = input_ids.scatter(index=position_A[:, None], dim=1, value=self.tokenizer.mask_token_id)
            pivot_A = batched_index(self.model(input_ids=pivot_input_ids, **batch).logits, position_A).argmax(dim=1)
        else:
            raise NotImplementedError("Unknown pivot mode")

        # Ensure position_A and position_B are masked
        input_ids = torch.scatter(input_ids, index=torch.stack([position_A, position_B], dim=1), dim=1,
                                  value=self.tokenizer.mask_token_id)

        # Compute position A contribution (i.e., we leave position A as [MASK] and cycle through choices of position B)
        A_log_probs = self._obtain_adjusted_log_probs_over_set(
            batch_original, position_A, position_B, set_A, set_B, pivot_A)

        # Compute position B contribution (i.e., position A is fixed to pivot_A)
        input_ids_masked_B = input_ids.scatter(index=position_A[:, None], src=pivot_A[:, None], dim=1)
        output_logits = batched_index(self.model(input_ids=input_ids_masked_B, **batch).logits, position_B)
        output_log_probs = output_logits.log_softmax(dim=1)
        selected_output_log_probs = output_log_probs.gather(index=set_B, dim=1)
        B_log_probs = selected_output_log_probs.unsqueeze(1).expand(-1, candidates_A, -1)

        # Compute joint
        log_joint = self._construct_joint(A_log_probs, B_log_probs)
        return log_joint

    @torch.no_grad()
    def pairwise_log_conditional_with_logits(self, batch, position_A, position_B, logits_A, logits_B):
        batch = batch.copy()
        input_ids = batch.pop("input_ids")
        assert input_ids.shape[0] == 1, "When supplying logits, we assume batch_size = 1"

        # Generate pivots
        if self.pivot_mode == "gold":
            raise NotImplementedError("Unsupported.")
        elif self.pivot_mode == "both":
            pivot_input_ids = input_ids.scatter(index=torch.stack([position_A, position_B], dim=1), dim=1,
                                                value=self.tokenizer.mask_token_id)
            pivot_A = batched_index(self.model(input_ids=pivot_input_ids, **batch).logits, position_A).argmax(dim=1)
        elif self.pivot_mode == "one":
            raise NotImplementedError("Unsupported.")
        else:
            raise NotImplementedError("Unknown pivot mode")

        # Ensure position_A and position_B are masked
        input_ids = torch.scatter(input_ids, index=torch.stack([position_A, position_B], dim=1), dim=1,
                                  value=self.tokenizer.mask_token_id)

        # Compute position A contribution
        A_log_probs = logits_A.log_softmax(dim=1) - batched_index(logits_A.log_softmax(dim=1), pivot_A).unsqueeze(1)

        # Compute position B contribution (i.e., position A is fixed to pivot_A)
        output_log_probs = logits_B[0, pivot_A, :].log_softmax(dim=1)
        B_log_probs = output_log_probs.unsqueeze(1).expand(-1, A_log_probs.shape[1], -1)

        # Compute joint
        log_joint = self._construct_joint(A_log_probs, B_log_probs)
        return log_joint


class IterConditionalModel(BaseConditionalModel):
    MASKING_MODE = MaskingMode.UNARY

    def __init__(self, iterations: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = iterations

    def _compute_joint(self, A_log_probs, B_log_probs):
        batch_size, candidates_A, candidates_B = A_log_probs.shape
        joint = torch.ones_like(A_log_probs) / (candidates_A * candidates_B)
        summed_probs = A_log_probs.exp() + B_log_probs.exp()
        for iteration in range(self.iterations):
            inv_marginal_A = 1 / joint.sum(dim=2, keepdim=True)
            inv_marginal_B = 1 / joint.sum(dim=1, keepdim=True)
            unnormalized_joint = summed_probs / (inv_marginal_A + inv_marginal_B)
            Z = unnormalized_joint.reshape(batch_size, -1).sum(dim=1)[:, None, None]
            joint = unnormalized_joint / Z
            del inv_marginal_A, inv_marginal_B, unnormalized_joint, Z

        return joint

    def _compute_log_joint(self, A_log_probs, B_log_probs):
        batch_size, candidates_A, candidates_B = A_log_probs.shape
        log_joint = (torch.ones_like(A_log_probs) / (candidates_A * candidates_B)).log()
        for iteration in range(self.iterations):
            log_marginal_A = log_joint.logsumexp(dim=2, keepdim=True).expand(-1, -1, candidates_B)
            log_marginal_B = log_joint.logsumexp(dim=1, keepdim=True).expand(-1, candidates_A, -1)
            log_numerator = torch.stack([A_log_probs, B_log_probs], dim=3).logsumexp(dim=3)
            log_denominator = torch.stack([-log_marginal_A, -log_marginal_B], dim=3).logsumexp(dim=3)
            unnormalized_log_joint = log_numerator - log_denominator
            log_Z = unnormalized_log_joint.reshape(batch_size, -1).logsumexp(dim=1)[:, None, None]
            log_joint = unnormalized_log_joint - log_Z

        return log_joint

    @torch.no_grad()
    def pairwise_log_conditional(self, batch, position_A, position_B, set_A, set_B):
        A_log_probs = self._obtain_logits_over_set(batch, position_A, position_B, set_A, set_B).log_softmax(dim=1)
        B_log_probs = self._obtain_logits_over_set(
            batch, position_B, position_A, set_B, set_A).log_softmax(dim=1).transpose(1, 2)
        joint = self._compute_joint(A_log_probs, B_log_probs)
        return joint.log()

    @torch.no_grad()
    def pairwise_log_conditional_with_logits(self, batch, position_A, position_B, logits_A, logits_B):
        A_log_probs = logits_A.log_softmax(dim=1)
        B_log_probs = logits_B.log_softmax(dim=2)

        # NOTE: If we were able to use the log joint instead, we'd use:
        # log_joint = self._compute_log_joint(A_log_probs, B_log_probs)
        # return log_joint.to(device)

        joint = self._compute_joint(A_log_probs, B_log_probs)
        return joint.log()


class CompatibilityConditionalModel(BaseConditionalModel):
    MASKING_MODE = MaskingMode.PAIR

    def __init__(self, compatibility_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compatibility_layer = compatibility_layer

    @torch.no_grad()
    def pairwise_log_conditional(self, batch, position_A, position_B, set_A, set_B):
        batch = batch.copy()

        # Mask both positions
        input_ids = batch.pop("input_ids")
        input_ids_masked = torch.scatter(input_ids, index=torch.stack([position_A, position_B], dim=1), dim=1,
                                         value=self.tokenizer.mask_token_id)
        outputs = self.model(input_ids=input_ids_masked, output_hidden_states=True, **batch)
        return self._pairwise_log_conditional(outputs, position_A, position_B, set_A, set_B, include_offset=False)

    @torch.no_grad()
    def _pairwise_log_conditional(self, outputs, position_A, position_B, set_A, set_B, *args, **kwargs):
        log_probs = outputs.logits
        log_conditional_both_A = batched_index(log_probs, position_A)
        log_conditional_both_B = batched_index(log_probs, position_B)
        representation_both_A = batched_index(outputs.hidden_states[-1], position_A)
        representation_both_B = batched_index(outputs.hidden_states[-1], position_B)

        # Construct joint (probably best to create a class for this)
        extra = {
            "representation_A": representation_both_A,
            "representation_B": representation_both_B,
            "position_A": position_A,
            "position_B": position_B,
        }
        log_learned = self.compatibility_layer(
            log_conditional_both_A, log_conditional_both_B, set_A, set_B, extra, *args, **kwargs
        )
        return log_learned

    @torch.no_grad()
    def pairwise_log_conditional_with_logits(self, batch, position_A, position_B, logits_A, logits_B):
        batch = batch.copy()
        batch_size = 1000  # NOTE: It would be nice to pass this in as an argument; hardcoding for now for simplicity

        # Ignore logits and recompute two-place-masked logits_both_A, logits_both_B
        input_ids = batch.pop("input_ids")
        assert input_ids.shape[0] == 1, "When supplying logits, we assume batch_size = 1"
        input_ids_masked = torch.scatter(input_ids, index=torch.stack([position_A, position_B], dim=1), dim=1,
                                         value=self.tokenizer.mask_token_id)
        outputs = self.model(input_ids=input_ids_masked, output_hidden_states=True, **batch)

        # Partition vocabulary into chunks of size batch_size
        offsets = []
        pbar = tqdm(list(chunked(range(self.tokenizer.vocab_size), batch_size)))
        for subset_A in pbar:
            set_A = torch.tensor(subset_A, device=input_ids.device).unsqueeze(0)
            offsets_B = []
            for subset_B in chunked(range(self.tokenizer.vocab_size), batch_size):
                set_B = torch.tensor(subset_B, device=input_ids.device).unsqueeze(0)
                local_offset = self._pairwise_log_conditional(
                    outputs, position_A, position_B, set_A, set_B, return_offset=True
                ).cpu()
                offsets_B.append(local_offset)

            offsets_B = torch.cat(offsets_B, dim=2)
            assert offsets_B.shape == (1, len(subset_A), self.tokenizer.vocab_size)
            offsets.append(offsets_B)

        offsets = torch.cat(offsets, dim=1)
        assert offsets.shape == (1, self.tokenizer.vocab_size, self.tokenizer.vocab_size)

        # Compute naive (i.e., independence-assumption) log joint
        log_probs = outputs.logits
        log_prob_both_A = batched_index(log_probs, position_A).log_softmax(dim=1).unsqueeze(2)
        log_prob_both_B = batched_index(log_probs, position_B).log_softmax(dim=1).unsqueeze(1)
        base_log_joint = (log_prob_both_A + log_prob_both_B).cpu()

        # Apply offsets and renormalize to obtain final log joint
        logits = base_log_joint + offsets
        log_Z = reduce(logits, "b sA sB -> b () ()", reduction=torch.logsumexp)
        log_joint = logits - log_Z
        return log_joint
