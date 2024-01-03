from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser
from util import generate_nli_split, generate_snli_split, prepare_nli_data_for_mlm, tokenize_data, collate_and_pad_fn, \
    generate_sets, compute_bert_conditionals_over_set, compute_kls, compute_bert_logits_whole_vocab, \
    compute_compatibility, add_model_cli_args, setup_derived_model_from_args, compute_jsds, get_rank_in_joint, \
    prepare_masks, generate_xsum_split, prepare_summarization_data_for_mlm
from functools import partial
import torch
import multiprocessing
from tqdm import tqdm
from pprint import pprint
from itertools import combinations
import warnings
import pickle

parser = ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("--schemes", nargs="+", choices=[
    "mrf", "mrf-local", "iter", "hcb-gold", "hcb-both", "hcb-one", "naive", "compatibility"
])
parser.add_argument("--compatibility-layer", type=str)
add_model_cli_args(parser)
parser.add_argument("--joint-size", type=int, default=None, help="Number of elements in joint.")
parser.add_argument("--debug-messages", action="store_true", default=False)
parser.add_argument("--dataset", type=str, choices=["snli", "xsum"], default="snli")
parser.add_argument("--max-datapoints", type=int)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--use-gpu", action="store_true", default=False)
parser.add_argument("--block-contiguous", action="store_true", default=False, help="Have blocks be over contiguous"
                    "tokens.")
parser.add_argument("--override-tokenizer", type=str)
parser.add_argument("--preprocessing-num-threads", type=int, default=multiprocessing.cpu_count())
parser.add_argument("--preprocessing-batch-size", type=int, default=100)
parser.add_argument("--output-file", type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = 0 if args.use_gpu else "cpu"
whole_vocab = args.joint_size is None
if whole_vocab:
    warnings.warn("Running in whole vocab mode. Model computations are done with batch_size=1, and batch_size CLI flag"
                  f" is being used for vocab computations (i.e., vocab_batch_size={args.batch_size}).")
    batch_size = 1
    vocab_batch_size = args.batch_size
else:
    batch_size = args.batch_size
    joint_size_A, joint_size_B = args.joint_size, args.joint_size

# Set up tokenizer and model
tokenizer_kwargs = {}
if "roberta" in args.model:
    tokenizer_kwargs = {"add_prefix_space": True}

if args.override_tokenizer:
    warnings.warn(f"Overriding tokenizer, using: {args.override_tokenizer}.")
    tokenizer = AutoTokenizer.from_pretrained(args.override_tokenizer, **tokenizer_kwargs)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
model.eval()

# Load data
if args.dataset == "multi_nli":
    dataset = generate_nli_split("mlm")
    dataset_prep_fn = prepare_nli_data_for_mlm
elif args.dataset == "snli":
    dataset = generate_snli_split("mlm")
    dataset_prep_fn = prepare_nli_data_for_mlm
elif args.dataset == "xsum":
    dataset = generate_xsum_split()
    dataset_prep_fn = prepare_summarization_data_for_mlm
else:
    raise NotImplementedError("Unsupported dataset.")

dataset = dataset.map(dataset_prep_fn, fn_kwargs=dict(tokenizer=tokenizer), batched=True,
                      batch_size=args.preprocessing_batch_size, remove_columns=dataset["train"].column_names,
                      num_proc=args.preprocessing_num_threads)
dataset = dataset.map(tokenize_data, fn_kwargs=dict(tokenizer=tokenizer, return_special_tokens_mask=True),
                      batched=True, batch_size=args.preprocessing_batch_size, num_proc=args.preprocessing_num_threads)
dataset = dataset.map(prepare_masks, fn_kwargs=dict(contiguous=args.block_contiguous), batched=True,
                      batch_size=args.preprocessing_batch_size, num_proc=1)
data_collator = partial(collate_and_pad_fn, tokenizer=tokenizer)

# Create data loader
python_columns = ["sentences"]
torch_columns = list(set(dataset["train"].column_names) - set(python_columns))
dataset.set_format(type="torch", columns=torch_columns, output_all_columns=True)
dataset = dataset["validation"]
if args.max_datapoints is not None:
    warnings.warn(f"Truncating dataset to first {args.max_datapoints} datapoints.")
    dataset = dataset.select(range(0, args.max_datapoints))

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, collate_fn=data_collator)


def append_to_metrics(schemas, metric, value):
    if frozenset(schemas) not in METRICS:
        METRICS[frozenset(schemas)] = {}

    if metric not in METRICS[frozenset(schemas)]:
        METRICS[frozenset(schemas)][metric] = []

    METRICS[frozenset(schemas)][metric].append(value)


derived_models = {
    scheme: setup_derived_model_from_args(scheme, tokenizer, model, args) for scheme in args.schemes
}

METRICS = {}
MISC = {
    "model": args.model,
    "sentences": [],
    "sentences_masked": [],
    "position_A": [],
    "position_B": [],
}


pbar = tqdm(iter(dataloader))
with torch.no_grad():
    for batch in pbar:
        lengths = batch.pop("lengths", None)      # noqa
        sentences = batch.pop("sentences")        # noqa
        special_tokens_mask = batch.pop("special_tokens_mask").bool().to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        blocks = batch.pop("tokens_to_mask")
        MISC["sentences"].extend(sentences)

        # Compute joint of blocks under derived model
        position_A, position_B = blocks.split(dim=1, split_size=1)
        MISC["position_A"].extend(position_A.squeeze(1).tolist())
        MISC["position_B"].extend(position_B.squeeze(1).tolist())
        MISC["sentences_masked"].extend([
            tokenizer.decode(
                [t for t in x.squeeze(0).tolist() if t != tokenizer.pad_token_id]
            ) for x in input_ids.scatter(dim=1, index=blocks, value=tokenizer.mask_token_id).split(dim=0, split_size=1)]
        )
        force_A = input_ids.gather(index=position_A, dim=1)
        force_B = input_ids.gather(index=position_B, dim=1)

        log_conditionals = {}
        if whole_vocab:
            bert_logit_A, bert_logit_B = compute_bert_logits_whole_vocab(
                batch, position_A, position_B, tokenizer, model, vocab_batch_size)

            # Compute all log conditionals for this batch--block pair
            for scheme, derived_model in derived_models.items():
                pairwise_log_conditional = derived_model.pairwise_log_conditional_with_logits(
                    batch, position_A.squeeze(1), position_B.squeeze(1), bert_logit_A, bert_logit_B)

                # Compute conditionals of blocks under derived model
                pairwise_log_conditional_A = \
                    pairwise_log_conditional - pairwise_log_conditional.logsumexp(dim=1, keepdim=True)
                pairwise_log_conditional_B = \
                    pairwise_log_conditional - pairwise_log_conditional.logsumexp(dim=2, keepdim=True)

                log_conditionals[scheme] = (pairwise_log_conditional.cpu(),
                                            pairwise_log_conditional_A.cpu(),
                                            pairwise_log_conditional_B.cpu())

                del pairwise_log_conditional, pairwise_log_conditional_A, pairwise_log_conditional_B

            # Construct log conditionals from logits
            bert_log_conditional_A = bert_logit_A.log_softmax(dim=1)
            bert_log_conditional_B = bert_logit_B.log_softmax(dim=2)

            # Clear up memory
            del bert_logit_A, bert_logit_B
        else:
            set_A, set_B = generate_sets(
                batch, position_A, position_B, joint_size_A, joint_size_B, tokenizer, model, force_A, force_B)

            # Check if gold needed to be forced or not (we use this for the harsh variants of the rank metrics)
            set_A_no_force, set_B_no_force = generate_sets(
                batch, position_A, position_B, joint_size_A, joint_size_B, tokenizer, model)

            gold_A_in_unforced_set = []
            for forced, unforced in zip(set_A.tolist(), set_A_no_force.tolist()):
                gold_A_in_unforced_set.append(set(forced) == set(unforced))

            gold_A_in_unforced_set = torch.tensor(gold_A_in_unforced_set)

            gold_B_in_unforced_set = []
            for forced, unforced in zip(set_B.tolist(), set_B_no_force.tolist()):
                gold_B_in_unforced_set.append(set(forced) == set(unforced))

            gold_B_in_unforced_set = torch.tensor(gold_B_in_unforced_set)

            # Compute conditionals of each of the two positions under BERT
            bert_log_conditional_A, bert_log_conditional_B = compute_bert_conditionals_over_set(
                batch, position_A, position_B, set_A, set_B, tokenizer, model)

            # Compute all log conditionals for this batch--block pair
            log_conditionals = {}
            for scheme, derived_model in derived_models.items():
                pairwise_log_conditional = derived_model.pairwise_log_conditional(
                    batch, position_A.squeeze(1), position_B.squeeze(1), set_A, set_B)

                # Compute conditionals of blocks under derived model
                pairwise_log_conditional_A = \
                    pairwise_log_conditional - pairwise_log_conditional.logsumexp(dim=1, keepdim=True)
                pairwise_log_conditional_B = \
                    pairwise_log_conditional - pairwise_log_conditional.logsumexp(dim=2, keepdim=True)
                log_conditionals[scheme] = \
                    pairwise_log_conditional, pairwise_log_conditional_A, pairwise_log_conditional_B

        # COMPUTE METRICS
        # Obtain gold tokens (note that when we are specifying sets, then (0, 0) is _always_ the gold token)
        if whole_vocab:
            gold_A, gold_B = force_A.item(), force_B.item()
        else:
            gold_A, gold_B = 0, 0

        # Compute BERT PPL
        bert_nll_pos_A = bert_log_conditional_A[:, gold_A, gold_B].cpu()
        bert_nll_pos_B = bert_log_conditional_B[:, gold_A, gold_B].cpu()
        append_to_metrics(["mlm-baseline"], "gold_singleton_nll", -(bert_nll_pos_A + bert_nll_pos_B) / 2)
        if args.debug_messages:
            print("bert nll", -(bert_nll_pos_A + bert_nll_pos_B) / 2)

        # Compute compatibility
        compatibility = compute_compatibility(bert_log_conditional_A, bert_log_conditional_B).cpu()
        append_to_metrics(["mlm-baseline"], "compatibility", compatibility)
        if args.debug_messages:
            print("compatibility", compatibility)

        # Compute model-specific metrics (e.g., KL of condtionals to BERT conditonals)
        for scheme in args.schemes:
            pairwise_log_conditional, pairwise_log_conditional_A, pairwise_log_conditional_B = log_conditionals[scheme]
            pairwise_log_conditional = pairwise_log_conditional.to(device)
            pairwise_log_conditional_A = pairwise_log_conditional_A.to(device)
            pairwise_log_conditional_B = pairwise_log_conditional_B.to(device)

            # Compute KLs
            A_kls_fwd = compute_kls(bert_log_conditional_A.transpose(1, 2), pairwise_log_conditional_A.transpose(1, 2)).cpu()  # noqa
            B_kls_fwd = compute_kls(bert_log_conditional_B, pairwise_log_conditional_B).cpu()
            A_kls_rev = compute_kls(pairwise_log_conditional_A.transpose(1, 2), bert_log_conditional_A.transpose(1, 2)).cpu()  # noqa
            B_kls_rev = compute_kls(pairwise_log_conditional_B, bert_log_conditional_B).cpu()
            A_kls_jsd = compute_jsds(bert_log_conditional_A.transpose(1, 2), pairwise_log_conditional_A.transpose(1, 2)).cpu()  # noqa
            B_kls_jsd = compute_jsds(bert_log_conditional_B, pairwise_log_conditional_B).cpu()
            batch_size = A_kls_rev.shape[0]

            # Perform intra-datapoint averaging
            append_to_metrics([scheme], "gold_kls_fwd", torch.stack([A_kls_fwd[:, gold_B], B_kls_fwd[:, gold_A]], dim=1).mean(dim=1))  # noqa
            append_to_metrics([scheme], "all_kls_fwd", torch.cat([A_kls_fwd, B_kls_fwd], dim=1).mean(dim=1))
            append_to_metrics([scheme], "gold_kls_rev", torch.stack([A_kls_rev[:, gold_B], B_kls_rev[:, gold_A]], dim=1).mean(dim=1))  # noqa
            append_to_metrics([scheme], "all_kls_rev", torch.cat([A_kls_rev, B_kls_rev], dim=1).mean(dim=1))
            append_to_metrics([scheme], "gold_kls_jsd", torch.stack([A_kls_jsd[:, gold_B], B_kls_jsd[:, gold_A]], dim=1).mean(dim=1))  # noqa
            append_to_metrics([scheme], "all_kls_jsd", torch.cat([A_kls_jsd, B_kls_jsd], dim=1).mean(dim=1))

            # Compute pairwise NLL of correct sample (used for PPL)
            append_to_metrics([scheme], "gold_pairwise_nll", -pairwise_log_conditional[:, gold_A, gold_B].cpu() / 2)

            # Compute single-position NLL of correct tokens (using gold tokens)
            nll_pos_A = pairwise_log_conditional_A[:, gold_A, gold_B].cpu()
            nll_pos_B = pairwise_log_conditional_B[:, gold_A, gold_B].cpu()
            append_to_metrics([scheme], "gold_singleton_nll", -(nll_pos_A + nll_pos_B) / 2)

            # Compute rank-based metrics
            # Note that these are only computed when we have a restricted joint, or else this is intractable
            if not whole_vocab:
                # Compute ranks
                token_at_A = force_A
                token_at_B = force_B
                ranks = get_rank_in_joint(pairwise_log_conditional, token_at_A, token_at_B, set_A, set_B)

                # Compute metrics
                rank_in_joint = (ranks * (ranks >= 0) + (joint_size_A * joint_size_B) * (ranks < 0)).float()
                reciprocal_rank = torch.nan_to_num(1 / (ranks + 1), posinf=0.)
                recall_at_1 = ((ranks >= 0) & (ranks < 1)).float()
                recall_at_5 = ((ranks >= 0) & (ranks < 5)).float()
                recall_at_10 = ((ranks >= 0) & (ranks < 10)).float()

                # Compute metrics (harsh variants)
                #   For this, we just replace the datapoint-specific metrics above with a penalized value whenever
                #   the golden token would not have appeared in the joint. The replaced values are:
                #       - rank_in_joint: rank set to (joint_size_A * joint_size_B) if forcing was required
                #       - all others: local value set to 0 if forcing was required
                gold_was_forced = ~(gold_A_in_unforced_set & gold_B_in_unforced_set)
                harsh_rank_in_joint = (rank_in_joint * ~gold_was_forced + (joint_size_A * joint_size_B) * gold_was_forced).float()  # noqa
                harsh_reciprocal_rank = reciprocal_rank * ~gold_was_forced
                harsh_recall_at_1 = recall_at_1 * ~gold_was_forced
                harsh_recall_at_5 = recall_at_5 * ~gold_was_forced
                harsh_recall_at_10 = recall_at_10 * ~gold_was_forced

                # Store metrics
                append_to_metrics([scheme], "rank_in_joint", rank_in_joint)
                append_to_metrics([scheme], "reciprocal_rank", reciprocal_rank)
                append_to_metrics([scheme], "recall_at_1", recall_at_1)
                append_to_metrics([scheme], "recall_at_5", recall_at_5)
                append_to_metrics([scheme], "recall_at_10", recall_at_10)

                append_to_metrics([scheme], "harsh_rank_in_joint", harsh_rank_in_joint)
                append_to_metrics([scheme], "harsh_reciprocal_rank", harsh_reciprocal_rank)
                append_to_metrics([scheme], "harsh_recall_at_1", harsh_recall_at_1)
                append_to_metrics([scheme], "harsh_recall_at_5", harsh_recall_at_5)
                append_to_metrics([scheme], "harsh_recall_at_10", harsh_recall_at_10)

            if args.debug_messages:
                print(scheme, -(pairwise_log_conditional[:, gold_A, gold_B].cpu()) / 2, -(nll_pos_A + nll_pos_B) / 2)

            del pairwise_log_conditional, pairwise_log_conditional_A, pairwise_log_conditional_B

        # Compute cross-model metrics (e.g., joint KL)
        if len(args.schemes) >= 2:
            for scheme_a, scheme_b in combinations(args.schemes, 2):
                pairwise_log_conditional_a, _, _ = log_conditionals[scheme_b]
                pairwise_log_conditional_b, _, _ = log_conditionals[scheme_a]
                batch_size = pairwise_log_conditional_a.shape[0]
                pairwise_log_conditional_a = pairwise_log_conditional_a.reshape(batch_size, -1)
                pairwise_log_conditional_b = pairwise_log_conditional_b.reshape(batch_size, -1)

                pairwise_jsd = compute_jsds(pairwise_log_conditional_a, pairwise_log_conditional_b).cpu()

                append_to_metrics([scheme_a, scheme_b], "jsd", pairwise_jsd)

                del pairwise_log_conditional_a, pairwise_log_conditional_b

        del bert_log_conditional_A, bert_log_conditional_B

# Average everything
non_averaged_metrics = {}
averaged_metrics = {}
for schemes, metrics_dict in METRICS.items():
    non_averaged_metrics[schemes] = {k: torch.cat(v, dim=0).cpu() for k, v in metrics_dict.items()}
    averaged_metrics[schemes] = {k: torch.cat(v, dim=0).mean().cpu() for k, v in metrics_dict.items()}

    if "gold_pairwise_nll" in averaged_metrics[schemes]:
        averaged_metrics[schemes]["gold_pairwise_ppl"] = averaged_metrics[schemes]["gold_pairwise_nll"].exp()

    if "gold_singleton_nll" in averaged_metrics[schemes]:
        averaged_metrics[schemes]["gold_singleton_ppl"] = averaged_metrics[schemes]["gold_singleton_nll"].exp()

# Output
pprint(averaged_metrics)

# Save
if args.output_file:
    print(f"Saving to {args.output_file}...")
    with open(args.output_file, "wb") as h:
        pickle.dump({"non-averaged": non_averaged_metrics, "averaged": averaged_metrics, "misc": MISC}, h)
