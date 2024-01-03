import torch
from more_itertools import chunked, padded
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import gc
from enum import Enum
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from einops import rearrange


NEG_INF = -1e20


class MaskingMode(Enum):
    """
    Unary masking means that we should mask each position, one at a time, to obtain the unary potentials.
    We then use those to propose which elements go into the joint.
    """
    UNARY = "unary"

    """
    Pair masking means that we should mask both positions at once and get the (independent) marginals by BERT.
    We then use those independent marginals to propose which elements go into the joint.
    """
    PAIR = "pair"


def tokenize_and_create_mask(tokenizer, chunk, words_to_mask):
    """
    Tokenizes a sentence so it is ready to be consumed by the model and produces a mask that masks out all subtokens
    associated with the words in `words_to_mask`
    """
    batch_size = len(chunk)
    encoded = tokenizer.batch_encode_plus(chunk, truncation=True, padding="longest", is_split_into_words=True)

    word_ids = torch.tensor([
        [w if w is not None else -1 for w in encoded.word_ids(i)] for i in range(batch_size)
    ])

    indices = torch.tensor(words_to_mask).unsqueeze(1)
    mask = (word_ids == indices).tolist()

    return encoded, mask


def prepare_batches(batch, tokenizer, words_field, A_field, B_field, include_unprepared = True):
    encoded, mask_A = tokenize_and_create_mask(tokenizer, batch[words_field], batch[A_field])
    _, mask_B = tokenize_and_create_mask(tokenizer, batch[words_field], batch[B_field])

    result = {
        **encoded,
        "mask_A": mask_A,
        "mask_B": mask_B,
    }

    if include_unprepared:
        result = {**batch, **result}

    return result


def collate_fn(data):
    """
    Collates all tensors by stacking, and non-tensor columns are "stacked" by creating an outer list to keep them.
    """
    collatable = []
    uncollatable = []
    uncollatable_columns = []

    for dp in data:
        dp_collatable = {k: v for k, v in dp.items() if isinstance(v, torch.Tensor)}
        dp_uncollatable = {k: v for k, v in dp.items() if not isinstance(v, torch.Tensor)}
        uncollatable_columns.extend(list(dp_uncollatable.keys()))
        collatable.append(dp_collatable)
        uncollatable.append(dp_uncollatable)

    torch_collated = torch.utils.data.default_collate(collatable)
    nontorch_collated = {}

    for col in uncollatable_columns:
        nontorch_collated[col] = [dp[col] for dp in uncollatable]

    return {**torch_collated, **nontorch_collated}


def collate_and_pad_fn(data, tokenizer):
    # Separate data into data that needs to be padded and data that does not need to be padded
    data_normal = []
    data_to_pad = []
    fields_to_pad = tokenizer.model_input_names + ["special_tokens_mask"]
    for dp in data:
        data_to_pad.append({k: dp.pop(k) for k in fields_to_pad})
        data_normal.append(dp)

    # Pad and collate data_to_pad. This is taken exactly from transformers.DataCollatorWithPadding
    batch = tokenizer.pad(
        data_to_pad,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )
    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    if "label_ids" in batch:
        batch["labels"] = batch["label_ids"]
        del batch["label_ids"]

    # Collate rest and return
    return {
        **batch,
        **collate_fn(data_normal)
    }


def prepare_batch_for_input(batch, model, prefix):
    """
    From a batch containing multiple tensorized inputs, select the inputs according to the prefix `prefix`
    """
    keys_to_keep = ["input_ids", "attention_mask", "token_type_ids"]

    keys_to_keep = [f"{prefix}_{k}" for k in keys_to_keep]
    return {k[len(prefix) + 1:]: v for k, v in batch.items() if k in keys_to_keep}


def loop_data_loader(data_loader):
    """
    Takes a data loader and transforms it into an infinite iterable
    """
    while True:
        epoch_iterator = iter(data_loader)
        for batch in epoch_iterator:
            yield batch


def generate_mlm_mask(pad_mask, num_tokens_to_mask):
    """
    Given a boolean tensor of dimension B x T, and a tensor with the number of tokens to mask in each data point,
    generates a mask by picking those positions in each data point uniformly at random, ensuring that positions
    corresponding to PAD tokens are ignored.

    Note that, because of this, the entries in num_tokens_to_mask has to be smaller than the number of non-PAD tokens
    in each data point.
    """
    batch_size, max_timesteps = pad_mask.shape

    # Randomly permute non-pad tokens an use this to create a mask for each datapoint
    ranked_indices = torch.argsort(torch.rand_like(pad_mask.float()) + -200 * pad_mask, dim=-1, descending=True)
    index_matrix = torch.arange(max_timesteps, device=pad_mask.device).unsqueeze(0).repeat(batch_size, 1)
    masked_tokens_mask = index_matrix < num_tokens_to_mask
    indices_to_mask = masked_tokens_mask * ranked_indices + ~masked_tokens_mask * ranked_indices[:, 0].unsqueeze(1)
    masking_mask = torch.zeros_like(pad_mask)
    masking_mask = masking_mask.scatter(dim=1, index=indices_to_mask, value=1)
    return masking_mask


def compute_log_probability_given_mask(tokenizer, model, batch, mask, drop_keys=None, normalize=True):
    """
    Given an already subtokenized input `batch` and a binary mask, this function [MASK]s out everything in
    `mask` pushes it through `model`, returning the log probabilities.
    """
    batch = batch.copy()
    input_ids = batch.pop("input_ids")
    input_ids_masked = (~mask * input_ids) + (mask * tokenizer.mask_token_id)
    inputs = {
        "input_ids": input_ids_masked,
        **batch
    }

    if drop_keys is not None:
        for dk in drop_keys:
            inputs.pop(dk)

    output = model(**inputs)

    if normalize:
        log_probs = output.logits.log_softmax(-1)
    else:
        log_probs = output.logits

    return log_probs


def compute_mlm_energy(batch, model, tokenizer, normalize_logits=False):
    # MLM energy is negative sum of position logits or log probs
    batch = batch.copy()
    input_ids = batch["input_ids"]
    selectable_positions = ~(batch.pop("special_tokens_mask").bool())

    log_probs = torch.zeros((input_ids.shape[0], input_ids.shape[1], tokenizer.vocab_size), device=input_ids.device)
    for t in range(input_ids.shape[1]):
        mask = torch.zeros_like(selectable_positions).bool()
        mask[:, t] = selectable_positions[:, t]
        log_probs[:, t] = compute_log_probability_given_mask(
            tokenizer, model, batch, mask, normalize=normalize_logits)[:, t]

    position_log_energies = log_probs.gather(dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
    position_log_energies = position_log_energies * selectable_positions
    return -position_log_energies.sum(dim=1)


@torch.no_grad()
def regenerate_masked_input_ids(tokenizer, model, batch, mask, strategy="sample", force_distinct=True):
    """
    Given an already subtokenized input `batch`, regenerates all the tokens that are `mask`ed and returns
    a new `input_ids` tensor where all the tokens in `mask` are replaced with their regenerated versions
    with the appropriate token_ids according to the tokenizer in `model`

    There are three possible regeneration strategies:
    1. sample: Resamples using the output log probabilities
    2. argmax: Picks the most likely output under model
    3. random: Samples a subtoken at random
    """
    # Fill in masked tokens
    if strategy == "random":
        tokens = torch.randint(
            high=tokenizer.vocab_size, size=(batch["input_ids"].shape[0],), device=batch["input_ids"].device)
        all_regenerated = tokens[:, None] * mask
    else:
        log_probs = compute_log_probability_given_mask(tokenizer, model, batch, mask)

        if force_distinct:
            # Forces the regenerated subtokens to be different from the original one
            neg_infty = -1e20
            all_ids = torch.arange(log_probs.shape[-1], device=batch["input_ids"].device).view(1, 1, -1)
            log_probs += (all_ids == batch["input_ids"].unsqueeze(2)) * neg_infty

        if strategy == "argmax":
            all_regenerated = log_probs.argmax(-1)
        elif strategy == "sample":
            all_regenerated = torch.distributions.Categorical(logits=log_probs).sample()
        else:
            raise Exception("Invalid regeneration strategy")

    return (~mask * batch["input_ids"]) + (mask * all_regenerated)


def batched_index(tensor, index):
    """
    Performs a gather along the second dimension. Basically, for every datapoint (i.e., dim 0) indexes along second
    dimension (i.e., dim 1) using the indices in `index`.

    Hence, if tensor is shape (B, T, D), and index is shape (B,) the output will be shape (B, D)
    """
    assert len(index) == tensor.shape[0]
    splits = [x.squeeze(0)[i] for i, x in zip(index.tolist(), tensor.split(dim=0, split_size=1))]
    return torch.stack(splits, dim=0)


def generate_nli_split(mode):
    assert mode in ["mlm", "clm"]

    # For the MultiNLI data, we split the training set (i.e., sentence pairs) into train/val/test split (with 99%/1%/1%)
    # of the data, and then re-split the training set into halves. We use the first half for MLM training.
    dataset = load_dataset("multi_nli")
    dataset = dataset.shuffle(seed=0)

    dataset_train_validtest = dataset["train"].train_test_split(test_size=0.01, shuffle=False, seed=0)
    dataset_valid_test = dataset_train_validtest["test"].train_test_split(test_size=0.5, shuffle=False, seed=0)

    if mode == "mlm":
        train_set_range = range(0, len(dataset_train_validtest["train"]) // 2)
    elif mode == "clm":
        train_set_range = range(len(dataset_train_validtest["train"]) // 2, len(dataset_train_validtest["train"]))

    # For NLI dataset, we use mismatched as the "test" set
    return DatasetDict({
        "train": dataset_train_validtest["train"].select(train_set_range),
        "validation": dataset_valid_test["train"],
        "test": dataset_valid_test["test"],
    })


def generate_snli_split(mode):
    assert mode in ["mlm", "clm"]

    # For the MultiNLI data, we split the training set (i.e., sentence pairs) into train/val/test split (with 99%/1%/1%)
    # of the data, and then re-split the training set into halves. We use the first half for MLM training.
    dataset = load_dataset("snli")
    dataset = dataset.shuffle(seed=0)

    dataset_train_validtest = dataset["train"].train_test_split(test_size=0.01, shuffle=False, seed=0)
    dataset_valid_test = dataset_train_validtest["test"].train_test_split(test_size=0.5, shuffle=False, seed=0)

    if mode == "mlm":
        train_set_range = range(0, len(dataset_train_validtest["train"]) // 2)
    elif mode == "clm":
        train_set_range = range(len(dataset_train_validtest["train"]) // 2, len(dataset_train_validtest["train"]))

    # For NLI dataset, we use mismatched as the "test" set
    return DatasetDict({
        "train": dataset_train_validtest["train"].select(train_set_range),
        "validation": dataset_valid_test["train"],
        "test": dataset_valid_test["test"],
    })


def generate_xsum_split():
    dataset = load_dataset("xsum")
    dataset = dataset.shuffle(seed=0)
    return dataset


def generate_bookcorpus_split():
    dataset = load_dataset("bookcorpus", split="train")
    dataset = dataset.shuffle(seed=0)

    # We don't need the whole dataset, so we use only the first 2M examples
    dataset = dataset.select(range(0, 2000000))

    dataset_train_validtest = dataset.train_test_split(test_size=0.005, shuffle=False, seed=0)
    dataset_valid_test = dataset_train_validtest["test"].train_test_split(test_size=0.5, shuffle=False, seed=0)
    dataset = DatasetDict({
        "train": dataset_train_validtest["train"],
        "validation": dataset_valid_test["train"],
        "test": dataset_valid_test["test"],
    })

    return dataset


def generate_wikipedia_split(tokenizer, preprocessing_batch_size, preprocessing_num_threads, generation: bool = False):
    dataset = load_dataset("wikipedia", "20220301.en", split="train")
    dataset = dataset.shuffle(seed=0)

    # We skip the first 150K examples since these are used for the compatibility evaluation
    if generation:
        # Ensures we use a disjoint subset of the data for evaluation purposes
        dataset = dataset.select(range(0, 150000))
    else:
        dataset = dataset.select(range(150000, len(dataset)))
        dataset = dataset.select(range(0, 400000))

    dataset = dataset.map(pretokenize_wikipedia_data, fn_kwargs=dict(tokenizer=tokenizer), batched=True,
                          batch_size=preprocessing_batch_size, num_proc=preprocessing_num_threads)

    dataset_train_validtest = dataset.train_test_split(test_size=0.001, shuffle=False, seed=0)
    dataset_valid_test = dataset_train_validtest["test"].train_test_split(test_size=0.5, shuffle=False, seed=0)
    dataset = DatasetDict({
        "train": dataset_train_validtest["train"],
        "validation": dataset_valid_test["train"],
        "test": dataset_valid_test["test"],
    })

    return dataset


def prepare_nli_data_for_mlm(batch, tokenizer=None):
    """
    Merges both text columns and computes the length of each datapoint in subtokens. Lengths can be used for sorting
    datapoints into compact batches.
    """
    result = {
        "sentences": batch["premise"] + batch["hypothesis"],
    }

    if tokenizer is not None:
        premise_lens = [len(tokens) for tokens in tokenizer(batch["premise"])["input_ids"]]
        hypothesis_lens = [len(tokens) for tokens in tokenizer(batch["hypothesis"])["input_ids"]]
        result["lengths"] = premise_lens + hypothesis_lens

    return result


def prepare_summarization_data_for_mlm(batch, tokenizer=None):
    """
    Renames the "summary" column and drop everything else.
    """
    return {
        "sentences": batch["summary"],
    }


def prepare_wikipedia_data_for_mlm(batch, tokenizer, chunk_size = 512, min_length = -1):
    """
    Takes documents and chunks data into segments of (at least) `chunk_size` tokens.
    Note that we do this by splitting on spaces, so in practice we might lose some data, but this shouldn't
    affect things much.
    """
    chunked_sentences = [chunked(sent.split(" "), chunk_size) for sent in batch["text"]]
    chunked_sentences = [
        " ".join(list(chunk))
        for chunked_sentence in chunked_sentences
        for chunk in chunked_sentence
        if len(chunk) > min_length
    ]

    return {
        "sentences": chunked_sentences,
    }


def pretokenize_wikipedia_data(batch, tokenizer):
    return {
        "text_tokenized": [tokenizer.tokenize(s) for s in batch["text"]],
    }


def chunk_wikipedia_data(batch, tokenizer, chunk_size: int = 512, min_chunk_size: int = 0):
    return {
        "sentences": [
            tokenizer.decode(tokenizer.convert_tokens_to_ids(chunk)) for sentence in batch["text_tokenized"]
            for chunk in chunked(sentence, chunk_size) if len(chunk) > min_chunk_size
        ]
    }


def prepare_bookcorpus_data_for_mlm(batch, tokenizer=None):
    result = {
        "sentences": batch["text"],
    }

    return result


def tokenize_data(batch, tokenizer, return_special_tokens_mask=False):
    """
    Tokenizes batches with appropriate amount of padding, and truncates long sequences if necessary
    """
    return tokenizer(batch["sentences"], truncation=True, return_special_tokens_mask=return_special_tokens_mask)


def prepare_masks(batch, contiguous: bool):
    special_tokens_mask = []
    max_length = max([len(x) for x in batch["special_tokens_mask"]])
    for mask in batch["special_tokens_mask"]:
        special_tokens_mask.append(list(padded(mask, fillvalue=1, n=max_length)))

    special_tokens_mask = torch.tensor(special_tokens_mask).bool()

    if contiguous:
        valid_positions = ~special_tokens_mask

        maskable_first_positions = torch.zeros_like(special_tokens_mask).bool()
        maskable_first_positions[:, :-1] = valid_positions[:, :-1] & valid_positions[:, 1:]
        maskable_first_positions[:, 1] = True

        first_tokens_to_mask = torch.multinomial(maskable_first_positions.float(), 1)
        second_tokens_to_mask = first_tokens_to_mask + 1
        blocks = torch.cat([first_tokens_to_mask, second_tokens_to_mask], dim=1)
    else:
        blocks = torch.multinomial((~special_tokens_mask).float(), 2)

    return {
        "tokens_to_mask": blocks.tolist()
    }


@torch.no_grad()
def generate_unary_sets(batch, position_A, position_B, size_set_A, size_set_B, tokenizer, model,
                        force_A=None, force_B=None):
    """
    Generates sets A (i.e., A_i) and B (i.e., A_j), with tokens that should be considered when constructing the joint
    over positions i and j, by masking out each position one at a time and obtaining the top size_set_* from each
    unary conditional.

    If force_A and/or force_B are provided, those elements will appear at the beginning of the supplied sets
    """
    assert size_set_A >= 1 and size_set_B >= 1, "Set size must be positive."

    batch = batch.copy()
    input_ids = batch.pop("input_ids")

    # Get logits at A (when only A is masked)
    input_ids_masked_A = input_ids.scatter(index=position_A, dim=1, value=tokenizer.mask_token_id)
    logits_A_mask_A = batched_index(model(input_ids=input_ids_masked_A, **batch).logits, position_A.squeeze(1))

    # Get logits at B (when only B is masked)
    input_ids_masked_B = input_ids.scatter(index=position_B, dim=1, value=tokenizer.mask_token_id)
    logits_B_mask_B = batched_index(model(input_ids=input_ids_masked_B, **batch).logits, position_B.squeeze(1))

    # Mask out forced tokens
    if force_A is not None:
        size_set_A = size_set_A - force_A.shape[1]
        logits_A_mask_A = logits_A_mask_A.scatter(index=force_A, dim=1, value=NEG_INF)

    if force_B is not None:
        size_set_B = size_set_B - force_B.shape[1]
        logits_B_mask_B = logits_B_mask_B.scatter(index=force_B, dim=1, value=NEG_INF)

    # :: Construct set A
    set_A = logits_A_mask_A.argsort(dim=1, descending=True)[:, :size_set_A]
    assert set_A.shape[1] == size_set_A

    # :: Construct set B
    set_B = logits_B_mask_B.argsort(dim=1, descending=True)[:, :size_set_B]
    assert set_B.shape[1] == size_set_B

    if force_A is not None:
        set_A = torch.cat([force_A, set_A], dim=1)

    if force_B is not None:
        set_B = torch.cat([force_B, set_B], dim=1)

    return set_A, set_B


@torch.no_grad()
def generate_pair_sets(batch, position_A, position_B, size_set_A, size_set_B, tokenizer, model,
                       force_A=None, force_B=None):
    """
    Generates sets A (i.e., A_i) and B (i.e., A_j), with tokens that should be considered when constructing the joint
    over positions i and j, by masking out both positions at once and obtaining the top size_set_* from each
    of the independent BERT conditionals.

    If force_A and/or force_B are provided, those elements will appear at the beginning of the supplied sets
    """
    assert size_set_A >= 1 and size_set_B >= 1, "Set size must be positive."

    batch = batch.copy()
    input_ids = batch.pop("input_ids")

    # Get logits when A and B are masked
    input_ids_masked_AB = input_ids.scatter(
        index=torch.cat([position_A, position_B], dim=1), dim=1, value=tokenizer.mask_token_id)
    outputs_logits_mask_AB = model(input_ids=input_ids_masked_AB, **batch).logits
    logits_A_mask_AB = batched_index(outputs_logits_mask_AB, position_A.squeeze(1))
    logits_B_mask_AB = batched_index(outputs_logits_mask_AB, position_B.squeeze(1))

    if force_A is not None:
        size_set_A = size_set_A - force_A.shape[1]
        logits_A_mask_AB = logits_A_mask_AB.scatter(index=force_A, dim=1, value=NEG_INF)

    if force_B is not None:
        size_set_B = size_set_B - force_B.shape[1]
        logits_B_mask_AB = logits_B_mask_AB.scatter(index=force_B, dim=1, value=NEG_INF)

    # :: Construct set A
    set_A = logits_A_mask_AB.argsort(dim=1, descending=True)[:, :size_set_A]
    assert set_A.shape[1] == size_set_A

    # :: Construct set B
    set_B = logits_B_mask_AB.argsort(dim=1, descending=True)[:, :size_set_B]
    assert set_B.shape[1] == size_set_B

    if force_A is not None:
        set_A = torch.cat([force_A, set_A], dim=1)

    if force_B is not None:
        set_B = torch.cat([force_B, set_B], dim=1)

    return set_A, set_B


def generate_sets_from_logits(logits, size_set, force=None):
    # Mask out forced tokens
    if force is not None:
        size_set = size_set - force.shape[1]
        logits_masked = logits.scatter(index=force, dim=1, value=NEG_INF)

    # :: Construct set
    final_set = logits_masked.argsort(dim=1, descending=True)[:, :size_set]
    assert final_set.shape[1] == size_set

    if force is not None:
        final_set = torch.cat([force, final_set], dim=1)

    return final_set


@torch.no_grad()
def generate_sets(batch, position_A, position_B, size_set_A, size_set_B, tokenizer, model, force_A=None, force_B=None):
    """
    Generates sets A (i.e., A_i) and B (i.e., A_j), with tokens that should be considered when constructing the joint
    over positions i and j.

    force_A and force_B should be (B, F) tensors of candidates that _must_ appear on the returned sets.
    """
    assert size_set_A >= 1 and size_set_B >= 1, "Set size must be positive."

    batch = batch.copy()
    input_ids = batch.pop("input_ids")

    # Get logits when A and B are masked
    input_ids_masked_AB = input_ids.scatter(
        index=torch.cat([position_A, position_B], dim=1), dim=1, value=tokenizer.mask_token_id)
    outputs_logits_mask_AB = model(input_ids=input_ids_masked_AB, **batch).logits
    logits_A_mask_AB = batched_index(outputs_logits_mask_AB, position_A.squeeze(1))
    logits_B_mask_AB = batched_index(outputs_logits_mask_AB, position_B.squeeze(1))

    # Get logits at A (when only A is masked)
    input_ids_masked_A = input_ids.scatter(index=position_A, dim=1, value=tokenizer.mask_token_id)
    logits_A_mask_A = batched_index(model(input_ids=input_ids_masked_A, **batch).logits, position_A.squeeze(1))

    # Get logits at B (when only B is masked)
    input_ids_masked_B = input_ids.scatter(index=position_B, dim=1, value=tokenizer.mask_token_id)
    logits_B_mask_B = batched_index(model(input_ids=input_ids_masked_B, **batch).logits, position_B.squeeze(1))

    # :: Construct set A
    # Add elements from p(x_i | x_{-i, -j}, x_j)
    if force_A is not None:
        set_A = force_A

        # Ensure we don't select elements that are forced to be in the set already
        set_A_to_add = logits_A_mask_A.scatter(index=set_A, dim=1, value=NEG_INF).argsort(
            dim=1, descending=True)[:, :(size_set_A - set_A.shape[1]) // 2]
        set_A = torch.cat([set_A, set_A_to_add], dim=1)
    else:
        set_A = logits_A_mask_A.argsort(dim=1, descending=True)[:, :size_set_A // 2]

    # Add elements from \sum_j p(x_i, x_j | x_{-i, -j})
    set_A_to_add = logits_A_mask_AB.scatter(index=set_A, dim=1, value=NEG_INF).argsort(
        dim=1, descending=True)[:, :size_set_A - set_A.shape[1]]
    set_A = torch.cat([set_A, set_A_to_add], dim=1)

    assert set_A.shape[1] == size_set_A

    # :: Construct set B
    if force_B is not None:
        set_B = force_B

        # Add elements from p(x_i | x_{-i, -j}, x_j)
        set_B_to_add = logits_B_mask_B.scatter(index=set_B, dim=1, value=NEG_INF).argsort(
            dim=1, descending=True)[:, :(size_set_B - set_B.shape[1]) // 2]
        set_B = torch.cat([set_B, set_B_to_add], dim=1)
    else:
        set_B = logits_B_mask_B.argsort(dim=1, descending=True)[:, :size_set_B // 2]

    # Add elements from \sum_j p(x_i, x_j | x_{-i, -j})
    set_B_to_add = logits_B_mask_AB.scatter(index=set_B, dim=1, value=NEG_INF).argsort(
        dim=1, descending=True)[:, :size_set_B - set_B.shape[1]]
    set_B = torch.cat([set_B, set_B_to_add], dim=1)

    assert set_B.shape[1] == size_set_B

    return set_A, set_B


@torch.no_grad()
def compute_bert_conditionals_over_set(batch, position_A, position_B, set_A, set_B, tokenizer, model, prune=True):
    batch = batch.copy()
    input_ids = batch.pop("input_ids")

    input_ids_masked_AB = torch.scatter(
        input_ids, index=torch.cat([position_A, position_B], dim=1), dim=1, value=tokenizer.mask_token_id)

    bert_log_conditional_A = []
    for replacements in set_B.split(dim=1, split_size=1):
        input_ids_masked_A = input_ids_masked_AB.scatter(index=position_B, src=replacements, dim=1)
        output_logits = batched_index(model(input_ids=input_ids_masked_A, **batch).logits, position_A.squeeze(1))
        if prune:
            selected_output_logits = output_logits.gather(index=set_A, dim=1)
            bert_log_conditional_A.append(selected_output_logits.log_softmax(dim=1))
        else:
            selected_output_logits = output_logits
            bert_log_conditional_A.append(selected_output_logits)

    bert_log_conditional_A = torch.stack(bert_log_conditional_A, dim=2)

    bert_log_conditional_B = []
    for replacements in set_A.split(dim=1, split_size=1):
        input_ids_masked_B = input_ids_masked_AB.scatter(index=position_A, src=replacements, dim=1)
        output_logits = batched_index(model(input_ids=input_ids_masked_B, **batch).logits, position_B.squeeze(1))
        if prune:
            selected_output_logits = output_logits.gather(index=set_B, dim=1)
            bert_log_conditional_B.append(selected_output_logits.log_softmax(dim=1))
        else:
            selected_output_logits = output_logits
            bert_log_conditional_B.append(selected_output_logits)

    bert_log_conditional_B = torch.stack(bert_log_conditional_B, dim=1)

    return bert_log_conditional_A, bert_log_conditional_B


@torch.no_grad()
def compute_bert_logits_whole_vocab(datapoint, position_A, position_B, tokenizer, model, batch_size):
    assert datapoint["input_ids"].shape[0] == 1, "Cannot consume a batch when collecting whole vocabulary."
    position_A, position_B = position_A.squeeze().item(), position_B.squeeze().item()
    datapoint = datapoint.copy()
    input_ids = datapoint.pop("input_ids")

    # Mask both positions
    input_ids_both_masked = input_ids.clone()
    input_ids_both_masked[0, position_A] = tokenizer.mask_token_id
    input_ids_both_masked[0, position_B] = tokenizer.mask_token_id
    device = input_ids_both_masked.device

    # Compute logits when masking position A
    logits_A = []
    for token_ids in tqdm(list(chunked(range(tokenizer.vocab_size), n=batch_size)), leave=False):
        batch = {k: v.expand(len(token_ids), -1) for k, v in datapoint.items()}
        input_ids_replaced = input_ids_both_masked.repeat(len(token_ids), 1)
        input_ids_replaced[:, position_B] = torch.tensor(token_ids, device=device)
        logits = model(input_ids=input_ids_replaced, **batch).logits[:, position_A].cpu()
        logits_A.append(logits)

    logits_A = torch.cat(logits_A, dim=0).transpose(0, 1).to(device)
    assert logits_A.shape == (tokenizer.vocab_size, tokenizer.vocab_size)

    # Compute logits when masking position B
    logits_B = []
    for token_ids in tqdm(list(chunked(range(tokenizer.vocab_size), n=batch_size)), leave=False):
        batch = {k: v.expand(len(token_ids), -1) for k, v in datapoint.items()}
        input_ids_replaced = input_ids_both_masked.repeat(len(token_ids), 1)
        input_ids_replaced[:, position_A] = torch.tensor(token_ids, device=device)
        logits = model(input_ids=input_ids_replaced, **batch).logits[:, position_B].cpu()
        logits_B.append(logits)

    logits_B = torch.cat(logits_B, dim=0).to(device)
    assert logits_A.shape == logits_B.shape

    return logits_A.unsqueeze(0), logits_B.unsqueeze(0)


def compute_kls(log_ps, log_qs):
    return (log_ps.exp() * (log_ps - log_qs)).sum(dim=-1)


def compute_jsds(log_ps, log_qs):
    """
    Same signature as compute_kls, but computes Jensen-Shannon divergence
    """
    log_ms = (0.5 * (log_ps.exp() + log_qs.exp())).log()
    return 0.5 * (compute_kls(log_ps, log_ms) + compute_kls(log_qs, log_ms))


def compute_compatibility(log_A, log_B):
    """
    Using (unnormalized) BERT logits, computes incompatibility between conditionals according to the Arnold & Press
    compatibility condition.
    """
    batch_size, size_A, size_B = log_A.shape

    log_C_ij = log_A - log_B
    log_C_dotdot = log_C_ij.reshape(batch_size, -1).logsumexp(dim=1)[:, None, None]
    log_C_idot = log_C_ij.logsumexp(dim=2, keepdim=True)
    log_C_dotj = log_C_ij.logsumexp(dim=1, keepdim=True)

    diffs = (log_C_ij + log_C_dotdot - log_C_idot - log_C_dotj) ** 2
    return diffs.reshape(batch_size, -1).sum(dim=1) / (size_A * size_B)


def debug_gpu_memory(label=None, enabled=True, threshold=20000):
    """
    Prints out information about all tensors that have one dimension that is at least of size `threshold`. This is
    meant to identify possible GPU memory leaks (i.e., tensors that are not released).

    Example usage: debug_gpu_memory("GPU usage after creating BERT logit tensors")
    """
    if enabled:
        if label:
            print(f"================ {label} =================")
        else:
            print("=================================")

        i = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    if max(obj.size()) > threshold:
                        print(i, type(obj), obj.size(), obj.device)
                        i += 1
            except:  # noqa
                pass

        print()
        print()


def setup_derived_model(scheme, tokenizer, model, iter_steps, compatibility_file, device):
    if scheme == "mrf":
        from conditional_model import MRFConditionalModel
        derived_model = MRFConditionalModel(tokenizer=tokenizer, model=model)
    elif scheme == "mrf-local":
        from conditional_model import MRFConditionalModel
        derived_model = MRFConditionalModel(tokenizer=tokenizer, model=model, normalize_logits=True)
    elif "hcb" in scheme:
        from conditional_model import HCBConditionalModel
        pivot_mode = scheme.split('-')[1]
        derived_model = HCBConditionalModel(tokenizer=tokenizer, model=model, pivot_mode=pivot_mode)
    elif scheme == "iter":
        from conditional_model import IterConditionalModel
        derived_model = IterConditionalModel(tokenizer=tokenizer, model=model, iterations=iter_steps)
    elif scheme == "naive":
        from conditional_model import NaiveConditionalModel
        derived_model = NaiveConditionalModel(tokenizer=tokenizer, model=model)
    else:
        raise NotImplementedError("Unsupported scheme.")

    return derived_model


def setup_derived_model_from_args(scheme, tokenizer, model, args):
    return setup_derived_model(scheme, tokenizer, model, **{
        "iter_steps": args.iter_steps,
        "compatibility_file": None,
        "device": 0 if args.use_gpu else "cpu",
    })


def add_model_cli_args(parser):
    parser.add_argument("--iter-steps", type=int, default=40, help="Number of iterations in 'iter' method.")


def tokenize_sentence(sentence, tokenizer):
    if tokenizer is None:
        # If no tokenizer is supplied, we use the NLTK default
        return word_tokenize(sentence)
    else:
        return tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))


def prepare_sentence_n_gram(sentence, tokenizer, N):
    tokenized = tokenize_sentence(sentence, tokenizer)
    bos_token, eos_token = "[BOS]", "[EOS]"
    if tokenizer is not None:
        bos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token

    sentence_ngrams = ngrams(
        tokenized, N, pad_left=True, pad_right=True, left_pad_symbol=bos_token, right_pad_symbol=eos_token
    )
    return list(sentence_ngrams)


@torch.no_grad()
def get_rank_in_joint(log_joint, token_at_A, token_at_B, set_A, set_B):
    batch_size = log_joint.shape[0]
    joint_size_B = log_joint.shape[2]

    match_A = (set_A == token_at_A).sum(dim=1)
    match_B = (set_B == token_at_B).sum(dim=1)
    assert torch.all(match_A <= 1) and torch.all(match_B <= 1)

    ranks = []
    for idx, individual_joint in enumerate(log_joint.split(dim=0, split_size=1)):
        if match_A[idx].bool() and match_B[idx].bool():
            correct_A = token_at_A[idx].item()
            correct_B = token_at_B[idx].item()
            correct_index_in_set_A = set_A[idx].tolist().index(correct_A)
            correct_index_in_set_B = set_B[idx].tolist().index(correct_B)
            correct_index_flattened = joint_size_B * correct_index_in_set_A + correct_index_in_set_B

            individual_joint = rearrange(individual_joint, "1 sA sB -> (sA sB)")
            all_ranks = individual_joint.argsort(descending=True)
            correct_rank = all_ranks.tolist().index(correct_index_flattened)
            ranks.append(correct_rank)
        else:
            ranks.append(-1)

    assert len(ranks) == batch_size
    ranks = torch.tensor(ranks)

    return ranks
