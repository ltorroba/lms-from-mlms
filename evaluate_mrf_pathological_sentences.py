from conditional_model import MRFConditionalModel, NaiveConditionalModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser
from util import compute_bert_logits_whole_vocab
import torch


parser = ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--use-gpu", action="store_true", default=False)
parser.add_argument("--batch-size", type=int, default=256)
args = parser.parse_args()


device = 0 if args.use_gpu else "cpu"
vocab_batch_size = args.batch_size

model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model.eval()


model_mrf = MRFConditionalModel(tokenizer=tokenizer, model=model)
model_mrf_l = MRFConditionalModel(tokenizer=tokenizer, model=model, normalize_logits=True)
model_mlm = NaiveConditionalModel(tokenizer=tokenizer, model=model)

sentences = ["The man is at the casino."]
positions = [[2, 3]]


def unflatten_index(index):
    index_A = torch.div(index,  tokenizer.vocab_size, rounding_mode="floor")
    index_B = index - index_A * tokenizer.vocab_size
    return index_A, index_B


def get_top_candidates(pairwise_conditional, top_k):
    assert pairwise_conditional.shape[0] == 1, "Batching not supported"
    flattened_conditional = pairwise_conditional.reshape(-1)
    return unflatten_index(flattened_conditional.topk(top_k).indices)


for sentence, (position_A, position_B) in zip(sentences, positions):
    print("Sentence:", sentence)

    batch = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = batch["input_ids"]

    position_A = torch.tensor([position_A], device=device)
    position_B = torch.tensor([position_B], device=device)
    correct_A = input_ids[0, position_A][0]
    correct_B = input_ids[0, position_B][0]

    masked_ids = input_ids.scatter(index=torch.tensor([[position_A, position_B]], device=device), dim=1, value=tokenizer.mask_token_id)  # noqa
    print("Masked sentence:", tokenizer.convert_ids_to_tokens(masked_ids[0]))

    # Compute unary logits under MLM
    bert_logit_A, bert_logit_B = compute_bert_logits_whole_vocab(
        batch, position_A, position_B, tokenizer, model, vocab_batch_size)

    # Compute pairwise conditionals under models
    pairwise_log_mrf = model_mrf.pairwise_log_conditional_with_logits(
        batch, position_A, position_B, bert_logit_A, bert_logit_B
    )
    pairwise_log_mrf_A = pairwise_log_mrf - pairwise_log_mrf.logsumexp(dim=1, keepdim=True)
    pairwise_log_mrf_B = pairwise_log_mrf - pairwise_log_mrf.logsumexp(dim=2, keepdim=True)

    pairwise_log_mrf_l = model_mrf_l.pairwise_log_conditional_with_logits(
        batch, position_A, position_B, bert_logit_A, bert_logit_B
    )
    pairwise_log_mrf_l_A = pairwise_log_mrf_l - pairwise_log_mrf_l.logsumexp(dim=1, keepdim=True)
    pairwise_log_mrf_l_B = pairwise_log_mrf_l - pairwise_log_mrf_l.logsumexp(dim=2, keepdim=True)

    pairwise_log_mlm = model_mlm.pairwise_log_conditional_with_logits(
        batch, position_A, position_B, bert_logit_A, bert_logit_B
    )
    pairwise_log_mlm_A = pairwise_log_mlm - pairwise_log_mlm.logsumexp(dim=1, keepdim=True)
    pairwise_log_mlm_B = pairwise_log_mlm - pairwise_log_mlm.logsumexp(dim=2, keepdim=True)

    # Get top candidates
    # top_k = 5
    # print(get_top_candidates(pairwise_log_mlm, top_k)) # OOM; try on collie?

    # Compute PNLL
    pnll_mrf = -pairwise_log_mrf[0, correct_A, correct_B]
    pnll_mrf_l = -pairwise_log_mrf_l[0, correct_A, correct_B]
    pnll_mlm = -pairwise_log_mlm[0, correct_A, correct_B]

    print("\tMRF PNLL:", pnll_mrf.item())
    print("\tMRF-L PNLL:", pnll_mrf_l.item())
    print("\tMLM PNLL:", pnll_mlm.item())
    print()

    # Compute UNLL
    unll_mrf = -0.5 * (pairwise_log_mrf_A[0, correct_A, correct_B] + pairwise_log_mrf_B[0, correct_A, correct_B])
    unll_mrf_l = -0.5 * (pairwise_log_mrf_l_A[0, correct_A, correct_B] + pairwise_log_mrf_l_B[0, correct_A, correct_B])
    unll_mlm = -0.5 * (pairwise_log_mlm_A[0, correct_A, correct_B] + pairwise_log_mlm_B[0, correct_A, correct_B])

    print("\tMRF UNLL:", unll_mrf.item())
    print("\tMRF-L UNLL:", unll_mrf_l.item())
    print("\tMLM UNLL:", unll_mlm.item())
    print()

    # Print out top items under conditionals
    top_k = 5

    # MRF
    top_k_mrf_A = pairwise_log_mrf_A[0, :, correct_B].topk(top_k)
    top_k_mrf_B = pairwise_log_mrf_B[0, correct_A, :].topk(top_k)
    print(f"\tp^{{MRF}}(.|{tokenizer.convert_ids_to_tokens([correct_B])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_mrf_A.indices[i]])}:", top_k_mrf_A.values[i].exp().item())
    print()

    print(f"\tp^{{MRF}}(.|{tokenizer.convert_ids_to_tokens([correct_A])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_mrf_B.indices[i]])}:", top_k_mrf_B.values[i].exp().item())
    print()

    # MRF-L
    top_k_mrf_l_A = pairwise_log_mrf_l_A[0, :, correct_B].topk(top_k)
    top_k_mrf_l_B = pairwise_log_mrf_l_B[0, correct_A, :].topk(top_k)
    print(f"\tp^{{MRF-L}}(.|{tokenizer.convert_ids_to_tokens([correct_B])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_mrf_l_A.indices[i]])}:", top_k_mrf_l_A.values[i].exp().item())  # noqa
    print()

    print(f"\tp^{{MRF-L}}(.|{tokenizer.convert_ids_to_tokens([correct_A])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_mrf_l_B.indices[i]])}:", top_k_mrf_l_B.values[i].exp().item())  # noqa
    print()

    # MLM
    top_k_mlm_A = pairwise_log_mlm_A[0, :, correct_B].topk(top_k)
    top_k_mlm_B = pairwise_log_mlm_B[0, correct_A, :].topk(top_k)
    print(f"\tp^{{MLM}}(.|{tokenizer.convert_ids_to_tokens([correct_B])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_mlm_A.indices[i]])}:", top_k_mlm_A.values[i].exp().item())
    print()

    print(f"\tp^{{MLM}}(.|{tokenizer.convert_ids_to_tokens([correct_A])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_mlm_B.indices[i]])}:", top_k_mlm_B.values[i].exp().item())
    print()

    # BERT unary
    log_unary_A = bert_logit_A - bert_logit_A.logsumexp(dim=1, keepdim=True)  # When masking position A
    log_unary_B = bert_logit_B - bert_logit_B.logsumexp(dim=2, keepdim=True)  # When masking position B

    top_k_unary_A = log_unary_A[0, :, correct_B].topk(top_k)  # We get a distribution over A
    top_k_unary_B = log_unary_B[0, correct_A, :].topk(top_k)  # We get a distribution over B
    print(f"\tp^{{Unary}}(.|{tokenizer.convert_ids_to_tokens([correct_B])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_unary_A.indices[i]])}:", top_k_unary_A.values[i].exp().item())  # noqa
    print(f"\t{tokenizer.convert_ids_to_tokens([correct_A])}:", log_unary_A[0, correct_A, correct_B].exp().item())  # noqa
    print()

    print(f"\tp^{{Unary}}(.|{tokenizer.convert_ids_to_tokens([correct_A])})")
    for i in range(top_k):
        print(f"\t\t{tokenizer.convert_ids_to_tokens([top_k_unary_B.indices[i]])}:", top_k_unary_B.values[i].exp().item())  # noqa
    print(f"\t{tokenizer.convert_ids_to_tokens([correct_B])}:", log_unary_B[0, correct_A, correct_B].exp().item())  # noqa
    print()

    import pudb; pudb.set_trace()
