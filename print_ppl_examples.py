import pickle
from argparse import ArgumentParser
from transformers import AutoTokenizer
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import spacy
from spacy.tokens.doc import Doc
import networkx as nx
import matplotlib.pyplot as plt  # noqa
from tqdm import tqdm
from plotly.subplots import make_subplots

pio.kaleido.scope.mathjax = None

nlp = spacy.load("en_core_web_sm")

parser = ArgumentParser()
parser.add_argument("file")
parser.add_argument("models", nargs="+")
parser.add_argument("tokenizer")
parser.add_argument("--min-diff", type=float)
parser.add_argument("--max-diff", type=float)
parser.add_argument("--distance", choices=["distance", "syntactic_distance"], required=True)
parser.add_argument("--graph-mode", choices=["pnll_diff", "unll_diff", "pnll", "unll"])
parser.add_argument("--average", default=False, action="store_true")
parser.add_argument("--average-min-count", type=int, default=20, help="Minimum number of datapoints to average.")
parser.add_argument("--output-image", type=str)
args = parser.parse_args()

distance_measure = args.distance
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

with open(args.file, "rb") as h:
    data = pickle.load(h)

available_models = [y for x in data["non-averaged"].keys() for y in x if len(x) <= 1 if y != "mlm-baseline"]
print(available_models)
for model in args.models:
    assert model in available_models, f"Available models: {available_models}"


sentences = data["misc"]["sentences"]
sentences_masked = data["misc"]["sentences_masked"]
position_A = data["misc"]["position_A"]
position_B = data["misc"]["position_B"]
models_pnlls, models_unlls = {}, {}

# Compute distance of masked tokens
masked_tokens, distances, syntactic_distances = [], [], []
for sentence, sentence_masked, position_A, position_B in tqdm(list(zip(sentences, sentences_masked, position_A, position_B))):  # noqa
    tokens = tokenizer.tokenize(sentence_masked)
    position_A = position_A - 1
    position_B = position_B - 1

    # COMPUTE TOKEN-DISTANCE
    masked = [idx for idx, tok in enumerate(tokens) if tok == tokenizer.mask_token]
    masked_tokens.append(masked)
    distance = masked[1] - masked[0] - 1
    distances.append(distance)

    # COMPUTE SYNTACTIC (AKA. DEPENDENCY-GRAPH) DISTANCE
    # Map to words (not subtokens)
    words = tokenizer.basic_tokenizer.tokenize(sentence, never_split=tokenizer.all_special_tokens)

    # Map subtokens to word ids
    subtokens_to_words = []
    subtokens = []
    for idx, word in enumerate(words):
        if word in tokenizer.basic_tokenizer.never_split:
            subtokens_to_words.append(idx)
            subtokens.append(word)
        else:
            for subtoken in tokenizer.wordpiece_tokenizer.tokenize(word):
                subtokens_to_words.append(idx)
                subtokens.append(subtoken)

    # Get dependency parse for `words`
    doc = Doc(vocab=nlp.vocab, words=words, sent_starts=[True] + [False] * (len(words) - 1))
    for name, proc in nlp.pipeline:
        doc = proc(doc)

    # Construct graph
    G = nx.Graph()
    for token in doc:
        G.add_node(token.i)
        for child in token.children:
            G.add_edge(token.i, child.i)

    # Get syntactic distance between two masked words
    real_word_A = subtokens_to_words[position_A]
    real_word_B = subtokens_to_words[position_B]
    if real_word_A == real_word_B:
        syntactic_distance = 0
    else:
        try:
            path = nx.shortest_path(G, source=real_word_A, target=real_word_B)
        except:  # noqa
            # nx.draw(G, with_labels=True, labels={token.i: token.text for token in doc})
            # plt.show()
            # import pudb; pudb.set_trace()'
            raise Exception("Could not compute shortest path.")

        syntactic_distance = len(path) - 1

    syntactic_distances.append(syntactic_distance)

df_data = []
for idx, (sentence, sentence_masked, masked_tokens, distance, syntactic_distance) in enumerate(zip(
        sentences, sentences_masked, masked_tokens, distances, syntactic_distances)):
    row = {"sentence": sentence, "sentence_masked": sentence_masked, "masked_A": masked_tokens[0], "masked_B":
           masked_tokens[1], "distance": distance, "syntactic_distance": syntactic_distance}
    for model in available_models:
        key_model = frozenset([model])
        row[f"unll_{model}"] = data["non-averaged"][key_model]["gold_pairwise_nll"][idx].tolist()
        row[f"pnll_{model}"] = data["non-averaged"][key_model]["gold_singleton_nll"][idx].tolist()

    df_data.append(row)

df = pd.DataFrame(df_data)

if len(args.models) == 2:
    # If diff >> 0, this means A_pnll > B_pnll, i.e., the log prob under A << log prob under B. This suggests that
    # model B is a better model for this sentence. The opposite holds if diff << 0.
    print("Diff >> 0 means that log prob under A << log prob under B, so model B thinks this sentence is more likely "
          "than model A.")

    model_A = args.models[0]
    model_B = args.models[1]

    df["pnll_diff"] = df[f"pnll_{model_A}"] - df[f"pnll_{model_B}"]
    df["unll_diff"] = df[f"unll_{model_A}"] - df[f"unll_{model_B}"]
    df = df.sort_values("pnll_diff")

    if args.min_diff is not None:
        df = df[df["pnll_diff"] > args.min_diff]

    if args.max_diff is not None:
        df = df[df["pnll_diff"] < args.max_diff]

    for idx, row in df.iterrows():
        sentence = row["sentence"]
        sentence_masked = row["sentence_masked"]
        dist = row["distance"]
        A_pnll, B_pnll = row[f"pnll_{model_A}"], row[f"pnll_{model_B}"]
        A_unll, B_unll = row[f"unll_{model_A}"], row[f"unll_{model_B}"]
        pnll_diff = row["pnll_diff"]
        unll_diff = row["unll_diff"]
        if args.min_diff is not None and pnll_diff < args.min_diff:
            continue

        if args.max_diff is not None and pnll_diff > args.max_diff:
            continue

        print(
            f"({idx})",
            "A_pnll:", A_pnll, " B_pnll:", B_pnll, f"(PNLL diff: {pnll_diff})",
            "A_unll:", A_unll, " B_unll:", B_unll, f"(UNLL diff: {unll_diff})",
            "dist:", dist
        )
        print("\t", sentence)
        print("\t", sentence_masked)
        print()

print(df.corr()["distance"])

model_labels = {
    "naive": "MLM",
    "mrf-local": r"MRF",
    "mrf": r"MRF (Logit)",
    "iter": "AG",
    "hcb-both": "HCB",
}

model_colors = {
    "naive": 'rgb(31, 119, 180)',
    "mrf-local": 'rgb(255, 127, 14)',  # MRF (regular)
    "mrf": 'rgb(44, 160, 44)',         # MRF (logit)
    "iter": 'rgb(214, 39, 40)',
    "hcb-both": 'rgb(148, 103, 189)',
}

mode_labels = {
    "pnll": "PNLL"
}

distance_labels = {
    "distance": "Token distance",
    "syntactic_distance": "Syntactic distance",
}



mode = "markers"
if args.average:
    mode = "lines+markers"
    if "diff" in args.graph_mode:
        aggregated_metrics = {f"{args.graph_mode}": "mean"}
    else:
        aggregated_metrics = {f"{args.graph_mode}_{model}": ["mean", "std"] for model in args.models}

    df = df.groupby([args.distance], as_index=False).agg({
        "sentence": "count",
        **aggregated_metrics
    })
    df = df.rename(columns={"sentence": "count"})
    df.columns = ['_'.join(column) if "nll" in column[0] else column[0] for column in df.columns]

    if args.average_min_count:
        df = df[df["count"] > args.average_min_count]


if args.output_image:
    # Generate dummy image to remove watermark
    print("Generating dummy image...")
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(args.output_image)

if args.graph_mode is not None:
    if "diff" in args.graph_mode:
        assert len(args.models) == 2, "Can only compute difference graphs if two models are provided."
        fig = px.scatter(df, x=args.distance, y=args.graph_mode, title=args.file)
        fig.show()
    else:
        if args.average:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(x=df[args.distance].values, y=df["count"].values, name="Counts", marker_color='rgb(0,0,0)',
                       opacity=0.2),
                secondary_y=True
            )

            for model in args.models:
                # TODO: Add std using `*_std` column
                fig.add_trace(go.Scatter(
                    x=df[args.distance], y=df[f"{args.graph_mode}_{model}_mean"], mode=mode, name=model_labels[model],
                    marker_color=model_colors[model], marker_size=4, line_width=4,
                    # error_y=dict(type="data", array=df[f"{args.graph_mode}_{model}_std"], visible=True),
                ), secondary_y=False)
        else:
            fig = go.Figure()
            for model in args.models:
                fig.add_trace(go.Scatter(
                    x=df[args.distance], y=df[f"{args.graph_mode}_{model}"], mode=mode, name=model_labels[model],
                    marker_color=model_colors[model], marker_size=4, line_width=4,
                ))

    fig.update_layout(
        # 800 x 500
        autosize=False, width=300, height=300,
        # margin=dict(l=20, r=20, t=20, b=20),
        margin=dict(l=0, r=10, t=0, b=0),
        font_family="Computer Modern",
        font_color="black",
        xaxis_title=distance_labels[args.distance],
        yaxis_title=mode_labels[args.graph_mode] if not args.average else "Average " + mode_labels[args.graph_mode],
        yaxis2=dict(title='Number of examples', overlaying='y', side='right'),
        showlegend=False,
    )

    if args.output_image:
        # fig.write_image(args.output_image, scale=4)
        fig.write_image(args.output_image, scale=4)
    else:
        fig.show()
