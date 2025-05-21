import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
from swift.utils import write_to_jsonl
from dataset import load_tg_dataset 

raw_edge_template = """Classify the following edge into one of the predefined categories: {category}.
Base your decision on the information given for the two end-points, the relation type, and the time stamp.
If the edge is ambiguous or does not fit clearly, choose the closest match.  Provide only the category name.

Edge description:
  Source node text: {u_text}
  Target node text: {i_text}
  Relation type: {relation_text}
  Time stamp: {timestamp}

Category:"""

query_rag_template = """Classify the following edge into one of the predefined categories: {category}.
Use the provided neighbouring edge descriptions to improve your understanding of context.  Base your decision on the
main theme across the texts and their temporal ordering. If the edge is ambiguous or does not fit clearly, choose the
closest match.  Provide only the category name.

Edge description:
  Source node text: {u_text}
  Target node text: {i_text}
  Relation type: {relation_text}
  Time stamp: {timestamp}

Neighbour edge references:
{references}

Category:"""

label_rag_edge_template = """Classify the following edge into one of the predefined categories: {category}.
Use the supplied labels of neighbouring edges as hints.  Base your decision on the edge description and how those
labels may relate. If the edge is ambiguous or does not fit clearly, choose the closest match.  Provide only the
category name.

Edge description:
  Source node text: {u_text}
  Target node text: {i_text}
  Relation type: {relation_text}
  Time stamp: {timestamp}

Neighbour edge labels:
{references}

Category:"""

few_shot_rag_edge_template = """Classify the following edge into one of the predefined categories: {category}.
Use the neighbouring edge descriptions together with their labels to refine your answer.  Base your decision on the
supplied information and temporal ordering. If the edge is ambiguous or does not fit clearly, choose the closest
match.  Provide only the category name.

Edge description:
  Source node text: {u_text}
  Target node text: {i_text}
  Relation type: {relation_text}
  Time stamp: {timestamp}

Neighbour edge examples:
{references}

Category:"""


def _node_text(node_id: int, entity_df: pd.DataFrame) -> str:
    txt = entity_df.loc[node_id, "text"]
    return txt

def _relation_text(rel_id: int, rel_df: pd.DataFrame) -> str:
    txt = rel_df.loc[rel_id, "text"]
    return txt

def _recent_neighbours(
    edge_row: pd.Series,
    edge_df: pd.DataFrame,
    k: int = 3
) -> pd.DataFrame:
    """Return up to k most recent edges touching either endpoint, with ts < target ts."""
    mask_u = (edge_df["u"] == edge_row.u) | (edge_df["i"] == edge_row.u)
    mask_i = (edge_df["u"] == edge_row.i) | (edge_df["i"] == edge_row.i)
    mask_u_i = (edge_df["u"] != edge_row.u) | (edge_df["i"] != edge_row.i)
    earlier = edge_df["ts"] <= edge_row.ts
    neigh = edge_df.loc[(mask_u | mask_i) & mask_u_i & earlier].sort_values("ts", ascending=False).head(k)
    return neigh

def _format_ref_text(neigh: pd.DataFrame,
                     entity_df: pd.DataFrame,
                     rel_df: pd.DataFrame) -> str:
    lines: List[str] = []
    for i, (_, row) in enumerate(neigh.iterrows()):
        ref = (
            f"({i+1}): time stamp={row.ts}\n"
            f"  Source text: {_node_text(row.u, entity_df)}\n"
            f"  Target text: {_node_text(row.i, entity_df)}\n"
            f"  Relation text: {_relation_text(row.r, rel_df)}\n"
        )
        lines.append(ref)
    return "\n\n".join(lines) if lines else "None"

def _format_ref_labels(neigh: pd.DataFrame) -> str:
    lines: List[str] = []
    for i, (_, row) in enumerate(neigh.iterrows()):
        ref = (
            f"({i+1}): time stamp={row.ts}\n"
            f"  Label: {row.label}"
        )
        lines.append(ref)
    return "\n\n".join(lines) if lines else "None"

def _format_ref_text_labels(neigh: pd.DataFrame,
                            entity_df: pd.DataFrame,
                            rel_df: pd.DataFrame) -> str:
    lines: List[str] = []
    for i, (_, row) in enumerate(neigh.iterrows()):
        ref = (
            f"({i+1}): time stamp={row.ts})\n"
            f"  Source text: {_node_text(row.u, entity_df)}\n"
            f"  Target text: {_node_text(row.i, entity_df)}\n"
            f"  Relation text: {_relation_text(row.r, rel_df)}\n"
            f"  Label: {row.label}"
        )
        lines.append(ref)
    return "\n\n".join(lines) if lines else "None"
    
def make_prompts(
    data: Dict[str, pd.DataFrame],
    category_list: List[str],
    k_neigh: int = 3
) -> pd.DataFrame:
    edge_df = data["edge_list"]
    ent_df  = data["entity_text"]
    rel_df  = data["relation_text"]
    cat_str = ", ".join(category_list)

    # only for test
    test_edge_df = edge_df[edge_df['split'] == 'test'].reset_index(drop=True)
    
    rows: List[Tuple[str, str, str, str, int]] = []  # store prompts

    for _, edge in tqdm(test_edge_df.iterrows(), total=len(test_edge_df)):
        # Core texts
        u_text = _node_text(edge.u, ent_df)
        i_text = _node_text(edge.i, ent_df)
        rel_text = _relation_text(edge.r, rel_df)

        # Gather neighbours
        neigh = _recent_neighbours(edge, edge_df, k=k_neigh)
        refs_text = _format_ref_text(neigh, ent_df, rel_df)
        refs_labels = _format_ref_labels(neigh)
        refs_text_labels = _format_ref_text_labels(neigh, ent_df, rel_df)

        # Fill each template
        p_raw = raw_edge_template.format(
            category=cat_str,
            u_text=u_text,
            i_text=i_text,
            relation_text=rel_text,
            timestamp=edge.ts
        )

        p_query_rag = query_rag_template.format(
            category=cat_str,
            u_text=u_text,
            i_text=i_text,
            relation_text=rel_text,
            timestamp=edge.ts,
            references=refs_text
        )

        p_label_rag = label_rag_edge_template.format(
            category=cat_str,
            u_text=u_text,
            i_text=i_text,
            relation_text=rel_text,
            timestamp=edge.ts,
            references=refs_labels
        )

        p_fewshot = few_shot_rag_edge_template.format(
            category=cat_str,
            u_text=u_text,
            i_text=i_text,
            relation_text=rel_text,
            timestamp=edge.ts,
            references=refs_text_labels
        )


        rows.append((p_raw, p_query_rag, p_label_rag, p_fewshot, edge.idx, edge.label))

    print(p_query_rag)
    print(p_label_rag)
    print(p_fewshot)
    prompt_df = pd.DataFrame(
        rows,
        columns=["raw", "query_rag", "label_rag", "few_shot_rag", "idx", "label"]
    )
    return prompt_df


for name in ['Amazon_movies', 'Enron', 'GDELT', 'Googlemap_CT', 'ICEWS1819', 'Stack_elec', 'Stack_ubuntu', 'Yelp']:
    data = load_tg_dataset(name)
    categories = list(data["edge_list"]["label"].unique())
    prompt_table = make_prompts(data, categories, k_neigh=3)
    
    # Show the first prompt of each type for sanity check
    print(prompt_table.iloc[0]["raw"], "\n")
    print(prompt_table.iloc[0]["query_rag"], "\n")
    print(prompt_table.iloc[0]["label_rag"], "\n")
    print(prompt_table.iloc[0]["few_shot_rag"], "\n")

    prompt_table.to_csv(f"edge_prompts/{name}.csv", index=False)

    response = prompt_table["label"]
    for col in ["raw", "query_rag", "label_rag", "few_shot_rag"]:
        query = df[col].values
        jsonl_data = [dict(query=q, response=r) for q, r in zip(query, response)][:10000]
        write_to_jsonl(f"edge_prompts/{name}_{col}.jsonl", jsonl_data)
    