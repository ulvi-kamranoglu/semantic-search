"""Korpus, sorğular və qrels arasındakı uyğunluğu yoxlayır."""
import polars as pl
from config import load_config

cfg = load_config()
P, Q, QR = cfg["paths"], cfg["qrels"], cfg["queries"]
thr = cfg["eval"]["relevant_threshold"]
pl.Config.set_tbl_rows(60)
pl.Config.set_fmt_str_lengths(70)

corpus = pl.read_parquet(P["corpus_raw"])
qrels = pl.read_csv(P["qrels"], schema_overrides={Q["query_col"]: pl.Utf8, Q["doc_col"]: pl.Utf8})
queries = pl.read_csv(P["queries"], schema_overrides={QR["id_col"]: pl.Utf8})

print("== Ölçülər ==")
print("korpus:", corpus.shape, "| unikal _id:", corpus["_id"].n_unique())
print("qrels:", qrels.shape, "| sorğular:", queries.shape)

print("\n== Korpusda boş mətn ==")
empty_text = corpus.filter(pl.col("text").is_null() | (pl.col("text").str.strip_chars() == ""))
empty_title = corpus.filter(pl.col("title").is_null() | (pl.col("title").str.strip_chars() == ""))
print("boş text:", empty_text.height, "| boş title:", empty_title.height)

print("\n== score paylanması ==")
print(qrels.group_by(Q["rel_col"]).len().sort(Q["rel_col"]))

print("\n== ID uyğunluğu ==")
missing = qrels.join(corpus.select("_id").unique(), left_on=Q["doc_col"], right_on="_id", how="anti")
print("qrels-də olub korpusda olmayan sənəd sətri:", missing.height)
q_ids, qr_ids = set(queries[QR["id_col"]]), set(qrels[Q["query_col"]])
print("qrels-də olub queries-də olmayan sorğu:", sorted(qr_ids - q_ids))
print("queries-də olub qrels-də olmayan sorğu:", sorted(q_ids - qr_ids))

per_q = (
    qrels.group_by(Q["query_col"])
    .agg(
        pl.len().alias("n_judged"),
        (pl.col(Q["rel_col"]) >= thr).sum().alias("n_relevant"),
        (pl.col(Q["rel_col"]) == 2).sum().alias("n_score2"),
    )
    .join(queries.select(QR["id_col"], QR["text_col"]), left_on=Q["query_col"], right_on=QR["id_col"])
    .with_columns(
        pl.col(QR["text_col"]).str.split(" ").list.len().alias("n_words"),
        (10 / pl.col("n_relevant")).clip(upper_bound=1.0).alias("max_possible_R@10"),
    )
    .sort(pl.col(Q["query_col"]).cast(pl.Int64))
)
print("\n== Hər sorğu üzrə ==")
print(per_q.select(Q["query_col"], "n_judged", "n_relevant", "n_score2", "n_words", "max_possible_R@10"))
print("\n== Xülasə ==")
print(per_q.select("n_judged", "n_relevant", "n_score2", "n_words", "max_possible_R@10").describe())
print("relevantı olmayan sorğu sayı:", per_q.filter(pl.col("n_relevant") == 0).height)

out = f"{P['results']}/qrels_per_query.csv"
per_q.write_csv(out)
print("yazıldı:", out)
