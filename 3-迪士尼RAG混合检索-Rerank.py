# -*- coding: utf-8 -*-
"""
迪士尼RAG知识库查询（混合检索 + 重排）

召回仍用 FAISS L2 与 BM25 融合的 Hybrid_scores。
文本取 Hybrid 排名前 15 条，再用 qwen3.7-text-rerank 精筛 Top-3 写入提示词。
价目表问句会额外补入来源为图片文件的 OCR 块，避免被其他文档挤出 Top-15。
只有问句要图片时才选图：来源是图片文件、且文件名与问题对得上的，返回归档原图；
多张时按 OCR 文本 Hybrid 排序，最多留 10 张，用 qwen3-vl-rerank 看图选 1 张。
对不上再退回 L2 短名单；价目表问句优先看 price_images 原图向量。
视频在关键词命中后直接返回。
"""
import os
import json
import base64
import numpy as np
import faiss
import jieba
import dashscope
from http import HTTPStatus
from openai import OpenAI
from rank_bm25 import BM25Okapi

# 配置（与构建脚本共用同一套索引文件）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"
TEXT_RERANK_MODEL = "qwen3.7-text-rerank"
VL_RERANK_MODEL = "qwen3-vl-rerank"
INDEX_FILE = os.path.join(BASE_DIR, "disney_index.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "disney_metadata.json")
RANK_PRINT_LIMIT = 20  # 全量混合排名只在控制台展示前若干名

# 关键词配置
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]
# 价目表相关问句：补文本召回、原图短名单优先
PRICE_QUERY_KEYWORDS = ["价格", "价目", "票价", "价目表", "价格表", "邮轮"]
MEDIA_DISTANCE_THRESHOLD = 3.0  # 来源对不上时，图片进入 L2 短名单的距离阈值
IMAGE_SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
FILENAME_OVERLAP_MIN = 2  # 来源文件名与问题至少重合这么多个内容词

# Hybrid = alpha * 向量归一化分数 + (1 - alpha) * BM25 归一化分数
# alpha 越大越偏向语义向量，越小越偏向关键词
HYBRID_ALPHA = 0.5
TEXT_RECALL_LIMIT = 15  # 文本按 Hybrid_scores 召回条数
PRICE_TEXT_BOOST = 5  # 价目表问句额外保证进入重排的图片来源文本条数
IMAGE_RERANK_LIMIT = 10  # 视觉重排短名单上限
# 文本重排按问答检索：能回答问题的段落优先
TEXT_RERANK_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."
# 图片重排看画面是否就是用户要看的那张，而不是文件名是否像
IMAGE_RERANK_INSTRUCT = "Retrieve the image that best matches the user query."


def load_index():
    """加载索引和元数据"""
    if not os.path.isfile(INDEX_FILE) or not os.path.isfile(METADATA_FILE):
        raise ValueError(
            f"错误：找不到索引文件，请先运行 1-迪士尼RAG知识库.py\n"
            f"  {INDEX_FILE}\n  {METADATA_FILE}"
        )
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"已加载索引: {index.ntotal} 条记录")
    return index, metadata


def tokenize(text):
    """中文分词，供 BM25 使用"""
    tokens = []
    for tok in jieba.lcut(text or ""):
        tok = tok.strip()
        if not tok:
            continue
        # 丢掉纯标点，避免关键词分数被符号撑起来
        if all(not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff") for ch in tok):
            continue
        tokens.append(tok)
    return tokens


def build_bm25(metadata):
    """按元数据顺序构建 BM25，下标与 FAISS / metadata 对齐"""
    corpus = [tokenize(item.get("content", "")) for item in metadata]
    bm25 = BM25Okapi(corpus)
    print(f"已构建 BM25 索引: {len(corpus)} 条")
    return bm25


def get_text_embedding(text):
    """文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def distance_to_similarity(distance):
    """L2距离转相似度 (0-1之间，越大越相似)"""
    return 1 / (1 + distance)


def minmax_normalize(scores):
    """把一组分线性缩放到 0-1，供 alpha 加权"""
    scores = np.asarray(scores, dtype=np.float64)
    low = float(scores.min())
    high = float(scores.max())
    if high == low:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def detect_media_intent(query):
    """检测query中是否包含图片/视频意图"""
    query_lower = query.lower()
    want_image = any(kw in query_lower for kw in IMAGE_KEYWORDS)
    want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
    return want_image, want_video


def is_price_query(query):
    """问句是否像在查价目表/邮轮价格"""
    return any(kw in query for kw in PRICE_QUERY_KEYWORDS)


def is_image_file_source(source):
    """来源是否为独立图片文件（价目表 OCR 文本的 source）"""
    return os.path.splitext(source or "")[1].lower() in IMAGE_SOURCE_EXTS


def is_price_image_record(meta):
    """图片记录是否属于价目表归档（原图或表格裁图）"""
    path = (meta.get("path") or "").replace("/", os.sep)
    content = meta.get("content") or ""
    source = meta.get("source") or ""
    if f"{os.sep}price_images{os.sep}" in path:
        return True
    return any(kw in content or kw in source for kw in PRICE_QUERY_KEYWORDS)


def content_tokens(text):
    """去掉图片意图词和单字，留下能对上文件名的内容词"""
    blocked = set(IMAGE_KEYWORDS)
    tokens = []
    for tok in tokenize(text):
        if tok in blocked or len(tok) < 2 or tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def filename_overlap(source, query):
    """来源文件名与问题重合的内容词"""
    name = os.path.splitext(os.path.basename(source))[0]
    return set(content_tokens(name)) & set(content_tokens(query))


def find_original_image(source, metadata):
    """按图片文件名找到归档目录里的 original 原图"""
    stem = os.path.splitext(os.path.basename(source))[0]
    ext = os.path.splitext(source)[1].lower()
    needle = f"__{stem}{os.sep}"
    marker = f"{os.sep}mineru_output{os.sep}"
    original_tail = f"{os.sep}__{stem}{os.sep}original{ext}"

    for item in metadata:
        if item.get("type") != "image":
            continue
        path = (item.get("path") or "").replace("/", os.sep)

        # 原图已向量化入库时，path 直接指向 original.ext
        if path.endswith(original_tail) or path.endswith(f"{os.sep}original{ext}"):
            if needle in path and os.path.isfile(path):
                return path

        if needle not in path or marker not in path:
            continue
        archive_dir = path.split(marker)[0]
        original = os.path.join(archive_dir, "original" + ext)
        if os.path.isfile(original):
            return original
    return None


def collect_source_originals(query, results, metadata):
    """
    文本来源是图片文件、且文件名对得上问题时，收成最多 10 张原图。
    多张时按该来源 OCR 文本的 Hybrid 分排序，便于日期/晚数区分 9 张价目表。
    """
    best = {}
    overlaps = {}
    for record in results:
        meta = record["metadata"]
        if meta.get("type") != "text":
            continue
        source = meta.get("source") or ""
        if not is_image_file_source(source):
            continue
        overlap = filename_overlap(source, query)
        if len(overlap) < FILENAME_OVERLAP_MIN:
            continue
        previous = best.get(source)
        if previous is None or record["hybrid_score"] > previous["hybrid_score"]:
            best[source] = record
            overlaps[source] = overlap

    ordered = sorted(best.values(), key=lambda item: item["hybrid_score"], reverse=True)
    shortlist = []
    for record in ordered:
        source = record["metadata"]["source"]
        path = find_original_image(source, metadata)
        if not path:
            continue
        shortlist.append({
            "metadata": {
                "path": path,
                "source": source,
                "type": "image",
                "content": source,
            },
            "hybrid_score": record["hybrid_score"],
            "distance": record.get("distance"),
            "overlap": overlaps[source],
        })
        if len(shortlist) >= IMAGE_RERANK_LIMIT:
            break
    return shortlist


def build_text_recall(query, results):
    """Hybrid 文本 Top-N；价目表问句再保证若干图片来源 OCR 块进入重排"""
    text_all = [r for r in results if r["metadata"]["type"] == "text"]
    recall = text_all[:TEXT_RECALL_LIMIT]
    if not is_price_query(query):
        return recall

    existing = {r["idx"] for r in recall}
    boosted = 0
    for record in text_all:
        if boosted >= PRICE_TEXT_BOOST:
            break
        if record["idx"] in existing:
            continue
        if not is_image_file_source(record["metadata"].get("source") or ""):
            continue
        recall.append(record)
        existing.add(record["idx"])
        boosted += 1
    if boosted:
        print(f"  价目表文本补召回 {boosted} 条（来源为图片文件的 OCR 块）")
    return recall


def build_image_l2_shortlist(query, results):
    """来源文件名未命中时的图片短名单；价目表问句优先 price_images，并按 Hybrid 排序"""
    candidates = [
        r for r in results
        if r["metadata"]["type"] == "image" and r["distance"] < MEDIA_DISTANCE_THRESHOLD
    ]
    if is_price_query(query):
        price_first = [r for r in candidates if is_price_image_record(r["metadata"])]
        others = [r for r in candidates if not is_price_image_record(r["metadata"])]
        price_first.sort(key=lambda x: x["hybrid_score"], reverse=True)
        others.sort(key=lambda x: x["distance"])
        shortlist = (price_first + others)[:IMAGE_RERANK_LIMIT]
    else:
        candidates.sort(key=lambda x: x["distance"])
        shortlist = candidates[:IMAGE_RERANK_LIMIT]
    return shortlist


def image_to_data_uri(path):
    """本地图片转成 qwen3-vl-rerank 接受的 Base64 Data URI"""
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".webp": "webp",
        ".bmp": "bmp",
        ".gif": "gif",
        ".tif": "tiff",
        ".tiff": "tiff",
    }[ext]
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def rerank_texts(query, candidates, top_n):
    """用 qwen3.7-text-rerank 对文本召回短名单精筛"""
    documents = [item["metadata"]["content"] for item in candidates]
    resp = dashscope.TextReRank.call(
        model=TEXT_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n,
        instruct=TEXT_RERANK_INSTRUCT,
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本重排失败: {resp.message}")

    ranked = []
    for item in resp.output["results"]:
        src = candidates[item["index"]]
        ranked.append({**src, "rerank_score": float(item["relevance_score"])})
    return ranked


def rerank_images(query, candidates):
    """用 qwen3-vl-rerank 看图，从短名单里选 1 张"""
    documents = []
    for item in candidates:
        documents.append({"image": image_to_data_uri(item["metadata"]["path"])})
    resp = dashscope.TextReRank.call(
        model=VL_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=1,
        instruct=IMAGE_RERANK_INSTRUCT,
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片重排失败: {resp.message}")

    best = resp.output["results"][0]
    chosen = candidates[best["index"]]
    return {**chosen, "rerank_score": float(best["relevance_score"])}


def search_with_details(query, index, metadata, bm25, alpha=HYBRID_ALPHA):
    """向量 L2 + BM25 混合检索，并打印 Hybrid_scores 详情"""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"混合检索: Hybrid_scores = {alpha:.2f} * 向量归一化 + {1 - alpha:.2f} * BM25归一化")
    print("向量分数来自 L2 距离: 1 / (1 + L2)，再做 min-max")
    print("=" * 60)

    query_vec = np.array([get_text_embedding(query)]).astype("float32")
    # 检索全部，供后续按类型筛选图片/视频；FAISS 索引为 IndexFlatL2
    distances, indices = index.search(query_vec, index.ntotal)

    n = len(metadata)
    l2_distances = np.full(n, np.inf, dtype=np.float64)
    vector_sims = np.zeros(n, dtype=np.float64)
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        l2_distances[idx] = float(dist)
        vector_sims[idx] = distance_to_similarity(float(dist))

    bm25_raw = np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float64)
    vector_norm = minmax_normalize(vector_sims)
    bm25_norm = minmax_normalize(bm25_raw)
    hybrid_scores = alpha * vector_norm + (1.0 - alpha) * bm25_norm

    print(f"\nHybrid_scores 排名 (越大越相关，仅展示前 {RANK_PRINT_LIMIT} 名):")
    print("-" * 108)
    print(f"{'排名':4s} {'ID':4s} {'类型':6s} {'Hybrid':8s} {'向量':8s} {'BM25':8s} {'L2距离':8s} 内容")
    print("-" * 108)

    results = []
    order = np.argsort(-hybrid_scores, kind="stable")
    for rank, idx in enumerate(order):
        m = metadata[int(idx)]
        dist = float(l2_distances[idx])
        sim = float(vector_sims[idx])
        content_preview = m["content"][:45].replace("\n", " ")
        type_tag = m["type"]

        marker = ""
        if type_tag == "image":
            marker = " <-- 图片"
        elif type_tag == "video":
            marker = " <-- 视频"

        if rank < RANK_PRINT_LIMIT:
            dist_text = f"{dist:8.4f}" if np.isfinite(dist) else "     inf"
            print(
                f"{rank + 1:4d} {int(idx):4d} [{type_tag:5s}] "
                f"{float(hybrid_scores[idx]):6.4f}  {float(vector_norm[idx]):6.4f}  "
                f"{float(bm25_norm[idx]):6.4f}  {dist_text}  {content_preview}...{marker}"
            )
        results.append({
            "idx": int(idx),
            "distance": dist,
            "similarity": sim,
            "vector_norm": float(vector_norm[idx]),
            "bm25": float(bm25_norm[idx]),
            "hybrid_score": float(hybrid_scores[idx]),
            "metadata": m,
        })

    return results


def rag_ask(query, index, metadata, bm25, k=3, alpha=HYBRID_ALPHA):
    """RAG问答。文本 Hybrid Top-15 后重排取 Top-k；图片视觉重排；视频关键词直接返回"""
    results = search_with_details(query, index, metadata, bm25, alpha=alpha)

    # 检测媒体意图
    want_image, want_video = detect_media_intent(query)
    print(f"\n意图检测: 需要图片={want_image}, 需要视频={want_video}")

    # 文本召回：Hybrid Top-15；价目表问句再补图片来源 OCR 块
    text_recall = build_text_recall(query, results)
    print(f"\n文本召回 Top-{len(text_recall)} (按 Hybrid_scores，送入 {TEXT_RERANK_MODEL}):")
    for i, r in enumerate(text_recall):
        print(f"  {i + 1:2d}. {r['metadata']['content'][:50]}... (Hybrid: {r['hybrid_score']:.4f})")

    top_results = rerank_texts(query, text_recall, top_n=k)
    print(f"\n文本精筛 Top-{len(top_results)} (按 {TEXT_RERANK_MODEL}):")
    for r in top_results:
        print(
            f"  - {r['metadata']['content'][:50]}... "
            f"(Rerank: {r['rerank_score']:.4f}, Hybrid: {r['hybrid_score']:.4f})"
        )

    # 要图片时：先用来源文件名对上原图，多张按 OCR Hybrid 排序后再视觉重排
    matched_image = None
    if want_image:
        source_shortlist = collect_source_originals(query, results, metadata)
        if source_shortlist:
            print(f"\n按来源文件名命中原图 {len(source_shortlist)} 张 (按 OCR Hybrid 排序):")
            for record in source_shortlist:
                words = "、".join(sorted(record["overlap"]))
                print(
                    f"  - {record['metadata']['source']} -> {record['metadata']['path']} "
                    f"(重合: {words}, Hybrid: {record['hybrid_score']:.4f})"
                )
            if len(source_shortlist) == 1:
                matched_image = source_shortlist[0]
                print(f"  -> 直接返回原图: {matched_image['metadata']['path']}")
            else:
                matched_image = rerank_images(query, source_shortlist)
                print(
                    f"  -> 视觉重排选中原图: {matched_image['metadata']['path']} "
                    f"(Rerank: {matched_image['rerank_score']:.4f})"
                )
        else:
            image_shortlist = build_image_l2_shortlist(query, results)
            print(f"\n来源文件名未命中，改用 L2 短名单 {len(image_shortlist)} 张 (L2 < {MEDIA_DISTANCE_THRESHOLD}):")
            for record in image_shortlist:
                print(
                    f"  - {record['metadata']['path']} "
                    f"(L2距离: {record['distance']:.4f}, Hybrid: {record['hybrid_score']:.4f})"
                )
            if image_shortlist:
                matched_image = rerank_images(query, image_shortlist)
                print(
                    f"  -> 视觉重排选中: {matched_image['metadata']['path']} "
                    f"(Rerank: {matched_image['rerank_score']:.4f}, "
                    f"L2距离: {matched_image['distance']:.4f})"
                )

    # 视频只有关键词命中才返回，不再按距离或视觉模型筛选
    matched_video = None
    if want_video:
        video_results = [r for r in results if r["metadata"]["type"] == "video"]
        if video_results:
            matched_video = video_results[0]
            print(f"  -> 关键词命中视频: {matched_video['metadata']['url']}")

    # 构建context
    context_str = ""
    for i, r in enumerate(top_results):
        m = r["metadata"]
        context_str += (
            f"背景知识 {i + 1} (来源: {m['source']}, Rerank: {r['rerank_score']:.4f}):\n"
            f"{m['content']}\n\n"
        )

    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识回答用户问题。

[背景知识]
{context_str}
[用户问题]
{query}
"""

    # 调用LLM
    print("\n调用LLM生成答案...")
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": "你是一个迪士尼客服助手。"},
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message.content

    # 附加匹配到的媒体
    if matched_image:
        answer += f"\n\n[相关图片]: {matched_image['metadata']['path']}"
    if matched_video:
        answer += f"\n\n[相关视频]: {matched_video['metadata']['url']}"

    print(f"\n最终答案:\n{answer}")
    return answer


if __name__ == "__main__":
    index, metadata = load_index()
    bm25 = build_bm25(metadata)

    print("\n" + "=" * 60)
    print("测试1: 文本查询 - 门票退款")
    rag_ask("我想了解一下迪士尼门票的退款流程", index, metadata, bm25, k=3)

    print("\n" + "=" * 60)
    print("测试2: 文本查询 - 带小朋友游玩")
    rag_ask("带小朋友去上海迪士尼可以玩哪些项目，有没有身高限制", index, metadata, bm25, k=3)

    print("\n" + "=" * 60)
    print("测试3: 文本查询 - 年票规则")
    rag_ask("迪士尼乐园年票怎么用，有哪些限制", index, metadata, bm25, k=3)

    print("\n" + "=" * 60)
    print("测试4: 文本查询 - 客诉升级")
    rag_ask("客人投诉服务态度很差，什么时候需要升级给宾客关系部", index, metadata, bm25, k=3)

    print("\n" + "=" * 60)
    print("测试5: 价目表文本 - 按日期房型问价（无图片意图，测 OCR 前缀与 Hybrid）")
    rag_ask(
        "2026年3月23日出发新加坡航线，迪士尼邮轮标准内舱双人间第一二人价和总价分别是多少",
        index, metadata, bm25, k=3
    )

    print("\n" + "=" * 60)
    print("测试6: 价目表图片 - 按晚数要原图（测 9 张表 Hybrid 排序与视觉重排）")
    rag_ask(
        "把迪士尼邮轮6天5晚新加坡那张价格表图片发给我看看",
        index, metadata, bm25, k=3
    )

    print("\n" + "=" * 60)
    print("测试7: 图片查询 - 邮轮价目表（泛问）")
    rag_ask("看看迪士尼邮轮价格表的图片", index, metadata, bm25, k=3)

    print("\n" + "=" * 60)
    print("测试8: 图片查询 - 园区地图")
    rag_ask("迪士尼乐园园区地图长什么样，有没有相关图片", index, metadata, bm25, k=3)

    print("\n" + "=" * 60)
    print("测试9: 视频查询 - 第一次去迪士尼")
    rag_ask("第一次去迪士尼有没有避坑攻略视频可以播放吗", index, metadata, bm25, k=3)
