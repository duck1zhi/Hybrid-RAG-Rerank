# -*- coding: utf-8 -*-
"""
迪士尼RAG知识库构建（MinerU4 统一解析 + 索引入库）

源文件目录：项目根目录下的「迪士尼RAG知识库」
全部文档/图片先经本机 MinerU 4 解析，产物写入项目根目录 parsed_archive，再做 embedding。
PDF 与图片走 standard 高质量解析；DOC/DOCX/PPT/PPTX 走 MinerU 原生 Office 解析。
"""
import os
import io
import re
import time
import json
import base64
import hashlib
import shutil
import zipfile
import requests
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from docx import Document as DocxDocument

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "迪士尼RAG知识库")
ARCHIVE_DIR = os.path.join(BASE_DIR, "parsed_archive")
PARSE_LOG_FILE = os.path.join(ARCHIVE_DIR, "parse_log.txt")
FAILED_LIST_FILE = os.path.join(ARCHIVE_DIR, "failed_files.txt")
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

# 输出文件（与查询脚本共用）
INDEX_FILE = os.path.join(BASE_DIR, "disney_index.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "disney_metadata.json")

# 切分参数
CHUNK_SIZE = 500  # 每个chunk的字符数
CHUNK_OVERLAP = 50  # chunk之间的重叠字符数

# MinerU 本机 API
MINERU_API_BASE = os.getenv("MINERU_API_BASE", "http://127.0.0.1:8000")
MINERU_TIER = "standard"
MINERU_POLL_INTERVAL = 2
MINERU_POLL_TIMEOUT = 1800
MINERU_PENDING_STUCK = 180  # 仅 queued 超过该秒数，视为队列无进展
MINERU_IN_PROGRESS = {"queued", "running", "waiting-file", "pending", "converting"}
MINERU_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".jp2": "image/jp2",
}


class MinerUQueueStuck(Exception):
    """MinerU 任务长时间停在排队，未进入 running。"""
    pass

# 价目表截图文件名关键词
PRICE_KEYWORDS = ["价格", "价目", "票价", "价目表", "价格表"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".jp2"}
PPT_EXTS = {".ppt", ".pptx"}
OFFICE_EXTS = {".doc", ".docx", ".ppt", ".pptx"}

# 视频知识库
VIDEO_KNOWLEDGE = [
    {
        "url": "https://desiney.oss-cn-beijing.aliyuncs.com/%E7%AC%AC%E4%B8%80%E6%AC%A1%E5%8E%BB%E8%BF%AA%E5%A3%AB%E5%B0%BC%E7%9A%84%E9%81%BF%E5%9D%91%E6%94%BB%E7%95%A5%E2%80%BC%EF%B8%8F.mp4",
        "description": "第一次去迪士尼的避坑攻略"
    }
]


def parse_docx(file_path):
    """解析 DOCX 文件，提取全部文本"""
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        # Word 文档里可能混入注释/处理指令，它们的 tag 不是字符串，直接跳过
        tag = element.tag
        if not isinstance(tag, str):
            continue
        if tag.endswith('p'):
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif tag.endswith('tbl'):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按固定长度切分文本"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def get_text_embedding(text):
    """文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_image_embedding(image_path):
    """图片embedding；同路径下已有 .npy 则直接复用"""
    cache_path = image_path + ".npy"
    if os.path.isfile(cache_path):
        return np.load(cache_path).astype(np.float32).tolist()

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    image_data = f"data:image/{ext};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'image': image_data}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.message}")
    embedding = resp.output['embeddings'][0]['embedding']
    np.save(cache_path, np.array(embedding, dtype=np.float32))
    return embedding


def get_video_embedding(video_url):
    """视频embedding（多帧取平均）"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'video': video_url}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.message}")

    embeddings = resp.output['embeddings']
    if len(embeddings) > 1:
        vectors = [np.array(e['embedding']) for e in embeddings]
        return np.mean(vectors, axis=0).tolist()
    return embeddings[0]['embedding']


def sanitize_name(name):
    """去掉Windows路径非法字符，用作存档目录名"""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name)
    cleaned = cleaned.strip(" .")
    return cleaned or "unnamed"


def make_archive_subdir(category, rel_path):
    """按相对路径生成存档子目录，避免重名覆盖"""
    rel_no_ext = os.path.splitext(rel_path)[0]
    flat = sanitize_name(rel_no_ext.replace("\\", "__").replace("/", "__"))
    return os.path.join(ARCHIVE_DIR, category, flat)


def is_price_image(filename):
    """文件名是否像价目表截图"""
    return any(keyword in filename for keyword in PRICE_KEYWORDS)


def mineru_headers():
    """MinerU 本机 JSON 请求头"""
    return {"Content-Type": "application/json"}


def mineru_mime_type(file_path):
    """按扩展名给出 MinerU 上传用的 MIME"""
    ext = os.path.splitext(file_path)[1].lower()
    return MINERU_MIME_TYPES.get(ext, "application/octet-stream")


def mineru_job_tier(file_path):
    """PDF/图片用 standard；Office 原生解析只能走 flash"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in OFFICE_EXTS:
        return "flash"
    return MINERU_TIER


def mineru_job_ocr_mode(file_path):
    """Office 原生解析用 auto；PDF 与图片强制 OCR"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in OFFICE_EXTS:
        return "auto"
    return "ocr"


def mineru_raise_http(action, resp):
    """把 MinerU HTTP 错误转成可读异常"""
    text = resp.text
    try:
        payload = resp.json()
        err = payload.get("error") or payload
        text = json.dumps(err, ensure_ascii=False)
    except Exception:
        pass
    raise Exception(f"{action}失败 HTTP {resp.status_code}: {text}")


def mineru_abs_url(url):
    """把本机返回的相对路径拼成可请求地址"""
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return MINERU_API_BASE.rstrip("/") + "/" + url.lstrip("/")


def mineru_download_file(file_id):
    """下载本机解析产物（markdown / zip）"""
    url = f"{MINERU_API_BASE}/v1/files/{file_id}/content"
    resp = requests.get(url, timeout=300, allow_redirects=True)
    if resp.status_code != 200:
        mineru_raise_http("MinerU下载产物", resp)
    return resp.content


def mineru_parse_local_file(file_path, archive_dir, data_id):
    """
    调用本机 MinerU V1 API：创建上传 -> PUT 内容 -> 完成上传 -> 提交解析任务 -> 轮询 -> 存档。
    返回 (markdown文本, 抽出的图片路径列表)
    """
    os.makedirs(archive_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()
    upload_name = f"{data_id}{ext}"
    file_size = os.path.getsize(file_path)
    mime_type = mineru_mime_type(file_path)

    health = requests.get(f"{MINERU_API_BASE}/v1/health", timeout=10)
    if health.status_code != 200:
        mineru_raise_http("MinerU健康检查", health)

    apply_resp = requests.post(
        f"{MINERU_API_BASE}/v1/uploads",
        headers=mineru_headers(),
        json={
            "filename": upload_name,
            "bytes": file_size,
            "mime_type": mime_type,
            "purpose": "parse"
        },
        timeout=60
    )
    if apply_resp.status_code != 200:
        mineru_raise_http("MinerU创建上传", apply_resp)
    upload_info = apply_resp.json()
    upload_id = upload_info["id"]
    print(f"    MinerU upload_id={upload_id}，开始上传")

    with open(file_path, "rb") as f:
        file_bytes = f.read()
    upload_url = mineru_abs_url(upload_info.get("upload_url"))
    if upload_url:
        put_headers = dict(upload_info.get("upload_headers") or {})
        if "Content-Type" not in put_headers:
            put_headers["Content-Type"] = "application/octet-stream"
        upload_resp = requests.put(upload_url, data=file_bytes, headers=put_headers, timeout=600)
    else:
        upload_resp = requests.put(
            f"{MINERU_API_BASE}/v1/uploads/{upload_id}/content",
            data=file_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600
        )
    if upload_resp.status_code not in (200, 201, 204):
        mineru_raise_http("MinerU文件上传", upload_resp)

    complete_resp = requests.post(
        f"{MINERU_API_BASE}/v1/uploads/{upload_id}/complete",
        headers=mineru_headers(),
        json={},
        timeout=60
    )
    if complete_resp.status_code != 200:
        mineru_raise_http("MinerU完成上传", complete_resp)
    complete_info = complete_resp.json()
    file_obj = complete_info.get("file") or {}
    file_id = file_obj.get("id")
    if not file_id:
        raise Exception(f"MinerU完成上传后没有 file_id: {complete_info}")
    print("    上传完成，提交解析任务")

    file_entry = {
        "source": {"type": "file_id", "file_id": file_id}
    }
    # 图片不支持 page_range，只有 PDF 才传
    if ext == ".pdf":
        file_entry["page_range"] = "all"
    job_resp = requests.post(
        f"{MINERU_API_BASE}/v1/parse/jobs",
        headers=mineru_headers(),
        json={
            "files": [file_entry],
            "tier": mineru_job_tier(file_path),
            "ocr_mode": mineru_job_ocr_mode(file_path),
            "output_formats": ["markdown", "zip"]
        },
        timeout=60
    )
    if job_resp.status_code not in (200, 202):
        mineru_raise_http("MinerU创建解析任务", job_resp)
    job_info = job_resp.json()
    job_id = job_info["job_id"]
    print(f"    MinerU job_id={job_id}，等待解析")

    started = time.time()
    pending_since = None
    result_item = None
    while True:
        if time.time() - started > MINERU_POLL_TIMEOUT:
            raise Exception(f"MinerU解析超时: {os.path.basename(file_path)}")

        poll_resp = requests.get(f"{MINERU_API_BASE}/v1/parse/jobs/{job_id}", timeout=30)
        if poll_resp.status_code != 200:
            mineru_raise_http("MinerU查询任务", poll_resp)
        poll_result = poll_resp.json()
        state = poll_result.get("status", "")
        files = poll_result.get("files") or []
        item = files[0] if files else {}
        file_state = item.get("status") or state

        if state in {"completed", "partial"} and file_state == "completed":
            result_item = item
            break
        if state in {"failed", "canceled"} or file_state == "failed":
            err = item.get("error") or poll_result
            raise Exception(f"MinerU解析失败: {err}")
        if state not in MINERU_IN_PROGRESS and file_state not in MINERU_IN_PROGRESS:
            raise Exception(f"MinerU未知任务状态: {state}/{file_state}")

        if state == "queued":
            if pending_since is None:
                pending_since = time.time()
            elif time.time() - pending_since > MINERU_PENDING_STUCK:
                raise MinerUQueueStuck(
                    f"MinerU队列无进展(state={state}): {os.path.basename(file_path)}"
                )
        else:
            pending_since = None

        progress = poll_result.get("progress") or {}
        completed = progress.get("completed")
        total = progress.get("total")
        if completed is not None and total is not None:
            print(f"    解析中: {state} {completed}/{total}")
        else:
            print(f"    解析中: {state}")
        time.sleep(MINERU_POLL_INTERVAL)

    extract_dir = os.path.join(archive_dir, "mineru_output")
    os.makedirs(extract_dir, exist_ok=True)
    output_files = result_item.get("output_files") or {}

    zip_ref = output_files.get("zip") or {}
    zip_id = zip_ref.get("file_id")
    if zip_id:
        print("    下载解析结果ZIP")
        zip_bytes = mineru_download_file(zip_id)
        zip_save_path = os.path.join(archive_dir, "mineru_result.zip")
        with open(zip_save_path, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(extract_dir)

    markdown_path = None
    for root, _dirs, files in os.walk(extract_dir):
        if "full.md" in files:
            markdown_path = os.path.join(root, "full.md")
            break
    if markdown_path is None:
        md_files = []
        for root, _dirs, files in os.walk(extract_dir):
            for name in files:
                if name.lower().endswith(".md"):
                    md_files.append(os.path.join(root, name))
        if md_files:
            markdown_path = md_files[0]
    if markdown_path is None:
        md_ref = output_files.get("markdown") or {}
        md_id = md_ref.get("file_id")
        if not md_id:
            raise Exception(f"MinerU结果中未找到 Markdown: {archive_dir}")
        markdown_bytes = mineru_download_file(md_id)
        markdown_path = os.path.join(extract_dir, "full.md")
        with open(markdown_path, "wb") as f:
            f.write(markdown_bytes)

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 再写一份 full.md，方便下次直接复用
    archive_md = os.path.join(archive_dir, "full.md")
    if os.path.abspath(markdown_path) != os.path.abspath(archive_md):
        with open(archive_md, "w", encoding="utf-8") as f:
            f.write(markdown_text)

    image_paths = []
    for root, _dirs, files in os.walk(extract_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()

    print(f"    存档完成: md={len(markdown_text)} 字符, 图片={len(image_paths)} 张")
    return markdown_text, image_paths


def convert_ppt_to_pdf(ppt_path, pdf_path):
    """用本机 PowerPoint 把 PPT/PPTX 转成 PDF"""
    import pythoncom
    import win32com.client

    ppt_path = os.path.abspath(ppt_path)
    pdf_path = os.path.abspath(pdf_path)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    pythoncom.CoInitialize()
    app = None
    presentation = None
    try:
        app = win32com.client.DispatchEx("PowerPoint.Application")
        app.Visible = 1
        app.DisplayAlerts = 0
        presentation = app.Presentations.Open(
            ppt_path,
            ReadOnly=True,
            Untitled=False,
            WithWindow=False
        )
        # 32 = ppSaveAsPDF；ExportAsFixedFormat 在 pywin32 下会把未传的 PrintRange 当成 Python 对象，触发 COM 类型错误
        presentation.SaveAs(pdf_path, 32)
    finally:
        try:
            if presentation is not None:
                presentation.Close()
        except Exception:
            pass
        try:
            if app is not None:
                app.Quit()
        except Exception:
            pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    if not os.path.exists(pdf_path):
        raise Exception(f"PPT转PDF失败，未生成文件: {ppt_path}")
    return pdf_path


def convert_doc_to_docx(doc_path, docx_path, retries=3):
    """用本机 Word 把旧版 .doc 转成 .docx"""
    import pythoncom
    import win32com.client

    doc_path = os.path.abspath(doc_path)
    docx_path = os.path.abspath(docx_path)
    os.makedirs(os.path.dirname(docx_path), exist_ok=True)

    last_error = None
    for attempt in range(1, retries + 1):
        if os.path.exists(docx_path):
            os.remove(docx_path)

        pythoncom.CoInitialize()
        app = None
        document = None
        try:
            app = win32com.client.DispatchEx("Word.Application")
            app.Visible = False
            app.DisplayAlerts = 0
            document = app.Documents.Open(doc_path, ReadOnly=True)
            # 16 = wdFormatXMLDocument (.docx)
            document.SaveAs2(docx_path, FileFormat=16)
        except Exception as e:
            last_error = e
            print(f"    DOC转换异常({attempt}/{retries}): {e}")
        finally:
            try:
                if document is not None:
                    document.Close(False)
            except Exception:
                pass
            try:
                if app is not None:
                    app.Quit()
            except Exception:
                pass
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

        if os.path.exists(docx_path):
            return docx_path
        time.sleep(2)

    raise Exception(f"DOC转DOCX失败，未生成文件: {doc_path}，最后错误: {last_error}")


def write_parse_log(message):
    """把解析过程写入 parsed_archive/parse_log.txt"""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PARSE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def reset_parse_log():
    """每次构建覆盖写入一份新的解析日志"""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PARSE_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[{stamp}] 开始构建知识库\n")


def write_failed_list(failed_items):
    """写出失败文件列表"""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    with open(FAILED_LIST_FILE, "w", encoding="utf-8") as f:
        if not failed_items:
            f.write("本次没有失败文件。\n")
            return
        f.write(f"失败文件共 {len(failed_items)} 个\n")
        f.write("=" * 60 + "\n")
        for item in failed_items:
            f.write(f"文件: {item['source']}\n")
            f.write(f"类别: {item['category']}\n")
            f.write(f"阶段: {item['stage']}\n")
            f.write(f"原因: {item['reason']}\n")
            f.write("-" * 60 + "\n")
    print(f"失败文件列表: {FAILED_LIST_FILE} （{len(failed_items)} 个）")


def parse_with_office_fallback(file_path, archive_dir, data_id, rel_path):
    """
    先走 MinerU 原生解析；失败后：
    DOC -> 转 DOCX 再解析；PPT/PPTX -> 转 PDF 再解析。
    """
    ext = os.path.splitext(file_path)[1].lower()
    write_parse_log(f"原生解析开始: {rel_path}")
    try:
        result = mineru_parse_local_file(file_path, archive_dir, data_id)
        write_parse_log(f"原生解析成功: {rel_path}")
        return result
    except MinerUQueueStuck:
        raise
    except Exception as native_err:
        write_parse_log(f"原生解析失败: {rel_path} | {native_err}")
        if ext == ".doc":
            converted = os.path.join(archive_dir, "converted.docx")
            print("    回退: DOC 转 DOCX 后再解析")
            write_parse_log(f"回退 DOC->DOCX: {rel_path}")
            convert_doc_to_docx(file_path, converted)
            write_parse_log(f"开始解析转换后的 DOCX: {rel_path}")
            result = mineru_parse_local_file(converted, archive_dir, data_id + "_docx")
            write_parse_log(f"回退解析成功: {rel_path}")
            return result
        if ext in PPT_EXTS:
            converted = os.path.join(archive_dir, "converted.pdf")
            print("    回退: PPT 转 PDF 后再解析")
            write_parse_log(f"回退 PPT->PDF: {rel_path}")
            convert_ppt_to_pdf(file_path, converted)
            write_parse_log(f"开始解析转换后的 PDF: {rel_path}")
            result = mineru_parse_local_file(converted, archive_dir, data_id + "_pdf")
            write_parse_log(f"回退解析成功: {rel_path}")
            return result
        raise


def collect_source_files():
    """扫描知识库，按类型分流"""
    if not os.path.isdir(DOCS_DIR):
        raise ValueError(f"错误：找不到知识库目录 {DOCS_DIR}")

    pdf_files = []
    ppt_files = []
    price_images = []
    other_images = []
    docx_files = []
    doc_files = []

    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.startswith("."):
                continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, DOCS_DIR)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                pdf_files.append((file_path, rel_path))
            elif ext in PPT_EXTS:
                ppt_files.append((file_path, rel_path))
            elif ext in IMAGE_EXTS and is_price_image(filename):
                price_images.append((file_path, rel_path))
            elif ext in IMAGE_EXTS:
                other_images.append((file_path, rel_path))
            elif ext == ".docx":
                docx_files.append((file_path, rel_path))
            elif ext == ".doc":
                doc_files.append((file_path, rel_path))
            else:
                print(f"  跳过未纳入本脚本的文件: {rel_path}")

    return pdf_files, ppt_files, price_images, other_images, docx_files, doc_files


def load_archived_markdown(archive_dir):
    """若存档目录已有 markdown，直接读取，避免重复解析"""
    preferred = ("full.md", "markdown.md")
    for name in preferred:
        md_path = os.path.join(archive_dir, name)
        if os.path.isfile(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
    found = []
    for root, _dirs, files in os.walk(archive_dir):
        for name in files:
            lower = name.lower()
            if lower in preferred:
                with open(os.path.join(root, name), "r", encoding="utf-8") as f:
                    return f.read()
            if lower.endswith(".md"):
                found.append(os.path.join(root, name))
    if found:
        with open(found[0], "r", encoding="utf-8") as f:
            return f.read()
    return None


def has_mineru_archive(archive_dir):
    """仅当存在 MinerU 产物时才复用，避免沿用旧的 python-docx 文本"""
    extract_dir = os.path.join(archive_dir, "mineru_output")
    if not os.path.isdir(extract_dir):
        return False
    return load_archived_markdown(archive_dir) is not None


def load_archived_images(archive_dir):
    """读取存档目录中的图片路径"""
    image_paths = []
    if not os.path.isdir(archive_dir):
        return image_paths
    for root, _dirs, files in os.walk(archive_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()
    return image_paths


def archive_text_document(category, rel_path, full_text, extra=None):
    """把解析后的文本写入存档目录，并返回存档信息"""
    archive_dir = make_archive_subdir(category, rel_path)
    os.makedirs(archive_dir, exist_ok=True)
    md_path = os.path.join(archive_dir, "full.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    entry = {
        "category": category,
        "source": rel_path,
        "archive_dir": archive_dir,
        "markdown": md_path
    }
    if extra:
        entry.update(extra)
    return archive_dir, entry


def price_text_prefix(filename, markdown_text):
    """价目表文本块前缀：文件名 + 首行标题，让 Hybrid/BM25 能命中邮轮/价格/晚数"""
    parts = [f"来源：{filename}"]
    for line in (markdown_text or "").splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                parts.append(title)
            break
    return " | ".join(parts)


def append_text_chunks(metadata_store, all_vectors, doc_id, source, text, content_prefix=None):
    """把文本切分后写入向量和元数据，返回新的 doc_id"""
    chunks = split_text(text)
    print(f"    文档长度: {len(text)} 字符, 切分为 {len(chunks)} 个chunk")
    for chunk in chunks:
        if content_prefix:
            chunk = f"{content_prefix}\n{chunk}"
        metadata = {
            "id": doc_id,
            "source": source,
            "type": "text",
            "content": chunk
        }
        vector = get_text_embedding(chunk)
        all_vectors.append(vector)
        metadata_store.append(metadata)
        doc_id += 1
    return doc_id


def append_image_record(metadata_store, all_vectors, doc_id, source_label, img_path, content_label=None):
    """按原脚本方式写入一条图片记录，返回新的 doc_id"""
    img_name = os.path.basename(img_path)
    display_name = content_label or img_name
    reused = os.path.isfile(img_path + ".npy")
    print(f"    - {img_name}" + (" 复用向量" if reused else ""))
    metadata = {
        "id": doc_id,
        "source": source_label,
        "type": "image",
        "path": img_path,
        "content": f"[图片] {display_name}"
    }
    vector = get_image_embedding(img_path)
    all_vectors.append(vector)
    metadata_store.append(metadata)
    return doc_id + 1


def write_manifest(entries):
    """写入存档清单，方便核对解析结果"""
    manifest_path = os.path.join(ARCHIVE_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"存档清单: {manifest_path}")


def build_and_save():
    """先解析存档，再构建知识库并保存"""
    print("\n--- 构建多模态知识库（MinerU） ---")
    print(f"切分参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"源文件目录: {DOCS_DIR}")
    print(f"解析存档目录: {ARCHIVE_DIR}")
    print(f"MinerU API: {MINERU_API_BASE} (PDF/图片={MINERU_TIER}, Office=flash 原生)")

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    reset_parse_log()

    pdf_files, ppt_files, price_images, other_images, docx_files, doc_files = collect_source_files()
    scan_msg = (
        f"扫描结果: PDF {len(pdf_files)}, PPT {len(ppt_files)}, "
        f"价目表截图 {len(price_images)}, 其他图片 {len(other_images)}, "
        f"DOCX {len(docx_files)}, DOC {len(doc_files)}"
    )
    print(scan_msg)
    write_parse_log(scan_msg)

    metadata_store = []
    all_vectors = []
    doc_id = 0
    manifest = []
    failed_items = []
    mineru_ok = True

    def ingest_mineru_file(file_path, archive_dir, data_id):
        """优先复用 MinerU 存档，否则调用本机 MinerU 4。"""
        if has_mineru_archive(archive_dir):
            print("    复用存档")
            return load_archived_markdown(archive_dir), load_archived_images(
                os.path.join(archive_dir, "mineru_output")
            )
        return mineru_parse_local_file(file_path, archive_dir, data_id)

    def process_with_mineru(category, files, label, keep_original=False, embed_original=False):
        """统一入口：MinerU 解析 -> parsed_archive -> 文本/图片 embedding。"""
        nonlocal doc_id, mineru_ok
        for file_path, rel_path in files:
            print(f"  {label}: {rel_path}")
            write_parse_log(f"{label}: {rel_path}")
            archive_dir = make_archive_subdir(category, rel_path)
            os.makedirs(archive_dir, exist_ok=True)

            original_copy = None
            if keep_original or embed_original:
                original_copy = os.path.join(
                    archive_dir, "original" + os.path.splitext(file_path)[1].lower()
                )
                if not os.path.isfile(original_copy):
                    shutil.copy2(file_path, original_copy)

            # 价目表要同时保留 original.ext 供查询回原图；其他独立图片再存一份真实文件名
            if embed_original and not keep_original:
                copied = os.path.join(archive_dir, os.path.basename(file_path))
                shutil.copy2(file_path, copied)
                original_copy = copied

            if has_mineru_archive(archive_dir):
                write_parse_log(f"复用存档: {rel_path}")
                markdown_text, image_paths = ingest_mineru_file(file_path, archive_dir, "")
            else:
                if not mineru_ok:
                    print("    跳过：MinerU队列无进展")
                    skip_msg = f"跳过：MinerU队列无进展 | {rel_path}"
                    write_parse_log(skip_msg)
                    failed_items.append({
                        "source": rel_path,
                        "category": category,
                        "stage": "跳过",
                        "reason": "MinerU队列无进展"
                    })
                    continue
                data_id = "id_" + hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
                try:
                    markdown_text, image_paths = parse_with_office_fallback(
                        file_path, archive_dir, data_id, rel_path
                    )
                except MinerUQueueStuck as e:
                    print(f"    {e}")
                    write_parse_log(str(e))
                    failed_items.append({
                        "source": rel_path,
                        "category": category,
                        "stage": "队列卡住",
                        "reason": str(e)
                    })
                    mineru_ok = False
                    continue
                except Exception as e:
                    print(f"    最终失败: {e}")
                    write_parse_log(f"最终失败: {rel_path} | {e}")
                    failed_items.append({
                        "source": rel_path,
                        "category": category,
                        "stage": "解析失败",
                        "reason": str(e)
                    })
                    continue

            entry = {
                "category": category,
                "source": rel_path,
                "archive_dir": archive_dir,
                "images": image_paths
            }
            converted_pdf = os.path.join(archive_dir, "converted.pdf")
            if os.path.isfile(converted_pdf):
                entry["converted_pdf"] = converted_pdf
            converted_docx = os.path.join(archive_dir, "converted.docx")
            if os.path.isfile(converted_docx):
                entry["converted_docx"] = converted_docx
            if original_copy:
                entry["original"] = original_copy
            manifest.append(entry)

            source_name = os.path.basename(file_path)
            if markdown_text:
                # 价目表 OCR 正文常无「邮轮/价格」，块前缀补上文件名与行程标题
                prefix = None
                if category == "price_images":
                    prefix = price_text_prefix(source_name, markdown_text)
                doc_id = append_text_chunks(
                    metadata_store, all_vectors, doc_id,
                    source_name, markdown_text, content_prefix=prefix
                )
            for img_path in image_paths:
                doc_id = append_image_record(
                    metadata_store, all_vectors, doc_id,
                    f"图片: {os.path.basename(img_path)}",
                    img_path
                )
            if embed_original and original_copy:
                # 价目表原图 content 用真实文件名，便于 BM25/Hybrid 命中
                content_label = source_name if category == "price_images" else None
                doc_id = append_image_record(
                    metadata_store, all_vectors, doc_id,
                    f"图片: {source_name}",
                    original_copy,
                    content_label=content_label
                )

    process_with_mineru("pdf", pdf_files, "处理PDF")
    process_with_mineru("docx", docx_files, "处理文档")
    process_with_mineru("doc", doc_files, "处理DOC")
    process_with_mineru("ppt", ppt_files, "处理PPT")
    process_with_mineru(
        "price_images", price_images, "处理价目表截图",
        keep_original=True, embed_original=True
    )
    process_with_mineru("images", other_images, "处理图片", embed_original=True)

    # 处理视频：URL 无法识别/提取视频时跳过解析与 embedding
    print("  处理视频...")
    for video_info in VIDEO_KNOWLEDGE:
        desc = video_info["description"]
        url = video_info["url"]
        print(f"    - {desc}")
        write_parse_log(f"处理视频: {desc} | {url}")
        try:
            vector = get_video_embedding(url)
        except Exception as e:
            # URL 不一定能提取到可 embedding 的视频，识别失败则跳过
            skip_msg = f"跳过视频（无法识别/embedding）: {desc} | {e}"
            print(f"    {skip_msg}")
            write_parse_log(skip_msg)
            failed_items.append({
                "source": desc,
                "category": "video",
                "stage": "跳过",
                "reason": f"无法识别视频URL或embedding失败: {e}"
            })
            continue

        metadata = {
            "id": doc_id,
            "source": f"视频: {desc}",
            "type": "video",
            "url": url,
            "description": desc,
            "content": f"[视频] {desc}"
        }
        all_vectors.append(vector)
        metadata_store.append(metadata)
        doc_id += 1
        manifest.append({
            "category": "video",
            "source": desc,
            "url": url
        })
        write_parse_log(f"视频入库成功: {desc}")

    write_manifest(manifest)
    write_failed_list(failed_items)
    write_parse_log(f"构建结束，失败 {len(failed_items)} 个")

    # 创建FAISS索引
    if all_vectors:
        dim = len(all_vectors[0])
        print(f"\n向量维度: {dim}")

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors).astype('float32'))

        faiss.write_index(index, INDEX_FILE)
        print(f"索引已保存: {INDEX_FILE}")

        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存: {METADATA_FILE}")

    text_count = sum(1 for m in metadata_store if m["type"] == "text")
    image_count = sum(1 for m in metadata_store if m["type"] == "image")
    video_count = sum(1 for m in metadata_store if m["type"] == "video")
    print(f"\n完成! 文本:{text_count}, 图片:{image_count}, 视频:{video_count}")

    print("\n--- 知识库内容 ---")
    for m in metadata_store:
        print(f"\n[{m['id']:2d}] [{m['type']:5s}] {m.get('source', '')}")
        print(f"    {m['content']}")


def is_price_metadata_entry(meta):
    """判断元数据是否属于价目表截图（文本或图片）"""
    source = meta.get("source") or ""
    path = (meta.get("path") or "").replace("/", os.sep)
    if f"{os.sep}price_images{os.sep}" in path:
        return True
    if is_price_image(source) and os.path.splitext(source)[1].lower() in IMAGE_EXTS:
        return True
    if source.startswith("图片:") and any(kw in source for kw in PRICE_KEYWORDS):
        return True
    return False


def rebuild_price_images_only():
    """
    只重建价目表：保留其余向量，重写价目表 OCR 文本前缀并嵌入 original 原图。
    用法: set REBUILD_PRICE_ONLY=1 后运行本脚本。
    """
    if not os.path.isfile(INDEX_FILE) or not os.path.isfile(METADATA_FILE):
        raise ValueError("找不到现有索引，请先完整构建知识库")

    print("\n--- 仅重建价目表索引 ---")
    old_index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        old_metadata = json.load(f)

    keep_vectors = []
    keep_metadata = []
    removed = 0
    for i, meta in enumerate(old_metadata):
        if is_price_metadata_entry(meta):
            removed += 1
            continue
        keep_vectors.append(old_index.reconstruct(i))
        keep_metadata.append(meta)
    print(f"保留非价目表记录: {len(keep_metadata)}，移除旧价目表记录: {removed}")

    _, _, price_images, _, _, _ = collect_source_files()
    print(f"扫描到价目表截图: {len(price_images)}")

    metadata_store = list(keep_metadata)
    all_vectors = [np.asarray(v, dtype=np.float32) for v in keep_vectors]
    doc_id = len(metadata_store)

    for file_path, rel_path in price_images:
        print(f"  处理价目表截图: {rel_path}")
        archive_dir = make_archive_subdir("price_images", rel_path)
        os.makedirs(archive_dir, exist_ok=True)

        original_copy = os.path.join(
            archive_dir, "original" + os.path.splitext(file_path)[1].lower()
        )
        if not os.path.isfile(original_copy):
            shutil.copy2(file_path, original_copy)

        if has_mineru_archive(archive_dir):
            markdown_text = load_archived_markdown(archive_dir)
            image_paths = load_archived_images(os.path.join(archive_dir, "mineru_output"))
        else:
            data_id = "id_" + hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
            markdown_text, image_paths = mineru_parse_local_file(
                file_path, archive_dir, data_id
            )

        source_name = os.path.basename(file_path)
        if markdown_text:
            prefix = price_text_prefix(source_name, markdown_text)
            doc_id = append_text_chunks(
                metadata_store, all_vectors, doc_id,
                source_name, markdown_text, content_prefix=prefix
            )
        for img_path in image_paths:
            # 归档里可能扫到 original，避免表格裁图与原图重复入库
            if os.path.basename(img_path).lower().startswith("original"):
                continue
            doc_id = append_image_record(
                metadata_store, all_vectors, doc_id,
                f"图片: {os.path.basename(img_path)}",
                img_path
            )
        doc_id = append_image_record(
            metadata_store, all_vectors, doc_id,
            f"图片: {source_name}",
            original_copy,
            content_label=source_name
        )

    # 重写 id，与 FAISS 下标对齐
    for new_id, meta in enumerate(metadata_store):
        meta["id"] = new_id

    dim = len(all_vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(all_vectors).astype("float32"))
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)

    text_count = sum(1 for m in metadata_store if m["type"] == "text")
    image_count = sum(1 for m in metadata_store if m["type"] == "image")
    video_count = sum(1 for m in metadata_store if m["type"] == "video")
    print(f"索引已更新: {INDEX_FILE}")
    print(f"元数据已更新: {METADATA_FILE}")
    print(f"完成! 文本:{text_count}, 图片:{image_count}, 视频:{video_count}")


if __name__ == "__main__":
    if os.getenv("REBUILD_PRICE_ONLY") == "1":
        rebuild_price_images_only()
    else:
        build_and_save()
