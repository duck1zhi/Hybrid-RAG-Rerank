Hybrid RAG + Rerank
================================

本项目把「迪士尼RAG知识库」目录中的文档、图片解析成文本和图片向量，写入 FAISS 索引；
查询时用 FAISS L2 + BM25 融合成 Hybrid_scores，可选再加重排，生成客服式回答。

适用环境：Windows + Python 3.12。
解析走本机 MinerU 4（默认 http://127.0.0.1:8000，会占用较大显存）。
向量、重排和问答依赖阿里云 DashScope。


一、下载后目录里有什么
------------------------

1-迪士尼RAG知识库.py              构建脚本：解析 -> 存档 -> embedding -> 索引
2-迪士尼RAG混合检索.py            查询：Hybrid 检索 -> Top-k 文本 -> 大模型回答
3-迪士尼RAG混合检索-Rerank.py     查询：Hybrid 召回 -> 文本/图片重排 -> 大模型回答
迪士尼RAG知识库\                  源文件目录，把 PDF/Word/PPT/图片放这里
parsed_archive\                   解析产物存档（运行构建脚本后生成）
parsed_archive\parse_log.txt      本次构建的解析日志
parsed_archive\failed_files.txt   失败/跳过的文件列表
parsed_archive\manifest.json      存档清单
disney_index.faiss                向量索引（构建完成后生成，查询脚本读取）
disney_metadata.json              文本/图片/视频元数据（构建完成后生成）
requirements.txt                  Python 依赖
readme.txt                        本说明

视频条目写在 1-迪士尼RAG知识库.py 顶部的 VIDEO_KNOWLEDGE 里，不是文件目录。


二、三个脚本在做什么
--------------------

【1-迪士尼RAG知识库.py】构建知识库

1. 扫描「迪士尼RAG知识库」文件夹，按类型分流：
   PDF、DOC/DOCX、PPT/PPTX、价目表截图、其他图片。
2. 全部交给本机 MinerU 4 解析：
   - PDF、图片：standard 档 + 强制 OCR
   - DOC/DOCX/PPT/PPTX：MinerU 原生 Office 解析（flash）
3. 失败回退（原生解析失败时）：
   - DOC -> 本机 Word 转 DOCX，再送 MinerU
   - PPT/PPTX -> 本机 PowerPoint 转 PDF，再送 MinerU
   - 某个文件回退仍失败：写入失败列表，继续处理后面的文件，不会整次中断
   - MinerU 队列卡住：不回退，并跳过后面尚未解析的文件
   - 回退需要本机安装 Word / PowerPoint；pywin32 已在 requirements.txt，Office 本身不是 pip 包
4. 解析结果写入 parsed_archive（markdown、抽出的图片、zip）。
   某文件目录下已有 mineru_output 时会复用，不再重复解析。
   每次构建还会写出：
   - parsed_archive/parse_log.txt
   - parsed_archive/failed_files.txt
5. 文本按 500 字切分、重叠 50 字；文本和图片调用 DashScope
   模型 tongyi-embedding-vision-plus 做向量。
   图片向量缓存在对应图片旁的 .npy，下次可复用。
6. 价目表截图（文件名含「价格/价目/票价/价目表/价格表」）单独处理：
   - OCR 正文每个 chunk 前加前缀：来源文件名 + 首行标题（如「4天3晚」），方便 Hybrid/BM25 命中
   - 归档 original 原图，并对原图也做图片 embedding（content 用真实文件名）
   - MinerU 裁出的表格图仍入库，供向量侧兜底
7. 其他独立图片：整图入库并 embedding。
8. 视频按 VIDEO_KNOWLEDGE 里的 url + description 做视频 embedding。
9. 写出 disney_index.faiss 和 disney_metadata.json。

仅重建价目表（保留其余向量，重写价目表文本前缀并嵌入原图）：

  $env:REBUILD_PRICE_ONLY = "1"
  python 1-迪士尼RAG知识库.py


【2-迪士尼RAG混合检索.py】混合检索查询（无重排）

1. 读取 disney_index.faiss / disney_metadata.json，构建 BM25。
2. 问题做成文本向量；FAISS 用 L2，关键词用 BM25，按 alpha 融合成 Hybrid_scores
   （默认 alpha=0.5：Hybrid = 0.5 * 向量归一化 + 0.5 * BM25 归一化）。
3. 文本：按 Hybrid 取 Top-k（默认 3）交给 qwen-flash。
   价目表问句会在 Top-k 中保底若干「来源为图片文件」的 OCR 块，避免被其他文档挤掉。
4. 要图片时（含「图片/海报/看看/长什么样」等）：
   - 先找文本来源是图片文件、且文件名与问题至少重合 2 个内容词的记录，映射到归档 original 原图
   - 多张时按该来源 OCR 的 Hybrid 排序，取最高 1 张
   - 对不上再退回 L2 短名单；价目表问句优先 price_images，再按 Hybrid / L2 选 1 张
5. 要视频时（含「视频/播放」等）：在 L2 阈值内取距离最小的视频 URL。
6. 脚本末尾示例题含：退款、小朋友项目、年票、客诉、价目表按日期问价、按晚数要原图、
   价目表泛问、园区地图、避坑视频。


【3-迪士尼RAG混合检索-Rerank.py】混合检索 + 重排

1. 召回与文件 2 相同：Hybrid_scores = alpha * 向量归一化 + (1-alpha) * BM25 归一化。
2. 文本：先按 Hybrid 召回前 15 条；价目表问句再补最多 5 条图片来源 OCR 块。
   再用 qwen3.7-text-rerank 精筛 Top-k（默认 3）写入提示词。
3. 要图片时：
   - 同样先按来源文件名命中 original 原图，多张按 OCR Hybrid 排序，最多留 10 张
   - 1 张直接返回；多张用 qwen3-vl-rerank 看图选 1 张
   - 文件名未命中：L2 短名单（价目表优先 price_images），再视觉重排选 1 张
4. 要视频时：关键词命中后直接返回视频 URL（不再按距离筛）。
5. 示例题与文件 2 对齐（价目表边界题 + 地图 + 视频等）。

文件 2 与文件 3 共用同一套索引；改价目表逻辑后若只动了价目表，可用 REBUILD_PRICE_ONLY 重建。


三、下载后怎么用
----------------

第 1 步 安装 Python 依赖

  打开 PowerShell，进入本项目根目录，执行：

    pip install -r requirements.txt

第 2 步 配置阿里云密钥

  需要环境变量 DASHSCOPE_API_KEY（embedding、重排、qwen-flash 都用它）。
  Windows 可在「系统属性 -> 环境变量」里新增用户变量，或在当前 PowerShell 临时设置：

    $env:DASHSCOPE_API_KEY = "你的密钥"

  新开终端后若用「系统环境变量」，一般会自动生效；临时设置只对当前窗口有效。

第 3 步 准备源文件

  把要入库的资料放到「迪士尼RAG知识库」目录（可分子文件夹）。
  支持：.pdf .doc .docx .ppt .pptx 以及常见图片（png/jpg/jpeg 等）。
  文件名含「价格/价目/票价」的图片会按价目表走 OCR 前缀 + 原图入库。

第 4 步 启动本机 MinerU 4（仅构建时需要）

  本机部署会占用较大显存，用完请关掉。
  默认地址：http://127.0.0.1:8000
  本机若按本环境安装，可在 PowerShell 执行：

    H:\MinerU\start-api.ps1

  若提示无法运行脚本，先执行：

    Set-ExecutionPolicy -Scope Process Bypass

  再重新运行 start-api.ps1。
  等到出现：Uvicorn running on http://127.0.0.1:8000
  这个窗口不要关。浏览器打开根地址出现 404 是正常的，接口在 /v1/health 和 /docs。
  若 API 不在 8000 端口，构建前设置：

    $env:MINERU_API_BASE = "http://127.0.0.1:你的端口"

第 5 步 构建索引

  另开一个 PowerShell，进入项目根目录：

    python 1-迪士尼RAG知识库.py

  首次会较慢（解析 + embedding）。看到「完成! 文本:xx, 图片:xx, 视频:xx」即成功。
  构建结束后到 MinerU 窗口按 Ctrl+C 停止服务，可释放显存。

  若只改价目表入库方式、其余文档已有索引，可：

    $env:REBUILD_PRICE_ONLY = "1"
    python 1-迪士尼RAG知识库.py

第 6 步 提问查询（不需要 MinerU）

  确认根目录已有 disney_index.faiss 和 disney_metadata.json，然后任选其一：

    python 2-迪士尼RAG混合检索.py
    python 3-迪士尼RAG混合检索-Rerank.py

  会依次跑脚本里的示例问题。要改问题，编辑对应脚本最底部的 rag_ask(...) 即可。
  要改视频，编辑 1-迪士尼RAG知识库.py 里的 VIDEO_KNOWLEDGE，再重新构建。


四、使用注意
------------

- 先构建，再查询。没有索引文件时查询脚本会直接报错。
- 查询只走 DashScope，不必开 MinerU。
- 解析会复用 parsed_archive 里的 MinerU 产物；全量构建时 embedding 仍会重建整份 FAISS。
  图片若已有 .npy 缓存则不再调图片向量接口。
- 价目表「问多少钱」主要靠 OCR 文本；「把那张表给我看」靠文件名命中 original 原图。
  构建时已把文件名写进文本块前缀，并给原图做了向量，查询侧再用 Hybrid 在多张表里排序。
- 旧版仅 python-docx 生成的 Word 存档（没有 mineru_output）会重新交给 MinerU 解析。
- 视频链接若带过期签名，失效后视频 embedding 会失败，需换成可访问地址后重建。
- 没有 Office 也能跑 MinerU 原生解析。
  只有 DOC/PPT 原生失败后的回退才需要本机 Word / PowerPoint。
  某个文件回退失败不会拦住后面的文件；只有 MinerU 队列卡住才会跳过后续解析。
  结果见 parsed_archive/parse_log.txt 和 parsed_archive/failed_files.txt。


五、常见问题
------------

1) 连接 127.0.0.1:8000 被拒绝
   MinerU 服务没启动，或启动窗口已关闭。先完成第 4 步。

2) 提示未设置 DASHSCOPE_API_KEY
   当前终端读不到该环境变量。设置后重新打开终端再跑。

3) 任务栏看不到 PowerShell
   若本机 MinerU 是在 Cursor 终端后台开的，不会出现独立窗口。到 Cursor 底部终端里找到
   含 Uvicorn running 的那个，按 Ctrl+C 停止。

4) 构建很慢或卡在「切分为 N 个 chunk」
   正在调用 DashScope 做文本向量，不是死机。每个 chunk 一次网络请求。

5) 问邮轮价格却答不到数字 / 选不到原图
   确认已用当前构建脚本重建过价目表（全量或 REBUILD_PRICE_ONLY=1）。
   文本块应带「来源：迪士尼邮轮价格N.jpeg」前缀；元数据里应有 path 指向 original.jpeg 的图片记录。
