# 文档维护

正式源文件在 `docs/`，主配置为 `mkdocs.yml`。`mkdocs.gh.yml` 继承主配置，保留 GitHub Pages 的 `/TinyAuton/` 路径。`site/` 是已有生成产物，验证输出到临时目录。

## 安装、构建与校验

在仓库根目录运行：

```powershell
python -m pip install -r DOC/requirements.txt
python DOC/tools/check_docs.py
python DOC/tools/check_preservation.py
python -m mkdocs build --strict -f DOC/mkdocs.yml -d "$env:TEMP/TinyAuton-doc-preview"
python DOC/tools/check_site.py "$env:TEMP/TinyAuton-doc-preview"
python DOC/tools/preview.py "$env:TEMP/TinyAuton-doc-preview" --port 9066
```

预览地址为 `http://127.0.0.1:9066/`，中文在 `/zh/`。`preview.py` 固定 JavaScript MIME 类型，避免 Windows 将搜索 Worker 当成纯文本。日常编辑可以用 `python -m mkdocs serve -f DOC/mkdocs.yml`。

GitHub Pages 配置单独验证：

```powershell
python -m mkdocs build --strict -f DOC/mkdocs.gh.yml -d "$env:TEMP/TinyAuton-doc-preview-gh"
python DOC/tools/check_site.py "$env:TEMP/TinyAuton-doc-preview-gh" --base-path /TinyAuton/
```

版本列表记录本次实际验证环境。RSS hook 修复多语言重复处理时间配置的问题；搜索使用浏览器支持的 Lunr 英语基础语言，中文按汉字边界切分，i18n 仍合并中英文条目。中文检索可用模块名、函数名和中文关键词，但不等同于词语语义分词。

## 内容保全

`tools/preservation_baseline.json` 基于整理前的 TinyAuton 工作区建立，记录 203 个原始页面、960 段围栏正文、19 个图像资源和 1,631 个章节锚点。它不是 TinySHM 的基线。原有英文 C 矩阵页末尾存在未闭合的空围栏，已补齐分隔符，围栏正文仍保留。

`check_preservation.py` 按页面比较代码块与日志正文的哈希及出现次数，允许移动、折叠和新增，禁止历史内容静默删除或改写。CRLF/LF 转换不影响检查。`check_site.py` 验证本地页面、资源、章节链接和原始锚点，不访问外部网站。不要为了让检查通过而重建基线。

原始章节 ID 尽量留在对应标题上；无法对应到改版后结构的旧 ID 放在该页的兼容锚点中。旧矩阵完整接口、源码与测试路由保留，新页面从中拆出阅读入口，不替代历史记录。

## 页面约定

- 中英文使用 `.en.md` / `.zh.md` 配对，导航引用不带语言后缀的路径。
- 设计页说明用途、假设、输入输出与约束；接口页说明布局、内存所有权和错误返回。
- 测试先呈现配置、结果与判据，再提供完整日志和源码。长源码用 `<details class="auton-source" markdown="1">` 折叠。
- 历史日志的数值、行序、已有省略号完整保留。新增摘要和计数放在围栏外，不把标签数当作独立测试用例数。
- 未记录的日期、硬件或提交明确写为未记录。源码日期不等于运行日期；文档构建不代表固件或硬件验证通过。
- 历史接口与当前实现不一致时保留历史摘录，并提供经当前头文件核对的调用。新示例与历史原文分别标明。

## 实现依据

Math 以 `CODE/AIoTNode-TinyAuton-MATH` 为依据，DSP 以 DSP 工程为依据，AI 以 AI 工程为依据，Toolbox 以 AI 工程为依据。三个工程默认入口与 DSP 副本差异见工程与版本页。

公共模块整理借鉴 TinySHM 的样式、工具和部分已核对内容，没有引入 TinySHM 的额外 C 数学模块或 SHM 应用模块。矩阵历史 C4.4 的 `[FAIL]`、AI 示例中浮点路径被命名为 INT8 的评估问题均如实说明。本次只整理文档，不修改这些固件行为。
