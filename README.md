# NInfer QSA — RTX 50 Ubuntu 一键部署

针对 Ubuntu x86_64、RTX 5070 Ti 16GB（也适用于其他消费级 RTX 50）的 **SM120a** 单架构预编译版本。GitHub Actions 在 Ubuntu 24.04 上用 CUDA 13.4.3 / GCC 13 编译，运行包发布到本仓库的 [Releases](https://github.com/cnnlei/ninfer-qsa-ubuntu-build/releases)，本地不用重新编译 CUDA。

- 上游：[13thblade/ninfer-all-QSA](https://github.com/13thblade/ninfer-all-QSA)，固定提交 [`29c79545429f`](https://github.com/13thblade/ninfer-all-QSA/commit/29c79545429f03c2ec9d74370b3b2c59b62cca0d)。
- [Actions 编译进度](https://github.com/cnnlei/ninfer-qsa-ubuntu-build/actions)。Release 只有在编译、架构检查、依赖检查与归档校验成功后才发布。
- GitHub runner 没有 NVIDIA GPU，因此编译与 CLI 验证不等于真实 GPU 推理验证。首次启动后请运行 `test`。
- 需要 Ubuntu 24.04 或更新版本（glibc ≥ 2.39）及兼容 CUDA 13.x 的 NVIDIA 驱动。Ubuntu 22.04 不适合此二进制。运行包带 CUDA 用户态库，不安装或替换系统显卡驱动，也不需要完整 CUDA Toolkit。

## 固定模型和 QSA v2 张量

| 文件 | 来源 | 大小 |
| --- | --- | --- |
| [`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf) | ISTA-DASLab 主模型，内置 MTP | 12.1GB |
| [`mmproj-Qwen3.8-27B-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/mmproj-Qwen3.8-27B-BF16.gguf) | ISTA-DASLab 官方视觉头；纯文本可省略 | 931MB |
| [`qsa-indexers.safetensors`](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-BF16/resolve/main/qsa-indexers.safetensors) | 与 [higa2515 QSA GGUF](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF) 相同的 2026-11-19 v2 索引器 | 约 210MB |

三个文件均校验固定 SHA-256；**不下载 GitHub Release 中的旧 QSA / GDN 张量**，不下载 higa2515 的整个 Q8/Q4 GGUF，也无需单独下载 MTP 草稿或完整 BF16 模型。转换器仍固定使用 GitHub 的 `13thblade/ninfer-all-QSA`。脚本把 HF v2 的 80 个 FP32 张量转换为上游接受的 GGUF 名称和 norm 编码（`1+w`），用其正式 graft recipe 嫁接；最终两个索引器投影遵循 NInfer 核的 BF16 格式、三个 norm 保持 FP32。GDN A/B controls 使用 base GGUF 自身的值（由转换器转为 BF16），不额外替换其他模型权重。`fp32_direct` 只写入生成的配置副本，不改原配置。

## Windows：双击 BAT 转换，再到 Ubuntu 启动

**推荐绿色版（免安装）：** 在 [GitHub Releases](https://github.com/cnnlei/ninfer-qsa-ubuntu-build/releases) 下载标题含 Windows x64 portable converter 的版本中的 `NInfer-QSA-Windows-Portable-x64.zip`，完整解压到例如 `D:\ninfer-qsa`，双击 `convert-windows-portable.bat`。ZIP 已包含 Python 3.12 x64、MinGit、CPU PyTorch、转换依赖、固定 GitHub 转换器及官方 Qwen 配置/分词器元数据；不包含模型权重。无需安装 Python/Git、WinGet 或 CUDA Toolkit，不写入系统 PATH、不需要管理员权限。三份 HF 输入文件已下载时，可离线转换。请保留完整目录，不要只复制 BAT 或在 ZIP 内直接运行。

如果你已克隆本仓库，也可 `git pull` 后双击 `convert-windows-portable.bat`；首次自动下载两份固定 SHA-256 的便携运行时并在 `.portable` 中准备依赖，无需系统安装。在只有源代码的目录里使用 `--prepare-only` 可提前准备这些依赖及元数据（不下载模型权重）。绿色版不创建不可直接搬移的 Python venv，而是在便携 Python 的本地目录中安装依赖。可以将完整目录移至其他位置，再继续双击使用。

普通版：

在 Windows 克隆或 `git pull` 本仓库后，直接双击 **`convert-windows.bat`**。需要 Git for Windows 和 64 位 Python 3.11/3.12；BAT 自动创建独立的 `.venv-win`，安装 CPU PyTorch，下载 HF v2 索引器和配置文件，并调用固定 GitHub 转换器。

如果 BAT 未找到 Python，双击运行时可选择 `Y` 通过 WinGet 安装官方 Python 3.11 x64，或者自行在 CMD/PowerShell 执行 `winget install -e --id Python.Python.3.11 --scope user --architecture x64` 后重新打开 BAT。已安装但不在 PATH 的常见目录会自动检测；自定义安装位置可在 CMD 使用 `set "QSA_PYTHON_EXE=D:\Python311\python.exe"` 后运行 BAT。

双击后可粘贴你已下载的主模型、BF16 mmproj、HF v2 `qsa-indexers.safetensors` 和输出路径。如果三份文件放在同一个目录，输入主模型路径后会自动识别另外两份文件，直接回车确认即可。已有文件通过 SHA-256 校验后就会复用，不会重新下载；默认目录里没有文件时才自动下载。也可直接用 CMD：

```bat
convert-windows.bat --gguf "D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" --mmproj "D:\models\mmproj-Qwen3.8-27B-BF16.gguf" --indexer "D:\models\qsa-indexers.safetensors" --output "D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-QSA-v2-mtp.ninfer"
```

绿色版和普通版支持相同的模型参数：`--auto` 使用默认路径自动下载及转换；`--no-vision` 只转换文本+MTP；`--check-only` 只跑转换预检；`--plan` 仅显示计划（不下载模型或转换）。Windows 和 Linux 可共用 `.ninfer` 权重文件，独立复制虚拟环境并非必要。转换成功后保留 `.ninfer`、`.ninfer.conversion.json` 和 `.ninfer.qsa-v2.json` 三个同名文件（最后一个记录 v2 来源哈希及真实 artifact 检查）。把它们放共享模型盘，在 Ubuntu 的 `qsa.env` 中填写 `QSA_MODEL`，运行 `bash qsa.sh install` 和 `bash qsa.sh start`，不必在 Ubuntu 再转换。

## Ubuntu 一条龙：克隆 → 转换 → 运行

```bash
git clone https://github.com/cnnlei/ninfer-qsa-ubuntu-build.git
cd ninfer-qsa-ubuntu-build
bash qsa.sh init
```

默认 `qsa.env` 的两条输入路径为空，执行转换时会自动下载指定模型和视觉头。也可以编辑 `qsa.env` 指向已有的正版文件（支持带空格路径）：

```bash
QSA_GGUF="/你的模型目录/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
QSA_MMPROJ="/你的模型目录/mmproj-Qwen3.8-27B-BF16.gguf"
```

已有这份 ISTA IQ3_S-mtp GGUF 可以复用。普通 `.ninfer` 文件本身没有 QSA 索引器，不能直接当 QSA 模型启动；应从原 GGUF 重新转换。MTP 已在主模型中，默认开启；不要图片输入时设置 `QSA_VISION=0`，不必准备 mmproj。

```bash
# 仅第一次缺系统依赖时运行（明确请求 sudo 安装依赖）：
bash qsa.sh deps

# 下载 GitHub Release 并检查 SHA256 → 准备转换环境 → 转换 → 启动
bash qsa.sh all
```

首次转换使用独立 Python venv、CPU 版 PyTorch。输出默认在 `models/Qwen3.8-27B-GSQ-RCO-IQ3_S-QSA-v2-mtp.ninfer`。请确保下载后仍有空间保存转换产物，保守预留约 30GB 或更多。源 GGUF、原有 `.ninfer` 和 mmproj 不会被改动或覆盖；转换报错时留下一份单独的 `.partial.*.ninfer` 供检查。旧版 `qsa.env` 请调整为上述 v2 输出名并清空旧 `QSA_DONOR_GGUF`；存在但没有 v2 来源证明的旧模型不会被误判成 v2。

## 日常控制

| 命令 | 用途 |
| --- | --- |
| `bash qsa.sh doctor` | 检查主机、GPU、内存、模型及运行包 |
| `bash qsa.sh install` | 下载并校验 Release，可单独执行 |
| `bash qsa.sh prepare` | 准备转换器、HF v2 索引器及元数据（不下载整个主模型） |
| `bash qsa.sh convert --check` | 验证转换输入，不写输出模型 |
| `bash qsa.sh convert` | 转换 GGUF 为 QSA `.ninfer` |
| `bash qsa.sh plan` | 预览完整启动参数，API key 不会打印 |
| `bash qsa.sh start` | 后台启动并等待 `/health` 就绪 |
| `bash qsa.sh stop` | SIGTERM 正常停止本工具启动的进程 |
| `bash qsa.sh restart` | 停止再启动，应用新参数 |
| `bash qsa.sh status` | 进程和就绪状态 |
| `bash qsa.sh logs -f` | 实时日志，Ctrl+C 仅退出日志查看 |
| `bash qsa.sh test` | 向本地 OpenAI 兼容 API 发送简单请求 |

`stop` 只根据 PID、启动时间和系统 boot id 来终止本工具自己的进程，不会 `pkill` 其他模型服务。正常停止超时会提示，只有明确运行 `stop --force` 才会强制结束该进程。配置和日志都只保存在本地，已加入 `.gitignore`。

默认 WebUI：`http://127.0.0.1:18201/`；OpenAI API：`http://127.0.0.1:18201/v1`；模型名 `qwen3.8-27b-qsa-v2-iq3s`。

如果要在局域网的 CLIProxyAPI / Agent 客户端连接，在 `qsa.env` 中设置 `QSA_BIND="0.0.0.0"`，同时设置 `QSA_API_KEY_FILE`，指向自己创建的 0600 单行密钥文件。无需把密钥上传 GitHub。该上游服务器通过进程参数接收密钥，本机有相应权限的账户可能看见它。不要将 API 直接暴露到公网。

## 16GB 显存的初始参数

这是可调整的初始配方，尚未在真实 5070 Ti 上验证；模型量化、桌面显存占用、视觉和草稿都会影响是否能启动。

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `QSA_CONTEXT` | 262144 | 256K 完整历史；改成 524288 或 1048576 将启用 YaRN |
| `QSA_TOP_BLOCKS` | 1024 | 每个 decode query 选择 4096 历史 tokens（每块 4 tokens） |
| `QSA_KV_RESIDENT` | 8192 | GPU 中的 KV 位置数，其余通过 pinned host RAM 流式搬运 |
| `QSA_KV_DTYPE` | int8 | 分支作者的实测 KV 精度 |
| `QSA_HOST_CACHE_MIB` | 16384 | 另外用于前缀/会话续接的 Host cache 预算 |
| `QSA_PREFILL_CHUNK` | 512 | 初始采用偏保守值；可逐级尝试 1024 |
| `QSA_MAX_CONCURRENCY` | 1 | 保持 CUDA Graph；按需测试 2 |
| `QSA_MAX_OUTPUT` | 32768 | 默认请求输出上限，含思考 |
| `QSA_MTP` / `QSA_DRAFT_TOKENS` | 1 / 3 | 仅开启模型自带 MTP；QSA 模式尚不支持 DFlash2 / DSpine |
| `QSA_VISION` | 1 | CPU 视觉，节省 GPU 显存 |
| `QSA_EMBEDDING_RESIDENCY` | host | 输入 embedding 放主机内存 |

以当前固定提交的源码逻辑为准，stream pool 至少容纳两个注意力窗口，即 `QSA_KV_RESIDENT >= QSA_TOP_BLOCKS * 8`，且须小于 context。`QSA_KV_RESIDENT=all` 表示全驻显存；仅该模式使用自动 KV 容量和 `QSA_HEADROOM_MIB`，流式模式使用明确的逻辑 KV 容量。并发时逻辑 KV 容量是共享的，不应理解为每条请求都能独占 256K。注意力窗口变小可能影响历史召回；建议先验证 256K 的效果，再尝试 512K 或 1M。

如果启动显存不足：先减少 `QSA_PREFILL_CHUNK`（256/128）、用 `QSA_TOP_BLOCKS=512` 配合 `QSA_KV_RESIDENT=4096`、关闭 vision/MTP（要求转换组件与启动选项一致）。不要盲目增加上下文或显存驻留数。

## 已有 QSA 模型 / 离线安装 / 更新

- 若已有真正的 QSA `.ninfer`，在 `qsa.env` 填写 `QSA_MODEL`，并按模型组件调整 `QSA_MTP` 和 `QSA_VISION`，运行 `install` + `start` 即可，不需要 `convert`。
- 手动从 Release 下载所有 `.tar.gz.NNN` 分卷和对应 `SHA256SUMS-*.txt` 到 `downloads/`，执行 `bash qsa.sh install --offline`。安装时校验归档以及包内每个文件，下载或已经安装的运行包不会被静默覆盖。
- 更新脚本用 `git pull`；私有 `qsa.env` 不在仓库中。需要重编译时在 Actions 运行 `Build Ubuntu 24.04 RTX 50 (SM120a)` 工作流。编译版本与下载器、转换器必须使用同一个固定 upstream commit。

独立脚本自测（使用模拟服务，不占 GPU、不需要模型或联网）：`python3 -m unittest discover -s tests -v`。

## 来源和许可

本仓库仅编译和部署上游开源项目，没有修改其推理数学。QSA 分支由 13thblade 开发，模型/索引器来源见其 [MODIFICATIONS.md](https://github.com/13thblade/ninfer-all-QSA/blob/29c79545429f03c2ec9d74370b3b2c59b62cca0d/MODIFICATIONS.md)。上游 Apache-2.0 许可、CUDA 运行库许可与 FFmpeg/dav1d 对应源码保存在 Release 运行包内。
