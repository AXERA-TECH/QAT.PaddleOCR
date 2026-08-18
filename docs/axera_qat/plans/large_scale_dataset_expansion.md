# 大规模数据集扩展计划(ICDR2015 方案验证完成后的下一步)

日期:2026-08-18

## 1. 背景与目标

ICDR2015 小规模数据上已完成方案验证:

- det:v5 mobile det ICDAR hmean 0.16527(1000 张训练图,数据量明显不足);
- rec:v5 mobile rec U8/S8 + 下采样 S16(exp16)val_acc 0.5927(约 1 万训练裁剪)。

下一步在大规模真实数据上训练,验证浮点与 QAT 精度是否达到可用水平。
约束:检测数据规模上限约 COCO 级(63k 张);识别数据越大越好。

## 2. 数据集选型结论

| 用途 | 数据集 | 规模 | 标注 | 下载量(实测) | 状态 |
| --- | --- | --- | --- | ---: | --- |
| det | **COCO-Text** | 63,686 张,145,859 词 | 词级框(转 4 点 polygon) | 13.5GB(train2014 镜像)+ ~1.3GB(annot) | URL 已验证 |
| det/rec | **HierText** | 11,639 张(train 8281),~1.2M 词 | **词级多边形** + 行/段落 | 3.9GB(tgz 三个 split) | URL 已验证 |
| rec | **TextOCR** | 28,143 张,907,774 词 | 词级转写 | 7.1GB(图片)+ 3×json | URL 已验证 |
| rec(可选) | LSVT(ICDAR2019) | 30,000 张,~450k 框 | 中文+英文 | 约 20GB | 需 RRC 注册,无法脚本化,手动 |

组合规模:det ≈ 74k 张(COCO-Text + HierText,另可并入本机 CTW1500 1000 张);
rec ≈ 150 万真实词(TextOCR 907k + HierText 1.2M 去重后,另可并入 COCO-Text
~142k 可读词)。

## 3. 存储评估(2026-08-18 实测)

```text
目标盘: /home(10.122.89.54:/ifs/car_home,NFS,共享)
总量 20T / 已用 19T(91%) / 可用 1.9T
需求:  下载 26GB + 解压 ~62GB ≈ 90GB(保守)
结论:  可用 1.9T >> 90GB,存储充足;但 NFS 共享盘整体 91% 使用率,下载期间
       关注其他占用(du 全量扫描很慢,建议只对数据集目录做增量统计)。
```

## 4. 下载(脚本 + tmux)

脚本位于 `tools/data/`(均已用 `bash -n` 检查):

| 脚本 | tmux 会话 | 内容 |
| --- | --- | --- |
| `download_cocotext.sh` | dl-cocotext | cocotext.v2.zip(GitHub release)+ train2014.zip(pjreddie 镜像) |
| `download_hiertext.sh` | dl-hiertext | images/{train,validation,test}.tgz + annotations/*.jsonl.gz |
| `download_textocr.sh` | dl-textocr | train_val_images.zip + TextOCR_0.1_{train,val,test}.json |
| `download_all.sh` | — | 存储检查 + 启动以上三个会话;`--check` 只查存储 |

用法:

```bash
bash tools/data/download_all.sh --check   # 先检查存储
bash tools/data/download_all.sh           # 启动三个 tmux 下载会话
tmux attach -t dl-cocotext                # 查看进度(Ctrl-b d 退出)
tail -f /home/heqi/dataset/logs/dl-cocotext.log
```

注意:

- 下载均 `wget -c` 断点续传 + 失败重试,可放心中断/重启;
- `images.cocodataset.org`(COCO 官方)在本机不可达(实测 000),train2014 改用
  pjreddie 镜像 `https://data.pjreddie.com/files/train2014.zip`(已验证 200);
- `DATASET_ROOT` 环境变量可覆盖默认 `/home/heqi/dataset`;
- LSVT 需在 https://rrc.cvc.uab.es/?ch=16 注册后手动下载(待用户决定)。

## 5. 目录布局与解压

```text
/home/heqi/dataset/
  cocotext/   cocotext.v2.zip、train2014.zip、train2014/、COCO_Text.json 等
  hiertext/   images/{train,validation,test}.tgz 与解压目录、annotations/*.jsonl.gz
  textocr/    images/train_val_images.zip 与解压目录、annotations/TextOCR_0.1_*.json
  logs/       dl-*.log
```

解压命令(下载完成后执行):

```bash
cd /home/heqi/dataset/cocotext && unzip -q cocotext.v2.zip && unzip -q train2014.zip
cd /home/heqi/dataset/hiertext/images && tar -xzf train.tgz -C . && tar -xzf validation.tgz -C . && tar -xzf test.tgz -C .
cd /home/heqi/dataset/hiertext && gzip -d annotations/*.jsonl.gz
cd /home/heqi/dataset/textocr/images && unzip -q train_val_images.zip
```

## 6. 格式转换计划(JSON → PaddleOCR 格式)

统一转换为 PaddleOCR 格式后直接喂 `tools/train.py`:

- det:`image_path<TAB>[{"transcription":"...","points":[[x,y],...]}]`(≥3 点 polygon);
- rec:`image_path<TAB>text`。

| 数据集 | 转换要点 |
| --- | --- |
| COCO-Text | 词级框(左上/右下)转 4 点 polygon;过滤 `legible=0`、`###`(含非字母数字)、无效框;`utf8_string` 作为 transcription;图片子集 = train2014 中出现的图片 |
| HierText | 取 word 级 polygon + `text`;过滤 `legible=False`/`###`/空文本;rec 用 word 裁剪(按 polygon 最小外接矩形 + 方向校正可选) |
| TextOCR | 词级 polygon + text;过滤 `###`/空文本;图片 = train_val_images.zip 中对应文件 |

过滤规则按数据合同(AGENTS.md):标签长度 ≤ `max_text_length`(v5 rec 为 25?),
字符须在对应字典内,全部字符未知的样本过滤;超长样本裁剪或丢弃。

转换脚本:待下载完成后新增 `tools/data/convert_*.py`(一个脚本处理一个数据集,
输出 train/val 两份标注,val 从各自官方 val/test 划分取,避免与训练重叠)。

## 7. 训练节奏与验收

1. **float 大数据训练**(det: COCO-Text+HierText 74k 张;rec: TextOCR+HierText
   ~150 万词),使用现有 `tools/train.py` 与模型 YAML,关闭增强/固定 shape 合同
   不变;验收:det hmean、rec acc 显著高于 ICDR 基线(v5 det 0.165、v5 rec
   float 0.5936);
2. **QAT(exp16 合同:U8/S8 + 下采样 S16 + LSQ)**:浮点达标后,用大数据全量或
   子集启动 QAT;先 smoke(随机输入 + 结构检查),再小规模 1 epoch 全链路验证,
   最后正式多 epoch;
3. **验证口径**:保留 ICDR2015 英文口径(rec_gt_test 2077 张)作回归;新增
   TextOCR val / HierText validation 作第二口径;
4. 每轮实验按合同记录命令、profile、指标、产物路径(不记录哈希)。

## 8. 风险与备选

1. NFS 共享盘 91% 使用率:下载前用 `download_all.sh --check` 复核;若空间被
   其他任务占用导致不足,暂停并报告;
2. COCO-Text 图片走 pjreddie 镜像(官方域名本机不可达),若镜像也失败,备选:
   只做 HierText+TextOCR(不依赖 COCO train2014),det 规模降到 11.6k 张;
3. TextOCR 图片与 HierText 图片均来自 Open Images,但为不同文件子集,无共享;
4. LSVT 需要注册下载,中文覆盖是后续可选增强,不影响英文主线;
5. 转换脚本与训练入口的标签格式差异(如 polygon 点数、坐标精度)以
   `pytorchocr/training/data/` 现有实现为准,转换后先跑 10 张样本 smoke。

## 9. 待办清单

- [x] 存储检查(1.9T 可用,需求 ~90GB,充足);
- [x] 下载脚本(tmux,`tools/data/`,URL 已实测);
- [ ] 运行 `download_all.sh` 启动三个下载会话;
- [ ] 解压 + gzip -d;
- [ ] 编写转换脚本 `tools/data/convert_cocotext.py` / `convert_hiertext.py` /
      `convert_textocr.py`(JSON → PaddleOCR 格式,含过滤规则);
- [ ] 转换后 smoke(10 张样本)确认与 train.py 兼容;
- [ ] float 大数据训练 + ICDR 回归 + 新口径验收;
- [ ] QAT(exp16 合同)smoke → 1 epoch 验证 → 正式训练;
- [ ] (可选)LSVT 注册下载,补中文覆盖。
