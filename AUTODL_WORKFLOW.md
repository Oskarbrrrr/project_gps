# AutoDL 协作工作流

这份文档约定一个简单稳定的工作模式：

- 本地负责改代码、看代码、整理问题
- AutoDL 负责装依赖、挂数据、正式训练、保存结果
- 代码改动统一通过 git 同步
- 不在本地和远程同时手工改同一份代码

## 1. 当前仓库信息

- 本地路径：`D:\code\project_gps`
- 远程仓库：`https://github.com/Oskarbrrrr/project_gps.git`

当前 `.gitignore` 已经忽略了这些大文件/目录：

- `Data/`
- `logs/`
- `checkpoints/`
- `*.pth`
- `*.pt`
- `*.npy`
- `*.npz`

所以正常情况下，代码可以安全提交，数据和训练产物不会被一起推上去。

## 2. 推荐工作闭环

每次都按这个顺序走：

1. 本地改代码
2. 本地查看改动
3. 本地提交 git
4. 本地 push 到 GitHub
5. AutoDL 上 `git pull`
6. AutoDL 上跑训练或可视化
7. 把报错、日志、结果图发回本地继续改

## 3. 本地常用命令

进入仓库：

```powershell
cd D:\code\project_gps
```

查看状态：

```powershell
git status
```

查看改了什么：

```powershell
git diff
```

添加改动：

```powershell
git add train.py src/dataset.py visualize_lidar.py AGENTS.md AUTODL_WORKFLOW.md
```

提交：

```powershell
git commit -m "fix lidar bev projection and dataset wiring"
```

推送：

```powershell
git push origin main
```

如果以后不是 `main` 分支，就把最后一条里的 `main` 改成对应分支名。

## 4. AutoDL 上常用命令

进入项目目录：

```bash
cd /root/autodl-tmp/project_gps
```

先确认当前分支：

```bash
git branch
```

拉取最新代码：

```bash
git pull origin main
```

运行训练：

```bash
python train.py
```

运行 LiDAR 可视化：

```bash
python visualize_lidar.py
```

如果你在 AutoDL 上用的是 conda 环境，先激活环境再跑：

```bash
conda activate your_env_name
```

## 5. 结果怎么带回来

因为 `logs/` 和 `checkpoints/` 被忽略了，所以训练结果默认不会进 git。

推荐带回本地的内容：

- 终端报错全文
- `logs/*.csv`
- `logs/*.txt`
- `logs/*.png`
- 关键 checkpoint 文件名

最简单的做法：

1. 在 AutoDL 上把 `logs/` 里的关键图和日志下载到本地
2. 或者直接把终端输出复制给我
3. 如果某个样本的 LiDAR 图有问题，把对应图片发回来

## 6. 强烈建议遵守的规则

### 规则 1：远程机不要手改代码

尽量只在本地改代码，然后 push，再让 AutoDL pull。

### 规则 2：每次跑之前先看 git 状态

在本地和远程都先执行：

```bash
git status
```

如果远程机出现你没见过的已修改文件，不要直接覆盖，先确认那是不是你以前在远程手改过的内容。

### 规则 3：一轮改动对应一次提交

比如：

- 修 dataset 接口一次提交
- 修 LiDAR BEV 一次提交
- 加可视化脚本一次提交

这样回看实验结果时，比较容易知道是哪版代码跑出来的。

## 7. 推荐提交信息写法

可以直接照着这类格式写：

- `fix dataset constructor usage`
- `fix lidar bev projection frame alignment`
- `add lidar bev visualization stats`
- `document autodl workflow`

## 8. 当前阶段最推荐的跑法

现在最适合你的节奏是：

1. 本地继续围绕 LiDAR BEV 改
2. 每次只改一个小点
3. push 到 GitHub
4. AutoDL 上只跑：
   - `visualize_lidar.py`
   - 或少量样本测试
   - 或单场景训练
5. 看结果再决定下一步

先把 BEV 的方向、稀疏度、帧间差异跑顺，比一上来完整训练三整个场景更省时间。
