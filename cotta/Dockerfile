# 步骤 1: 选择一个包含 Miniconda 的基础镜像
# 这是一个官方维护的、干净的 Miniconda 环境
FROM continuumio/miniconda3:latest

# 步骤 2: 安装必要的系统工具
# 我们需要 git，因为你的 environment.yml 中有一个从 GitHub 安装的 pip 包
# procps 是为了方便在容器内使用 ps 等命令，便于调试
RUN apt-get update && apt-get install -y \
    git \
    procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 步骤 3: 设置工作目录
# 之后的所有操作都会在这个目录下进行
WORKDIR /app

# 步骤 4: 将环境配置文件复制到镜像中
# 这个文件需要和 Dockerfile 放在同一个目录下
COPY environment.yml .

# 步骤 5: 使用 conda 创建环境
# conda env create 会读取 yml 文件并安装所有指定的依赖
# 这会是构建过程中最耗时的一步
RUN conda env create -f environment.yml

# 步骤 6 (可选，但推荐): 清理 conda 缓存以减小镜像体积
RUN conda clean -af

# 步骤 7: 配置 shell，让之后的所有命令默认在 cotta 环境中运行
# 这是一个很方便的技巧，避免每次都要手动 conda activate
SHELL ["conda", "run", "-n", "cotta", "/bin/bash", "-c"]

# 步骤 8: 设置容器启动后的默认命令
# 这里我们设置为启动一个 bash shell，这样你就可以直接进入容器进行交互
# 因为上面的 SHELL 指令，这个 bash 已经是在 cotta 环境中了
CMD ["/bin/bash"]