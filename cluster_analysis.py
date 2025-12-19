import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain # 对应 python-louvain 库
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime, timedelta

# ==========================================
# 1. 模拟数据生成 (如果你有真实数据，请读取CSV替代此部分)
# ==========================================
def generate_mock_data(num_users=50, num_comments=200):
    users = [f"User_{i}" for i in range(num_users)]
    
    # 模拟正常用户：发帖时间随机，内容多样
    data = []
    base_time = datetime(2025, 11, 18, 19, 0, 0)
    
    # 模拟一个恶意集群 (Cluster A): 5个账号，在 19:10 分集中发帖，内容相似
    cluster_users = ['User_0', 'User_1', 'User_2', 'User_3', 'User_4']
    cluster_texts = ["必须严查此事！", "这件事必须严查", "严查！太离谱了", "必须给个说法，严查", "支持严查"]
    
    for _ in range(num_comments):
        user = random.choice(users)
        
        if user in cluster_users and random.random() > 0.3: # 集群账号有大概率表现异常
            # 异常行为：时间高度集中 (19:10:00 - 19:10:30) [cite: 61]
            time_offset = random.randint(600, 630) 
            text = random.choice(cluster_texts) # 文本高度相似 [cite: 73]
        else:
            # 正常行为：时间分散
            time_offset = random.randint(0, 3600)
            text = f"这是正常的评论内容_{random.randint(1, 100)}"
            
        post_time = base_time + timedelta(seconds=time_offset)
        data.append({"user_id": user, "content": text, "timestamp": post_time})
        
    return pd.DataFrame(data)

# ==========================================
# 2. 特征计算与相似度矩阵 [cite: 90-95]
# ==========================================
def build_similarity_matrix(df):
    users = df['user_id'].unique()
    n_users = len(users)
    user_idx = {u: i for i, u in enumerate(users)}
    
    # 初始化相似度矩阵
    sim_matrix = np.zeros((n_users, n_users))
    
    # A. 文本相似度 (使用 TF-IDF) [cite: 94]
    # 注意：真实中文环境建议先用 jieba 分词
    vectorizer = TfidfVectorizer()
    # 将每个用户的所有评论合并为一个长文本进行分析
    user_contents = df.groupby('user_id')['content'].apply(lambda x: " ".join(x)).reindex(users).fillna("")
    tfidf_matrix = vectorizer.fit_transform(user_contents)
    text_sim = cosine_similarity(tfidf_matrix)
    
    # B. 时间相似度 (简化版：计算平均发帖时间的接近度) [cite: 93]
    # 真实项目中可用时间序列分箱或 DTW
    avg_times = df.groupby('user_id')['timestamp'].apply(lambda x: np.mean([t.timestamp() for t in x])).reindex(users)
    time_sim = np.zeros((n_users, n_users))
    for i in range(n_users):
        for j in range(n_users):
            if i == j: continue
            diff = abs(avg_times[i] - avg_times[j])
            # 如果平均发帖时间差小于60秒，认为时间相似度极高
            time_sim[i][j] = 1.0 / (1.0 + diff/60.0) 

    # C. 综合加权 
    w1, w2 = 0.4, 0.6 # 权重可调整
    final_sim = w1 * time_sim + w2 * text_sim
    
    return users, final_sim

# ==========================================
# 3. 图构建与聚类 [cite: 106-110]
# ==========================================
def detect_communities(users, sim_matrix, threshold=0.6):
    G = nx.Graph()
    
    # 添加节点
    G.add_nodes_from(users)
    
    # 添加边：只有相似度超过阈值才连线 [cite: 107]
    n_users = len(users)
    for i in range(n_users):
        for j in range(i + 1, n_users):
            if sim_matrix[i][j] > threshold:
                G.add_edge(users[i], users[j], weight=sim_matrix[i][j])
    
    # 使用 Louvain 算法进行社区检测 
    # 注意：如果图是空的或没有边，可能会报错，需处理
    if G.number_of_edges() == 0:
        return G, {u: 0 for u in users}
        
    partition = community_louvain.best_partition(G)
    return G, partition

# ==========================================
# 4. 可视化 [cite: 151]
# ==========================================
def visualize_clusters(G, partition):
    pos = nx.spring_layout(G, k=0.5) # k越大节点越分散
    plt.figure(figsize=(10, 8))
    
    # 根据社区分类设置颜色
    cmap = plt.get_cmap('viridis')
    max_cluster = max(partition.values()) + 1
    colors = [cmap(partition[node] / max_cluster) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif') # 这里的字体可能需要根据系统调整以显示中文
    
    plt.title("Account Cluster Analysis Graph")
    plt.axis('off')
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("正在生成模拟数据...")
    df = generate_mock_data()
    print(f"生成了 {len(df)} 条评论数据。")
    
    print("正在计算相似度矩阵...")
    users, sim_matrix = build_similarity_matrix(df)
    
    print("正在进行图构建与社区检测...")
    G, partition = detect_communities(users, sim_matrix, threshold=0.7)
    
    # 打印结果：找到的疑似集群
    cluster_dict = {}
    for user, cluster_id in partition.items():
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(user)
    
    print("\n检测到的集群结果：")
    for cid, members in cluster_dict.items():
        if len(members) > 2: # 只显示超过2人的团簇
            print(f"集群 ID {cid}: 成员 {members}")
            
    print("\n正在绘图...")
    visualize_clusters(G, partition)