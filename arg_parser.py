import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="TrustGuard: 基于图卷积网络的信任预测模型")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="../data/bitcoinotc-rating.txt",  # bitcoinalpha-rating.txt
                        help="Edge list txt.")

    parser.add_argument("--data_path",
                        nargs="?",
                        default="../data/bitcoinotc.csv",  # bitcoinalpha.csv
                        help="Original dataset that covers time information.")

    parser.add_argument("--single_prediction",
                        type=bool,
                        default=True,
                        help="For single-timeslot prediction or multi-timeslot prediction.")

    parser.add_argument("--time_slots",
                        type=int,
                        default=10,
                        help="Total of timeslots.")

    parser.add_argument("--train_time_slots",
                        type=int,
                        default=2,  # set as 2 to make sure that we have at least two snapshots for learning temporal patterns
                        help="Number of training timeslots.")

    parser.add_argument("--seed",
                        type=int,
                        default=42)  # 40-44

    parser.add_argument("--attention_head",
                        type=int,
                        default=16,  # 8 for BitcoinOTC, 16 for BitcoinAlpha
                        help="Number of attention heads for self-attention in the temporal aggregation layer.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 32 32.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.005,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10 ** -5,
                        help="权重衰减系数用于L2正则化。默认值为10^-5。")

    parser.add_argument("--regularization-type", 
                        type=str,
                        default="L2",
                        choices=["L1", "L2", "ELASTIC_NET", "none"],
                        help="正则化类型，可选：L1、L2、ELASTIC_NET或none。")
    
    parser.add_argument("--l1-lambda", 
                        type=float,
                        default=1e-5,
                        help="L1正则化系数，默认为1e-5。")
    
    parser.add_argument("--l2-lambda", 
                        type=float,
                        default=1e-4,
                        help="L2正则化系数（当不使用weight_decay时）。默认为1e-4。")

    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="训练轮数。默认为50。")

    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability) of spatial aggregation.')

    parser.add_argument("--use_adaptive_agg", 
                      type=bool,
                      default=True,
                      help="Whether to use adaptive aggregation.")

    parser.add_argument("--elastic-net-ratio",
                        type=float,
                        default=0.5,
                        help="弹性网络的混合比例，0到1之间。")

    parser.add_argument("--use-smote",
                        action="store_true",
                        help="Whether to use SMOTE to handle imbalanced data.")

    parser.add_argument("--smote-strategy",
                        type=str,
                        default="auto",
                        choices=["auto", "minority", "not majority", "all"],
                        help="SMOTE sampling strategy.")

    parser.add_argument("--random-seed",
                        type=int,
                        default=42,
                        help="Random seed for SMOTE.")

    # Node2Vec 相关参数
    parser.add_argument("--use-node2vec",
                        action="store_true",
                        help="是否使用Node2Vec代替随机初始化节点嵌入。")
    
    parser.add_argument("--node2vec-p",
                        type=float,
                        default=1.0,
                        help="Node2Vec的返回参数p。较低的值(< 1)偏向DFS，较高的值(> 1)偏向BFS。")
    
    parser.add_argument("--node2vec-q",
                        type=float,
                        default=1.0,
                        help="Node2Vec的进出参数q。较低的值鼓励对外探索，较高的值鼓励局部探索。")
    
    parser.add_argument("--node2vec-dimensions",
                        type=int,
                        default=64,
                        help="Node2Vec嵌入向量维度，建议与模型的隐藏层维度匹配。")
    
    parser.add_argument("--node2vec-walk-length",
                        type=int,
                        default=30,
                        help="Node2Vec每次随机游走的长度。较长的游走可以捕获更多的全局结构信息。")
    
    parser.add_argument("--node2vec-num-walks",
                        type=int,
                        default=200,
                        help="Node2Vec每个节点的随机游走次数。增加此值可以提高嵌入质量，但会增加计算时间。")
    
    parser.add_argument("--node2vec-window",
                        type=int,
                        default=10,
                        help="Node2Vec训练过程中的上下文窗口大小。较大的窗口可以捕获更长距离的依赖关系。")
    
    parser.add_argument("--node2vec-workers",
                        type=int,
                        default=4,
                        help="Node2Vec训练的并行线程数。根据CPU核心数调整。")
    
    parser.add_argument("--node2vec-use-cache",
                        action="store_true",
                        help="是否使用缓存的Node2Vec嵌入。可以加快重复实验的速度。")
    
    parser.add_argument("--node2vec-weighted",
                        action="store_true",
                        help="是否在Node2Vec中考虑边权重。启用后将使用信任值作为边权重。")
    
    parser.add_argument("--node2vec-normalize",
                        action="store_true",
                        help="是否对生成的Node2Vec嵌入进行归一化处理。")

    parser.set_defaults(layers=[32,64,32])  # hidden embedding dimension

    return parser.parse_args()
