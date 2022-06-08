from statistics import mode
import sys
import torch
import random

from dataCenter import *
from layer import *
from options import args_parser
from model import *


args = args_parser()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print("using device", device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
# device = "cuda:1"
print("DEVICE:", device, flush=True)

if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=sys.maxsize)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mode = args.mode
    h_feats = args.h_feats
    lr = args.lr
    batch_size = args.batch_size

    # load data
    dataCenter = DataCenter(mode, device)
    dataCenter.load_dataSet(batch_size)
    print("Load Data Finished ", flush=True)

    train_loader = getattr(dataCenter, "train_loader")
    test_loader = getattr(dataCenter, "test_loader")
    in_feats = getattr(dataCenter, "in_feats")
    num_classes = getattr(dataCenter, "num_classes")

    # init models
    model = Model(train_loader, mode, in_feats, h_feats,
                  num_classes, lr, device).to(device)

    test_f1 = 0
    test_f1_list = []
    max_test_f1 = 0

    """开始训练"""
    for epoch in range(1, args.epochs + 1):
        """开始训练"""
        model.train()
        loss = model.supervisedTrain()

        """test"""
        with torch.no_grad():
            model.eval()
            test_f1 = model.test(test_loader)
            test_f1_list.append(test_f1)
            max_test_f1 = max(max_test_f1, test_f1_list[-1])

        """打印结果"""
        print("-----epoch", epoch, "test f1:",
              test_f1_list[-1], " -----", flush=True)

    print("max f1:", max_test_f1)
