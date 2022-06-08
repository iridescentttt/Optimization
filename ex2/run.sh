# nohup python3 main.py --mode=SGD --lr=0.001 --h_feats=256 --batch_size=8 >SGD_8.txt 2>&1 &
# nohup python3 main.py --mode=SGD --lr=0.001 --h_feats=256 --batch_size=16 >SGD_16.txt 2>&1 &
# nohup python3 main.py --mode=SGD --lr=0.001 --h_feats=256 --batch_size=32 >SGD_32.txt 2>&1 &
# nohup python3 main.py --mode=SGD --lr=0.001 --h_feats=256 --batch_size=64 >SGD_64.txt 2>&1 &
# nohup python3 main.py --mode=SGD --lr=0.001 --h_feats=256 --batch_size=128 >SGD_128.txt 2>&1 &

nohup python3 main.py --mode=GD --lr=0.1 --h_feats=256 >GD.txt 2>&1 &

nohup python3 main.py --mode=SAG --lr=0.001 --h_feats=256 >SAG.txt 2>&1 &