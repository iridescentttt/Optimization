p_list=(0.01 0.1 1 10 100)
for p in "${p_list[@]}"
do
# nohup python3 main.py --mode=pg --p=$p>result/pg/$p.txt 2>&1 &
# nohup python3 main.py --mode=sg --p=$p>result/sg/$p.txt 2>&1 &
nohup python3 main.py --mode=admm --p=$p >result/admm/$p.txt 2>&1 &
done